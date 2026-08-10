---
name: add-atomic-action
description: Add a new simulation atomic action or motion primitive to EmbodiChain's typed planning and execution framework. Use when implementing a new skill, goal contract, action planner, symbolic effect, registration entry, documentation, and tests for AtomicActionEngine.
---

# Add Atomic Action

Add an action-owned goal and a side-effect-free `AtomicAction.plan()`
implementation. The engine owns all motion-planning resources; action
constructors accept only optional typed default options. Keep task-graph/MLLM
logic, simulator stepping, controller I/O, and physical-effect commits outside
the action.

## Read the current contracts

Inspect only the files relevant to the requested skill:

| Purpose | Path |
|---|---|
| Base action and descriptors | `embodichain/lab/sim/atomic_actions/core.py` |
| Goals and dynamic pose references | `embodichain/lab/sim/atomic_actions/goals.py` |
| Role-to-resource binding | `embodichain/lab/sim/atomic_actions/bindings.py` |
| Invocation, options, and resolved request | `embodichain/lab/sim/atomic_actions/invocation.py` |
| Control-part semantic commands | `embodichain/lab/sim/atomic_actions/control.py` |
| Invocation policies | `embodichain/lab/sim/atomic_actions/policies.py` |
| Robot/task/scene state | `embodichain/lab/sim/atomic_actions/state.py` |
| Effects and plans | `embodichain/lab/sim/atomic_actions/effects.py`, `plans.py` |
| Trajectory helpers | `embodichain/lab/sim/atomic_actions/trajectory.py` |
| Engine-owned planning resources | `embodichain/lab/sim/atomic_actions/runtime.py` |
| Reference implementations | `embodichain/lab/sim/atomic_actions/primitives/` |
| Static compiler and execution session | `engine.py`, `execution.py` |
| Controller-facing execution ports | `runner.py`, `sim_adapter.py` |

The public contract is:

```python
plan = engine.plan(invocation: ActionInvocation[Goal], context: PlanningContext)
```

Use `engine.plan_action(action, invocation, context)` for a configured action
that is intentionally not in the stable skill registry. Never pass a motion
generator to an action constructor.

Do not add compatibility code for `ActionTarget`, `WorldState`, `ActionResult`,
`execute()`, or `AtomicActionEngine.run()`.

## 1. Define the goal

Place a frozen action-owned dataclass beside the action. Add a stable
`goal_kind: ClassVar[str]`. Do not inherit a marker base merely for dispatch;
declare the accepted type on the action instead.

```python
from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True, slots=True, eq=False)
class PushGoal:
    goal_kind: ClassVar[str] = "push"
    contact_pose: torch.Tensor
```

Keep only semantic intent in the goal. Do not include arm/hand names, planner
options, retry counts, live state, or a generic optional field bag. Use
`SceneEntityPose` for a pose that must be resolved again when the scene moves.
Use `ObjectActionGoal` only when the shared `semantics` field is genuinely
required.

## 2. Define runtime options and control commands

Define a frozen `ActionOptions` subclass only when skill behavior may vary by
invocation. Examples include distances, grasp constraints, and phase split
counts. If no such behavior exists, use the base `ActionOptions`.

Do not put `motion_source`, planner choice, sample count, control period,
velocity limits, collision policy, or recovery thresholds in skill options;
those belong to `MotionPolicy` or `RecoveryPolicy`.

```python
from dataclasses import dataclass

from embodichain.lab.sim.atomic_actions import ActionOptions


@dataclass(frozen=True, slots=True, eq=False)
class PushOptions(ActionOptions):
    push_distance: float = 0.05
```

Do not put arm/hand names, hand qpos, or named robot postures in options. Bind
participants with `ActionBinding`. Register embodiment-specific commands such
as `open`, `grasp`, or `ready` on `ControlPartCommandProfile`; use
`ActionControlOverrides` only for one invocation revision.

## 3. Implement the planner

Inherit `AtomicAction[PushGoal, PushOptions]` directly. Declare stable metadata and resolve
resources from semantic binding roles.

```python
from typing import ClassVar

from embodichain.lab.sim.atomic_actions import (
    ResolvedActionRequest,
    ActionPlan,
    AtomicAction,
    PlanningContext,
    StateDelta,
)


class Push(AtomicAction[PushGoal, PushOptions]):
    skill_id: ClassVar[str] = "push"
    GoalType: ClassVar[type] = PushGoal
    OptionsType: ClassVar[type] = PushOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(self, default_options: PushOptions | None = None) -> None:
        super().__init__(default_options)

    def plan(
        self,
        request: ResolvedActionRequest[PushGoal, PushOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        options = request.skill_options
        manipulator = request.binding.manipulator("primary")
        control_part = manipulator.name
        joint_ids = list(manipulator.joint_ids)
        start_qpos = context.robot.qpos[:, joint_ids]

        # Build planner states and generate controlled-joint motion using
        # request.motion_policy. Embed it into full robot DoF.
        result = self.builder.generate_arm_plan(
            target_states,
            start_qpos,
            request.motion_policy.sample_count,
            control_part=control_part,
            arm_dof=manipulator.dof,
            cfg=request.motion_policy,
        )
        success, trajectory = self.builder.to_full_robot_trajectory(
            result,
            base_qpos=context.robot.qpos,
            joint_ids=joint_ids,
            env_ids=context.env_ids,
            control_dt=request.motion_policy.control_dt,
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=trajectory,
            expected_effects=StateDelta(),
        )
```

Follow these invariants:

- Let the engine supply `self.robot`, `self.motion_generator`, and the shared
  `self.builder`; use `_on_bind()` only for robot/device-dependent setup.
- Call `require_goal()` before planning.
- Plan from `context.robot.qpos`, never an implicit live robot start state.
- Return full-robot `(B, N, robot.dof)` motion as a tensor or
  `TimedTrajectory` with matching `env_ids`.
- Preserve backend timing/derivatives when available.
- Return `failed_plan(request, context, message=...)` for an expected soft
  planning failure.
- Never mutate the context, step simulation, send commands, or claim a physical
  effect occurred.
- Declare attachment/task changes with `StateDelta`; the execution runtime
  applies them only after verification.
- Set `scene_dependencies` indirectly by using `SceneEntityPose` in the goal;
  `build_plan()` records them for dynamic invalidation.

## 4. Register and invoke

Register an instance by its class-level `skill_id`:

```python
engine.register(Push())
```

Use the global registry only for discoverable third-party classes:

```python
register_action(Push)
```

Construct a grounded invocation explicitly:

```python
invocation = ActionInvocation(
    skill_id="push",
    goal=PushGoal(contact_pose),
    binding=ActionBinding(manipulators={"primary": "left_arm"}),
    motion_policy=MotionPolicy(sample_count=60),
    recovery_policy=RecoveryPolicy(max_replans=2),
)
compiled = engine.compile((invocation,))
```

For dynamic scene updates or online error recovery, create a session with
`engine.start(...)`, then connect it to observation, command, and clock ports
through `ExecutionRunner`. Use non-blocking `runner.step()` in an existing event
loop or `runner.run_until_blocked()` in a simple application.

## 5. Export and document

Export the goal, options, and action from:

1. `embodichain/lab/sim/atomic_actions/primitives/__init__.py`
2. `embodichain/lab/sim/atomic_actions/__init__.py`

Add the stable skill ID, goal, roles, and effect to
`docs/source/overview/sim/atomic_actions/builtin_actions.md`. Update API docs for
new public classes. Do not create a compatibility re-export module or a closed
built-in-goal union.

## 6. Test behavior

Add pure pytest tests under `tests/sim/atomic_actions/`. Cover:

- descriptor `skill_id`, `GoalType`, and required roles;
- invalid goal and missing binding rejection;
- per-environment planning success/failure masks;
- full-robot trajectory shape, `env_ids`, timing, and failed-row hold behavior;
- side-effect-free context handling;
- masked `StateDelta` application for task effects;
- `SceneEntityPose` replanning when the action accepts a dynamic goal;
- effect verification when the action declares a non-empty delta.

Run focused tests, format changed Python files with the pinned Black version,
then use the `pre-commit-check` skill before committing.

## Common mistakes

| Mistake | Required correction |
|---|---|
| Inherit another action | Inherit `AtomicAction` directly; compose helpers. |
| Add one generic target with many optional fields | Define a narrow action-owned goal. |
| Put hardware names in the goal | Bind semantic roles through `ActionBinding`. |
| Put arm/hand control-part names in skill options | Use `ActionBinding` as their only source. |
| Bind a joint, link, TCP frame, or arbitrary name | Every binding value must be a key in `RobotCfg.control_parts`. |
| Put hand qpos or named robot postures in skill options | Register semantic commands on the concrete control-part profile. |
| Put planner/recovery knobs in skill options | Move them to invocation policies. |
| Pass a motion generator to each action | Pass it once to `AtomicActionEngine`; construct actions from default options only. |
| Read `robot.get_qpos()` inside `plan()` | Use `context.robot.qpos`. |
| Return an arm-only tensor | Embed into full robot DoF. |
| Mutate held state after planning | Declare a `StateDelta`. |
| Treat `plan_success` as physical success | Verify effects during execution. |
| Step the simulator from the action | Emit plans; connect execution through `ExecutionRunner`. |
