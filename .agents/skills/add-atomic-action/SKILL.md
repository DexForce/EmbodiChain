---
name: add-atomic-action
description: Add a new simulation atomic action or motion primitive to EmbodiChain's typed planning and execution framework. Use when implementing a new skill, goal contract, action planner, symbolic effect, registration entry, documentation, and tests for AtomicActionEngine.
---

# Add Atomic Action

Add an action-owned goal and a side-effect-free `AtomicAction._plan()`
implementation. The inherited public `plan()` entry point binds the current
collision scene before calling the skill hook. The engine owns all
motion-planning resources; action constructors accept only optional typed
default options. Keep task-graph/MLLM logic, simulator stepping, controller
I/O, and physical-effect commits outside the action.

## Read the current contracts

Inspect only the files relevant to the requested skill:

| Purpose | Path |
|---|---|
| Base action and descriptors | `embodichain/lab/sim/atomic_actions/core.py` |
| Goals and dynamic pose references | `embodichain/lab/sim/atomic_actions/goals.py` |
| Skill endpoint requirements | `embodichain/lab/sim/atomic_actions/requirements.py` |
| Resolved endpoint bindings and targets | `embodichain/lab/sim/atomic_actions/bindings.py` |
| Invocation, options, and resolved request | `embodichain/lab/sim/atomic_actions/invocation.py` |
| Control-part semantic commands | `embodichain/lab/sim/atomic_actions/control.py` |
| Invocation policies | `embodichain/lab/sim/atomic_actions/policies.py` |
| Robot/task/scene state | `embodichain/lab/sim/atomic_actions/state.py` |
| Dynamic scene provider contract | `embodichain/lab/sim/atomic_actions/scene.py` |
| Effects and plans | `embodichain/lab/sim/atomic_actions/effects.py`, `plans.py` |
| Runtime command frames and payloads | `embodichain/lab/sim/atomic_actions/runtime_commands.py` |
| Endpoint command transports | `embodichain/lab/sim/atomic_actions/transports.py` |
| Trajectory helpers | `embodichain/lab/sim/atomic_actions/trajectory_ops.py` |
| Engine-owned planning resources | `embodichain/lab/sim/atomic_actions/runtime.py` |
| Declarative robot resources and adapters | `embodichain/lab/task_program/semantics/profiles.py` |
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
invocation. Examples include distances, grasp constraints, and segment split
counts. If no such behavior exists, use the base `ActionOptions`.

Do not put `strategy`, planner choice, sample count, control period,
velocity limits, collision policy, or recovery thresholds in skill options;
those belong to `MotionPolicy` or `RecoveryPolicy`.

```python
from dataclasses import dataclass

from embodichain.lab.sim.atomic_actions import ActionOptions


@dataclass(frozen=True, slots=True, eq=False)
class PushOptions(ActionOptions):
    push_distance: float = 0.05
```

Do not put arm/hand names, hand qpos, or named robot postures in options.
Declare robot-independent participant slots and endpoints with
`SkillBindingContract`; the engine or a bound robot skill profile produces the
engine-owned `ActionBinding`. Register embodiment-specific commands such as
`open`, `grasp`, or `ready` on `ControlPartCommandProfile`; use
`ActionControlOverrides` only for one invocation revision.

## 3. Implement the planner

Inherit `AtomicAction[PushGoal, PushOptions]` directly. Declare stable metadata
and an explicit, robot-independent endpoint contract. Every concrete action
class must declare `binding_contract` in its own class body; use
`SkillBindingContract()` for a skill that consumes no robot resource.

```python
from typing import ClassVar

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    AtomicAction,
    CARTESIAN_POSE_CAPABILITY,
    JointPositionTarget,
    PlanningContext,
    ResolvedActionRequest,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    StateDelta,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    to_full_robot_trajectory,
)


class Push(AtomicAction[PushGoal, PushOptions]):
    skill_id: ClassVar[str] = "push"
    GoalType: ClassVar[type] = PushGoal
    OptionsType: ClassVar[type] = PushOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    ),
                ),
            ),
        ),
    )

    def __init__(self, default_options: PushOptions | None = None) -> None:
        super().__init__(default_options)

    def _plan(
        self,
        request: ResolvedActionRequest[PushGoal, PushOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        motion_target = request.binding.endpoint(
            "primary", "motion"
        ).require_target(JointPositionTarget)
        control_part = motion_target.control_part
        joint_ids = list(motion_target.joint_ids)
        start_qpos = context.robot.qpos[:, joint_ids]
        target_poses = goal.contact_pose

        # Build planner states and generate controlled-joint motion using
        # request.motion_policy. Embed it into full robot DoF.
        result = self.motion_generator.generate(
            build_pose_plan_states(target_poses),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_qpos,
                control_part=control_part,
            ),
        )
        success, trajectory = to_full_robot_trajectory(
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

- Let the engine supply `self.robot` and `self.motion_generator`; use
  `_on_bind()` only for robot/device-dependent setup.
- Keep slot and endpoint IDs semantic and robot-independent. Declare all-of
  capabilities, required typed commands, and disjointness constraints in the
  `SkillBindingContract`; do not infer resources from endpoint names.
- Resolve an endpoint with `request.binding.endpoint(slot_id, endpoint_id)` and
  call `require_target(ExpectedTarget)` before using target-specific fields.
- Import pure target-shaping, interpolation, pose-translation, and full-robot
  embedding helpers directly from `atomic_actions.trajectory_ops`; keep
  stateful planning inside `MotionGenerator`.
- Call `require_goal()` before planning.
- Implement `_plan()` rather than overriding the framework-owned public
  `plan()` method; the latter injects the latest dynamic obstacle poses into a
  copied planner policy.
- Plan from `context.robot.qpos`, never an implicit live robot start state.
- For joint-backed motion, return full-robot `(B, N, robot.dof)` motion as a
  tensor or `TimedTrajectory` with matching `env_ids` through `build_plan()`.
- Preserve row-local planner success. `build_plan()` normalizes the mask and
  replaces unsuccessful trajectory rows with the context's observed qpos.
- Preserve backend timing/derivatives when available.
- For a composite trajectory, pass an ordered `segment_lengths` mapping with
  the actual returned waypoint counts. Segments are inspection metadata, not
  independent planning or recovery boundaries.
- Return `failed_plan(request, context, message=...)` for an expected soft
  planning failure.
- Never mutate the context, step simulation, send commands, or claim a physical
  effect occurred.
- Declare attachment/task changes with `StateDelta`; the execution runtime
  applies them only after verification.
- Set `scene_dependencies` indirectly by using `SceneEntityPose` in the goal;
  `build_plan()` records them for dynamic invalidation.
- Do not add dynamic-obstacle arguments to a skill. A `SceneProvider` declares
  `collision_entity_ids`; supported planners receive those entity poses through
  the framework-owned `plan()` entry point.

## 4. Emit generic runtime commands when needed

Use `build_command_plan()` when a skill targets a mobile base, whole-body
controller, tool, or another non-joint transport. Build immutable endpoint
commands; keep live controller and device handles in the transport:

```python
target = request.binding.endpoint("primary", "tool").require_target(ToolTarget)
frames = tuple(
    RuntimeCommandFrame(
        commands=(EndpointCommand(target=target, payload=ToolPayload(value)),),
        active_mask=torch.ones(
            context.batch_size,
            dtype=torch.bool,
            device=context.robot.qpos.device,
        ),
        env_ids=context.env_ids,
        hold_duration=torch.full(
            (context.batch_size,),
            request.motion_policy.control_dt,
            device=context.robot.qpos.device,
        ),
    )
    for value in command_values
)
return self.build_command_plan(
    request,
    context,
    success=success,
    commands=TimedCommandSequence(frames=frames, env_ids=context.env_ids),
)
```

For a new transport kind:

1. Define an immutable `RuntimeEndpointTarget` and `RuntimeCommandPayload` with
   the same stable `transport_id`; both must return independently owned
   snapshots. Payloads also expose `batch_size` and `device`. If target-specific
   addressing or safe hold depends on fields beyond the exact target type,
   `transport_id`, and `target_id`, override `address_fingerprint` to include
   those immutable fields; frames, replans, and revisions preserve it.
2. If declarative robot profiles select it, define a `ResourceEndpoint` and an
   exact-type `ResourceEndpointAdapter` that returns `EndpointResolution` with
   the runtime target and physical claim metadata.
3. Implement `EndpointCommandTransport.send()`, `hold()`, and `cancel()`, then
   register it in `EndpointCommandRouter` used as the `ExecutionRunner` command
   sink. The router validates payload types before dispatch.

The default command-plan feedback mode is timed and `joint_trajectory` is
optional. Use joint-position feedback only when a matching full-robot
`joint_trajectory` is supplied. Test target/payload snapshot ownership, frame
batch/device consistency, routing, acknowledgement, hold, and cancel behavior.

## 5. Register and invoke

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
binding = engine.bind_control_parts(
    "push",
    {"primary": {"motion": "left_arm"}},
)
invocation = ActionInvocation(
    skill_id="push",
    goal=PushGoal(contact_pose),
    binding=binding,
    motion_policy=MotionPolicy(sample_count=60),
    recovery_policy=RecoveryPolicy(max_replans=2),
)
compiled = engine.compile((invocation,))
```

For dynamic scene updates or online error recovery, create a session with
`engine.start(...)`, then connect it to observation, command, and clock ports
through `ExecutionRunner`. Use non-blocking `runner.step()` in an existing event
loop or `runner.run_until_blocked()` in a simple application.

`engine.bind_control_parts()` is the explicit direct-core path for joint-backed
control parts. When a `RobotSkillProfile` is installed, prefer
`engine.skill_profile.resolve("push", selections).action_binding` so capability,
command, resource-claim, and custom-adapter validation remain declarative.

## 6. Export and document

Export the goal, options, and action from:

1. `embodichain/lab/sim/atomic_actions/primitives/__init__.py`
2. `embodichain/lab/sim/atomic_actions/__init__.py`

Add the stable skill ID, goal, binding slots/endpoints, and effect to
`docs/source/overview/sim/atomic_actions/builtin_actions.md`. Update API docs for
new public classes. Do not create a compatibility re-export module or a closed
built-in-goal union.

### Task Program handoff

An Atomic Skill is not automatically a Task Program Semantic Call. If the user
also requests high-level declarative exposure, finish and test the Atomic Skill
contract first, then invoke `$add-semantic-call`. Prefer a registered Semantic
Call extension unless the concept is intentionally promoted to a stable
built-in language primitive.

If the new skill requires new logical resources, endpoints, capabilities, or
commands on a reusable robot, update them through
`$add-embodiment-component`; do not encode task or profile metadata in the
Atomic Action.

## 7. Test behavior

Add pure pytest tests under `tests/sim/atomic_actions/`. Cover:

- descriptor `skill_id`, `GoalType`, and explicit binding contract;
- invalid goal, wrong binding owner, and missing/extra endpoint rejection;
- per-environment planning success/failure masks;
- full-robot trajectory shape, `env_ids`, timing, and failed-row hold behavior;
- generic command target/payload ownership, frame batch/device consistency, and
  optional `joint_trajectory` behavior when the skill emits command frames;
- side-effect-free context handling;
- masked `StateDelta` application for task effects;
- `SceneEntityPose` replanning when the action accepts a dynamic goal;
- collision-world revision replanning when the action uses a dynamic-world
  planner;
- effect verification when the action declares a non-empty delta.

Run focused tests, format changed Python files with the pinned Black version,
then use the `pre-commit-check` skill before committing.

## Common mistakes

| Mistake | Required correction |
|---|---|
| Inherit another action | Inherit `AtomicAction` directly; compose helpers. |
| Add one generic target with many optional fields | Define a narrow action-owned goal. |
| Put hardware names in the goal | Declare semantic slots/endpoints and resolve an engine-owned binding. |
| Put arm/hand control-part names in skill options | Read typed runtime targets from bound endpoints. |
| Declare legacy role tuples on the action | Declare a class-local `SkillBindingContract`. |
| Use role-specific binding accessors | Use `binding.endpoint(...).require_target(...)`. |
| Construct a binding from role dictionaries | Use a bound skill profile, or `engine.bind_control_parts()` for the direct joint path. |
| Pass an arbitrary joint/link/TCP name to the direct path | `bind_control_parts()` values must be keys in `RobotCfg.control_parts`; add an endpoint adapter for another resource kind. |
| Put hand qpos or named robot postures in skill options | Register semantic commands on the concrete control-part profile. |
| Put planner/recovery knobs in skill options | Move them to invocation policies. |
| Pass a motion generator to each action | Pass it once to `AtomicActionEngine`; construct actions from default options only. |
| Read `robot.get_qpos()` inside `plan()` | Use `context.robot.qpos`. |
| Return an arm-only tensor | Embed into full robot DoF. |
| Collapse or manually mask batched planner success | Return the row-local mask; let `build_plan()` hold unsuccessful rows. |
| Model one action as independently recoverable phases | Return one trajectory with optional named segment metadata. |
| Mutate held state after planning | Declare a `StateDelta`. |
| Treat `plan_success` as physical success | Verify effects during execution. |
| Step the simulator from the action | Emit plans; connect execution through `ExecutionRunner`. |
| Put live controller handles in targets or payloads | Keep immutable addressing/data in values and own handles in the transport. |
| Force a non-joint endpoint into a fake trajectory | Emit typed frames with `build_command_plan()` and install its transport. |
| Override public `plan()` | Implement `_plan()` so scene binding cannot be bypassed. |
