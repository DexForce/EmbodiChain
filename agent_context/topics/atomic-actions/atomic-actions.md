# Atomic actions

## Current contract

Atomic actions are side-effect-free, environment-batched planners:

```python
plan = engine.plan(invocation: ActionInvocation, context: PlanningContext)
```

There is no `ActionTarget`, `WorldState`, `ActionResult`, `execute()`, or
`AtomicActionEngine.run()` compatibility surface.

`ActionInvocation` separates:

- an action-owned typed goal (`goal_kind` is its stable discriminator);
- `ActionBinding`, which maps semantic roles to names from the engine robot's
  `control_parts` mapping;
- reusable `MotionPolicy` planner/timing choices;
- bounded `RecoveryPolicy` thresholds and retry budgets;
- optional typed `skill_options` and role-scoped `control_overrides` for one
  invocation revision.

`PlanningContext` separates measured `RobotObservation`, verified symbolic
`TaskState`, versioned `SceneSnapshot`, and environment IDs. An `ActionPlan`
contains per-environment planning success, one or more `PlannedPhase` objects,
full-robot `TimedTrajectory` data, diagnostics, completion conditions, and an
uncommitted `StateDelta`.

Each `AtomicActionEngine` exclusively owns one `ActionPlanningServices`
instance, which contains its robot, motion generator, planner backend, shared
`TrajectoryBuilder`, and control-part command profiles. Actions retain only an
owned copy of typed default options and borrow engine services after
`engine.register(action)` or
`engine.plan_action(action, invocation, context)`. A bound action cannot be
reused by another engine.

## Static compilation

Register configured action instances by their class-level stable `skill_id`,
then call:

```python
compiled = engine.compile(invocations, context=None)
```

Compilation does not step simulation. It concatenates timed trajectories and
applies successful expected effects only to `compiled.projected_context`, so a
following action can be checked against hypothetical state. Failed rows hold
their last successful qpos.

Use `engine.plan_action(...)` for an unregistered configured instance when an
application needs multiple variants with the same stable `skill_id`; the action
still uses the engine's single motion generator.

## Dynamic execution and recovery

`SceneEntityPose(entity_id, relative_pose)` is resolved from the latest scene
snapshot every time the action plans. Its entity ID is recorded in
`PhaseSpec.scene_dependencies`.

```python
session = engine.start(invocations, initial_context)
tick = session.tick(latest_context, effect_success=None)
```

An `ExecutionSession` emits at most one `JointCommand` per tick and monitors:

- joint tracking error against the previous command;
- translation/rotation drift of referenced scene entities;
- phase timeout;
- planner and semantic-effect failure.

It replans from the latest observation within per-environment budgets. Unknown
or exhausted failures are reported as structured `ExecutionEvent` objects. A
non-empty `StateDelta` is not committed until the caller supplies an external
`effect_success` mask.

Recovery replans reuse the current immutable `ResolvedActionRequest`. To change
a goal, option, policy, binding, or control command during execution, submit a
strictly newer revision explicitly:

```python
session.revise_current(revised_invocation)
```

The replacement must keep the active `skill_id` and `invocation_id`. The
session resolves a new snapshot, resets that revision's recovery budgets, and
replans from the latest context.

## Parameter ownership

Goal dataclasses carry only semantic task intent. They do not carry robot part
names, planner configuration, retry policy, or runtime state.

`MotionPolicy` owns planner selection, motion source, sample count, fallback
control period, limits, and typed planner options. `RecoveryPolicy` owns
tracking/dynamic-goal thresholds, timeouts, and budgets. Each built-in has a
frozen `*Options` value for invocation-varying phase counts, offsets, and grasp
selection behavior. An action constructor may accept `default_options`; an
invocation's `skill_options` replaces them for that call. There is no
`ActionCfg` or built-in `*Cfg` layer.

Every `ActionBinding` value is a `RobotCfg.control_parts` key. It is not a link,
TCP-frame, joint, or scene-object name. Planning services validate those names
and resolve immutable `ResolvedControlPart` values containing full-robot joint
indices. Built-ins use the binding as the only source for participating arm and
hand names; attachment state and `StateDelta` keys use the bound manipulator.

Embodiment-specific joint commands do not belong to Action options. Register
them once by actual control-part name:

```python
engine = AtomicActionEngine(
    motion_generator,
    control_profiles={
        "left_hand": ControlPartCommandProfile.joint_positions(
            open=left_open_qpos,
            grasp=left_grasp_qpos,
        ),
        "left_arm": ControlPartCommandProfile.joint_positions(ready=ready_qpos),
    },
)
```

Actions request semantic commands (`open`, `grasp`, or a named joint target)
from the `ResolvedControlPart`. `ActionControlOverrides` may replace commands
by semantic binding role for one invocation revision. Joint limits constrain
commands but do not define semantic open/grasp states; a robot integration or
tutorial may derive a simple profile from limits explicitly.

## Built-ins

| Skill ID | Goal type | Roles |
|---|---|---|
| `move_end_effector` | `EndEffectorPoseGoal` | manipulator `primary` |
| `move_joints` | `JointPositionGoal` (`target` is explicit qpos or a profile command name) | manipulator `primary` |
| `pick_up` | `GraspGoal` | manipulator/end effector `primary` |
| `move_held_object` | `HeldObjectPoseGoal` | manipulator/end effector `primary` |
| `place` | `PlaceGoal`, `AssembleGoal` | manipulator/end effector `primary` |
| `press` | `PressGoal` | manipulator/end effector `primary` |
| `coordinated_pickment` | `CoordinatedPickGoal` | `left`, `right` |
| `coordinated_placement` | `CoordinatedPlacementGoal` | `placing`, `support` |
| `hand_over` | `GraspGoal` | `source`, `destination` |

## Extension rules

1. Define a frozen action-owned goal dataclass with `goal_kind`.
2. Define a frozen `ActionOptions` subclass only when runtime behavior exists.
3. Declare `skill_id`, `GoalType`, `OptionsType`, and required semantic roles.
4. Validate with `require_goal(request)` and consume only the resolved binding.
5. Plan from `context.robot.qpos`; never read an implicit live start state.
6. Return full-robot positions or a `TimedTrajectory` through `build_plan()`.
7. Declare symbolic changes with `StateDelta`; do not mutate context or commit
   physical effects during planning.
8. Keep scene stepping, controller I/O, and task-graph/MLLM logic outside the
   atomic action.
