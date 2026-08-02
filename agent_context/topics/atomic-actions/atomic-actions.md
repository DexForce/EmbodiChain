# Atomic actions

## Current contract

Atomic actions are side-effect-free, environment-batched planners:

```python
plan = action.plan(invocation: ActionInvocation, context: PlanningContext)
```

There is no `ActionTarget`, `WorldState`, `ActionResult`, `execute()`, or
`AtomicActionEngine.run()` compatibility surface.

`ActionInvocation` separates:

- an action-owned typed goal (`goal_kind` is its stable discriminator);
- `ActionBinding`, which maps semantic roles to concrete robot resources;
- reusable `MotionPolicy` planner/timing choices;
- bounded `RecoveryPolicy` thresholds and retry budgets.

`PlanningContext` separates measured `RobotObservation`, verified symbolic
`TaskState`, versioned `SceneSnapshot`, and environment IDs. An `ActionPlan`
contains per-environment planning success, one or more `PlannedPhase` objects,
full-robot `TimedTrajectory` data, diagnostics, completion conditions, and an
uncommitted `StateDelta`.

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

## Dynamic execution and recovery

`SceneEntityPose(entity_id, relative_pose)` is resolved from the latest scene
snapshot every time the action plans. Its entity ID is recorded in
`PhaseSpec.scene_dependencies`.

```python
session = engine.start(invocations, initial_context)
runner = ExecutionRunner(
    session,
    observation_provider,
    command_sink,
    clock=execution_clock,
)
result = runner.step(effect_success=None)
```

`ExecutionSession` owns deterministic planning progress and recovery state. It
emits at most one `JointCommand` per tick and monitors:

- joint tracking error against the previous command;
- translation/rotation drift of referenced scene entities;
- phase timeout;
- planner and semantic-effect failure.

It replans from the latest observation within per-environment budgets. Unknown
or exhausted failures are reported as structured `ExecutionEvent` objects. A
non-empty `StateDelta` is not committed until the caller supplies an external
`effect_success` mask.

`ExecutionRunner` owns the controller-facing lifecycle around a session:

- `ObservationProvider.observe(task_state)` supplies a fresh, monotonically
  timestamped `PlanningContext` when a feedback cycle is due;
- `CommandSink.send/hold/cancel` returns a `CommandAcknowledgement` with
  `accepted`, `rejected`, or `timed_out` status;
- `ExecutionClock` supplies monotonic time and backend waiting;
- non-blocking `step()` dispatches only when the current command's
  `hold_duration` has elapsed;
- `run_until_blocked()` is a convenience loop that waits through the clock and
  stops at a terminal state or an unhandled effect-verification boundary; the
  runner remembers that boundary so a later verifier call can resume it;
- cancellation, observation/session exceptions, and negative acknowledgements
  enter a best-effort cancel-then-hold path.

`TimedTrajectory.dt[:, i]` is the interval leading to sample `i`.
`ExecutionSession` maps this to `JointCommand.hold_duration`; the final sample
uses its own interval as a settling window before terminal validation. Batched
execution currently advances at a synchronized barrier using the longest active
row interval.

`SimulationExecutionAdapter` implements observation, command, and clock ports
for a `SimulationManager`/`Robot` pair. Its `sleep()` advances an integral
number of physics steps, so simulation execution does not depend on wall time.
Stable context IDs are correlation identifiers; the adapter maps command rows
to simulation robot indices rather than using those IDs as array indices.
Real-device adapters should implement the same protocols and enforce the passed
acknowledgement timeout in their transport/controller layer.

The latest validated session context is retained for safe hold if the first
live observation fails. Environment IDs must remain stable and ordered for the
entire session; robot and scene timestamps and scene versions must be monotonic.

## Parameter ownership

Goal dataclasses carry only semantic task intent. They do not carry robot part
names, planner configuration, retry policy, or runtime state.

`MotionPolicy` owns planner selection, motion source, sample count, fallback
control period, limits, and typed planner options. `RecoveryPolicy` owns
tracking/dynamic-goal thresholds, timeouts, and budgets. Action configs retain
only implementation-specific behavior such as gripper poses, phase splits,
lift distances, and grasp constraints.

## Built-ins

| Skill ID | Goal type | Roles |
|---|---|---|
| `move_end_effector` | `EndEffectorPoseGoal` | manipulator `primary` |
| `move_joints` | `JointPositionGoal`, `NamedJointPositionGoal` | manipulator `primary` |
| `pick_up` | `GraspGoal` | manipulator/end effector `primary` |
| `move_held_object` | `HeldObjectPoseGoal` | manipulator/end effector `primary` |
| `place` | `PlaceGoal`, `AssembleGoal` | manipulator/end effector `primary` |
| `press` | `PressGoal` | manipulator/end effector `primary` |
| `coordinated_pickment` | `CoordinatedPickGoal` | `left`, `right` |
| `coordinated_placement` | `CoordinatedPlacementGoal` | `placing`, `support` |
| `hand_over` | `GraspGoal` | `source`, `destination` |

## Extension rules

1. Define a frozen action-owned goal dataclass with `goal_kind`.
2. Declare `skill_id`, `GoalType`, and required semantic roles on the action.
3. Validate with `require_goal(invocation)`.
4. Plan from `context.robot.qpos`; never read an implicit live start state.
5. Return full-robot positions or a `TimedTrajectory` through `build_plan()`.
6. Declare symbolic changes with `StateDelta`; do not mutate context or commit
   physical effects during planning.
7. Keep scene stepping, controller I/O, and task-graph/MLLM logic outside the
   atomic action. Put execution-loop I/O behind the runner protocols rather than
   calling a simulator or device from `plan()` or `ExecutionSession`.
