# Atomic actions

```{toctree}
:hidden:

builtin_actions
```

Atomic actions turn typed, grounded skill requests into full-robot timed motion.
Planning is side-effect free; execution is incremental and controller-independent.

```text
Action Agent / task graph
        |
        | semantic skill call
        v
grounder + capability binder
        |
        | ActionInvocation
        v
AtomicAction.plan(invocation, PlanningContext)
        |
        | ActionPlan + StateDelta
        +---------------------------------+
        |                                 |
        v                                 v
AtomicActionEngine.compile          ExecutionSession
(fixed-scene/offline)          (recovery state machine)
                                          |
                            ExecutionRunner.step / run_until_blocked
                              ^                         |
                    fresh observation           timed JointCommand
                              |                         v
                    ObservationProvider             CommandSink
                                          + ExecutionClock
```

## Contracts

- `ActionGoal`: structural protocol implemented by action-owned frozen goal
  dataclasses. There is no common target object or closed union.
- `ActionBinding`: semantic role to robot-resource mapping. Goals do not carry
  arm or hand names.
- `MotionPolicy`: planner, interpolation, sample count, timing, and limits.
- `RecoveryPolicy`: replan/retry budgets, tracking and dynamic-goal thresholds,
  and phase timeout.
- `PlanningContext`: measured `RobotObservation`, verified `TaskState`, versioned
  `SceneSnapshot`, and stable environment IDs.
- `ActionPlan`: one or more scene-bound phases, timed trajectories, completion
  conditions, diagnostics, and uncommitted `StateDelta` effects.
- `ObservationProvider`: captures a fresh `PlanningContext` for each due
  feedback cycle.
- `CommandSink`: sends, holds, or cancels commands and returns a structured
  acknowledgement.
- `ExecutionClock`: supplies monotonic scheduling; simulation adapters advance
  physics while real backends use wall or controller time.

## Static and dynamic use

`AtomicActionEngine.compile()` plans a fixed sequence and returns a
`CompiledTrajectory`. It projects terminal qpos and expected task effects only
inside the returned context; it never changes simulator state.

`AtomicActionEngine.start()` creates an `ExecutionSession`. Each `tick()` takes
the latest context and emits at most one `JointCommand`. The session detects
tracking error, phase timeout, and movement of entities referenced by
`SceneEntityPose`, then replans from the latest observation within the configured
budget. Non-empty symbolic effects require external verification before commit.

`ExecutionRunner` owns the outer execution lifecycle. Its non-blocking `step()`
observes only when the next waypoint is due, dispatches commands using the
trajectory's per-sample `dt`, and records acknowledgements. Its convenience
`run_until_blocked()` loop sleeps or advances simulation through an injected
clock. Observation errors, rejected or timed-out commands, session failures, and
explicit cancellation trigger a best-effort cancel-then-hold sequence.

`SimulationExecutionAdapter` implements all three ports for a simulation robot.
Real hardware integrations implement the same observation and command protocols
without changing action planning or recovery state.

## Example

```python
binding = ActionBinding(manipulators={"primary": "left_arm"})
invocation = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(target_pose),
    binding=binding,
    motion_policy=MotionPolicy(sample_count=80),
)

engine = AtomicActionEngine(motion_generator)
engine.register(MoveEndEffector(motion_generator, MoveEndEffectorCfg()))
compiled = engine.compile((invocation,))
```

For feedback-driven execution:

```python
adapter = SimulationExecutionAdapter(sim, robot)
initial = adapter.observe(TaskState.empty(robot.get_qpos().shape[0], robot.device))
session = engine.start((invocation,), initial)
runner = ExecutionRunner(session, adapter, adapter, clock=adapter)
result = runner.run_until_blocked()
```

See [Built-in actions](builtin_actions.md) for the shipped skill catalog and
[the tutorial](../../../tutorial/atomic_actions.rst) for closed-loop usage and
the runnable tracking-error recovery example.
