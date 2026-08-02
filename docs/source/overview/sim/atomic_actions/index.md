# Atomic actions

```{toctree}
:hidden:

builtin_actions
```

Atomic actions turn typed, grounded skill requests into full-robot timed motion.
Planning is side-effect free and execution is incremental.

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
        +------------------------------+
        |                              |
        v                              v
AtomicActionEngine.compile       ExecutionSession.tick
(fixed-scene/offline)            (dynamic/closed-loop)
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

## Static and dynamic use

`AtomicActionEngine.compile()` plans a fixed sequence and returns a
`CompiledTrajectory`. It projects terminal qpos and expected task effects only
inside the returned context; it never changes simulator state.

`AtomicActionEngine.start()` creates an `ExecutionSession`. Each `tick()` takes
the latest context and emits at most one `JointCommand`. The session detects
tracking error, phase timeout, and movement of entities referenced by
`SceneEntityPose`, then replans from the latest observation within the configured
budget. Non-empty symbolic effects require external verification before commit.

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

See [Built-in actions](builtin_actions.md) for the shipped skill catalog and
[the tutorial](../../../tutorial/atomic_actions.rst) for closed-loop usage.
