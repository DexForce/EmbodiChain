(atomic-actions)=

# Atomic actions

```{toctree}
:hidden:

builtin_actions
```

```{currentmodule} embodichain.lab.sim.atomic_actions
```

Atomic actions are the typed planning and execution boundary between a semantic
task request and robot joint commands. A caller describes **what** should happen
with an action-owned goal, grounds semantic roles onto robot resources, and
supplies the latest measured context. The action returns a full-robot,
time-aware plan without stepping simulation or claiming that a physical effect
has occurred.

```{note}
The current built-ins focus on arm-and-gripper manipulation. They already emit
full-robot-DoF trajectories, but dexterous-hand policies, lower-body locomotion,
and whole-body control are not implemented by this module yet.
```

## Architecture and responsibility boundary

```text
Action Agent / task graph
        |
        | semantic skill call (object, destination, constraints)
        v
grounder + capability binder
        |
        | ActionInvocation + PlanningContext
        v
+-------------------------------------------------------------+
| AtomicActionEngine                                          |
|                                                             |
| owns exactly one ActionPlanningServices                     |
|   +-- Robot                                                 |
|   +-- MotionGenerator / planner backend                     |
|   +-- device and shared TrajectoryBuilder                   |
|                                                             |
|   registered AtomicAction.plan(...) -> ActionPlan           |
+--------------------------+----------------------------------+
                           |
              +------------+-------------+
              |                          |
              v                          v
      compile(...)                 start(...) / tick(...)
      fixed projection             observed closed loop
              |                          |
              v                          v
   CompiledTrajectory              JointCommand + events
```

The boundary is deliberate:

| Concern | Owner | Contract |
|---|---|---|
| Task intent and sequencing | Action Agent or task graph | Selects a skill and semantic goal |
| Perception and grounding | Application adapter | Builds scene snapshots, object semantics, and resource bindings |
| Deterministic motion planning | Atomic action module | Produces an `ActionPlan` from an invocation and context |
| Motion-generation resources | `AtomicActionEngine` | Owns one robot, motion generator, planner backend, device, and trajectory builder |
| Robot/simulator stepping | Application control loop | Consumes `JointCommand`; the session never steps the simulator itself |
| Physical-effect verification | Application observer | Verifies grasp, release, handover, and other symbolic effects |

## Core contracts

The public contracts separate values with different owners and lifetimes. This
keeps goals small and prevents robot-specific or planner-specific parameters
from leaking into an Action Agent schema.

| Contract | Contains | Does not contain |
|---|---|---|
| `ActionGoal` | Action-specific desired outcome, such as an EEF pose or object pose | Arm names, planner instances, recovery counters |
| `ActionBinding` | Semantic-role mappings such as `primary -> left_arm` and `primary -> left_hand` | Motion settings or task geometry |
| `ActionCfg` | Implementation and hardware constants: hand qpos, grasp-selection rules, phase structure | Per-call goal, motion generator, generic recovery settings |
| `MotionPolicy` | Motion source, sample count, timing, limits, collision option, typed planner options | Skill semantics or robot-resource names |
| `RecoveryPolicy` | Replan/retry budgets, tracking and dynamic-goal thresholds, phase timeout | Controller state or mutable counters |
| `PlanningContext` | Measured `RobotObservation`, verified `TaskState`, versioned `SceneSnapshot`, stable environment IDs | Hypothetical simulator mutation |
| `ActionPlan` | Per-environment planning result, scene-bound phases, timed trajectories, diagnostics, expected `StateDelta` | Proof that a grasp/release/contact physically succeeded |

Goals follow the structural `ActionGoal` protocol: each action owns one or more
frozen dataclasses with a stable `goal_kind`. There is no shared `ActionTarget`
base class and no closed union that must change whenever a skill is added.

### Semantic resource binding

Bindings make an invocation portable across embodiments:

```python
binding = ActionBinding(
    manipulators={"primary": "left_arm"},
    end_effectors={"primary": "left_hand"},
)
```

Single-resource skills use `primary`; handover uses `source` and `destination`;
coordinated pick uses `left` and `right`; coordinated placement uses `placing`
and `support`. The action descriptor declares which roles are required, so a
grounder can validate a call before planning.

Simple motion skills resolve their concrete control part entirely from the
binding. Some current multi-phase manipulation implementations also keep
concrete arm/hand names in their action config because their preconfigured hand
qpos and phase assembly are hardware-specific. For those actions, the binding
must match the configured resources. This is an implementation constraint, not
a reason to put resource names back into goals.

### Engine-owned planning resources

One engine owns one motion generator. Actions borrow its planning services only
after `register()` or `plan_action()` binds them:

```python
engine = AtomicActionEngine(motion_generator)
engine.register(MoveEndEffector(MoveEndEffectorCfg()))
engine.register(MoveJoints(MoveJointsCfg()))
```

Consequences of this ownership model:

- action constructors contain only skill configuration;
- every action in an engine sees the same robot, device, backend, caches, and
  collision world;
- an action instance cannot be silently reused by a different engine;
- one registered instance exists per stable `skill_id` in an engine.

When two differently configured instances share the same stable skill ID, keep
one or both outside the registry and call `engine.plan_action(...)` explicitly:

```python
left_pick = PickUp(left_pick_cfg)
right_pick = PickUp(right_pick_cfg)

left_plan = engine.plan_action(left_pick, left_invocation, latest_context)
right_plan = engine.plan_action(right_pick, right_invocation, latest_context)
```

Both instances still borrow the same engine-owned motion generator.

## Which planning API to use

| API | Use it for | Result / behavior |
|---|---|---|
| `AtomicAction.plan(invocation, context)` | Implementing a skill | Action-owned side-effect-free planning hook; application code normally calls it through the engine |
| `engine.plan(invocation, context)` | Planning one registered skill | Resolves the registered action, binds shared resources, and validates its plan |
| `engine.plan_action(action, invocation, context)` | Planning an unregistered configured instance | Supports multiple configurations with one `skill_id` and one engine backend |
| `engine.compile(invocations, context)` | Fixed-scene/offline sequence planning | Returns one concatenated `CompiledTrajectory` and a hypothetical projected context |
| `engine.start(invocations, context)` | Observed incremental execution | Returns an `ExecutionSession`; each `tick()` emits at most one command and recovery events |

`AtomicAction.plan()` is therefore not a second execution API. It is the
polymorphic implementation point used by the engine. Neither it nor the engine
mutates the simulator.

## Static compilation

`compile()` plans invocations in order. For every successful action it projects
the terminal qpos and expected task-state effect into a new context so the next
action can be checked against a hypothetical result. The observed context and
simulator remain unchanged.

```python
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    EndEffectorPoseGoal,
    ExecutionEventKind,
    ExecutionStatus,
    MotionPolicy,
    MoveEndEffector,
    RecoveryPolicy,
    SceneEntityPose,
)

engine = AtomicActionEngine(motion_generator)
engine.register(MoveEndEffector())

invocation = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=target_pose),
    binding=ActionBinding(manipulators={"primary": "left_arm"}),
    motion_policy=MotionPolicy(sample_count=80, control_dt=1.0 / 60.0),
)

compiled = engine.compile((invocation,))
if compiled.plan_success.all():
    positions = compiled.trajectory.positions  # (B, N, robot_dof)
```

When no context is supplied, the engine captures robot qpos/qvel and creates an
empty task state and scene snapshot. Supply an explicit context whenever goals
depend on perceived entities or a previous verified attachment.

## Dynamic goals and closed-loop recovery

Pose-valued goals can use `SceneEntityPose` instead of freezing an object pose
at invocation creation time:

```python
moving_goal = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(
        xpos=SceneEntityPose(
            entity_id="moving_tray",
            relative_pose=tray_to_tcp,
            minimum_confidence=0.8,
        )
    ),
    binding=ActionBinding(manipulators={"primary": "left_arm"}),
    recovery_policy=RecoveryPolicy(
        max_replans=3,
        max_phase_retries=2,
        tracking_error_threshold=0.05,
        goal_translation_threshold=0.02,
        goal_rotation_threshold=0.087,
        phase_timeout=30.0,
    ),
)

latest_context = initial_context
session = engine.start((moving_goal,), latest_context)
while session.status is ExecutionStatus.RUNNING:
    tick = session.tick(latest_context)
    if tick.command is not None:
        send_joint_command(tick.command)
    latest_context = observe_context()
```

On each tick, the session can detect:

- joint tracking error relative to the previously emitted command;
- translation or rotation of a `SceneEntityPose` dependency beyond policy
  thresholds;
- phase timeout;
- planning or terminal-goal failure for individual batch rows.

Recovery is bounded. A session replans from the latest observation, retries an
action only within the configured budgets, freezes ineligible environment rows,
and emits structured events when recovery is exhausted.

```{attention}
Automatic dynamic-goal invalidation is dependency-driven. A goal must contain a
`SceneEntityPose` for the session to track that scene entity. A primitive that
directly queries a simulation entity during planning will use its latest pose
when planning happens, but that query alone does not trigger scene-motion
replanning.
```

## Planning success versus physical success

`ActionPlan.plan_success` only means a valid trajectory was produced for an
environment row. Pick, place, handover, and coordinated skills also return an
uncommitted `StateDelta` describing the attachment state expected after
execution.

At the terminal waypoint, an `ExecutionSession` requests an external
per-environment verification mask before committing a non-empty effect:

```python
tick = session.tick(latest_context)
if any(event.kind is ExecutionEventKind.EFFECT_VERIFICATION_REQUIRED for event in tick.events):
    effect_success = verify_grasp_or_release()
    tick = session.tick(latest_context, effect_success=effect_success)
```

This prevents a collision-free plan or well-tracked trajectory from being
misreported as a successful grasp, release, or handover.

## Action Agent integration

An MLLM should not construct `ActionInvocation` by copying arbitrary JSON into
runtime objects. An adapter should expose stable `SkillDescriptor` metadata and
agent-facing goal schemas, validate the semantic call, resolve object references
and embodiment capabilities, then produce the typed invocation:

```text
MLLM SkillCallSpec
    -> schema validation
    -> object / scene grounding
    -> capability and role binding
    -> ActionInvocation
    -> AtomicActionEngine
    -> ActionPlan / execution events
```

This keeps learned reasoning independent of planner instances and concrete
joint groups, while invocation IDs, planner diagnostics, and execution events
provide structured feedback for the agent's next decision.

## Extending the module

A new primitive should:

1. define a frozen, action-owned goal dataclass with a stable `goal_kind`;
2. declare `skill_id`, `GoalType`, required semantic roles, and agent visibility;
3. keep embodiment constants in its `ActionCfg` and reusable motion/recovery
   choices in invocation policies;
4. implement side-effect-free `plan(invocation, context)` using the
   engine-owned planning services;
5. return full-robot timed motion, per-environment planning success,
   diagnostics, and uncommitted effects;
6. add registration coverage, contract tests, execution/recovery tests, a
   runnable example, and documentation.

See {doc}`builtin_actions` for the shipped skill catalog and visual demos, and
{doc}`/tutorial/atomic_actions` for complete usage patterns and runnable scripts.

## Further reading

- {doc}`../planners/motion_generator` — the motion generator owned by the engine
- {doc}`../sim_robot` — robot control parts and kinematic configuration
- `scripts/tutorials/atomic_action/` — focused examples for every built-in skill
