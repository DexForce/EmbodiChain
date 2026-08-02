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
+--------------------------------+    +--------------------------------+
| Action Agent / semantic graph  |    | User-authored application      |
| skill call + object references |    | typed goal + binding + policy  |
+---------------+----------------+    +---------------+----------------+
                |                                     |
                v                                     |
 agent adapter: schema validation,                    |
 scene grounding, capability binding                  |
                |                                     |
                +------------------+------------------+
                                   |
                                   | ActionInvocation
                                   | + PlanningContext
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
| Task intent and sequencing | Action Agent, task graph, or user-authored application | Selects skills, goals, and execution order |
| Invocation construction | Agent adapter or user-authored code/config loader | Produces the same typed `ActionInvocation`; the engine has no agent-only interface |
| Perception and grounding | Agent adapter or user application | Builds scene snapshots and resource bindings, or supplies already-grounded values directly |
| Deterministic motion planning | Atomic action module | Produces an `ActionPlan` from an invocation and context |
| Motion-generation resources | `AtomicActionEngine` | Owns one robot, motion generator, planner backend, device, trajectory builder, and control-part command profiles |
| Robot/simulator stepping | Application control loop | Consumes `JointCommand`; the session never steps the simulator itself |
| Physical-effect verification | Application observer | Verifies grasp, release, handover, and other symbolic effects |

### Caller entry points

The engine supports two first-class caller paths. An Action Agent emits a
semantic skill call that an adapter validates, grounds, and converts into an
`ActionInvocation`. A user can instead author the typed invocation directly in
Python or load it from an application-owned configuration layer:

```python
manual_invocation = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=target_pose),
    binding=ActionBinding(manipulators={"primary": "left_arm"}),
    motion_policy=MotionPolicy(sample_count=80, control_dt=1.0 / 60.0),
    recovery_policy=RecoveryPolicy(max_replans=2),
)

# Inspect a single plan, compile a fixed sequence, or execute with recovery.
plan = engine.plan(manual_invocation, latest_context)
compiled = engine.compile((manual_invocation,), latest_context)
session = engine.start((manual_invocation,), latest_context)
```

A manual caller may bypass the semantic-schema adapter only when its target and
robot-resource binding are already grounded. Scene-relative goals still need a
current `PlanningContext`, and object names or semantic roles still need to be
resolved by the user application (or by reusing the same grounding adapter as
the Agent path).

Both paths converge at `ActionInvocation + PlanningContext`. They therefore use
the same goal validation, capability checks, planning backend, execution
events, bounded recovery, and physical-effect verification. Manual authoring is
an alternative orchestration entry point, not a lower-level path around the
engine contracts.

## Core contracts

The public contracts separate values with different owners and lifetimes. This
keeps goals small and prevents robot-specific or planner-specific parameters
from leaking into an Action Agent schema.

| Contract | Contains | Does not contain |
|---|---|---|
| `ActionGoal` | Action-specific desired outcome, such as an EEF pose or object pose | Arm names, planner instances, recovery counters |
| `ActionBinding` | Semantic-role mappings to keys from the engine robot's `control_parts`, such as `primary -> left_arm` and `primary -> left_hand` | Link/TCP names, arbitrary scene objects, motion settings, or task geometry |
| `ActionOptions` / built-in `*Options` | Frozen invocation-varying skill behavior: phase counts, offsets, grasp-selection rules | Robot resource names, hand qpos, planner backend |
| `ControlPartCommandProfile` | Embodiment-specific semantic commands such as `open`, `grasp`, and `ready`, keyed by actual control-part name | Action roles, task goals, recovery state |
| `ActionControlOverrides` | Optional role-scoped command replacements for one invocation revision | Persistent robot configuration |
| `MotionPolicy` | Motion source, sample count, timing, limits, collision option, typed planner options | Skill semantics or robot-resource names |
| `RecoveryPolicy` | Replan/retry budgets, tracking and dynamic-goal thresholds, phase timeout | Controller state or mutable counters |
| `PlanningContext` | Measured `RobotObservation`, verified `TaskState`, versioned `SceneSnapshot`, stable environment IDs | Hypothetical simulator mutation |
| `ActionPlan` | Per-environment planning result, scene-bound phases, timed trajectories, diagnostics, expected `StateDelta` | Proof that a grasp/release/contact physically succeeded |

Goals follow the structural `ActionGoal` protocol: each action owns one or more
frozen dataclasses with a stable `goal_kind`. There is no shared `ActionTarget`
base class and no closed union that must change whenever a skill is added.

### Semantic resource binding

A **role** is an action-owned semantic participant slot: it describes the job a
robot resource performs in that action, not the identity of the resource. Each
`AtomicAction` declares its required slots through `manipulator_roles` and
`end_effector_roles`; the same declarations are exposed through its
`SkillDescriptor` so an Agent adapter or manual caller can construct a complete
binding before planning.

Role names are local to both the skill and the resource category. For example,
`primary` in `manipulators` and `primary` in `end_effectors` are two separate
slots. Using the same role name expresses that the selected arm and hand/tool
serve the same functional participant in the action:

```python
binding = ActionBinding(
    manipulators={"primary": "left_arm"},
    end_effectors={"primary": "left_hand"},
)
```

In this example, `primary` is the role and `left_arm` / `left_hand` are the
bound resources. `primary` does not mean left, right, the first configured
arm, or a globally preferred arm; it simply denotes the principal participant
of a single-participant skill. Changing the values can bind the same action to
another compatible arm and tool without changing its goal or implementation.

Every bound value is the name of a control part declared by the engine-owned
robot. Both `left_arm` and `left_hand` must therefore be keys in
`robot.control_parts` (originating from `RobotCfg.control_parts`). They are not
joint names, link names, TCP frame names, or scene-object identifiers.
`end_effectors` specifically selects the actuated tool/hand control part; the
manipulator's IK/TCP frame remains part of the robot and solver configuration.
The engine validates every name and resolves its full-robot joint indices
before calling the action planner.

The validation boundary is intentionally narrow: the engine verifies required
roles, `control_parts` membership, resolvable joint indices, command type, and
command dimensions. The Agent adapter or application binder remains responsible
for capability compatibility, such as pairing an arm with the hand mounted on
it and choosing a semantic command supported by that tool.

Role names should describe action responsibilities rather than robot-specific
joint, link, or model names. Single-resource skills use `primary`; handover uses
`source` and `destination`; coordinated placement uses `placing` and `support`.
The current coordinated-pick contract uses `left` and `right` because its goal
geometry also distinguishes left/right grasps. New skills should prefer
functional roles unless a spatial distinction is intrinsic to their semantics.

All built-ins resolve participating arm and hand control parts from the binding.
They obtain hardware-specific `open` and `grasp` commands from the resolved
end-effector profile; no action or option duplicates arm names, hand names, or
hand qpos. Attachment state and expected effects are keyed by the bound
manipulator control-part name.

### Control-part semantic commands

Register embodiment commands once when constructing the engine. The keys are
concrete names from `robot.control_parts`; the command names remain semantic:

```python
engine = AtomicActionEngine(
    motion_generator,
    control_profiles={
        "left_hand": ControlPartCommandProfile.joint_positions(
            open=left_open_qpos,
            grasp=left_grasp_qpos,
        ),
        "left_arm": ControlPartCommandProfile.joint_positions(
            ready=left_ready_qpos,
        ),
    },
)
```

`MoveJoints(JointPositionGoal("ready"))` resolves `ready` from its bound
manipulator. Manipulation primitives resolve `open` and/or `grasp` from their
bound end effectors. A one-dimensional `JointPositionCommand` broadcasts over
the planning batch; a two-dimensional value must match the selected batch.

For a one-off change, override by action role rather than by concrete robot
name:

```python
invocation = ActionInvocation(
    skill_id="pick_up",
    goal=goal,
    binding=binding,
    control_overrides=ActionControlOverrides(
        end_effectors={
            "primary": {
                "grasp": JointPositionCommand(object_specific_grasp_qpos),
            }
        }
    ),
    revision=1,
)
```

The engine merges the override after resolving `primary` and captures the
result in `ResolvedActionRequest`. Automatic recovery for revision 1 sees the
same command snapshot. Joint limits remain constraints; they do not define the
semantic meaning of `open` or `grasp`. Tutorials may explicitly derive a simple
profile from limits, while a robot integration should normally provide
calibrated commands.

### Engine-owned planning resources

One engine owns one motion generator. Actions borrow its planning services only
after `register()` or `plan_action()` binds them:

```python
engine = AtomicActionEngine(motion_generator, control_profiles=profiles)
engine.register(MoveEndEffector())
engine.register(MoveJoints())
```

Consequences of this ownership model:

- action constructors optionally contain only typed default options;
- every action in an engine sees the same robot, device, backend, caches, and
  collision world;
- an action instance cannot be silently reused by a different engine;
- one registered instance exists per stable `skill_id` in an engine.

Prefer invocation `skill_options` when behavior varies per call. If an
application still needs two instances with different default options and the
same stable skill ID, keep one or both outside the registry and call
`engine.plan_action(...)` explicitly:

```python
left_pick = PickUp(default_options=left_pick_options)
right_pick = PickUp(default_options=right_pick_options)

left_plan = engine.plan_action(left_pick, left_invocation, latest_context)
right_plan = engine.plan_action(right_pick, right_invocation, latest_context)
```

Both instances still borrow the same engine-owned motion generator.

## Which planning API to use

| API | Use it for | Result / behavior |
|---|---|---|
| `AtomicAction.plan(request, context)` | Implementing a skill | Consumes an engine-resolved immutable request; application code normally calls it through the engine |
| `engine.plan(invocation, context)` | Planning one registered skill | Resolves the registered action, binds shared resources, and validates its plan |
| `engine.plan_action(action, invocation, context)` | Planning an unregistered configured instance | Supports multiple configurations with one `skill_id` and one engine backend |
| `engine.compile(invocations, context)` | Fixed-scene/offline sequence planning | Returns one concatenated `CompiledTrajectory` and a hypothetical projected context |
| `engine.start(invocations, context)` | Observed incremental execution | Returns an `ExecutionSession`; each `tick()` emits at most one command and recovery events |
| `session.revise_current(invocation)` | Explicit runtime parameter/goal update | Requires a newer revision of the active logical invocation and replans from the latest context |

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

Recovery does not re-read a mutable Action object or invocation. The engine
resolves each call once into a `ResolvedActionRequest` containing its binding,
policies, options, control commands, invocation ID, and revision. Every local
replan for that revision reuses the same request and varies only the measured
context.

Use an explicit newer revision when the application or Action Agent decides to
change runtime behavior:

```python
revised = ActionInvocation(
    skill_id=current.skill_id,
    goal=updated_goal,
    binding=current.binding,
    motion_policy=updated_motion_policy,
    recovery_policy=current.recovery_policy,
    skill_options=updated_options,
    control_overrides=updated_commands,
    invocation_id=current.invocation_id,
    revision=current.revision + 1,
)
session.revise_current(revised)
```

`skill_id` and `invocation_id` must still identify the active logical call.
Revision replacement preserves verified task state and environment eligibility,
resets the new revision's local recovery counters, emits
`INVOCATION_REVISED`, and replans from the latest context.

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
    -> safe skill-option selection
    -> semantic command selection (never raw qpos)
    -> ActionInvocation
    -> AtomicActionEngine
    -> ActionPlan / execution events
```

The adapter may expose a curated subset of `OptionsType`, but engine-only
profiles, `JointPositionCommand` payloads, planner instances, and concrete joint
groups should remain outside the MLLM schema. If the agent needs an
object-specific grasp mode, it should choose a semantic command or capability;
the grounding layer turns that choice into `ActionControlOverrides`. Invocation
IDs and monotonic revisions correlate updated agent decisions with planner
diagnostics and execution events, providing structured feedback for the next
decision without mutating an in-flight request implicitly.

## Extending the module

A new primitive should:

1. define a frozen, action-owned goal dataclass with a stable `goal_kind`;
2. define a frozen `ActionOptions` subclass only for behavior that can vary per
   invocation;
3. declare `skill_id`, `GoalType`, `OptionsType`, required semantic roles, and
   agent visibility;
4. put reusable embodiment commands on control-part profiles and generic
   motion/recovery choices in invocation policies;
5. implement side-effect-free `plan(request, context)` using the
   engine-owned planning services;
6. return full-robot timed motion, per-environment planning success,
   diagnostics, and uncommitted effects;
7. add registration coverage, contract tests, execution/recovery tests, a
   runnable example, and documentation.

See {doc}`builtin_actions` for the shipped skill catalog and visual demos, and
{doc}`/tutorial/atomic_actions` for complete usage patterns and runnable scripts.

## Further reading

- {doc}`../planners/motion_generator` — the motion generator owned by the engine
- {doc}`../sim_robot` — robot control parts and kinematic configuration
- `scripts/tutorials/atomic_action/` — focused examples for every built-in skill
