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
|   resolves requests and calls AtomicAction.plan(...)        |
+------------------------------+------------------------------+
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
          engine.plan()  engine.compile()  engine.start()/tick()
          one ActionPlan fixed projection  observed closed loop
                               |             |
                               v             v
                      CompiledTrajectory  JointCommand + events
                                               |
                                               v
                                      ExecutionRunner
                               observe / schedule / dispatch
                                               |
                                               v
                          ObservationProvider + CommandSink + Clock
```

The boundary is deliberate:

| Concern | Owner | Contract |
|---|---|---|
| Task intent and sequencing | Action Agent, task graph, or user-authored application | Selects skills, goals, and execution order |
| Invocation construction | Agent adapter or user-authored code/config loader | Produces the same typed `ActionInvocation`; the engine has no agent-only interface |
| Perception and grounding | Agent adapter or user application | Builds scene snapshots and resource bindings, or supplies already-grounded values directly |
| Deterministic motion planning | Atomic action module | Produces an `ActionPlan` from an invocation and context |
| Motion-generation resources | `AtomicActionEngine` | Owns one robot, motion generator, planner backend, device, trajectory builder, and control-part command profiles |
| Recovery state | `ExecutionSession` | Consumes fresh contexts, emits at most one `JointCommand` per tick, and owns bounded recovery/revision state |
| Scene observation | `SceneProvider` | Captures ordered entities plus monotonic global or per-environment collision-world revisions |
| Scheduling and controller lifecycle | `ExecutionRunner` | Observes only when due, dispatches timed commands, records acknowledgements, and performs safe stop |
| Robot/simulator I/O | `ObservationProvider`, `CommandSink`, and `ExecutionClock` adapters | Isolates observation, command transport, and time/physics advancement from planning and session state |
| Physical-effect verification | Application observer | Verifies grasp, release, handover, and other symbolic effects |

`ExecutionRunner.step()` is non-blocking. Its convenience
`run_until_blocked()` loop waits or advances simulation through an injected
clock. Observation errors, rejected or timed-out commands, session failures,
and explicit cancellation trigger a best-effort cancel-then-hold sequence.
`SimulationExecutionAdapter` implements all three ports for a simulation robot;
real hardware integrations implement the same protocols without changing
action planning or recovery state.

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

# Choose one entry point according to the planning/execution requirement.
single_plan = engine.plan(manual_invocation, latest_context)
static_program = engine.compile((manual_invocation,), latest_context)
live_session = engine.start((manual_invocation,), latest_context)
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

## Choosing an engine entry point

Application code normally chooses between these three public entry points:

| API | Choose it when | Returns | State and observation behavior |
|---|---|---|---|
| `engine.plan(invocation, context)` | You need to inspect or plan exactly one registered action | `ActionPlan` | Reads one context; does not project its terminal qpos or expected task effect for another action |
| `engine.compile(invocations, context)` | All goals for an ordered static sequence are known before execution | `CompiledTrajectory` | Plans in order and propagates hypothetical qpos and expected effects through `projected_context`; never observes execution |
| `engine.start(invocations, context)` | Commands must be issued incrementally from fresh observations with bounded recovery | `ExecutionSession` | `tick(latest_context)` consumes measured state, emits at most one command, requests effect verification, and can replan |

The short selection rule is:

```text
one action to inspect or plan             -> plan
one or more actions in a fixed scene      -> compile
observed execution and error recovery     -> start, then tick
```

All three leave simulator stepping and controller I/O to the application.
`plan()` and `compile()` only return planning data. An `ExecutionSession` also
does not step the simulator itself; its `tick()` method returns commands for the
application to send.

Calling `compile()` with one invocation is valid and gives a uniform
`CompiledTrajectory` result, but it is not required for a single action. More
importantly, `compile()` cannot observe physical execution. If a later goal
depends on the measured result of an earlier action, end the compiled phase,
observe a new `PlanningContext`, and plan or compile the next phase. Use
`start()` when that observe/replan loop should be managed continuously by an
`ExecutionSession`.

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
| `MotionPolicy` | Motion source, sample count, timing, limits, dynamic-collision mode, typed planner options | Skill semantics or robot-resource names |
| `RecoveryPolicy` | Replan/retry budgets, tracking and dynamic-goal thresholds, phase timeout | Controller state or mutable counters |
| `ExecutionRunnerCfg` | Runner-level acknowledgement deadlines, minimum feedback cadence, and completion hold policy | Skill behavior, planning resources, or invocation revision data |
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

One engine owns one motion generator. At initialization it creates a fresh
instance of every action type in `BUILTIN_ACTION_TYPES` and binds those
instances to the engine's planning services:

```python
engine = AtomicActionEngine(motion_generator, control_profiles=profiles)

# All nine built-ins are immediately usable by stable skill ID.
assert "move_end_effector" in engine.actions
assert "pick_up" in engine.actions
```

Consequences of this ownership model:

- built-in action constructors use their typed default options unless an
  invocation supplies `skill_options`;
- every action in an engine sees the same robot, device, backend, caches, and
  collision world;
- an action instance cannot be silently reused by a different engine;
- one registered instance exists per stable `skill_id` in an engine.

The framework-owned `AtomicAction.plan()` template method binds collision
entity poses from the current `SceneSnapshot` into copied backend options, then
calls the skill-specific `_plan()` hook. Individual skills therefore do not
own dynamic-obstacle parameters or mutate caller-owned motion policies.

Registration means that an implementation is installed, not that every robot
can execute it. Required roles, control parts, profiles, and task-state
preconditions are validated while an invocation is resolved and planned. Agent
adapters must additionally filter the catalog by `agent_visible` and
embodiment capability instead of exposing every `engine.actions` entry blindly.

Use invocation `skill_options` whenever behavior varies per call. Two variants
with the same stable skill ID therefore share one built-in implementation:

```python
left_invocation = ActionInvocation(
    skill_id="pick_up",
    goal=left_goal,
    binding=left_binding,
    skill_options=left_pick_options,
)
right_invocation = ActionInvocation(
    skill_id="pick_up",
    goal=right_goal,
    binding=right_binding,
    skill_options=right_pick_options,
)

left_plan = engine.plan(left_invocation, latest_context)
right_plan = engine.plan(right_invocation, latest_context)
```

`register()` remains the extension point for a custom skill. Replacing a
built-in implementation requires an explicit `replace=True`; isolated tests or
fully custom engines can opt out of the catalog with `load_builtins=False`:

```python
custom_engine = AtomicActionEngine(motion_generator, load_builtins=False)
custom_engine.register(MyAction())

engine.register(CustomPickUp(), replace=True)
```

The module-level `register_action()` catalog is only for process-wide extension
type discovery. It does not mutate existing engines or join their default
built-in set; instantiate a discovered extension and pass it to
`engine.register()` explicitly.

### Implementation and advanced APIs

The similarly named `AtomicAction.plan()` method is not a fourth application
entry point. It is a framework-owned template method called by the engine after
resolving an invocation; skill implementations provide `_plan()`:

| API | Intended caller | Behavior |
|---|---|---|
| `AtomicAction.plan(request, context)` | `AtomicActionEngine` | Binds the current collision scene into a copied policy, then delegates to `_plan()` |
| `AtomicAction._plan(request, context)` | Atomic-action implementer | Consumes the prepared immutable `ResolvedActionRequest` and returns an `ActionPlan` |
| `engine.plan_action(action, invocation, context)` | Extension or isolated test | Temporarily binds and plans an unregistered action instance; built-in parameter variants should use invocation `skill_options` instead |
| `session.revise_current(invocation)` | Runtime orchestrator or Action Agent | Replaces the active logical call with a newer revision and replans from the latest observed context |
| `runner.step(effect_success=...)` | Non-blocking controller integration | Observes and dispatches only when the next timed command is due |
| `runner.run_until_blocked(...)` | Simple blocking application or tutorial | Advances the injected clock until terminal or external effect verification is required |
| `runner.cancel(reason)` | Explicit safe stop | Requests controller cancellation followed by an observed-position hold |

Application code should start with `engine.plan()`, `engine.compile()`, or
`engine.start()` unless it specifically needs one of these extension points.

## Planning one action

Use `engine.plan()` when one registered action needs to be inspected, tested,
or integrated into application-owned orchestration:

```python
invocation = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=target_pose),
    binding=ActionBinding(manipulators={"primary": "left_arm"}),
    motion_policy=MotionPolicy(sample_count=80, control_dt=1.0 / 60.0),
)

plan = engine.plan(invocation, latest_context)
if plan.plan_success.all():
    positions = plan.trajectory.positions
```

The result contains that action's trajectory, diagnostics, completion
conditions, and uncommitted expected effects. `plan()` does not automatically
create a next context. If another action must be planned against this action's
hypothetical result, use `compile()` instead of manually reproducing its state
projection rules.

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
    MotionPolicy,
)

engine = AtomicActionEngine(motion_generator)
binding = ActionBinding(manipulators={"primary": "left_arm"})
motion_policy = MotionPolicy(sample_count=80, control_dt=1.0 / 60.0)

approach = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=approach_pose),
    binding=binding,
    motion_policy=motion_policy,
)
retreat = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=retreat_pose),
    binding=binding,
    motion_policy=motion_policy,
)

initial_context = engine.initial_context()
compiled = engine.compile((approach, retreat), initial_context)
if compiled.plan_success.all():
    positions = compiled.trajectory.positions  # (B, N, robot_dof)
    approach_plan, retreat_plan = compiled.action_plans
    final_context = compiled.projected_context
```

When no context is supplied, the engine captures robot qpos/qvel and creates an
empty task state and scene snapshot. Supply an explicit context whenever goals
depend on perceived entities or a previous verified attachment.

Do not compile across a boundary where execution feedback changes a later goal.
For example, `scripts/tutorials/atomic_action/coordinated_placement.py` compiles
the two pick-ups, executes them, rebuilds the held-object state from measured
poses, and only then compiles placement.

## Dynamic goals and closed-loop recovery

Pose-valued goals can use `SceneEntityPose` instead of freezing an object pose
at invocation creation time:

```python
from embodichain.lab.sim.atomic_actions import (
    ExecutionStatus,
    RecoveryPolicy,
    SceneEntityPose,
)

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

For most applications, use `ExecutionRunner` to keep scheduling and controller
acknowledgement handling outside the session:

```python
scene_provider = RigidObjectSceneProvider({"moving_tray": moving_tray})
adapter = SimulationExecutionAdapter(sim, robot, scene_provider=scene_provider)
initial_context = adapter.observe(
    TaskState.empty(robot.get_qpos().shape[0], robot.device)
)
session = engine.start((moving_goal,), initial_context)
runner = ExecutionRunner(session, adapter, adapter, clock=adapter)
result = runner.run_until_blocked()
```

For a lightweight scene source that does not need environment correlation IDs,
pass a `scene_supplier(timestamp)` callback instead. `scene_provider` and
`scene_supplier` are mutually exclusive.

`ExecutionRunner.step()` is the non-blocking entry point for an application
that already owns its event loop. It observes only when the previous command's
`hold_duration` has elapsed, dispatches active commands through `CommandSink`,
and records accepted, rejected, or timed-out acknowledgements. Cancellation,
observation/session exceptions, and negative acknowledgements enter a
best-effort cancel-then-hold path.

`TimedTrajectory.dt[:, i]` is the interval leading to sample `i`.
`ExecutionSession` maps each following arrival interval onto the preceding
command's post-dispatch hold, while the final sample reuses its own interval as
a settling window before terminal validation. A batched runner uses the longest
active row interval as its synchronized barrier.
`SimulationExecutionAdapter.sleep()` converts that interval to an integral
number of physics steps instead of using wall-clock sleep. Stable `env_ids`
remain correlation identifiers and are not used as simulator array indices.

On each tick, the session can detect:

- joint tracking error relative to the previously emitted command;
- translation or rotation of a `SceneEntityPose` dependency beyond policy
  thresholds;
- a newer collision-world revision for a collision-sensitive phase;
- phase timeout;
- planning or terminal-goal failure for individual batch rows.

Recovery is bounded. A session replans from the latest observation, retries an
action only within the configured budgets, freezes ineligible environment rows,
and emits structured events when recovery is exhausted.

Eligibility, retry counters, and replan counters are per environment. Execution
cursors are intentionally batch-synchronized in this runtime: when any eligible
row is allowed to replan, the session regenerates the current action for the
active cohort and restarts the shared phase waypoint cursor. Rows that did not
trigger recovery keep their eligibility and do not spend recovery budget, but
they receive the regenerated plan from its batch barrier. Fully asynchronous
per-environment phase scheduling belongs in a higher-level scheduler rather than
this atomic-action session.

`SceneProvider.snapshot(timestamp=..., env_ids=...)` is the scene-observation
boundary. `SceneSnapshot.collision_entity_ids` identifies obstacle poses
consumed by a planner, while `collision_world_revision` can be global or
per-environment. `RigidObjectSceneProvider` tracks live simulation objects,
filters sub-threshold pose noise, and advances those revisions. Backends opt in
through `supports_collision_world_updates` and `with_collision_world()`;
`MotionGenerator.bind_collision_world()` owns that backend boundary, and cuRobo
maps the snapshot poses to `CuroboPlanOptions.dynamic_obstacle_poses`. A newer
revision invalidates only affected rows before synchronized cohort replanning.

`MotionPolicy.dynamic_collision_mode` controls this live-scene path. `AUTO`
(the default) consumes collision entities when the selected motion source and
planner support them, `OFF` ignores snapshot collision entities and their
revisions, and `REQUIRED` fails planning unless a compatible motion generator
and collision entities are available. This mode does not enable or disable the
planner's configured static-world or self-collision checks.

Recovery does not re-read a mutable Action object or invocation. The engine
resolves each call once into a `ResolvedActionRequest` containing an owned goal
snapshot, binding, policies, options, control commands, invocation ID, and
revision. Every local replan for that revision reuses the same request and
varies only the measured context. Mutable goal values such as tensors and
metadata containers are copied, while simulator-backed `BatchEntity` handles
retain their runtime identity.

Each emitted `JointCommand` carries a per-environment `hold_duration` derived
from the plan's `TimedTrajectory.dt`. The application control loop must respect
that timing after dispatching the command and before requesting the next
observation. `dt[:, i]` is the arrival interval leading to waypoint `i`, so the
first waypoint is dispatched immediately and command `i` carries `dt[:, i + 1]`
until the next waypoint is due. The final command reuses `dt[:, -1]` as a
settling window. For a synchronized batch, the caller should wait for the
longest duration among active rows. A passive hold command has zero duration.

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

Dynamic collision invalidation is provider-driven. Only registered,
pose-updatable collision entities are supported; adding/removing obstacles or
changing their geometry requires rebuilding the planner world.
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
5. implement side-effect-free `_plan(request, context)` using the engine-owned
   planning services; do not override the framework-owned public `plan()`;
6. return full-robot timed motion, per-environment planning success,
   diagnostics, and uncommitted effects;
7. add registration coverage, contract tests, execution/recovery tests, a
   runnable example, and documentation.

See {doc}`builtin_actions` for the shipped skill catalog and visual demos, and
{doc}`/tutorial/atomic_actions` for complete usage patterns and runnable scripts.

## Further reading

- {doc}`../planners/motion_generator` — the motion generator owned by the engine
- {doc}`../sim_robot` — robot control parts and kinematic configuration
- {doc}`/tutorial/atomic_actions` — static, closed-loop, and recovery examples
- `scripts/tutorials/atomic_action/moving_target_recovery.py` — runnable runner
  example that visibly moves a late-bound target, replans, and picks up the cube
- `scripts/tutorials/atomic_action/dynamic_obstacle_recovery.py` — cuRobo
  collision-world revision and obstacle-pose replanning example
