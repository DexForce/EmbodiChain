(atomic-actions)=

# Atomic actions

```{toctree}
:hidden:

builtin_actions
robot_skill_profiles
```

```{currentmodule} embodichain.lab.sim.atomic_actions
```

Atomic actions are the typed planning and execution boundary between a semantic
task request and runtime endpoint commands. A caller describes **what** should
happen with an action-owned goal, selects resources for the skill's participant
slots, and supplies the latest measured context. The action returns a
transport-neutral, time-aware plan without stepping simulation or claiming that
a physical effect has occurred.

```{note}
The current built-ins focus on arm-and-gripper manipulation and retain an
optional full-robot joint trajectory for planning feedback and inspection. The
binding and runtime-command contracts are not limited to joints: locomotion,
whole-body, or other controllers add capabilities, endpoint adapters, command
payloads, and transports without adding fixed resource categories to the core.
```

## Architecture and responsibility boundary

```text
+--------------------------------+    +--------------------------------+
| Action Agent / semantic graph  |    | User-authored application      |
| skill call + object references |    | typed goal + binding + policy  |
+---------------+----------------+    +---------------+----------------+
                |                                     |
                v                                     |
 SemanticSkillCompiler / SemanticSkillRuntime:         |
 schema validation, SceneRegistry grounding, binding  |
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
|   +-- device and control-part command profiles              |
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
                      CompiledTrajectory  RuntimeCommandFrame + events
                                               |
                                               v
                                      ExecutionRunner
                               observe / schedule / dispatch
                                               |
                                               v
                 ObservationProvider + EndpointCommandRouter + Clock
                                               |
                                               v
                             EndpointCommandTransport(s)
```

The boundary is deliberate:

| Concern | Owner | Contract |
|---|---|---|
| Task intent and sequencing | Action Agent, task graph, or user-authored application | Selects skills, goals, and execution order |
| Invocation construction | Agent adapter or user-authored code/config loader | Produces the same typed `ActionInvocation`; the engine has no agent-only interface |
| Perception and grounding | `SceneRegistry` on the canonical path; adapter or user application on the advanced path | Normalizes aliases to canonical typed references and publishes snapshots, or supplies already-grounded values directly |
| Deterministic motion planning | Atomic action module | Produces an `ActionPlan` from an invocation and context |
| Motion-generation resources | `AtomicActionEngine` | Owns one robot, motion generator, planner backend, device, trajectory builder, and control-part command profiles |
| Recovery state | `ExecutionSession` | Consumes fresh contexts, emits at most one `RuntimeCommandFrame` per tick, and owns bounded recovery/revision state |
| Scene observation | Registry-derived `SceneProvider` | Captures canonical ordered entities plus monotonic global or per-environment collision-world revisions |
| Scheduling and controller lifecycle | `ExecutionRunner` | Observes only when due, dispatches timed commands, records acknowledgements, and performs safe stop |
| Robot/simulator I/O | `ObservationProvider`, `EndpointCommandRouter`, `EndpointCommandTransport`, and `ExecutionClock` adapters | Isolates observation, per-controller command transport, and time/physics advancement from planning and session state |
| Physical-effect verification | Application observer | Verifies grasp, release, handover, and other symbolic effects |

`ExecutionRunner.step()` is non-blocking. Its convenience
`run_until_blocked()` loop waits or advances simulation through an injected
clock. Observation errors, rejected or timed-out commands, session failures,
and explicit cancellation trigger a best-effort cancel-then-hold sequence.
`SimulationExecutionAdapter` provides observation, clock, and the built-in
`robot.joint_position` transport for a simulation robot. Register it with an
`EndpointCommandRouter`; real hardware integrations provide transports for the
same or additional endpoint kinds without changing action planning or recovery
state.

### Caller entry points

The engine supports two first-class caller paths. An Action Agent or
configuration-driven application can emit a semantic call for
{class}`~embodichain.lab.sim.skills.SemanticSkillCompiler` and
{class}`~embodichain.lab.sim.skills.SemanticSkillRuntime` to validate, ground,
and convert into an `ActionInvocation`. A user can instead author the typed
invocation directly in Python or load it from an application-owned
configuration layer:

```python
binding = engine.bind_control_parts(
    "move_end_effector",
    {"primary": {"motion": "left_arm"}},
)
manual_invocation = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=target_pose),
    binding=binding,
    motion_policy=MotionPolicy(sample_count=80),
    recovery_policy=RecoveryPolicy(max_replans=2),
)

# Choose one entry point according to the planning/execution requirement.
single_plan = engine.plan(manual_invocation, latest_context)
static_program = engine.compile((manual_invocation,), latest_context)
live_session = engine.start((manual_invocation,), latest_context)
```

A manual caller may bypass the semantic-schema adapter only when its target and
robot-resource endpoints are already grounded. Scene-relative goals still need
a current `PlanningContext`, and object names or participant selections still
need to be resolved by the user application (or by reusing the same grounding
adapter as the Agent path).

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
| `engine.compile(invocations, context)` | All goals are known and every action provides an inspectable joint trajectory | `CompiledTrajectory` | Plans in order and propagates hypothetical qpos and expected effects through `projected_context`; never observes execution |
| `engine.start(invocations, context)` | Commands must be issued incrementally from fresh observations with bounded recovery | `ExecutionSession` | `tick(latest_context)` consumes measured state, emits at most one command, requests effect verification, and can replan |

The short selection rule is:

```text
one action to inspect or plan             -> plan
joint-trajectory actions in a fixed scene -> compile
observed execution and error recovery     -> start, then tick
```

All three leave simulator stepping and controller I/O to the application.
`plan()` and `compile()` only return planning data. An `ExecutionSession` also
does not step the simulator itself; its `tick()` method returns commands for the
application to send.

Calling `compile()` with one invocation is valid and gives a uniform
`CompiledTrajectory` result, but it is not required for a single action. More
importantly, `compile()` cannot observe physical execution. If a later goal
depends on the measured result of an earlier action, end the compiled stage,
observe a new `PlanningContext`, and plan or compile the next stage. Use
`start()` when that observe/replan loop should be managed continuously by an
`ExecutionSession`.

`compile()` is intentionally an offline **joint-trajectory** projection API. It
rejects an `ActionPlan` whose optional `joint_trajectory` is absent. Generic
non-joint command plans remain valid for `plan()` and `start()`; composing their
hypothetical state requires a future endpoint-specific projection contract.

## Core contracts

The public contracts separate values with different owners and lifetimes. This
keeps goals small and prevents robot-specific or planner-specific parameters
from leaking into an Action Agent schema.

| Contract | Contains | Does not contain |
|---|---|---|
| Action-owned goal dataclass | Action-specific desired outcome, such as an EEF pose or object pose | Arm names, planner instances, recovery counters |
| `SkillBindingContract` | Skill-local participant slots, required endpoint capabilities and commands, and disjointness constraints | Concrete robot resources, controller handles, or transport configuration |
| `ActionBinding` / `EndpointBinding` | Engine-owned endpoint snapshots keyed by `(slot_id, endpoint_id)`, including capabilities, semantic commands, claims, and an immutable runtime target | Live controllers, planner settings, task geometry, or caller-owned mutable mappings |
| `ActionOptions` / built-in `*Options` | Frozen invocation-varying skill behavior: segment counts, offsets, grasp-selection rules | Robot resource names, hand qpos, planner backend |
| `ControlPartCommandProfile` | Embodiment-specific semantic commands such as `open`, `grasp`, and `ready`, keyed by actual control-part name | Skill slots/endpoints, task goals, recovery state |
| `ActionControlOverrides` | Optional `(slot, endpoint)`-scoped command replacements for one invocation revision | Persistent robot configuration |
| `MotionPolicy` | Motion strategy, sample count, dynamic-collision mode, typed planner options | Execution cadence, skill semantics, or robot-resource names |
| `RecoveryPolicy` | Action replan/retry budgets, tracking and dynamic-goal thresholds, action-attempt timeout | Controller state or mutable counters |
| `ExecutionRunnerCfg` | Runner-level acknowledgement deadlines, minimum feedback cadence, and completion hold policy | Skill behavior, planning resources, or invocation revision data |
| `PlanningContext` | Measured `RobotObservation`, verified `TaskState`, versioned `SceneSnapshot`, stable environment IDs, and optional explicit control cadence for action-owned interpolation | Hypothetical simulator mutation or a planner timing fallback |
| `ActionPlan` | Per-environment result, `TimedCommandSequence`, optional joint trajectory, named segments, action-level recovery metadata, diagnostics, expected `StateDelta` | Proof that a grasp/release/contact physically succeeded; independently recoverable segment boundaries |
| `RuntimeCommandFrame` | Synchronized endpoint commands, active rows, stable environment IDs, and per-row hold duration | Live transport or controller objects |

`MotionPolicy.strategy` accepts exactly `"motion_gen"` or `"ik_interp"`; the
same value is forwarded to `MotionGenOptions.strategy` without an adapter layer.
Every planner result that contains positions must also contain per-waypoint
`dt`; its per-environment `duration` is derived from those intervals. Every
action passes a `TimedTrajectory` to `build_plan()`; raw position tensors are
rejected. For
action-owned deterministic interpolation, the integration supplies its
authoritative cadence as `PlanningContext.control_dt` (normally
`BaseEnv.step_dt`). The engine never supplies or guesses missing timing.
Planner-backend compatibility is a profile-level concern expressed by
`SkillPolicyPreset.required_planner`, not a per-invocation motion choice.

Each action owns one or more frozen goal dataclasses and declares the accepted
type through `AtomicAction.GoalType`. The action validates that type when the
engine resolves an invocation. There is no marker protocol, shared
`ActionTarget` base class, or closed union that must change whenever a skill is
added.

### Skill contracts and endpoint binding

The canonical semantic path uses a
{doc}`RobotSkillProfile <robot_skill_profiles>` to match skill-local slots and
endpoint capabilities against a generic robot resource graph. It validates
participant pairing, typed commands, physical claims, complete defaults, and
policy presets before producing the engine-owned `ActionBinding` used by an
invocation.

Each `AtomicAction` declares one explicit `SkillBindingContract`. A **slot** is
an action-local participant such as `primary`, `source`, or `destination`. Each
slot contains one or more named endpoint requirements. An endpoint name is also
local to the skill contract: current manipulation skills use `motion` and
`grasp`, while a future navigation or whole-body skill can declare different
names and open, namespaced capabilities. There are no global `manipulator`,
`end_effector`, `base`, or `whole_body` fields to extend.

For example, `PickUp` requires `primary.motion` with its motion capabilities and
`primary.grasp` with the `interaction.grasp` capability plus typed `open` and
`grasp` commands. Its contract also requires those two endpoint views to have
disjoint physical claims. A profile can satisfy that contract with a composite
participant resource whose endpoints resolve to an arm and hand. Another skill
may deliberately permit overlapping views of one coupled whole-body
controller.

The canonical path resolves the skill through a bound profile:

```python
resolved = engine.skill_profile.resolve(
    "pick_up",
    selections={"primary": "left_participant"},
)
binding = resolved.action_binding
```

Advanced direct-core code can select joint-backed endpoints by actual
`Robot.control_parts` names. Use the engine helper rather than constructing an
`ActionBinding` manually; the helper validates the installed skill's contract,
resolves joint indices and commands, and stamps the engine ownership identity:

```python
binding = engine.bind_control_parts(
    "pick_up",
    {
        "primary": {
            "motion": "left_arm",
            "grasp": "left_hand",
        }
    },
)
```

The resulting `ActionBinding` is generic. Each `EndpointBinding` records its
`slot_id`, `endpoint_id`, logical `resource_id`, adapter ID, capabilities,
commands, claim tokens, and a typed `RuntimeEndpointTarget`. A target contains
only immutable addressing information such as transport ID and destination ID;
the live simulator entity, hardware client, or controller belongs to the
registered transport. Profile endpoint adapters can therefore return a mobile,
whole-body, joint-position, or custom target without changing `ActionBinding`.

Slot names describe action responsibilities rather than robot-specific joint,
link, or model names. Single-participant skills use `primary`; handover uses
`source` and `destination`; coordinated placement uses `placing` and `support`.
The current coordinated-pick contract uses `left` and `right` because its goal
geometry distinguishes left/right grasps. New skills should prefer functional
slot names unless a spatial distinction is intrinsic to their semantics.

Current built-ins resolve joint-backed `motion` and `grasp` endpoints from the
binding. They obtain hardware-specific `open` and `grasp` commands from the
resolved grasp endpoint; no action or option duplicates arm names, hand names,
or hand qpos. Their attachment state and expected effects are currently keyed
by the motion endpoint's control-part target.

### Control-part semantic commands

On the canonical semantic path, declare embodiment commands on the
{doc}`RobotSkillProfile <robot_skill_profiles>` and pass the profile through the
engine's `skill_profile` argument. For a direct-core integration, register the
same command profiles explicitly when constructing the engine. Profile command
IDs are generic and selected by endpoint adapters; the built-in control-part
adapter defaults them to concrete `robot.control_parts` names. Direct-core
engine keys are always concrete control-part names. The command names remain
semantic:

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
`primary.motion` endpoint. Manipulation primitives resolve `open` and/or
`grasp` from their bound grasp endpoints. A one-dimensional
`JointPositionCommand` broadcasts over the planning batch; a two-dimensional
value must match the selected batch.

For a one-off change, override by action-local slot and endpoint rather than by
concrete robot name:

```python
invocation = ActionInvocation(
    skill_id="pick_up",
    goal=goal,
    binding=binding,
    control_overrides=ActionControlOverrides(
        endpoints={
            "primary": {
                "grasp": {
                    "grasp": JointPositionCommand(object_specific_grasp_qpos),
                }
            }
        }
    ),
    revision=1,
)
```

The engine merges the override into `primary.grasp` and captures the result in
`ResolvedActionRequest`. Automatic recovery for revision 1 sees the same
command snapshot. Joint limits remain constraints; they do not define the
semantic meaning of `open` or `grasp`. Tutorials may explicitly derive a simple
profile from limits, while a robot integration should normally provide
calibrated commands.

### Engine-owned planning resources

One engine owns one motion generator. At initialization it creates a fresh
instance of every action type in `BUILTIN_ACTION_TYPES` and binds those
instances to the engine's planning services:

```python
engine = AtomicActionEngine(motion_generator, control_profiles=profiles)

# All eleven built-ins are immediately usable by stable skill ID.
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

For a canonical integration, construct the snapshot provider and collision
world from one {doc}`SceneRegistry <../scene_registry>`. Direct use of
`RigidObjectSceneProvider` remains an advanced-core path.

Registration means that an implementation is installed, not that every robot
can execute it. `engine.actions` contains direct-core implementations;
`engine.skills` contains installed, agent-visible implementations with an
explicit generic binding contract; and `engine.skill_profile.skills` applies
embodiment capability filtering. Required task-state preconditions remain
runtime conditions and are validated while an invocation is resolved and
planned.

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

Registration is deliberately engine-local. Construct an extension and pass it
to `engine.register()` explicitly; there is no separate process-wide catalog
whose contents can drift from the actions installed in an engine.

### Implementation and advanced APIs

The similarly named `AtomicAction.plan()` method is not a fourth application
entry point. It is a framework-owned template method called by the engine after
resolving an invocation; skill implementations provide `_plan()`:

This is a deliberate hard extension boundary. Defining `plan()` on a subclass
raises `TypeError` at class definition and has no compatibility adapter. Migrate
an older custom action by renaming its implementation to `_plan()`.

| API | Intended caller | Behavior |
|---|---|---|
| `AtomicAction.plan(request, context)` | `AtomicActionEngine` | Binds the current collision scene into a copied policy, then delegates to `_plan()` |
| `AtomicAction._plan(request, context)` | Atomic-action implementer | Consumes the prepared immutable `ResolvedActionRequest` and returns an `ActionPlan` |
| `engine.plan_action(action, invocation, context)` | Extension or isolated test | Temporarily binds and plans an unregistered action instance; built-in parameter variants should use invocation `skill_options` instead |
| `engine.start(invocations, context, eligible_mask=...)` | Runtime orchestrator | Starts a session whose owned row cohort can only shrink across action barriers and recovery |
| `session.revise_current(invocation)` | Manually ticked runtime orchestrator | Replaces the active logical call with a newer same-destination revision and replans from the latest observed context |
| `runner.revise_current(invocation)` | Runner-driven runtime orchestrator or Action Agent | Snapshots a revision, preserves the current frame deadline, then replans from a fresh due-time observation |
| `runner.deactivate_rows(mask, reason=...)` | Runner-driven runtime orchestrator | Permanently removes rows and refreshes the runner's cached effect request; prefer it over direct session mutation |
| `runner.step(effect_result=..., effect_verifier=...)` | Non-blocking controller integration | Observes only when due; accepts either an asynchronous correlated result or a synchronous verifier, never both |
| `runner.run_until_blocked(...)` | Simple blocking application or tutorial | Advances the injected clock until terminal or external effect verification is required |
| `runner.cancel(reason)` | Explicit safe stop | Requests controller cancellation followed by an observed-position hold |

Application code should start with `engine.plan()`, `engine.compile()`, or
`engine.start()` unless it specifically needs one of these extension points.

## Planning one action

Use `engine.plan()` when one registered action needs to be inspected, tested,
or integrated into application-owned orchestration:

```python
binding = engine.bind_control_parts(
    "move_end_effector",
    {"primary": {"motion": "left_arm"}},
)
invocation = ActionInvocation(
    skill_id="move_end_effector",
    goal=EndEffectorPoseGoal(xpos=target_pose),
    binding=binding,
    motion_policy=MotionPolicy(sample_count=80),
)

plan = engine.plan(invocation, latest_context)
if plan.plan_success.all():
    command_frames = plan.commands.frames
    if plan.joint_trajectory is not None:
        positions = plan.joint_trajectory.positions
```

The result always contains that action's transport-neutral command sequence. A
joint-planned action may additionally retain `joint_trajectory` for feedback,
inspection, and static qpos projection. The plan also contains named segment
ranges, diagnostics, action-level recovery metadata, and uncommitted expected
effects. `plan()` does not automatically create a next context. If another
action must be planned against this action's hypothetical result, use
`compile()` instead of manually reproducing its state projection rules.

`AtomicAction.build_plan()` normalizes scalar or per-environment planner success
and replaces unsuccessful rows with the context's observed joint position.
Primitive implementations therefore preserve row-local failures in
`plan_success`; they do not need to duplicate failure-row hold logic.
It accepts only `TimedTrajectory`. Interpolation code can construct one with
`TimedTrajectory.from_uniform_step(..., step_dt=context.require_control_dt())`;
planner-backed code should preserve the planner's explicit `dt`.

`TrajectorySegment.start` and `.stop` form an action-local half-open waypoint
range. `plan.segment(name)` resolves that local metadata, while
`compiled.segment(action_index, name)` shifts it into the concatenated
trajectory. Composite built-ins publish their actual planner-returned segment
lengths; actions without explicit structure receive one segment named by their
`skill_id`. Segments support tracing and tutorial callbacks only—the session
still replans and retries the enclosing action as one unit.

## Static compilation

`compile()` plans joint-trajectory invocations in order. For every successful
action it projects the terminal qpos and expected task-state effect into a new
context so the next action can be checked against a hypothetical result. The
observed context and simulator remain unchanged.

```python
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    EndEffectorPoseGoal,
    MotionPolicy,
)

engine = AtomicActionEngine(motion_generator)
binding = engine.bind_control_parts(
    "move_end_effector",
    {"primary": {"motion": "left_arm"}},
)
motion_policy = MotionPolicy(sample_count=80)

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
    binding=engine.bind_control_parts(
        "move_end_effector",
        {"primary": {"motion": "left_arm"}},
    ),
    recovery_policy=RecoveryPolicy(
        max_replans=3,
        max_action_retries=2,
        tracking_error_threshold=0.05,
        goal_translation_threshold=0.02,
        goal_rotation_threshold=0.087,
        action_timeout=30.0,
    ),
)

latest_context = initial_context
session = engine.start((moving_goal,), latest_context)
while session.status is ExecutionStatus.RUNNING:
    tick = session.tick(latest_context)
    if tick.command is not None:
        dispatch_runtime_frame(tick.command)
    latest_context = observe_context()
```

For most applications, use `ExecutionRunner` to keep scheduling and controller
acknowledgement handling outside the session. The following snippet shows the
advanced direct-core provider path:

```python
scene_provider = RigidObjectSceneProvider({"moving_tray": moving_tray})
adapter = SimulationExecutionAdapter(sim, robot, scene_provider=scene_provider)
router = EndpointCommandRouter((adapter,))
initial_context = adapter.observe(
    TaskState.empty(robot.get_qpos().shape[0], robot.device)
)
session = engine.start((moving_goal,), initial_context)
runner = ExecutionRunner(session, adapter, router, clock=adapter)
result = runner.run_until_blocked()
```

For a lightweight scene source that does not need environment correlation IDs,
pass a `scene_supplier(timestamp)` callback instead. `scene_provider` and
`scene_supplier` are mutually exclusive.

`ExecutionRunner.step()` is the non-blocking entry point for an application
that already owns its event loop. It observes only when the previous
`RuntimeCommandFrame.hold_duration` has elapsed, dispatches active endpoint
commands through `EndpointCommandRouter`, and records accepted, rejected, or
timed-out acknowledgements. The router preflights a whole frame, groups commands
by exact `transport_id`, and aggregates transport acknowledgements, so an
unknown or incompatible transport cannot cause partial dispatch. Cancellation,
observation/session exceptions, and negative acknowledgements enter a
best-effort cancel-then-hold path for every armed runtime target.

Pass an owned `eligible_mask` to `engine.start()` when only a subset of rows may
enter the invocation sequence. This cohort is sticky: eligibility can only
shrink across action barriers and replans. Later failures outside the atomic
runtime should call `runner.deactivate_rows(mask, reason=...)`; the operation is
idempotent, the next command neutralizes changed rows, and removing the final
eligible row fails and terminates the session. When effect verification is
pending, deactivation narrows the request and assigns a new
`verification_id`. Do not mutate `session` directly while its runner owns
scheduling, because the runner must refresh its cached effect boundary.

The engine authorizes every emitted command against the immutable target and
physical claims in the resolved binding. A command cannot address an unbound
destination, substitute target metadata, or overlap another endpoint's joints
or claim tokens. Non-empty frames and recovery plans keep one stable
destination set. If a failed replan emits no frames, the session retains the
previous targets so the runner can still request a transport-owned hold.

An inactive row is not equivalent to omitting a write: each transport must
actively neutralize inactive rows for every addressed target. The simulation
joint-position transport holds observed positions for those rows; a velocity
transport would normally send zero velocity.

Each `RuntimeCommandFrame` carries the delay before the next frame. A batched
runner uses the longest active row duration as its synchronized barrier. The
joint-trajectory lowering helper derives these holds from trajectory arrival
intervals; non-joint planners set them directly when building their
`TimedCommandSequence`.
`SimulationExecutionAdapter.sleep()` converts that interval to an integral
number of physics steps instead of using wall-clock sleep. Stable `env_ids`
remain correlation identifiers and are not used as simulator array indices.

On each tick, the session can detect:

- joint tracking error relative to the previously emitted command;
- translation or rotation of a `SceneEntityPose` dependency beyond policy
  thresholds;
- a newer collision-world revision for a collision-sensitive action;
- action-attempt timeout;
- planning or terminal-goal failure for individual batch rows.

Recovery is bounded. A session replans from the latest observation, retries an
action only within the configured budgets, freezes ineligible environment rows,
and emits structured events when recovery is exhausted.

Eligibility, retry counters, and replan counters are per environment. Execution
cursors are intentionally batch-synchronized in this runtime: when any eligible
row is allowed to replan, the session regenerates the current action for the
active cohort and restarts the shared action waypoint cursor. Rows that did not
trigger recovery keep their eligibility and do not spend recovery budget, but
they receive the regenerated plan from its batch barrier. Fully asynchronous
per-environment action scheduling belongs in a higher-level scheduler rather than
this atomic-action session.

`SceneProvider.snapshot(timestamp=..., env_ids=...)` is the scene-observation
boundary. On the canonical planning path,
`SceneRegistry.make_planning_scene_provider()` derives an independent provider
and eagerly validates its collision contract against the motion generator.
Its snapshots expose canonical registry IDs only.
The registry owns static identity, aliases, geometry, affordances, hierarchy,
and collision roles; `SceneSnapshot` owns versioned dynamic pose/confidence and
collision revisions. Snapshot states are defensively copied on construction and
public read.

`SceneSnapshot.collision_entity_ids` identifies obstacle poses consumed by a
planner, while `collision_world_revision` can be global or per-environment.
Registry-derived providers filter sub-threshold pose noise and advance those
revisions from the last materially published pose, so cumulative motion cannot
remain hidden indefinitely. Backends opt in through
`supports_collision_world_updates` and `with_collision_world()`;
`MotionGenerator.bind_collision_world()` owns that backend boundary, and cuRobo
maps the snapshot poses to `CuroboPlanOptions.dynamic_obstacle_poses`. A newer
revision invalidates only affected rows before synchronized cohort replanning.

`make_planning_scene_provider()` requires two exact canonical-ID agreements:
the registry's complete `STATIC ∪ DYNAMIC` set must equal the planner's
complete collision-world set, and the registry, derived provider, and planner
dynamic subsets must equal one another. It also requires planner update support
for a non-empty dynamic subset and matching shared or per-environment world
semantics. A one-environment registry may infer `SHARED`; a multi-environment
dynamic registry must choose `SHARED` or `PER_ENV` explicitly. External
perception/hardware providers use
`validate_collision_integration(..., scene_provider=...)` directly. Plain
`make_scene_provider()` and `RigidObjectSceneProvider` are perception or
advanced direct-core paths without eager planner agreement. See
{doc}`../scene_registry` for setup.

`MotionPolicy.dynamic_collision_mode` controls this live-scene path. `AUTO`
(the default) consumes collision entities when the selected motion strategy and
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

Each emitted `RuntimeCommandFrame` carries a per-environment `hold_duration`.
The application control loop must respect that timing after dispatch and before
requesting the next observation. For a synchronized batch, the caller waits
for the longest duration among active rows. Safe stop is a separate transport
lifecycle: the runner cancels every armed target and then asks each transport to
hold that target from the latest observed context.

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
runner.revise_current(revised)
```

`skill_id` and `invocation_id` must still identify the active logical call.
Revision replacement preserves verified task state and environment eligibility,
resets the new revision's local recovery counters, emits
`INVOCATION_REVISED`, and replans from the latest context. Once the current
action owns runtime destinations, the revision must declare the same non-empty
destination set and preserve every target's exact address fingerprint, including
its safe-hold footprint. Switching to a base, another arm, or another controller
is a new invocation boundary. `runner.revise_current()` stages the owned request,
keeps the current frame deadline, and plans only after collecting the next due
observation. A physical effect awaiting verification cannot be abandoned by a
revision; verify it first, or cancel and start a new invocation. Callers that
drive `ExecutionSession.tick()` directly can use `session.revise_current()` and
should pass their fresh context explicitly.

```{attention}
Automatic dynamic-goal invalidation is dependency-driven. A goal must contain a
`SceneEntityPose`, or an object-centric primitive must explicitly declare the
`ObjectSemantics.entity_id` whose snapshot pose it consumes. `PickUp` and the
implicit-initial-pose path of coordinated pickup declare that dependency
automatically. The deprecated live-entity fallback does not trigger
scene-motion replanning.

`ActionPlan.scene_dependency_monitor_until` may assign each dependency an
exclusive command-frame cutoff. Omitted dependencies remain monitored for the
whole action. `PickUp` stops monitoring its semantic object ID after `approach`
is dispatched so contact-, close-, and lift-induced object movement does not
trigger a false replan. Joint-tracking and collision-world checks remain active.

Dynamic collision invalidation is provider-driven. Only registered,
pose-updatable collision entities are supported; adding/removing obstacles or
changing their geometry requires rebuilding the planner world.
```

## Planning success versus physical success

`ActionPlan.plan_success` only means a valid command plan was produced for an
environment row. Pick, place, handover, and coordinated skills also return an
uncommitted `StateDelta` describing the attachment state expected after
execution.

`TaskState.held_objects` uses one `HeldObjectState` per bound manipulator.
Multi-arm grasps use multiple entries that share the same `ObjectSemantics`;
there is no parallel coordinated-attachment representation to synchronize.
Consumers query per-environment active and exclusive-hold masks from that one
map. A single-arm transport, release, or handover row fails safely while a
second manipulator still holds the same semantic object or live entity.

At the terminal waypoint, an `ExecutionSession` requests an external
correlated per-environment result before committing a non-empty effect:

```python
from embodichain.lab.sim.atomic_actions import EffectVerificationResult

tick = session.tick(latest_context)
if tick.pending_effect is not None:
    request = tick.pending_effect
    success_mask, failure_mask = verify_grasp_or_release(request.env_mask)
    effect_result = EffectVerificationResult(
        verification_id=request.verification_id,
        success_mask=success_mask,
        failure_mask=failure_mask,
    )
    tick = session.tick(latest_context, effect_result=effect_result)
```

This prevents a collision-free or well-tracked command plan from being
misreported as a successful grasp, release, or handover. The typed
`EffectVerificationRequest` persists on subsequent ticks while waiting;
`EFFECT_VERIFICATION_REQUIRED` remains a one-time observability event. Success
and failure masks are disjoint subsets of the request mask; omitted request rows
remain unresolved. Request IDs change after mask shrinkage or whole-action
retry, so a delayed result cannot commit a newer attempt.

`request.deadline` is expressed in the robot-observation timestamp domain.
`RecoveryPolicy.action_timeout` covers both trajectory execution and the
terminal effect wait; a retry invalidates the old request ID. With
`ExecutionRunner.step()`, a call made before the next due cycle does not consume
its `effect_result`: schedule another call using `wait_duration`, re-read the
current request, and submit a result for that current ID. Partial resolution and
row deactivation can also replace the request before the delayed result arrives.

## Action Agent integration

An MLLM should not construct `ActionInvocation` by copying arbitrary JSON into
runtime objects. The `embodichain.lab.sim.skills` package provides the semantic
boundary: stable call descriptors, immutable call values, scene/profile
manifests, a compiler, and a runtime facade. The agent selects among
`SemanticSkillRuntime.available_calls` and supplies declarative object-centric
values; the compiler performs validation and grounding before the atomic engine
sees the request:

```text
MLLM / application SemanticCallSpec
    -> SemanticCallCatalog discovery
    -> SemanticIntegrationManifest validation
    -> SemanticSkillCompiler.analyze()
       object / affordance / resource / effect-flow validation
    -> SemanticSkillCompiler.ground(latest_context)
       participant binding + safe options + ActionInvocation
    -> SemanticSkillRuntime / AtomicActionEngine
    -> verified task state + structured execution events
```

Engine-only profiles, `JointPositionCommand` payloads, planner instances, live
objects, and concrete joint groups remain outside semantic call payloads. A
registered extension accepts only declarative data and requires an explicitly
installed version-matched lowerer. Invocation IDs and monotonic revisions
correlate compatible in-flight updates with planner diagnostics and execution
events without mutating a request implicitly.

The semantic runtime is also useful without an agent. `run()` executes one
known workflow, while `open_task()` and `run_segment()` retain verified state
across safe application decision boundaries. Call-local recovery remains owned
by `ExecutionRunner`; automatic skill replacement or symbolic-state
reconciliation after a terminal failure is intentionally not provided. See
{doc}`../semantic_skills` for the complete compiler/runtime and dynamic-task
contract.

## Extending the module

A new primitive should:

1. define a frozen, action-owned goal dataclass;
2. define a frozen `ActionOptions` subclass only for behavior that can vary per
   invocation;
3. declare `skill_id`, `GoalType`, `OptionsType`, an explicit
   `SkillBindingContract`, and agent visibility;
4. put reusable embodiment commands on control-part profiles and generic
   motion/recovery choices in invocation policies;
5. implement side-effect-free `_plan(request, context)` using the engine-owned
   planning services; do not override the framework-owned public `plan()`—the
   class definition is rejected if it does;
6. return a `TimedCommandSequence`, per-environment planning success, optional
   joint-trajectory and named-segment metadata, diagnostics, and uncommitted
   effects; joint planners can use `build_plan()`, while other endpoint types
   use `build_command_plan()`;
7. add registration coverage, contract tests, execution/recovery tests, a
   runnable example, and documentation.

See {doc}`builtin_actions` for the shipped skill catalog and visual demos, and
{doc}`/tutorial/atomic_actions` for complete usage patterns and runnable scripts.

## Further reading

- {doc}`../scene_registry` — canonical scene identity, snapshots, and collision integration
- {doc}`../semantic_skills` — semantic calls, compilation, runtime execution, and dynamic task boundaries
- {doc}`../planners/motion_generator` — the motion generator owned by the engine
- {doc}`../sim_robot` — robot control parts and kinematic configuration
- {doc}`/tutorial/atomic_actions` — static, closed-loop, and recovery examples
- `scripts/tutorials/atomic_action/moving_target_recovery.py` — runnable runner
  example that visibly moves a late-bound target, replans, and picks up the cube
- `scripts/tutorials/atomic_action/dynamic_obstacle_recovery.py` — cuRobo
  collision-world revision and obstacle-pose replanning example
