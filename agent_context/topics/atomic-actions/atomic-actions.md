# Atomic actions

## Current contract

Atomic actions are side-effect-free, environment-batched planners:

```python
plan = engine.plan(invocation: ActionInvocation, context: PlanningContext)
```

There is no `ActionTarget`, `WorldState`, `ActionResult`, `execute()`, or
`AtomicActionEngine.run()` compatibility surface.

`ActionInvocation` separates:

- an action-owned typed goal, validated against the action's `GoalType`;
- `ActionBinding`, which maps semantic roles to names from the engine robot's
  `control_parts` mapping;
- reusable `MotionPolicy` planner/timing choices;
- bounded `RecoveryPolicy` thresholds and retry budgets;
- optional typed `skill_options` and role-scoped `control_overrides` for one
  invocation revision.

`PlanningContext` separates measured `RobotObservation`, verified symbolic
`TaskState`, versioned `SceneSnapshot`, and environment IDs. An `ActionPlan`
contains per-environment planning success, one full-robot `TimedTrajectory`,
action-level recovery and scene-invalidation metadata, planner diagnostics,
named `TrajectorySegment` ranges, and an uncommitted `StateDelta`. Segments are
inspection/tracing metadata inside one trajectory; they are not independently
replannable execution boundaries.

`AtomicAction.build_plan()` normalizes the success mask and freezes unsuccessful
trajectory rows at the context's observed qpos; skill implementations should
return row-local success instead of duplicating failure-row masking.
Use `plan.segment(name)` for action-local half-open ranges and
`compiled.segment(action_index, name)` for concatenated coordinates; do not
recompute private sample splits in callers.

Each `AtomicActionEngine` exclusively owns one `ActionPlanningServices`
instance, which contains its robot, one `MotionGenerator`/planner backend, and
the legacy core's control-part command profiles. `MotionGenerator.generate()` is the only
stateful motion-planning entry point. `MotionPolicy.to_motion_gen_options()`
passes the invocation's `strategy` directly into `MotionGenOptions`; it is either
`"motion_gen"` or `"ik_interp"`. Target shaping, world-frame pose translation,
hand/joint interpolation used by composite actions, and full-robot trajectory
embedding are pure functions in `trajectory_ops.py`. Actions retain only an
owned copy of typed default options and borrow engine services. Engine
construction creates and binds a fresh instance of every type in
`BUILTIN_ACTION_TYPES`; use `load_builtins=False` only for isolated tests or a
fully custom action set. A bound action cannot be reused by another engine.

## Engine entry points

Choose the public engine entry point by lifecycle, not by skill type:

| Entry point | Use | Result and state behavior |
|---|---|---|
| `engine.plan(invocation, context)` | Inspect or plan one registered action | Returns one `ActionPlan`; does not project a context for another action |
| `engine.compile(invocations, context)` | Plan an ordered sequence against a fixed scene | Returns a concatenated `CompiledTrajectory`; propagates hypothetical qpos and expected effects through `projected_context` |
| `engine.start(invocations, context)` | Execute incrementally from observations | Returns an `ExecutionSession`; `tick(latest_context)` emits commands and performs bounded recovery |

None steps simulation directly. `compile()` never observes physical execution;
split compilation at observation boundaries when later goals depend on measured
results. Use `start()` when observation, effect verification, and replanning
must remain active during execution.

`AtomicAction.plan(request, context)` is the framework-owned template method
called by the engine, not a fourth application entry point. It binds collision
entities from the current scene into a copied motion policy before delegating
to the skill-specific `_plan()` hook. New actions implement `_plan()` and must
not override `plan()`. Custom actions must be installed with
`engine.register()` before using the same public entry points.

The `_plan()` extension boundary is an intentional hard break with no legacy
adapter. A subclass that defines `plan()` raises `TypeError` at class definition;
migrate an older custom action by renaming that implementation to `_plan()`.

## Robot skill profiles and resource binding

`embodichain.lab.sim.skills.RobotSkillProfile` is the authoritative
embodiment-level catalog for semantic resource binding. Its resource model is a
generic DAG, not a fixed arm/tool schema:

- `RobotResource.resource_id` is a stable logical ID. `endpoints` maps
  skill-local endpoint protocol names such as `motion` or `grasp` to
  `ResourceEndpoint` values, and `members` declares physical composition.
- `members` determines transitive claim closure only. It does not inherit or
  synthesize endpoint capabilities. A whole-body composite must declare its own
  whole-body capability and endpoint explicitly.
- `ResourceEndpoint` is the extension boundary for controller kinds. An exact
  endpoint-type `ResourceEndpointAdapter` resolves each declaration against the
  engine into an `EndpointResolution`: lowering values, an optional generic
  command-profile key, joint IDs, adapter-defined claim tokens, and exclusivity.
  `ControlPartEndpointAdapter` is installed by default for
  `ControlPartEndpoint`; integrations pass additional `endpoint_adapters` to
  profile or engine binding for mobile bases, whole-body controllers, or other
  endpoint kinds. Registration is by exact endpoint type, and the built-in
  adapter cannot be overridden; distinct controller semantics use a distinct
  endpoint subtype.
- Resources, profiles, and resolved bindings own independent endpoint
  snapshots. A custom endpoint whose nested payload cannot be deep-copied must
  override `snapshot()` and return a new value of its exact type.
- Binding snapshots adapter output as a `ResolvedResourceEndpoint`, including
  its resolved commands and claims. An exclusive resolution must declare at
  least one joint ID or claim token; a deliberately non-exclusive endpoint may
  omit both.
- A leaf must expose at least one endpoint. Member references must exist and the
  graph must be acyclic. On engine binding, physical leaves must own disjoint
  robot joints and adapter claim tokens; a composite endpoint may control only
  joints already covered by its transitive members.

Skills own the robot-independent side of the contract. A concrete
`AtomicAction` opts into semantic discovery by declaring a
`SkillBindingContract` in its own class body. The contract contains
skill-local `SkillResourceSlot` values; every slot requires named
`SkillEndpointRequirement` values with all-of capabilities, optional typed
semantic commands, and an optional `ActionBindingRoute`. Selecting one resource
per slot keeps related endpoints together, so a manipulation participant cannot
silently combine one arm with an unrelated tool. Endpoint views within that
resource may overlap by default, which permits an arm, mobile base, and
whole-body view to describe the same physical system. Add
`DisjointSlotEndpoints` to a slot only when selected endpoint views must be
physically disjoint. `DisjointResourceSlots` separately expresses pairwise
claim separation between selected participant resources.

`ActionBindingRoute` is only a transition adapter into the current core's
`manipulators` and `end_effectors` maps. Contract routes must cover the action's
declared core roles exactly. `BoundRobotSkillProfile.resolve()` returns a
`ResolvedSkillBinding` that retains the selected logical resources, the lowered
concrete `ActionBinding`, each resource's resolved endpoint data, and one
combined `ResourceClaim`. Direct-core callers may still construct
`ActionBinding` themselves, but that path does not perform profile capability
matching.

Discovery boundaries are distinct:

- `engine.actions` contains every installed action instance and is the
  direct-core registry.
- `engine.skills` contains descriptors only for installed, `agent_visible`
  actions whose concrete class explicitly declares a binding contract. A
  subclass does not inherit semantic exposure implicitly.
- `engine.skill_profile.skills` filters `engine.skills` again to contracts with
  at least one valid assignment on the bound robot. Registering or replacing an
  action invalidates the engine's bound profile; an independently retained
  `BoundRobotSkillProfile` also rejects use after the engine skill catalog
  changes and must be rebound.

Binding and policy authority is split deliberately:

- the action class owns its slot/endpoint/command requirement contract;
- the `RobotSkillProfile` owns the resource DAG, capability declarations,
  complete per-skill default `ResourceBinding` values, semantic command
  profiles keyed by generic profile IDs, and named `SkillPolicyPreset`
  snapshots; endpoint declarations or adapters select those profile IDs;
- the bound robot owns actual control-part membership and joint IDs, and its
  configured solver is checked for known solver-backed capabilities;
- endpoint adapters own controller-specific validation, physical claims, and
  lowering metadata;
- the engine owns installed actions, one planner backend, and the legacy
  control-part command profiles used by the current action core.

Constructing `AtomicActionEngine(..., skill_profile=profile)` makes the
profile's generic `command_profiles` the single authoritative constructor
source; passing `control_profiles` at the same time is rejected.
`command_profiles` values currently use `ControlPartCommandProfile` as their
immutable command container, but their mapping keys are generic profile IDs
rather than necessarily being control-part names.
`ControlPartEndpointAdapter` plus `RobotSkillProfile.action_control_profiles()`
is only the bridge that lowers applicable endpoint commands into the current
core's control-part-keyed profiles. Binding a profile to an already constructed
engine instead requires equivalent bridge commands to have been installed
already. A profile `JointPositionCommand` is one-dimensional and sized to the
adapter-resolved endpoint joint IDs; invocation `ActionControlOverrides` remain
the authority for one revision's per-environment replacements. Resolving a
custom endpoint's commands does not by itself add their controller transport to
the current action core.

Resolution selects a sole valid assignment automatically. If several remain,
it uses only a complete, currently valid per-skill default or enough explicit
slot selections; partial defaults and mapping/lexical order never disambiguate.
Preset lookup order is explicit preset, per-skill preset, then profile default,
and every returned preset is an owned snapshot. Planner-pinned presets must
match the engine's configured planner.

`ResourceClaim` contains transitive leaf-resource IDs, sorted concrete joint
IDs, and adapter-defined `claim_tokens`. Claims conflict when any category
overlaps, so a `whole_body` composite conflicts with a contained arm even when
their endpoint or control-part names differ. This is deterministic conflict
metadata only: there is no resource lease manager, parallel scheduler,
joint-mask command merger, or concurrency guarantee yet. `ExecutionSession`
and `ExecutionRunner` still emit, cancel, and hold full-robot joint commands. A
custom mobile/base endpoint can bind and participate in capability matching
once its adapter resolves it, including a controller claim token, but that does
not create a reusable navigation skill, planner/controller path, or command
transport. Do not treat successful binding or a non-conflicting claim as proof
of safe parallel or mobile execution.

## Object identity and pose grounding

`ObjectSemantics.entity_id` is the typed core's canonical snapshot-key lowering
target. The registry-backed path obtains it from a resolved `SceneEntityRef`.
It remains optional for advanced direct-core compatibility but, when supplied,
must be a non-empty string. Pose grounding with an explicit ID is strict:
resolve it only from the current `PlanningContext.scene`; a missing snapshot
entry is an error and never falls back to the live `entity`. Only when no ID is
supplied may the core read `ObjectSemantics.entity`; that path emits
`DeprecationWarning`, reads live state, and cannot declare a scene-motion
dependency.

`ObjectSemantics` is shallow-frozen. Top-level fields such as `entity_id`,
`entity`, and `label` cannot be rebound after construction; create a new
semantics value to change identity. Nested affordance and metadata objects may
remain mutable, but they never establish identity.

`SceneSnapshot` owns copies of input entity states and returns a defensive
`EntityState`/pose copy on every public mapping lookup. Mutating an input tensor
or a previously returned pose cannot change the published snapshot. Publish a
new scene version for every material dynamic-state change.

## Scene registry integration

`embodichain.lab.sim.skills.SceneRegistry` is the canonical integration catalog.
It owns immutable registration metadata: typed identity, aliases, pose source,
parent relationships, backend-local names, dynamics, geometry, collision role,
semantic type, and affordance data. A `SceneSnapshot` does not duplicate that
catalog; it contains only versioned dynamic pose/confidence and collision
revision state.

All object, articulation, link, and affordance IDs occupy one flat globally
unique namespace. Store link/affordance ancestry in
`SceneEntityRegistration.parent`, not by nesting or qualifying the ID. String
lookups may resolve aliases once to a canonical typed reference. An already
typed ref must contain a canonical ID and match the registered ref class.
Duplicate IDs, ambiguous aliases, alias/canonical collisions, missing parents,
and type mismatches fail at construction or lookup. Within one reference type,
the same `(parent, native_name)` physical source cannot be assigned multiple
canonical IDs; the same local name remains valid under different parents or
for different reference types.

`SceneRegistry.from_simulation()` is explicit opt-in. Its `rigid_objects` and
`articulations` mappings are `registry_id -> simulation_uid`; selected UIDs are
installed as aliases, and unlisted simulation entities are never scanned.
Collision participation defaults to `NONE`, and every static/dynamic collision
registration requires a geometry provider.

`registry.make_planning_scene_provider(motion_generator, batch_size=...)`
returns a fresh `RegistrySceneProvider` with independent baselines and revision
counters after eager registry/provider/planner validation. Snapshots expose
canonical IDs only. The provider requires stable ordered `env_ids` and
monotonic timestamps, derives relative affordance poses from the same
observation, compares movement against the last materially published pose, and
maintains per-row collision revisions. Plain `make_scene_provider()` is only
for perception and advanced direct-core consumers without planner agreement.

For an external perception/hardware provider, call
`registry.validate_collision_integration(..., scene_provider=provider)`
directly. The registry's complete `STATIC ∪ DYNAMIC` ID set must exactly
match `MotionGenerator.collision_world_entity_ids`; separately, the registry,
provider, and planner dynamic ID sets must match exactly in the canonical
namespace. The planner must support live updates for a non-empty dynamic set,
and planner/registry batch mode must agree. With dynamic entities, one
environment may infer `SHARED`; multiple environments must explicitly select
`SceneCollisionWorldMode.SHARED` or `PER_ENV`.

Construct a registry-backed cuRobo world with
`registry.collision_geometry_by_id()`. Its default mapping includes only
`STATIC` and `DYNAMIC` registrations and excludes `NONE`. Mapping keys are
canonical logical/source IDs for cache identity and full-world validation. With
`cuboid` or `mesh`, they are also the physical YAML and runtime-update keys.
Static `sphere` sources expand to backend names such as `id_0`; dynamic sphere
configuration is rejected, while cache/full-world identity stays on `id`.
Registry mappings fail fast when a source lacks geometry required by the chosen
representation. List-valued cuRobo worlds and `RigidObjectSceneProvider` remain
advanced direct-core paths.

Stable object identity follows these exact rules:

1. The same `ObjectSemantics` instance is identical to itself.
2. If either side has an explicit `entity_id`, both sides must have an explicit
   ID and the strings must match. Never compare an explicit ID directly with a
   legacy UID, even when the spellings are equal.
3. Only when both explicit IDs are absent, compare non-empty legacy
   `entity.uid` values. If either side has a valid UID, both must have one and
   the strings must match.
4. Only when neither side has an explicit ID or valid UID may identity fall back
   to the same live entity handle. `label` is descriptive and never establishes
   identity.

The direct-core identity rules do not perform alias resolution; normalization
belongs only to `SceneRegistry`. Partial-batch `StateDelta` attachment merges
use the same stable identity rules, so equivalent semantic wrappers update one
held object instead of creating label-based duplicates.

For both individual and coordinated attachments, a same-identity partial merge
preserves scalar metadata: if any previously active environment row remains,
the merged relation keeps `previous.semantics` and selects only the per-row
mask, transforms, and grasp poses from previous/candidate values. It adopts
`candidate.semantics` only when no previously active row survives the update.
This prevents an update for some environments from silently replacing the
semantic metadata shared by untouched rows.

Scene dependencies must match the poses each primitive actually consumes:

| Primitive | Scene dependencies |
|---|---|
| `MoveEndEffector` | A `SceneEntityPose` in `xpos`. |
| `MoveJoints` | None; its target is qpos or a named control-profile command. |
| `PickUp` | Always its semantic `entity_id`, when present, because the object pose is grounded once and reused; plus any goal-owned `SceneEntityPose`, such as `grasp_xpos`. |
| `CoordinatedPickment` | Goal-owned target/initial `SceneEntityPose` values; the semantic `entity_id` only when `object_initial_pose` is omitted and semantic grounding supplies that pose. |
| `Place` | A `SceneEntityPose` in ordinary `xpos`; for `AssembleGoal`, `base_pose` when supplied. Omitting `base_pose` uses the deprecated live `AssembleAffordance.base_object_entity` fallback with no dependency. |
| `MoveHeldObject` | A `SceneEntityPose` in `object_target_pose`; current object orientation is derived from observed EEF pose plus verified `object_to_eef`, not a scene-object read. |
| `Press` | `PressGoal.target_pose` when it is a `SceneEntityPose`; affordance data is entity-free. |
| `Slide` | `SlideGoal.target_pose` when it is a `SceneEntityPose`; the local grasp mesh does not own the link. |
| `Twist` | `TwistGoal.target_pose` when it is a `SceneEntityPose`; affordance data is entity-free. |
| `CoordinatedPlacement` | `SceneEntityPose` values in the placing or support object target pose. |
| `HandOver` | No semantic-object scene dependency. It verifies stable attachment identity and derives current pose from held state; its middle/final option poses are tensors, and the reused `GraspGoal.grasp_xpos` field is ignored. |

`collect_scene_dependencies()` deliberately stops at `ObjectSemantics`.
Therefore, a custom action that consumes a snapshot pose through semantic data
must override `_scene_dependencies()`, union `super()` dependencies, and add the
consumed semantic ID. Do not declare an ID merely because semantics are present.

## Static compilation

Built-ins are already registered by their class-level stable `skill_id`; call:

```python
compiled = engine.compile(invocations, context=None)
```

Compilation does not step simulation. It concatenates timed trajectories and
applies successful expected effects only to `compiled.projected_context`, so a
following action can be checked against hypothetical state. Failed rows hold
their last successful qpos.

Use invocation `skill_options` for multiple variants with the same stable
`skill_id`; do not create per-variant built-in instances.

Composite actions allocate their named trajectory segments from the total
sample budget with `split_three_segments()`. The first motion allocation rounds
`(sample_count - hand_interp_steps) * first_segment_ratio`; callers must not
reproduce that calculation or assume truncation.

## Dynamic execution and recovery

`SceneEntityPose(entity_id, relative_pose)` is resolved from the latest scene
snapshot every time the action plans. Its entity ID is recorded in
`ActionPlan.scene_dependencies`.

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
emits at most one `JointCommand` per tick. The command's per-environment
`hold_duration` schedules the next feedback cycle from `TimedTrajectory.dt`:
command `i` carries the arrival interval `dt[:, i + 1]` leading to the next
waypoint. The final command reuses its own interval as a settling window. The
session monitors:

- joint tracking error against the previous command;
- translation/rotation drift of referenced scene entities;
- per-environment collision-world revision changes for collision-sensitive
  actions;
- action-attempt timeout;
- planner and semantic-effect failure.

It replans from the latest observation within per-environment budgets. The
budgets and eligibility masks are row-local, while the action waypoint cursor
is batch-synchronized: one allowed replan regenerates the active cohort and
restarts its action trajectory without charging unaffected rows. Unknown
or exhausted failures are reported as structured `ExecutionEvent` objects. A
non-empty `StateDelta` is not committed until the caller supplies an external
`effect_success` mask. While verification is outstanding,
`ExecutionTick.pending_effect` retains a typed `EffectVerificationRequest` on
every tick; `EFFECT_VERIFICATION_REQUIRED` is only the one-time audit event.

Recovery replans reuse the current immutable `ResolvedActionRequest`, including
its owned goal snapshot. Mutable goal values are copied, while simulator-backed
`BatchEntity` handles retain their runtime identity. To change a goal, option,
policy, binding, or control command during execution, submit a strictly newer
revision explicitly:

```python
session.revise_current(revised_invocation)
```

The replacement must keep the active `skill_id` and `invocation_id`. The
session resolves a new snapshot, resets that revision's recovery budgets, and
replans from the latest context.

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
`ExecutionSession` dispatches sample zero immediately, then maps each following
arrival interval to the preceding command's `JointCommand.hold_duration`. The
final sample uses its own interval again as a settling window before terminal
validation. Batched execution currently advances at a synchronized barrier
using the longest active row interval.

`SimulationExecutionAdapter` implements observation, command, and clock ports
for a `SimulationManager`/`Robot` pair. Its `sleep()` advances an integral
number of physics steps, so simulation execution does not depend on wall time.
Stable context IDs are correlation identifiers; the adapter maps command rows
to simulation robot indices rather than using those IDs as array indices.
Real-device adapters should implement the same protocols and enforce the passed
acknowledgement timeout in their transport/controller layer.

`SceneProvider.snapshot(timestamp=..., env_ids=...)` is the scene-observation
boundary used by execution adapters. `SceneSnapshot.collision_entity_ids`
identifies obstacle poses consumed by a planner, while
`collision_world_revision` is either global or per environment. A newer
revision invalidates only affected batch rows. `RegistrySceneProvider` is the
canonical provider and derives its entity/collision sets from one immutable
`SceneRegistry`. It filters sub-threshold pose noise, advances the general scene
version, and maintains per-environment collision revisions. Thresholds are
measured from the last materially published pose per entity and environment, so
repeated sub-threshold motion eventually becomes observable.
`RigidObjectSceneProvider` retains that lower-level revision behavior for
advanced direct-core integrations.
For lightweight sources that do not need environment correlation IDs,
`SimulationExecutionAdapter` also accepts a mutually exclusive
`SceneSnapshotSupplier(timestamp)` callback.

The public `AtomicAction.plan()` copies `MotionPolicy` and binds collision entity
poses through `MotionGenerator.bind_collision_world()`. The motion generator
owns option copying and the backend capability boundary, then forwards the
update through `BasePlanner.with_collision_world()`. Backends opt in via
`supports_collision_world_updates`; cuRobo implements this bridge using
`CuroboPlanOptions.dynamic_obstacle_poses`. Replanning therefore consumes the
same scene snapshot that triggered invalidation without adding obstacle
parameters to each skill. Add/remove/geometry mutations are not yet supported
by this pose-update path; providers should revision only pose-updatable
registered obstacles.

`BasePlanner.collision_world_entity_ids`, `dynamic_collision_entity_ids`, and
`collision_world_batch_mode` expose the backend's complete world, dynamic
subset, and batching contract. `MotionGenerator` validates and forwards those
properties for `SceneRegistry.make_planning_scene_provider()`. External
providers call `validate_collision_integration(..., scene_provider=...)`.
These construction checks are separate from per-plan
`bind_collision_world()`.

Runnable closed-loop examples live under `scripts/tutorials/atomic_action/`:
`tracking_error_recovery.py`, `moving_target_recovery.py`, and
`dynamic_obstacle_recovery.py`. Each injects one disturbance, reports the
structured invalidation/replan events, and requires terminal completion.

The latest validated session context is retained for safe hold if the first
live observation fails. Environment IDs must remain stable and ordered for the
entire session; robot and scene timestamps and scene versions must be monotonic.
Collision-world revisions must also remain monotonic per environment.

## Parameter ownership

Goal dataclasses carry only semantic task intent. They do not carry robot part
names, planner configuration, retry policy, or runtime state.

`MotionPolicy` owns planner selection, motion strategy, sample count, fallback
control period, limits, dynamic-collision mode, and typed planner options.
`DynamicCollisionMode.AUTO` consumes a live collision world when available,
`OFF` ignores snapshot collision entities and their revisions, and `REQUIRED`
fails unless the motion strategy, scene, and planner support that path. These
modes do not toggle backend-configured static-world or self-collision checks.
`RecoveryPolicy` owns tracking/dynamic-goal thresholds, timeouts, and budgets.
Each built-in has a frozen `*Options` value for invocation-varying segment counts,
offsets, and grasp selection behavior. An action constructor may accept
`default_options`; an invocation's `skill_options` replaces them for that call.
There is no `ActionCfg` or built-in `*Cfg` layer.

`engine.register(action)` is reserved for custom skill implementations. A
built-in can be replaced only with explicit `replace=True`. Registration means
an implementation is installed; it does not prove that the current embodiment
has compatible control parts, profiles, bindings, or task state. Capability
has compatible control parts, profiles, bindings, or task state. `engine.skills`
contains only agent-visible installed actions whose concrete classes explicitly
declare a `binding_contract`; when a robot profile is bound,
`engine.skill_profile.skills` further filters that catalog to valid resource
assignments. Registration is engine-local; there is no independent process-wide
action catalog. Construct extensions explicitly and install them with
`engine.register()` so discovery and execution cannot observe disconnected
registries.

`ExecutionRunnerCfg` is intentionally separate from action options. It
configures controller acknowledgement deadlines, scheduler cadence, and final
safe-hold behavior for one runner instance; it does not change skill planning
semantics and does not belong in `ActionInvocation` or an invocation revision.

Every `ActionBinding` value is a `RobotCfg.control_parts` key. It is not a link,
TCP-frame, joint, or scene-object name. Planning services validate those names
and resolve immutable `ResolvedControlPart` values containing full-robot joint
indices. Built-ins use the binding as the only source for participating arm and
hand names; attachment state and `StateDelta` keys use the bound manipulator.
`TaskState.held_objects` is the sole attachment map. A multi-manipulator grasp
stores one `HeldObjectState` per manipulator with the same `ObjectSemantics`
instance. `TaskState.held_object_mask()` exposes active rows, while
`exclusive_held_object_mask()` excludes rows where another manipulator holds
the same semantic object or live entity. Single-arm transport, release, and
handover operations only succeed on exclusive rows; coordinated placement
likewise requires two distinct, exclusively held objects.

Embodiment-specific joint commands do not belong to Action options. A caller
using the legacy direct-core path without a `RobotSkillProfile` registers them
by actual control-part name:

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
tutorial may derive a simple profile from limits explicitly. Profile-based
integrations instead own commands under generic `command_profiles` IDs and let
endpoint declarations/adapters resolve those IDs; only
`action_control_profiles()` converts applicable control-part endpoints back to
the legacy core mapping.

## Built-ins

| Skill ID | Goal type | Roles |
|---|---|---|
| `move_end_effector` | `EndEffectorPoseGoal` | manipulator `primary` |
| `move_joints` | `JointPositionGoal` (`target` is explicit qpos or a profile command name) | manipulator `primary` |
| `pick_up` | `GraspGoal` | manipulator/end effector `primary` |
| `move_held_object` | `HeldObjectPoseGoal` | manipulator/end effector `primary` |
| `place` | `PlaceGoal`, `AssembleGoal` | manipulator/end effector `primary` |
| `press` | `PressGoal` | manipulator/end effector `primary` |
| `slide` | `SlideGoal` | manipulator/end effector `primary` |
| `twist` | `TwistGoal` | manipulator/end effector `primary` |
| `coordinated_pickment` | `CoordinatedPickGoal` | `left`, `right` |
| `coordinated_placement` | `CoordinatedPlacementGoal` | `placing`, `support` |
| `hand_over` | `GraspGoal` | `source`, `destination` |

`PressAffordance`, `SlideAffordance`, and `TwistAffordance` contain only
target-local geometry and interaction semantics. Their goals own an explicit
`target_pose`, which may be a deterministic tensor snapshot or a late-bound
`SceneEntityPose`. Never put an `Articulation`, `RigidObject`, or live link pose
reader in these affordances.

`Press` and `Slide` use dense axis-aligned Cartesian targets for their contact
motion. The linear motion-generator path solves every output sample with IK;
it does not resample sparse IK endpoints in joint space. `Press` has a distinct
contact segment before penetration. `TwistAffordance.axis_origin` and
`twist_axis` together define the full 3D rotation axis.

These three motion-centric primitives declare `SkillDescriptor.open_loop=True`
and an empty `StateDelta`. Their completion means motion execution only, not
verified button actuation, grasp retention, or articulation travel. Applications
that need semantic completion must observe and verify those physical outcomes.

`GraspGoal.grasp_xpos` accepts an explicit pose tensor, a late-bound
`SceneEntityPose`, or `None` for affordance sampling. A `SceneEntityPose`
registers the referenced entity as a recovery dependency, allowing an executing
`PickUp` to replan when the grasp target moves. `PickUp` also resolves its
semantic object's pose once per planning attempt and declares the semantic
`entity_id` because grasp sampling, upright adjustment, and the held
`object_to_eef` relation all consume that same pose.

`AssembleGoal.base_pose=SceneEntityPose(...)` is the canonical assembly anchor
and becomes a recovery dependency. An omitted `base_pose` permits the deprecated
live `AssembleAffordance.base_object_entity` fallback for direct-core callers
only; it is not dependency-tracked. The current `assemble.py` tutorial exercises
that legacy fallback, while `moving_target_recovery.py` is the canonical
snapshot-grounded object example.

## Extension rules

1. Define a frozen action-owned goal dataclass.
2. Define a frozen `ActionOptions` subclass only when runtime behavior exists.
3. Declare `skill_id`, `GoalType`, `OptionsType`, and required core roles. Also
   declare a class-local `SkillBindingContract` when the skill should appear in
   `engine.skills`; route every current core role exactly once.
4. Implement `_plan()`; do not override the framework-owned `plan()` method.
5. Validate with `require_goal(request)` and consume only the resolved binding.
6. Plan from `context.robot.qpos`; never read an implicit live start state.
7. If planning consumes a semantic object's snapshot pose, override
   `_scene_dependencies()`, preserve `super()` dependencies, and add exactly
   that semantic ID.
8. Return full-robot positions or a `TimedTrajectory` through `build_plan()`.
   Build batched `list[PlanState]`, translate the policy with
   `request.motion_policy.to_motion_gen_options()`, and call
   `self.motion_generator.generate()`. Import pure operations directly from
   `trajectory_ops.py`.
9. Declare symbolic changes with `StateDelta`; do not mutate context or commit
   physical effects during planning. For partial attachment updates, retain
   previous scalar semantics while any previous row remains; merge only batched
   masks/transforms and adopt candidate semantics only on full replacement.
10. Keep scene stepping, controller I/O, and task-graph/MLLM logic outside the
   atomic action. Put execution-loop I/O behind the runner protocols rather than
   calling a simulator or device from `plan()` or `ExecutionSession`.
