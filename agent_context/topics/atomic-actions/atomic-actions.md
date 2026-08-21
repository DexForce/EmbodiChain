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
- an engine-owned `ActionBinding`, which covers the skill contract by exact
  `(slot_id, endpoint_id)` keys and terminates every endpoint at an immutable
  `RuntimeEndpointTarget`;
- reusable `MotionPolicy` strategy, sampling, collision, and backend options;
- bounded `RecoveryPolicy` thresholds and retry budgets;
- optional typed `skill_options` and endpoint-scoped `control_overrides` for
  one invocation revision.

`PlanningContext` separates measured `RobotObservation`, verified symbolic
`TaskState`, versioned `SceneSnapshot`, environment IDs, and an optional
explicit `control_dt` used only by action-owned interpolation. An `ActionPlan`
contains per-environment planning success, an authoritative
`TimedCommandSequence` in `commands`, an optional full-robot `TimedTrajectory`
in `joint_trajectory`, action-level recovery and scene-invalidation metadata,
planner diagnostics, named `TrajectorySegment` frame ranges, and an uncommitted
`StateDelta`. Segments are inspection/tracing metadata inside one command
sequence; they are not independently replannable execution boundaries.

`AtomicAction.build_plan()` is the planner-backed joint convenience path: it
normalizes the success mask, freezes unsuccessful trajectory rows at the
context's observed qpos, and lowers the trajectory through bound
`JointPositionTarget` values. `AtomicAction.build_command_plan()` is the generic
extension boundary for transport-neutral command sequences. Both mask failed
rows; skill implementations should return row-local success instead of
duplicating that work.
Use `plan.segment(name)` for action-local half-open ranges and
`compiled.segment(action_index, name)` for concatenated coordinates; do not
recompute private sample splits in callers.

Each `AtomicActionEngine` exclusively owns one `ActionPlanningServices`
instance, which contains its robot, one `MotionGenerator`/planner backend, and
its direct control-part command-profile snapshot. It also issues an opaque
binding-owner ID, so an `ActionBinding` cannot cross engine instances. It does
not own a timing fallback. Planner results with positions require explicit `dt`;
`duration` is derived from it. Actions must pass a complete `TimedTrajectory` to
`build_plan()`. Environment-backed integrations put `BaseEnv.step_dt` on
`PlanningContext.control_dt` when action-owned interpolation needs a cadence.
`MotionGenerator.generate()` is the only stateful motion-planning entry point.
`MotionPolicy.to_motion_gen_options()` passes the invocation's `strategy`
directly into `MotionGenOptions`; it is either `"motion_gen"` or `"ik_interp"`.
Target shaping, world-frame pose translation, hand/joint interpolation used by
composite actions, and full-robot trajectory embedding are pure functions in
`trajectory_ops.py`. Actions retain only an owned copy of typed default options
and borrow engine services. Engine construction creates and binds a fresh
instance of every type in `BUILTIN_ACTION_TYPES`; use `load_builtins=False` only
for isolated tests or a fully custom action set. A bound action cannot be
reused by another engine.

## Engine entry points

Choose the public engine entry point by lifecycle, not by skill type:

| Entry point | Use | Result and state behavior |
|---|---|---|
| `engine.plan(invocation, context)` | Inspect or plan one registered action | Returns one `ActionPlan`; does not project a context for another action |
| `engine.compile(invocations, context)` | Plan an ordered sequence against a fixed scene | Returns a concatenated `CompiledTrajectory`; propagates hypothetical qpos and expected effects through `projected_context` |
| `engine.start(invocations, context, *, eligible_mask=None)` | Execute incrementally from observations | Returns an `ExecutionSession`; the optional initial cohort is sticky, and `tick(latest_context)` emits commands and performs bounded recovery |

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
  engine into an `EndpointResolution`: a `RuntimeEndpointTarget`, an optional
  generic command-profile key, joint IDs, adapter-defined claim tokens, and
  exclusivity. `ControlPartEndpointAdapter` is installed by default for
  `ControlPartEndpoint` and produces a `JointPositionTarget`. Integrations pass
  additional `endpoint_adapters` to profile or engine binding for mobile bases,
  whole-body controllers, or other endpoint kinds. Registration is by exact
  endpoint type, and the built-in adapter cannot be overridden; distinct
  controller semantics use a distinct endpoint subtype.
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
semantic commands, and no fixed arm/tool role or route layer. Selecting one
resource per slot keeps related endpoints together, so a participant cannot
silently combine endpoint views from unrelated resources. Endpoint views within
that resource may overlap by default, which permits an arm, mobile base, and
whole-body view to describe the same physical system. Add
`DisjointSlotEndpoints` to a slot only when selected endpoint views must be
physically disjoint. `DisjointResourceSlots` separately expresses pairwise
claim separation between selected participant resources.

Profile binding lowers every selected endpoint directly into an
`EndpointBinding`. Its `target` supplies immutable runtime addressing
(`transport_id`, `target_id`); its semantic commands, capabilities, and claim
tokens remain attached to the same endpoint. `BoundRobotSkillProfile.resolve()`
returns a `ResolvedSkillBinding` that retains the selected logical resources,
the engine-owned `ActionBinding`, each resource's resolved endpoint data, and
one combined `ResourceClaim`.

Advanced callers without a profile use
`engine.bind_control_parts(skill, endpoints)` with an exact nested
`slot -> endpoint -> control_part` mapping. The engine accepts an installed
skill ID or an explicit action instance later passed to `plan_action()`, checks
contract coverage, control-part existence, required commands, ownership, and
disjointness, then emits the same generic `ActionBinding` with
`JointPositionTarget` endpoints. Callers do not construct bindings manually,
and this path deliberately does not perform profile resource discovery or
capability matching.

`engine.make_invocation(skill_id, goal, ...)` is the convenience construction
boundary when callers do not need to retain a binding separately. Pass
`control_parts` for the direct path, or rely on a bound `RobotSkillProfile` and
optionally pass `resources` as `slot -> resource_id` selections. The two binding
sources are mutually exclusive. Without a profile, `control_parts` is required;
with a profile, omitting `resources` uses unique or configured-default profile
resolution. The method returns an ordinary `ActionInvocation` and does not plan
or execute it. It resolves bindings only; profile policy presets and runner
configuration remain semantic-runtime concerns.

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
  immutable runtime-target lowering;
- runtime payload types own immutable command values, while
  `EndpointCommandTransport` implementations own live controller/client state
  and execute only payloads whose `transport_id` matches their targets;
- the engine owns installed actions, one planner backend, its binding identity,
  and direct control-part command-profile snapshots.

Constructing `AtomicActionEngine(..., skill_profile=profile)` makes the
profile's generic `command_profiles` the single authoritative constructor
source; passing `control_profiles` at the same time is rejected.
`command_profiles` values currently use `ControlPartCommandProfile` as their
immutable command container, but their mapping keys are generic profile IDs
rather than necessarily being control-part names.
`ControlPartEndpointAdapter` plus `RobotSkillProfile.action_control_profiles()`
provides the direct control-part lookup used by built-in joint planners when an
engine is constructed from a profile; it is not a binding route. Binding a
profile to an already constructed engine instead requires equivalent direct
control-part commands to have been installed already. Profile resolution still
places all resolved semantic commands, including commands for custom endpoint
types, on their `EndpointBinding`. A profile `JointPositionCommand` is
one-dimensional and sized to the adapter-resolved endpoint joint IDs;
invocation `ActionControlOverrides` remain the authority for one revision's
per-environment endpoint-command replacements.

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
or concurrency guarantee yet. Dynamic execution can dispatch multiple
endpoint commands in one synchronized frame, but that does not imply resource
scheduling or safe parallelism. A custom mobile/base or whole-body endpoint is
executable only when its adapter supplies a target, the action emits a matching
runtime payload, and the target's transport is registered with the
`EndpointCommandRouter`. Successful binding or a non-conflicting claim alone is
not proof that a planner/controller path or safe concurrent execution exists.

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

`SceneEntityMetadata` is the single provider-free scene declaration model.
`SceneEntityManifest` specializes that value without redeclaring its fields,
and both `SceneManifest` and `SceneRegistry` use the same canonical ID, alias,
parent, native-member, and affordance index. A `SceneManifest` additionally
snapshots `collision_world_mode`; `SemanticIntegrationManifest.bind()` rejects
live metadata or collision-mode drift before installing a robot profile.

## Semantic workflow compilation

Semantic calls (`Pick`, `Place`, `HandOver`, and catalog-registered values) are
robot-independent declarations. `SemanticCallDescriptor` has one canonical
atomic `target_descriptor`; its `skill_id` and `binding_contract` are derived
views, not separately stored values. Curated call targets cannot be remapped.
Registered calls require an explicit agent-visible target plus an installed
`RegisteredSemanticLowerer` with a matching call ID and schema version.

`SemanticSkillCompiler.analyze()` performs provider-free linking, resource and
affordance validation, held-object flow analysis, and first-release look-ahead.
A `Place` with no explicit `primary` resource inherits the workflow's known
holder resource, and a `HandOver` with no explicit `source` does the same. The
inferred selection is snapshotted onto the canonical linked call before
binding; an explicit conflicting selection still fails with
`held_resource_mismatch`. Inference never crosses a registered-call boundary.
`HandOver` selects participants only through the `source` and `destination`
resource slots; there is no separate receiver alias.
A pick therefore owns zero or one downstream object target rather than an
arbitrary target tuple. Relation targets retain affordance payload type and
revision metadata and stay late-bound through an explicitly installed
`RelationTargetGrounder`; handover poses stay behind a named
`HandOverPoseProvider` selected by the robot profile.

`SemanticSkillCompiler.ground()` lowers exactly one analyzed call from the
latest `PlanningContext` and returns a `GroundedSemanticCall`. Its
`eligible_mask` is an owned snapshot and must be handed to execution together
with the invocation:

```python
grounded = compiler.ground(workflow, call_index, context, eligible_mask=cohort)
session = engine.start(
    (grounded.invocation,),
    context,
    eligible_mask=grounded.eligible_mask,
)
```

The compiler identity prevents a workflow from crossing lowerer/grounder
registries. Engine/profile staleness is checked through the bound integration;
the workflow does not duplicate engine-owner or catalog-revision fields.

## Semantic runtime facade

`embodichain.lab.sim.skills.SemanticSkillRuntime` is the application-facing
orchestration layer. `bind()` connects an explicit manifest, registry, engine,
observation provider, command sink, and clock; `from_simulation()` assembles the
standard joint-position simulation ports while still requiring an explicit
registry, robot profile, and motion generator. Its optional `control_dt`
selects a command cadence independently of the simulation physics period. A
runtime-level `runner_cfg` overrides all calls; when omitted, each grounded
call uses the `ExecutionRunnerCfg` owned by its selected `SkillPolicyPreset`.
The runtime exposes only calls supported by both the semantic catalog and the
currently bound robot profile, and allows exactly one active `SemanticTask`
because no resource scheduler or lease manager exists.

`SemanticSkillRuntime.run()` is the blocking one-segment convenience path and
requires a `SemanticEffectVerifier`. Use `start()` when effect verification is
asynchronous. A `SemanticTask` retains externally verified `TaskState`, stable
environment IDs, and the sticky eligible cohort across several independently
analyzed segments. `run_segment()` supports dynamic application decisions at
safe semantic-call boundaries; submit all known calls in one segment when Pick
look-ahead should account for a downstream Place or HandOver target.

`SemanticExecution` always JIT-grounds and starts one invocation at a time. It
uses a fresh observation before each call, delegates local recovery and safe
stop to `ExecutionRunner`, commits only verified effects, then carries the
session's task state and eligibility into the next grounding boundary. Manual
execution reports `WAITING_FOR_EFFECT` and resumes through `step(effect_success=...)`;
compatible in-place call changes use `revise_current()`, which reanalyzes the
workflow and still inherits the runner's same-skill, same-invocation, and
same-runtime-address restrictions. Runtime failures remain terminal; automatic
task-level skill replacement or symbolic-state reconciliation is not provided.
A failed or cancelled segment closes its task and releases runtime ownership;
successful dynamic segments retain ownership until `finish()` or cancellation.

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
| `PickUp` | Always its semantic `entity_id`, when present, because the object pose is grounded once and reused; plus any goal-owned `SceneEntityPose`, such as `grasp_xpos`. These dependencies are monitored only through the `approach` segment. |
| `CoordinatedPickment` | Goal-owned target/initial `SceneEntityPose` values; the semantic `entity_id` only when `object_initial_pose` is omitted and semantic grounding supplies that pose. |
| `Place` | A `SceneEntityPose` in ordinary `xpos`; for `AssembleGoal`, `base_pose` when supplied. Omitting `base_pose` uses the deprecated live `AssembleAffordance.base_object_entity` fallback with no dependency. |
| `MoveHeldObject` | A `SceneEntityPose` in `object_target_pose`; current object orientation is derived from observed EEF pose plus verified `object_to_eef`, not a scene-object read. |
| `Press` | `PressGoal.target_pose` when it is a `SceneEntityPose`; affordance data is entity-free. |
| `Slide` | `SlideGoal.target_pose` when it is a `SceneEntityPose`; the local grasp mesh does not own the link. |
| `Twist` | `TwistGoal.target_pose` when it is a `SceneEntityPose`; affordance data is entity-free. |
| `CoordinatedPlacement` | `SceneEntityPose` values in the placing or support object target pose. |
| `HandOver` | `SceneEntityPose` values in `HandOverOptions.middle_object_pose` or `final_object_pose`. Its current held-object pose is derived from verified attachment state and observed EEF pose; the reused `GraspGoal.grasp_xpos` field is ignored. |

`collect_scene_dependencies()` deliberately stops at `ObjectSemantics`.
Therefore, a custom action that consumes a snapshot pose through semantic data
must override `_scene_dependencies()`, union `super()` dependencies, and add the
consumed semantic ID. Do not declare an ID merely because semantics are present.
`ActionPlan.scene_dependency_end_segment` can bound dynamic-goal monitoring to
the reversible part of a staged action. `PickUp` stops monitoring after its
approach is dispatched: target motion before contact still replans, while
contact-, grasp-, and lift-induced object motion is not misclassified as an
external target update. Collision-world and joint-tracking checks remain
independent of this boundary.

## Static compilation

Built-ins are already registered by their class-level stable `skill_id`; call:

```python
compiled = engine.compile(invocations, context=None)
```

Compilation does not step simulation. It concatenates timed trajectories and
applies successful expected effects only to `compiled.projected_context`, so a
following action can be checked against hypothetical state. Because
`CompiledTrajectory` is a joint-trajectory result, every action plan in a
compiled sequence must own `joint_trajectory`; `compile()` rejects a generic
runtime-command plan without one. Use `start()` plus an execution runner for
plans whose authoritative `commands` target non-joint transports. Failed joint
rows hold their last successful qpos.

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
session = engine.start(
    invocations,
    initial_context,
    eligible_mask=initial_eligible_mask,
)
runner = ExecutionRunner(
    session,
    observation_provider,
    command_sink,
    clock=execution_clock,
)
result = runner.step(effect_success=None)
```

`ExecutionSession` owns deterministic planning progress and recovery state. It
emits at most one synchronized `RuntimeCommandFrame` per tick from the plan's
authoritative `TimedCommandSequence`. A frame contains one or more
`EndpointCommand` values, a shared environment batch and active mask, and a
per-environment `hold_duration`. Every command pairs a
`RuntimeEndpointTarget` with a `RuntimeCommandPayload`; their `transport_id`
values must match, destinations must be unique within the frame, and joint
targets may not overlap. `ExecutionFeedbackMode.JOINT_POSITION` requires an
owned `joint_trajectory` and joint-position targets/payloads; generic command
plans default to timed completion and retain external semantic-effect
verification. Framework authorization replaces every emitted target with its
binding-owned snapshot and rejects unbound destinations, target substitution,
and endpoint claim conflicts. A plan's non-empty frames and its recovery
replans retain a stable destination set. Empty failed plans retain previously
active targets so the caller can still hold them. The session monitors:

- joint tracking error against the previous command in joint-position mode;
- translation/rotation drift of referenced scene entities;
- per-environment collision-world revision changes for collision-sensitive
  actions;
- action-attempt timeout;
- planner and semantic-effect failure.

The optional initial `eligible_mask` is copied onto the engine device and must
be a boolean tensor with one value per environment. Initially ineligible rows
never re-enter the cohort. They are excluded from every command, replan, effect
verification, and later invocation barrier. An all-false cohort creates a
failed session without invoking any action planner.

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
runner.revise_current(revised_invocation)
```

The replacement must keep the active `skill_id` and `invocation_id`. The
session resolves a new snapshot, resets that revision's recovery budgets, and
replans from the latest context. Once runtime destinations are owned, the
replacement must preserve a non-empty destination set and every target address
fingerprint; changing a base, whole-body, arm, controller, or safe-hold
footprint requires a new invocation. The runner snapshots the revision, keeps
the current frame deadline, then observes and installs it at the next due
boundary. Pending physical effects must be verified first, or the caller must
cancel and start a new invocation. A caller that owns manual session ticks may
use `session.revise_current(..., context=fresh_context)` directly.

`ExecutionRunner` owns the controller-facing lifecycle around a session:

- `ObservationProvider.observe(task_state)` supplies a fresh, monotonically
  timestamped `PlanningContext` when a feedback cycle is due;
- `CommandSink.send(frame)`, `hold(targets, context)`, and `cancel(targets)`
  return a `CommandAcknowledgement` with `accepted`, `rejected`, or
  `timed_out` status;
- `EndpointCommandRouter` is the standard mixed-controller sink. It preflights
  every frame, groups commands or targets by exact transport ID, dispatches to
  registered `EndpointCommandTransport` implementations, and accepts only when
  every addressed transport accepts;
- `ExecutionClock` supplies monotonic time and backend waiting;
- non-blocking `step()` dispatches only when the current command's
  `hold_duration` has elapsed;
- `revise_current()` stages an owned same-address revision, preserves the active
  frame deadline, and replans it from the next due observation;
- `run_until_blocked()` is a convenience loop that waits through the clock and
  stops at a terminal state or an unhandled effect-verification boundary; the
  runner remembers that boundary so a later verifier call can resume it;
- before dispatch, the runner records every target that may become armed by
  `(transport_id, target_id)`; cancellation, observation/session/controller
  exceptions, and negative acknowledgements enter target-scoped safe stop:
  cancel all recorded targets first, then hold them from a fresh observation or
  the last validated context when one is available.

Every transport must actively neutralize inactive rows for each addressed
target. Omission is unsafe for persistent controllers: position transports
hold those rows and velocity transports normally command zero velocity.

`TimedTrajectory.dt[:, i]` is the interval leading to sample `i`.
The built-in joint lowerer dispatches sample zero immediately, then maps each
following arrival interval to the preceding `RuntimeCommandFrame`'s
`hold_duration`. The final frame uses its own interval again as a settling
window before terminal validation. Generic action implementations set frame
hold durations directly. Batched execution currently advances at a
synchronized barrier using the longest active row interval.

`SimulationExecutionAdapter` implements observation and clock ports plus the
exact `robot.joint_position` endpoint transport for a
`SimulationManager`/`Robot` pair. It can serve directly as the command sink for
joint-only plans or be registered in an `EndpointCommandRouter` beside mobile,
whole-body, or device-specific transports. Its `sleep()` advances an integral
number of physics steps, so simulation execution does not depend on wall time.
Stable context IDs are correlation identifiers; the adapter maps command rows
to simulation robot indices rather than using those IDs as array indices.
Real-device transports should implement `EndpointCommandTransport` and enforce
the passed acknowledgement timeout in their controller/client layer.

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
update through `BasePlanner.with_collision_world()`. Backends opt in through
`BasePlanner.collision_world_info.supports_updates`; cuRobo implements this
bridge using `CuroboPlanOptions.dynamic_obstacle_poses`. Replanning therefore
consumes the same scene snapshot that triggered invalidation without adding
obstacle parameters to each skill. Add/remove/geometry mutations are not yet
supported by this pose-update path; providers should revision only
pose-updatable registered obstacles.

`BasePlanner.collision_world_info` exposes the backend's complete world,
dynamic subset, batching mode, and update capability as one immutable contract.
`MotionGenerator` validates and forwards it for
`SceneRegistry.make_planning_scene_provider()`. External providers call
`validate_collision_integration(..., scene_provider=...)`. These construction
checks are separate from per-plan `bind_collision_world()`.

Runnable closed-loop examples live under `scripts/tutorials/atomic_action/`:
`tracking_error_recovery.py`, `moving_target_recovery.py`, and
`dynamic_obstacle_recovery.py`. Each injects one disturbance, reports the
structured invalidation/replan events, and requires terminal completion.

Semantic integration tutorials live under `scripts/tutorials/semantic_skill/`.
Both examples separate `create_*_application()` (scene/profile/runtime and
default verifier wiring), `create_*_task()` (robot-independent semantic calls),
and the application-facing `app.run(task, ...)` entry. `app` remains a
`SemanticSkillRuntime`; there is no tutorial-specific facade. `place.py`
executes `Pick -> Place`, verifying the observed lift, planned object-to-EEF
relation, release pose, and open hand. `hand_over.py` demonstrates disjoint
dual-arm resources plus an explicit `RegisteredSemanticLowerer`, then verifies
source release and receiver ownership at the final target. Both report
structured recovery events and use `--diagnose_plan` only for a separate
offline compile that projects hypothetical effects without executing them.
Release and ownership-transfer presets disable whole-action effect retries
because those physical changes are not safely repeatable without state
reconciliation.

Human-facing architecture and lifecycle documentation lives in
`docs/source/overview/sim/semantic_skills.md`; the runnable walkthrough is
indexed at `docs/source/tutorial/semantic_skills.rst`.

The latest validated session context is retained for safe hold if the first
live observation fails. Environment IDs must remain stable and ordered for the
entire session; robot and scene timestamps and scene versions must be monotonic.
Collision-world revisions must also remain monotonic per environment.

## Parameter ownership

Goal dataclasses carry only semantic task intent. They do not carry robot part
names, planner configuration, retry policy, or runtime state.

`MotionPolicy` owns motion strategy, sample count, dynamic-collision mode, and
typed planner options. Optional planner-backend compatibility belongs to
`SkillPolicyPreset.required_planner`; velocity and acceleration constraints
belong to the selected backend's typed `PlanOptions`. Timing belongs to the
trajectory producer: planners return explicit `dt` with derived `duration`,
while custom or composite interpolation constructs a `TimedTrajectory` using
an explicit cadence such as `PlanningContext.require_control_dt()`. Missing
timing is an error rather than an engine-owned default.
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
discovery is separate: `engine.skills`
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

`ActionBinding` is an engine-owned tuple of `EndpointBinding` values, not a map
of arm/tool roles. Each endpoint is addressed by the contract's exact
`(slot_id, endpoint_id)` key and contains its logical `resource_id`, adapter ID,
capabilities, semantic commands, claims, and immutable runtime target. A
`RuntimeEndpointTarget` is controller addressing, not a live controller: its
`transport_id` selects a transport and its `target_id` selects the destination
within that transport. `JointPositionTarget` is the built-in target for a named
`RobotCfg.control_parts` entry and additionally owns its full-robot joint IDs.
Built-in joint primitives explicitly require that target type when they need
IK, joint interpolation, or current attachment keys; a custom mobile or
whole-body skill is not required to masquerade as an arm or hand.

Attachment state and `StateDelta` keys use the bound target's concrete control
part. `TaskState.held_objects` is the sole attachment map. A multi-manipulator
grasp stores one `HeldObjectState` per manipulator with the same
`ObjectSemantics` instance. `TaskState.held_object_mask()` exposes active rows,
while `exclusive_held_object_mask()` excludes rows where another manipulator
holds the same semantic object or live entity. Single-arm transport, release,
and handover operations only succeed on exclusive rows; coordinated placement
likewise requires two distinct, exclusively held objects.

Embodiment-specific semantic commands do not belong to action options. A caller
using direct control-part binding without a `RobotSkillProfile` registers them
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

Actions request semantic commands (`open`, `grasp`, or a named target) from an
`EndpointBinding`; `joint_positions()` is the typed convenience for a
`JointPositionCommand`. `ActionControlOverrides` may replace commands under the
exact `slot -> endpoint -> command` path for one invocation revision. Joint
limits constrain commands but do not define semantic open/grasp states; a robot
integration or tutorial may derive a simple profile from limits explicitly.
Profile-based integrations instead own commands under generic
`command_profiles` IDs and let endpoint declarations/adapters resolve those
IDs. `action_control_profiles()` additionally exposes applicable control-part
commands to the built-in joint planning helpers; custom endpoint commands stay
on their resolved endpoint.

## Built-ins

| Skill ID | Goal type | Required slot endpoints |
|---|---|---|
| `move_end_effector` | `EndEffectorPoseGoal` | `primary.motion` |
| `move_joints` | `JointPositionGoal` (`target` is explicit qpos or a profile command name) | `primary.motion` |
| `pick_up` | `GraspGoal` | `primary.motion`, `primary.grasp` |
| `move_held_object` | `HeldObjectPoseGoal` | `primary.motion`, `primary.grasp` |
| `place` | `PlaceGoal`, `AssembleGoal` | `primary.motion`, `primary.grasp` |
| `press` | `PressGoal` | `primary.motion`, `primary.grasp` |
| `slide` | `SlideGoal` | `primary.motion`, `primary.grasp` |
| `twist` | `TwistGoal` | `primary.motion`, `primary.grasp` |
| `coordinated_pickment` | `CoordinatedPickGoal` | `left.motion`, `left.grasp`, `right.motion`, `right.grasp` |
| `coordinated_placement` | `CoordinatedPlacementGoal` | `placing.motion`, `placing.grasp`, `support.motion`, `support.grasp` |
| `hand_over` | `GraspGoal` | `source.motion`, `source.grasp`, `destination.motion`, `destination.grasp` |

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
3. Declare `skill_id`, `GoalType`, `OptionsType`, and a class-local
   `SkillBindingContract` when the skill should appear in `engine.skills`.
   Express only semantic slots, endpoint requirements, capabilities, required
   commands, and any real disjointness constraints; do not introduce arm/tool
   roles for a mobile-base or whole-body endpoint.
4. Implement `_plan()`; do not override the framework-owned `plan()` method.
5. Validate with `require_goal(request)` and consume endpoints only through
   `request.binding.endpoint(slot_id, endpoint_id)`. Require a concrete target
   subtype only when the planner or payload implementation genuinely needs it.
6. Plan from `context.robot.qpos`; never read an implicit live start state.
7. If planning consumes a semantic object's snapshot pose, override
   `_scene_dependencies()`, preserve `super()` dependencies, and add exactly
   that semantic ID.
8. For planner-backed joint motion, return a full-robot `TimedTrajectory`
   through `build_plan()`; raw position tensors are rejected. Preserve planner
   `dt`, or use `TimedTrajectory.from_uniform_step()` with an explicitly
   selected cadence for action-owned interpolation. Build batched
   `list[PlanState]`, translate the policy with
   `request.motion_policy.to_motion_gen_options()` (including
   `interpolation_dt=context.control_dt` when applicable), call
   `self.motion_generator.generate()`, and import pure operations directly from
   `trajectory_ops.py`. For mobile, whole-body, or other controller-native
   motion, build `EndpointCommand` frames and a `TimedCommandSequence`, then use
   `build_command_plan()`. A new transport family must define matching
   `RuntimeEndpointTarget` and `RuntimeCommandPayload` types with the same
   `transport_id`, plus an `EndpointCommandTransport` registered in the runner's
   router.
9. Declare symbolic changes with `StateDelta`; do not mutate context or commit
   physical effects during planning. For partial attachment updates, retain
   previous scalar semantics while any previous row remains; merge only batched
   masks/transforms and adopt candidate semantics only on full replacement.
10. Keep scene stepping, controller I/O, and task-graph/MLLM logic outside the
   atomic action. Put execution-loop I/O behind the runner protocols rather than
   calling a simulator or device from `plan()` or `ExecutionSession`.
