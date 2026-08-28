# Atomic actions

## Current contract

Atomic actions are side-effect-free, environment-batched planners:

```python
plan = engine.plan(invocation: ActionInvocation, context: PlanningContext)
```

There is no `ActionTarget`, `WorldState`, `ActionResult`, `execute()`, or
`AtomicActionEngine.run()` surface.

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
planner diagnostics with a typed retryable/non-retryable `PlanningFailure`,
named `TrajectorySegment` frame ranges, and an uncommitted `StateDelta`.
Segments are inspection/tracing metadata inside one command sequence; they are
not independently replannable execution boundaries. Execution emits one
`TRAJECTORY_SEGMENT_ENTERED` event at each named boundary and preserves the
segment name on subsequent events for observability.

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
its direct control-part command-profile snapshot. It may also contain
standalone `GraspPoseGenerator` services keyed by grasp endpoint runtime target
ID. These services are siblings of `MotionGenerator`, not motion-generator
features: direct callers may use them without atomic actions, while `PickUp`,
`HandOver`, `Slide`, and `CoordinatedPickment` resolve them through their bound
grasp endpoints. `AntipodalAffordance` owns target-local mesh geometry only.
`embodichain.toolkits.graspkit` owns the backend-neutral
`GraspPoseGenerator`, `ParallelJawGraspPoseGenerator`, and gripper-model
contracts. The toolkit has no dependency on `embodichain.lab`; simulation,
atomic actions, Expert Program, and handwritten environments are consumers of
the same service API. Its `pg_grasp` package exposes
`AntipodalGraspPoseGenerator` as the sole antipodal generator entry point;
mesh-specific sampling, annotation, collision, and on-disk cache state live
behind a private backend rather than a second public generator/configuration
pair.
The engine-scoped registry retains generator instances by reference, so a
composition root may reuse an already prepared service in a handwritten
environment. The engine also issues an opaque binding-owner ID, so an
`ActionBinding` cannot cross engine instances. It does not own a timing
fallback. Planner results with positions require explicit `dt`;
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

An engine may additionally borrow a default `SceneProvider`. In that case,
`engine.initial_context()` captures a `SceneSnapshot` from the provider using
the robot observation timestamp and generated environment IDs. An explicitly
supplied `scene=` snapshot takes precedence, and engines without either source
retain the empty-scene behavior. The provider is only an initial-context
convenience for direct-core planning; execution observations and scene revision
advancement remain owned by the runtime's `ObservationProvider`.

Direct simulation callers that only need selected rigid-object poses should use
`create_simulation_atomic_action_engine(..., scene_entities=(...))`. The factory
derives canonical direct-core IDs from the supplied objects' `uid` values and
installs the default provider; it never scans `SimulationManager`. Actions then
select the entries they consume through their goal and semantic entity IDs.
Articulation/link observations, aliases, collision roles, dynamic execution,
and external perception remain explicit `SceneProvider` or `SceneRegistry`
integration paths.

## Engine entry points

Choose the public engine entry point by lifecycle, not by skill type:

| Entry point | Use | Result and state behavior |
|---|---|---|
| `engine.plan(invocation, context)` | Inspect or plan one registered action | Returns one `ActionPlan`; does not project a context for another action |
| `engine.compile(invocations, context)` | Plan an ordered sequence against a fixed scene | Returns a concatenated `CompiledTrajectory`; propagates hypothetical qpos and expected effects through `projected_context` |
| `engine.start(invocations, context, *, eligible_mask=None)` | Execute incrementally from observations | Returns an `ExecutionSession`; the optional initial cohort is sticky, and `tick(latest_context)` emits commands, exposes effect boundaries, and performs bounded recovery |

None steps simulation directly. `compile()` never observes physical execution;
split compilation at observation boundaries when later goals depend on measured
results. Use `start()` when observation, effect verification, and replanning
must remain active during execution. Non-empty expected effects always expose a
correlated effect boundary. `SkillRuntime` resolves that boundary from the
selected semantic monitor; when no monitor is configured, it projects the
planned state without claiming physical task-success evidence.

`AtomicAction.plan(request, context)` is the framework-owned template method
called by the engine, not a fourth application entry point. It binds collision
entities from the current scene into a copied motion policy before delegating
to the skill-specific `_plan()` hook. New actions implement `_plan()` and must
not override `plan()`. Custom actions must be installed with
`engine.register()` before using the same public entry points.

The `_plan()` extension boundary is strict. A subclass that defines `plan()`
raises `TypeError` at class definition; custom actions implement `_plan()`.

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
  additional `endpoint_adapters` to `RobotSkillProfile.bind()` for mobile bases,
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
`engine.bind_control_parts(skill_id, endpoints)` with an exact nested
`slot -> endpoint -> control_part` mapping. The engine accepts an installed
skill ID, checks contract coverage, control-part existence, required commands,
ownership, and disjointness, then emits the same generic `ActionBinding` with
`JointPositionTarget` endpoints. Callers do not construct bindings manually,
and this path deliberately does not perform profile resource discovery or
capability matching.

`engine.make_invocation(skill_id, goal, ..., control_parts=...)` is the
direct-core convenience construction boundary. It resolves only the explicit
`slot -> endpoint -> control_part` mapping and returns an ordinary
`ActionInvocation`; it never imports, binds, or stores a `RobotSkillProfile`.
Profile-based callers use `RobotSkillProfile.bind(engine, ...)`, resolve a
binding through the returned `BoundRobotSkillProfile`, and construct an
`ActionInvocation` directly. `SemanticSkillCompiler` owns that path for semantic
workflows.

Discovery boundaries are distinct:

- `engine.actions` contains every installed action instance and is the
  direct-core registry.
- `engine.skills` contains descriptors only for installed, `agent_visible`
  actions whose concrete class explicitly declares a binding contract. A
  subclass does not inherit semantic exposure implicitly.
- `bound_profile.skills` filters `engine.skills` again to contracts with at least
  one valid assignment on the bound robot. Registering or replacing an action
  advances `engine.skill_catalog_revision`; every retained
  `BoundRobotSkillProfile` then rejects use and must be rebound.

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

The atomic core never imports or owns a `RobotSkillProfile`. A profile-aware
composition root first constructs `AtomicActionEngine` with
`control_profiles=profile.action_control_profiles()`, installs any custom
actions, and then calls `profile.bind(engine, endpoint_adapters=...)`. The
returned `BoundRobotSkillProfile` belongs to the semantic integration layer.
`command_profiles` values currently use `ControlPartCommandProfile` as their
immutable command container, but their mapping keys are generic profile IDs
rather than necessarily being control-part names.
`ControlPartEndpointAdapter` plus `RobotSkillProfile.action_control_profiles()`
provides the direct control-part lookup used by built-in joint planners; it is
not a binding route. Binding requires equivalent direct control-part commands
to have been installed on the engine already. Profile resolution still places
all resolved semantic commands, including commands for custom endpoint types,
on their `EndpointBinding`. A profile `JointPositionCommand` is
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
their endpoint or control-part names differ. `ParallelSkillRuntime` uses these
claims for deterministic preflight and rejects overlapping branches, but there
is no general resource lease manager outside that coordinator. Non-conflicting
claims are not proof of collision safety: the parallel coordinator requires a
`ParallelCommandSafetyValidator` before merged command frames can leave it. A
custom mobile/base or whole-body endpoint is executable only when its adapter
supplies a target, the action emits a matching runtime payload, and the target's
transport is registered with the `EndpointCommandRouter`.

## Object identity and pose grounding

`ObjectSemantics.entity_id` is the typed core's canonical snapshot-key lowering
target and is required. The registry-backed path obtains it from a resolved
`SceneEntityRef`; direct-core callers supply the same non-empty string. Pose
grounding resolves only from the current `PlanningContext.scene`, and a missing
snapshot entry is an error.

`ObjectSemantics` is shallow-frozen. Top-level fields such as `entity_id`,
`affordance`, and `label` cannot be rebound after construction; create a new
semantics value to change identity. Nested affordance and metadata objects may
remain mutable, but they never establish identity.

`SceneSnapshot` owns copies of input entity states and returns a defensive
`EntityState`/pose copy on every public mapping lookup. Mutating an input tensor
or a previously returned pose cannot change the published snapshot. Publish a
new scene version for every material dynamic-state change.

`OpenDoorGoal.open_fraction` owns the desired absolute hinge state: `0` maps to
the `OpenDoorAffordance` closed legal endpoint and `1` to its open endpoint.
`OpenDoorAffordance.opening_direction` owns the closed-to-open joint-coordinate
direction and defaults to increasing qpos; reverse-coordinate hinges configure
`-1` at affordance construction. `OpenDoorAffordance.from_articulation()`
consumes only `Articulation.get_parent_joint_chain()`: automatic hinge
selection skips fixed joints and requires exactly one active revolute ancestor;
prismatic ancestors, latch joints, and other multi-active chains require an
explicit `hinge_joint_name`. The planner automatically matches the
affordance-resolved parent revolute joint name to one unique
`SceneSnapshot.articulation_joints` observation, computes a row-local opening
delta from measured qpos, holds rows already at target, and fails rows with
invalid observations, illegal targets, or targets behind the current opening
state. Interpolation density, approach/retract distances, and joint comparison
tolerance remain `OpenDoorOptions` policy values.

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
`RegisteredSemanticLowerer` with a matching call ID and target descriptor.
Their payloads carry task intent, while the selected `SkillPolicyPreset` is the
sole action-option source; a lowerer may read its owned option template for goal
grounding but must not mirror those options into the payload as a second policy
configuration.

`SemanticSkillCompiler.analyze()` performs provider-free linking, resource and
affordance validation, held-object flow analysis, and first-release look-ahead.
A `Place` with no explicit `primary` resource inherits the workflow's known
holder resource, and a `HandOver` with no explicit `source` does the same. The
inferred selection is snapshotted onto the canonical linked call before
binding; an explicit conflicting selection still fails with
`held_resource_mismatch`. Holder-resource inference never crosses a
registered-call boundary.
`HandOver` selects participants only through the `source` and `destination`
resource slots; there is no separate receiver alias.
A registered lowerer is opaque to pickup look-ahead by default. It may override
`pick_lookahead_targets()` to certify that it retains the same picked object on
the same bound `primary` resource and expose an exact ordered tuple of
intermediate object poses. Returning `None` remains the conservative barrier;
an empty tuple retains the chain without adding a target. A pick therefore owns
an ordered downstream target sequence through its first release. Relation
targets retain affordance payload type and revision metadata and stay
late-bound through an explicitly installed `RelationTargetGrounder`; handover
poses stay behind a named
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

## Semantic skill runtime

`embodichain.lab.sim.skills.SkillRuntime` is the canonical execution service.
`start()` analyzes the complete semantic-call window once, then JIT-grounds and
starts exactly one invocation, `ExecutionSession`, and `ExecutionRunner` per
executed call. Before every call it obtains a fresh `PlanningContext`; only
verified `TaskState` and the shrinking eligible cohort cross call barriers.
`execution_prefix_length` lets a selected future suffix participate in static
look-ahead without grounding or executing that suffix. A runtime owns at most
one active workflow, while `fork()` creates an independent lane sharing the
compiler, observation/evidence providers, clock, and optional runner override.

The selected `SkillPolicyPreset` owns the default `ExecutionRunnerCfg` for each
call. A runtime-level `runner_cfg` is an explicit all-call override; omitting it
does not synthesize a second default. Motion and recovery policy continue to be
lowered by `SemanticSkillCompiler` into the invocation.

Physical verification flows through `EffectEvidenceCollectorPort`, the
grounded `SemanticEffectSpec`, and its selected `EffectMonitor`. `SkillResult`
contains tensor-owning row masks, verified task state, call/plan/effect traces,
and JSON-safe metadata. `SkillFailure` exposes a stable `code` and `phase`;
post-analysis preparation failures preserve an original `SemanticDiagnostic`
when available, while low-level execution events remain in the call trace.
Terminal failure first applies the core-owned symbolic reconciliation selected
from per-expectation physical outcomes. `WorkflowRecoveryPolicy` then provides
a bounded, row-local recovery budget. Rows whose reconciled state still proves
the failed call's source relation retry that call from a fresh observation;
rows whose source relation was invalidated execute a real semantic Pick on the
resolved source resource and then retry the original call. Already successful
rows wait at the shared call barrier, and every recovery call uses normal
analysis, grounding, planning, command dispatch, effect verification, and
trace metadata.

`SkillPolicyPreset.effect_monitors` is the semantic verification switch at
exact call-ID granularity. Omitting the constructor argument installs the
built-in Pick/Place/HandOver monitors; an explicit empty mapping selects
trajectory-only execution, and a partial mapping verifies only its selected
calls. The compiler leaves `effect_spec` and `effect_monitor` unset for an
unselected call, and `SkillRuntime` automatically projects the plan's expected
state when that monitor is absent. A configured monitor whose
factory or evidence provider is unavailable fails closed instead of silently
becoming open-loop.

`SkillRuntime.from_simulation()` is the standard explicit simulation factory.
It combines an optional application verifier and step observer with the typed
terminal monitors, phase-effect gates, and held-object guards selected by the
compiler. `AtomicSkills` is a small application facade over that same runtime;
it does not own a second compiler or execution loop. Its simulation constructor
delegates to `SkillRuntime.from_simulation()`, `available_skills` exposes the
bound profile's immutable atomic skill descriptors, and `availability()`
returns a structured semantic diagnostic instead of reducing capability checks
to a boolean.

The core runtime and Expert Program adapter share one provider-free semantic
assembly path for the scene manifest, robot-profile binding, catalog, and
compiler. The standard Gym registration, compilation, bridge, and task
integration contracts are routed separately through
`agent_context/topics/expert-programs/expert-programs.md`.

`ParallelSkillRuntime` coordinates two or more forked semantic lanes on one
clock. It rejects overlapping `ResourceClaim` values and symbolic writes,
requires a `ParallelCommandSafetyValidator`, merges at most one synchronized
command action per coordinator step, and currently supports fail-fast recovery
only. Parallel success adopts verified branch state at the shared barrier.

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
2. Otherwise, the required canonical `entity_id` strings must match.
3. `label`, affordance payloads, and live simulator handles never establish
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
| `PickUp` | Always its semantic `entity_id`, because the object pose is grounded once and reused; plus any goal-owned `SceneEntityPose`, such as `grasp_xpos`. The semantic object ID is monitored only through `approach`; other dependencies keep their plan-declared window. |
| `CoordinatedPickment` | Goal-owned target/initial `SceneEntityPose` values; the semantic `entity_id` when `object_initial_pose` is omitted and semantic grounding supplies that pose. |
| `Place` | A `SceneEntityPose` in ordinary `xpos`; `AssembleGoal` always declares its required `base_pose`. |
| `MoveHeldObject` | A `SceneEntityPose` in `object_target_pose`; the exact object target is composed with the verified `object_to_eef` attachment, without implicit reorientation. After successful semantic calls, the runtime refreshes held relations from terminal object observations and EEF forward kinematics when available. |
| `PushObject` | Its semantic object ID plus a `SceneEntityPose` in `target_pose`. Both dependencies are monitored through `approach`; contact and push intentionally move the object. |
| `Press` | `PressGoal.target_pose` when it is a `SceneEntityPose`; affordance data is entity-free. |
| `Slide` | `SlideGoal.target_pose` when it is a `SceneEntityPose`; the local grasp mesh does not own the link. |
| `OpenDoor` | `OpenDoorGoal.target_pose` when it is a `SceneEntityPose`; monitoring stops after the `reach` segment so grasp- and hinge-induced handle motion does not trigger recovery. |
| `Twist` | `TwistGoal.target_pose` when it is a `SceneEntityPose`; affordance data is entity-free. |
| `CoordinatedPlacement` | `SceneEntityPose` values in the placing or support object target pose. |
| `HandOver` | Its semantic object ID plus a `SceneEntityPose` in `HandOverGoal.target_pose`. The unified action observes the object before pickup, derives its middle transfer pose from the two arm roots, and owns pickup through final release. |

`collect_scene_dependencies()` deliberately stops at `ObjectSemantics`.
Therefore, a custom action that consumes a snapshot pose through semantic data
must override `_scene_dependencies()`, union `super()` dependencies, and add the
consumed semantic ID. Do not declare an ID merely because semantics are present.
`ActionPlan.scene_dependency_end_segment` can bound dynamic-goal monitoring to
the reversible part of a staged action for every dependency.
`ActionPlan.scene_dependency_monitor_until` can assign each dependency an
exclusive command-frame cutoff. An omitted dependency remains monitored for the
whole action unless the global segment boundary applies. `PickUp` stops
monitoring its semantic object ID after approach: object motion before contact
still replans, while contact-, grasp-, and lift-induced motion is not
misclassified as an external update. Collision-world and joint-tracking checks
remain independent of these dependency windows.

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
result = runner.step(effect_result=None)
```

`ExecutionSession` owns deterministic planning progress and recovery state. It
emits at most one synchronized `RuntimeCommandFrame` per tick from the plan's
authoritative `TimedCommandSequence`. A frame contains one or more
`EndpointCommand` values, a shared environment batch and active mask, and a
per-environment `hold_duration`. Every command pairs a
`RuntimeEndpointTarget` with a `RuntimeCommandPayload`; their `transport_id`
values must match, destinations must be unique within the frame, and joint
targets may not overlap. `TrackingPolicy` separates optional in-flight checks
from terminal acceptance. `AtomicAction` projects command payloads into a
command-aligned `TimedTrackingSequence` through each endpoint's typed tracking
channel, while `TrackingRuntime` resolves exact-version feedback providers,
command projectors, and metric evaluators. `TrackingPolicy.joint_position()`
installs feedback-based in-flight and terminal joint metrics;
`TrackingPolicy.timed()` uses explicit terminal settling without feedback.
Invalid required feedback fails only the affected rows closed, while feedback
that exceeds the configured consecutive-violation budget can trigger a replan.
Framework authorization replaces every emitted target with its binding-owned
snapshot and rejects unbound destinations, target substitution, and endpoint
claim conflicts. A plan's non-empty frames and its recovery replans retain a
stable destination set. Empty failed plans retain previously active targets so
the caller can still hold them. The session monitors:

- typed in-flight tracking metrics and terminal acceptance;
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

It replans from the latest observation within per-environment budgets. Pass an
owned boolean `eligible_mask` to `engine.start()` when a previous semantic call
has already deactivated rows. Eligibility can only shrink; use
`runner.deactivate_rows(mask, reason=...)` while a runner owns scheduling so its
cached effect request stays correlated. The budgets, verified task state, and
eligibility masks are row-local, while the action waypoint cursor and call
barrier are batch-synchronized. One allowed replan regenerates the still-pending
cohort without charging unaffected rows. Exhausted rows hold and never become
eligible again.

A non-empty `StateDelta` is not committed until the caller supplies a
correlated `EffectVerificationResult`. Its disjoint `success_mask` and
`failure_mask` must be subsets of the current request mask; requested rows in
neither mask remain unresolved. Partial successes commit immediately while
unresolved rows keep the barrier pending. `EffectVerificationRequest` carries a
monotonic `verification_id`, stable `requested_at`/`deadline` values in the
robot-observation timestamp domain, a session-local `attempt_generation`, and
an owned effect snapshot. Mask shrinkage creates a new ID without extending the
deadline or changing the generation; installing a replacement plan increments
the generation. Results for an old ID are rejected. `RecoveryPolicy.action_timeout`
covers the trajectory and terminal effect wait together, and only timestamps
strictly greater than the deadline time out. While verification is outstanding,
`ExecutionTick.pending_effect` retains the request on every tick;
`EFFECT_VERIFICATION_REQUIRED` is only the one-time audit event.

For synchronous verification, pass `effect_verifier(context, request)` to
`runner.step()` or `run_until_blocked()`. The runner calls it after the fresh
due-cycle observation and supplies its result to `session.tick()` in that same
cycle. It does not call the verifier when the observation timestamp is already
past the request deadline. A verifier must return an exact
`EffectVerificationResult`; all-false masks mean unresolved. External
asynchronous integrations instead pass `effect_result` explicitly on a due
`step()` call.

```python
import torch

request = tick.pending_effect
effect_result = EffectVerificationResult(
    verification_id=request.verification_id,
    success_mask=observed_success,
    failure_mask=observed_failure,
    invalidation_mask=observed_failure,
    retry_mask=torch.zeros_like(observed_failure),
)
result = runner.step(effect_result=effect_result)
```

Both failure-policy masks must be subsets of `failure_mask`.
`invalidation_mask` applies only the request-owned, removal-only
`failure_invalidation` delta; a verifier cannot inject replacement state.
`retry_mask` selects rows whose physical preconditions still permit replay of
the same invocation. Other failures cross a typed recovery boundary, while
unresolved terminal evidence is reconciled fail-closed at the action deadline.

Curated semantic calls also install physical checks inside an action. A
`HeldObjectGuardRequest` observes negative invariants before commands in named
segments and applies removal-only reconciliation when attachment loss is
proven. A `PhaseEffectGateRequest` blocks entry to a named segment until its
positive physical transition is verified, replaying the preceding command for
the synchronized active cohort while evidence is unresolved. Gate success
unlocks motion but does not commit `TaskState`; the terminal monitor remains
authoritative.

Pick gates attachment before `lift`; Place gates release before `retract`.
The unified HandOver action gates source pickup before `pickup_transport`,
destination pickup before `handover_release`, and source release before
`place`. Its source-held guard covers pickup transport through receiver close,
and its destination-held guard covers source release and placement. All checks
are observational: they never create simulator constraints, freeze bodies, or
override poses.

Cause events (`ACTION_PLANNING_FAILED`, `EFFECT_VERIFICATION_FAILED`, and
`EFFECT_VERIFICATION_TIMEOUT`) are distinct from the `ACTION_RETRY` recovery
event. An `ACTION_PLANNING_FAILED` event carries the plan's stable failure code
and retryability. Non-retryable failures deactivate their affected rows without
spending an action retry budget. `AtomicAction.build_command_plan()` supplies a
retryable `planning_failed` classification when a failed result omitted one;
direct `ActionPlan` construction requires explicit failure diagnostics.
`SESSION_COMPLETED` and `SESSION_FAILED` are distinct terminal events.

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
structured invalidation/replan events, and requires terminal completion. The
dynamic-obstacle example additionally uses dense `morphit` robot collision
spheres, proves that the moved cuboid intersects the original TCP path, and
requires the replanned TCP path to retain a positive minimum clearance.

Semantic integration tutorials live under `scripts/tutorials/semantic_skill/`.
Both examples separate `create_*_application()` (scene/profile/runtime and
default verifier wiring), `create_*_task()` (robot-independent semantic calls),
and the application-facing `app.run(task, ...)` entry. Both examples use the
canonical `SkillRuntime.from_simulation()` factory; there is no tutorial-specific
execution loop. `place.py`
executes `Pick -> Place`, verifying the observed lift, planned object-to-EEF
relation, release pose, and open hand. `hand_over.py` demonstrates disjoint
dual-arm resources plus an explicit `RegisteredSemanticLowerer`, then verifies
the unified pickup, transfer, placement, and final release. Both report
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
declare a `binding_contract`; `BoundRobotSkillProfile.skills` further filters
that catalog to valid resource assignments. Registration is engine-local; there
is no independent process-wide action catalog. Construct extensions explicitly
and install them with `engine.register()` so discovery and execution cannot
observe disconnected registries.

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
| `axis_align` | `AxisAlignGoal` | `primary.motion`, `primary.grasp` |
| `move_held_object` | `HeldObjectPoseGoal` | `primary.motion`, `primary.grasp` |
| `pour` | `PourGoal` | `primary.motion`, `primary.grasp` |
| `push_object` | `PushObjectGoal` | `primary.motion`, `primary.grasp` |
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

`PushObject` is a free-object planar interaction with an empty `StateDelta`.
Its options separate object/support contact geometry from per-control-part tool
calibration, and a completion tolerance lets a corrective invocation hold when
the latest measured object pose is already close. A task still owns settling
and measured landing-pose validation; action completion alone does not claim
placement.

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
`AxisAlign` likewise returns a single-manipulator `HeldObjectState`; it preserves
the object-to-EEF relation established at grasp and records the final aligned
EEF pose as the projected grasp pose.

`AssembleGoal.base_pose=SceneEntityPose(...)` is the canonical assembly anchor
and is required, so the base is always snapshot-grounded and registered as a
recovery dependency. The `assemble.py` tutorial publishes the matching scene
snapshot; `moving_target_recovery.py` demonstrates dynamic snapshot recovery.

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
