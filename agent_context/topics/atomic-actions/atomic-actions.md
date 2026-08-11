# Atomic actions

## Current contract

Atomic actions are side-effect-free, environment-batched planners:

```python
plan = engine.plan(invocation: ActionInvocation, context: PlanningContext)
```

There is no `ActionTarget`, `WorldState`, `ActionResult`, `execute()`, or
`AtomicActionEngine.run()` compatibility surface.

`ActionInvocation` separates:

- an action-owned typed goal (`goal_kind` is its stable discriminator);
- an engine-owned `ActionBinding`, which covers the skill contract by exact
  `(slot_id, endpoint_id)` keys and terminates every endpoint at an immutable
  `RuntimeEndpointTarget`;
- reusable `MotionPolicy` planner/timing choices;
- bounded `RecoveryPolicy` thresholds and retry budgets;
- optional typed `skill_options` and endpoint-scoped `control_overrides` for
  one invocation revision.

`PlanningContext` separates measured `RobotObservation`, verified symbolic
`TaskState`, versioned `SceneSnapshot`, and environment IDs. An `ActionPlan`
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
binding-owner ID, so an `ActionBinding` cannot cross engine instances.
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
| `engine.start(invocations, context)` | Execute incrementally from observations | Returns an `ExecutionSession`; `tick(latest_context)` emits commands and performs bounded recovery |

None steps simulation directly. `compile()` never observes physical execution;
split compilation at observation boundaries when later goals depend on measured
results. Use `start()` when observation, effect verification, and replanning
must remain active during execution.

`AtomicAction.plan(request, context)` is the framework-owned template method
called by the engine, not a fourth application entry point. It binds collision
entities from the current scene into a copied motion policy before delegating
to the skill-specific `_plan()` hook. New actions implement `_plan()` and must
not override `plan()`. `engine.plan_action(...)` is only an extension/testing
escape hatch for an unregistered instance.

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
  snapshots that also select exact semantic-effect monitors; endpoint
  declarations or adapters select those profile IDs;
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
metadata only: a `ResourceClaim` by itself is not a resource lease manager,
parallel scheduler, or concurrency guarantee. The separate explicit
`ParallelSkillRuntime` described below coordinates analyzed branch lanes and
still requires an authoritative safety validator. Dynamic execution can
dispatch multiple endpoint commands in one synchronized frame, but that alone
does not imply resource scheduling or safe parallelism. A custom mobile/base or whole-body endpoint is
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
| `Press` | A `SceneEntityPose` in `xpos`. |
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
session = engine.start(invocations, initial_context)
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
`invalidation_mask` selects rows on which the core applies the request-owned,
removal-only `failure_invalidation` delta; it does not let the verifier inject
state. `retry_mask` is reserved for rows whose physical preconditions still
make replay of the same invocation valid. Other failed rows require external
recovery. Unresolved evidence at the action deadline is reconciled fail-closed
when the pending effect covers active verified state.

The semantic layer keeps physical observation separate from symbolic effect
commit. `SkillPolicyPreset.effect_monitors` maps exact semantic call IDs to
versioned, bounded-declarative `EffectMonitorRef` values. Omitting the mapping
selects the built-in `builtin.composite_effect@1` monitor for `pick`,
`place`, and `hand_over`; an explicit empty mapping disables the default and
makes analysis of those curated calls fail with `missing_effect_monitor`.
`SemanticIntegrationManifest` rejects monitor keys absent from its call
catalog. `SemanticSkillCompiler.analyze()` resolves the exact factory and
validates monitor parameters without observing scene providers or constructing
stateful monitors.

Grounding creates an immutable `SemanticEffectSpec` and an independent monitor
for the call. The spec separates typed symbolic state expectations from typed
physical clauses. Pick declares an attached destination, Place a detached
source with an owned pre-effect pose baseline, and HandOver both. Endpoint
adapters publish immutable `EffectEvidenceSourceRef` values and a logical
`task_state_key`; evidence routes use `EffectEvidenceAddress`, never the
command-only `RuntimeEndpointTarget`. This keeps motion, mobile, whole-body,
articulation, and custom controller transports extensible without treating a
control part as symbolic state identity.

`HeldObjectState` is verified symbolic knowledge only. Neither it nor the
standard effect runtime creates simulator joints, managed attachments,
kinematic parents, frozen bodies, or pose overrides. Physical grasp retention
therefore depends on the configured controller, collision geometry, materials,
contact solver, and rigid-body parameters. A command-state evidence value is
only accepted controller intent and never physical contact proof by itself.

Providers emit raw `PoseRelationEvidenceBatch`, `BinaryEffectEvidenceBatch`,
`ScalarEffectEvidenceBatch`, or `JointStateEvidenceBatch` values with stable
environment IDs, per-row validity/acquisition diagnostics, timestamps, and
observation revisions. Providers do not apply policy thresholds. The composite
monitor evaluates clauses as a conjunction per state expectation, applies
pose/force/joint hysteresis, treats invalid rows as unresolved, and reports
explicit contradictory evidence as failure. It never uses `TaskState` as
physical proof. The `SkillRuntime` adapter validates the decision, attaches
only the current verification ID, and returns an exact
`EffectVerificationResult` in the same due observation cycle. Request shrink
within one `attempt_generation` preserves remaining-row hysteresis; installing
a retry/replan/revision increments the generation and resets it. Evidence at
the exact deadline is allowed; evidence after it is rejected and normal runner
timeout/recovery remains authoritative.

Cause events (`ACTION_PLANNING_FAILED`, `EFFECT_VERIFICATION_FAILED`, and
`EFFECT_VERIFICATION_TIMEOUT`) are distinct from the `ACTION_RETRY` recovery
event. `SESSION_COMPLETED` and `SESSION_FAILED` are distinct terminal events.

Effect verification is currently a terminal action boundary. There is no
in-flight physical-invariant monitor for the held-object relation during Pick
lift or HandOver transfer/release/delivery. A slip can therefore be detected at
the terminal monitor but cannot interrupt the trajectory at the frame where it
occurs. On a failed HandOver, the success-only `StateDelta` is not committed,
but an already verified source-held relation also is not reconciled from
failure evidence; blindly retrying after both grippers lost the object can use
stale symbolic state. Pure-dynamics recovery needs a typed, phase-aware
in-flight guard plus failure-outcome reconciliation rather than a simulator-side
attachment.

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

## Semantic runtime and Expert Programs

`embodichain.lab.sim.skills` is the semantic frontend over the core contracts.
`Pick`, `Place`, `HandOver`, `OperateArticulation`, and registered extension
calls are immutable, robot-independent intent values. `SemanticSkillCompiler`
performs provider-free workflow analysis first, then grounds exactly one call
from a fresh `PlanningContext`. It resolves the authoritative `SceneRegistry`,
profile resource binding and preset, downstream target look-ahead, typed goal,
effect specification, and effect monitor before producing one
`ActionInvocation`.

`SkillRuntime` owns the shared call barrier and persistent verified `TaskState`.
Every call creates exactly one one-invocation `ExecutionSession` and re-observes
before the next call. Eligibility, success, failure, cancellation, recovery,
and effect state are row-local; active rows share the call boundary. The
runtime exposes non-blocking `start()`/`step()` and synchronous `run()` over the
same path. `AtomicSkills` is a convenience facade. `AtomicSkills.from_env()`
accepts only an explicit `SkillRuntimeProvider` and never scans arbitrary
environment attributes; Gym demo environments use the lazy bridge below so
commands cannot bypass `env.step()`.

`embodichain.lab.gym.envs.expert_program` owns strict declarative programs.
Schema version 1 supports bounded `Sequence`, `Repeat`, `Segment`, and `Invoke`;
version 2 adds deterministic `Parallel` branches and explicit `Barrier` nodes.
The decoder rejects unknown fields/discriminators, duplicate serialized keys,
unsupported versions, executable values, dotted environment traversal,
unbounded expansion, and invalid registry/catalog references before runtime.
JSON and YAML files are loaded with `load_expert_program()`. A Gym config can
select one with `expert_program_path`, resolved relative to that config file.

`ExpertProgramCompiler` expands program/demo segments lazily while preserving
typed target selections, post-policies, validators, and parallel blocks.
`AtomicDemoBridge` assembles each segment around the canonical runtime and a
buffered command sink. A `ProcessedEnvAction` marks controller-ready output so
the action manager does not transform it twice, but every command and
post-policy hold still passes through ordinary `env.step()`. `BaseEnv.step_dt`
is authoritative; frame durations must be integral multiples of that cadence.
Parallel lanes are aligned on that strict grid and shorter lanes repeat their
last safe target as hold padding; fractional frames are rejected rather than
implicitly resampled. Early generator termination performs the bridge's
explicit cancel-then-hold handshake before the iterator is closed.

Bridge creation materializes the bounded segment stream and performs
provider-aware semantic preflight before the first command is emitted.
Sequential stretches analyze their remaining downstream calls together, so a
Pick retains target look-ahead across logical segment boundaries; an explicit
parallel block is a conservative look-ahead barrier. Runtime grounding remains
just-in-time against the latest observation. Relation Place calls require an
exact typed/versioned `RelationTargetGrounder`, and HandOver requires the
profile-selected `HandOverPoseProvider`; neither provider is inferred from
names.

The production simulation path is
`create_simulation_expert_program_adapter(environment, scene_binding=...,
robot_profile_binding=...)`. `SimulationSceneBinding` declares canonical/native
scene data, while `SimulationRobotSkillProfileBinding` declares reusable robot
resources, capabilities, commands, defaults, and presets. The factory creates
the registry, profile, motion generator, engine, shared-tick observation/evidence
port, command encoder, runtime, and segment policy port. Task classes combine an
external declarative program with typed scene/profile integration declarations
and install the returned adapter; they do not assemble skill trajectories.

`SimulationRobotSkillProfileBinding` accepts generic `RobotResourceBinding`
declarations containing arbitrary typed `ResourceEndpoint` values;
`ControlPartResourceBinding` is the joint-backed convenience. Mobile-base,
whole-body, and non-joint integrations install a matching
`ResourceEndpointAdapter` and `RuntimeTransportActionEncoder` through the same
standard simulation factory. Task-level Expert Programs remain unchanged. This
is an extension seam rather than built-in locomotion: current curated semantic
skills do not consume the example base/whole-body capabilities. A reusable
production capability also installs its semantic descriptor/lowerer, atomic
skill, payload, safe-state transport behavior, and effect integration as
applicable.

The standard Gym encoder currently composes custom transports over a full-qpos
hold and the standard simulation factory owns a `MotionGenerator`. A robot may
omit named control parts, but a truly jointless or natively structured mobile
controller still needs a reusable base-action composition/provider
integration. That integration must not add base- or whole-body-shaped fields to
the generic resource, binding, runner, or router contracts.

Task vertical slices may keep typed profile bindings locally during API
stabilization, but repeated use should promote them into an embodiment-owned
profile catalog rather than duplicate robot data across tasks.

The Open Drawer vertical slice has completed its supported-simulation physical
run and reached the configured drawer joint target. Repeated cube pick/place has
completed one physical Pick/Place/settle/validator cycle; the full three-cycle
run remains in threshold calibration. The dual-UR5/PGI HandOver slice has
completed three consecutive supported-simulation Pick/transfer/settle/validator
runs using contact dynamics only. Its calibrated profile drives only the PGI
master joints, keeps mimic-child drives disabled, uses a 0.011 close target with
stiffness 2000, damping 50, and maximum effort 140, models the can at 0.33 kg,
and uses 200 motion samples. The default 0.05-rad tracking gate and bounded
replanning remain active.

When no explicit contact or constraint callback is installed, simulation grasp
and release evidence combines the live object-to-endpoint pose relation with
`ControlCommandStateEvidenceTracker`. The tracker changes row-local state only
after an exact profile-owned `open` or `grasp` command is successfully encoded
and buffered. Intermediate commands and inactive rows retain prior state;
cancel, discard, or observer failure invalidates affected evidence. Stable
`env_ids`, not simulator array assumptions, correlate full and subset batches.
This command state is evidence of accepted controller intent, not physical
contact by itself.

`DynamicSettleMonitor` is shared by reset events and the Expert Program
`wait_stable` post-policy. It owns threshold, cadence, consecutive-check,
settled, and timeout state but never steps simulation. Eligible rows reuse live
target qpos so a contact-blocked position gripper retains closure preload;
initially inactive rows use fresh measured-qpos holds. Early-settled eligible
rows keep their targets until the active cohort terminates. Every action still
passes through the normal environment-step path, and segment validators remain
a separate dataset/task boundary.

The standard simulation factory lowers both `MotionPolicy.control_dt` and
`ExecutionRunnerCfg.minimum_cycle_time` to the authoritative Gym `step_dt`.
When `hold_during_effect_verification=False`, runner polling emits no
observed-position HOLD; the bridge advances physics by replaying the last
accepted environment action. HandOver also sets `hold_on_completion=False`, so
its subsequent `wait_stable` policy continues the existing targets rather than
neutralizing the gripper at its contact-displaced qpos. Cancellation and
failure still perform cancel followed by an observed-position safe hold.

This staged B behavior solves the validated joint-position HandOver path but is
not a generic continuation contract for mobile-base or whole-body transports.
Those endpoints need a typed transport-owned continuation command rather than a
joint-qpos latch. `wait_stable` also runs only after terminal effect verification
and symbolic-state commit, so its timeout is a post-policy failure and does not
trigger atomic-action recovery.

Runtime and demo results expose deterministic JSON-safe metadata. Call traces
include invocation identity, masks, command counts, execution/recovery events,
plan-attempt trajectory segments, scene/collision revisions and dependencies,
plus effect decisions and monitor evidence. Segment metadata adds post-policy
settling and validator results. Trajectory segments are trace ranges inside an
atomic plan and never own separate recovery, effect, or timeout state.

Parallel execution is an explicit schema/runtime layer rather than a second
atomic scheduler. Static analysis rejects overlapping `ResourceClaim` values.
Independent lane runtimes share one clock and barrier, command frames are
merged only after destination/claim/safety validation, failure handling is
row-local, and verified `StateDelta` values merge deterministically at the
barrier. Parallel execution also requires an authoritative
`ParallelCommandSafetyValidator`; resource disjointness alone is never promoted
to physical-safety evidence, and a missing validator fails closed. Schema
version 2 intentionally uses strict task-state key-level merge conflicts;
mask-aware same-key branch merges are not part of this version.

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
built-in can be replaced only with explicit `replace=True`. Registration puts
the implementation in `engine.actions`; it does not prove that the current
embodiment supports it. Semantic exposure additionally requires a concrete
class-local `binding_contract` for `engine.skills` and a valid profile assignment
for `engine.skill_profile.skills`. The module-level `register_action()` API is a
process-wide extension-type discovery catalog only; it neither binds actions nor
changes an engine's default built-in set.

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
| `coordinated_pickment` | `CoordinatedPickGoal` | `left.motion`, `left.grasp`, `right.motion`, `right.grasp` |
| `coordinated_placement` | `CoordinatedPlacementGoal` | `placing.motion`, `placing.grasp`, `support.motion`, `support.grasp` |
| `hand_over` | `GraspGoal` | `source.motion`, `source.grasp`, `destination.motion`, `destination.grasp` |
| `operate_articulation` | `OperateArticulationGoal` | `primary.motion`, `primary.interaction` |

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

1. Define a frozen action-owned goal dataclass with `goal_kind`.
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
8. For planner-backed joint motion, return full-robot positions or a
   `TimedTrajectory` through `build_plan()`: build batched `list[PlanState]`,
   translate the policy with `request.motion_policy.to_motion_gen_options()`,
   call `self.motion_generator.generate()`, and import pure operations directly
   from `trajectory_ops.py`. For mobile, whole-body, or other controller-native
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
