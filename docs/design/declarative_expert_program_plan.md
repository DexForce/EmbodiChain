# Declarative Expert Programs and Unified Semantic Skill Runtime

- Status: core contracts are implemented through Phase 7 on stacked feature
  branches. Open Drawer has completed its supported-simulation physical run;
  repeated cube pick/place has completed one Pick/Place/settle/validator cycle,
  while the full three-cycle run remains in threshold calibration.
- Baseline: `main@bcccb787e8f9165e9c8acf6f39f165ba6ac752a4`
- Last updated: 2026-08-11
- Related issues: [#471](https://github.com/DexForce/EmbodiChain/issues/471),
  [#474](https://github.com/DexForce/EmbodiChain/issues/474)
- Related implementation:
  [#475](https://github.com/DexForce/EmbodiChain/pull/475),
  [#517](https://github.com/DexForce/EmbodiChain/pull/517),
  [#487](https://github.com/DexForce/EmbodiChain/pull/487)

## 1. Executive summary

EmbodiChain should add a declarative **Expert Program** for task authors, but it
must not become an "Action Bank 2" or a third execution framework. The Expert
Program is one frontend of a unified semantic skill system:

```text
Python facade ---------+
ExpertProgramCfg ------+--> SemanticCallSpec --> SemanticSkillCompiler
future MLLM calls -----+                              |
                                                      v
                                               ActionInvocation
                                                      |
                                                      v
                                              AtomicActionEngine
                                                      |
                                                      v
                                                SkillRuntime
                                                      |
                              +-----------------------+--------------------+
                              |                                            |
                              v                                            v
                    application/hardware                         AtomicDemoBridge
                                                                       |
                                                                       v
                                                                 DemoSegment
```

The existing typed atomic-action engine remains the advanced planning and
execution core. A new semantic layer owns object-centric calls, scene identity,
robot capability binding, workflow look-ahead, policy presets, and built-in
effect monitoring. Both Python and configuration users compile through that
same layer and run through one runtime built on `ExecutionRunner`.

The target authoring cost is:

- a new task that uses existing semantic capabilities: Expert Program
  configuration plus typed scene/profile integration declarations, and
  optionally a declarative validator, with no task-specific motion code;
- a new robot: one reusable `RobotSkillProfile`, not task-specific motion code;
- a genuinely new physical interaction: one reusable capability bundle
  containing its semantic skill/compiler/monitor and controller integration as
  applicable, after which tasks select it through program and integration data.

This design preserves the core direction of #471. Issue #474 changes the
middle of the architecture: ordinary configuration must describe semantic
intent instead of raw `ActionBinding`, `MotionPolicy`, grasp transforms,
sessions, or verifiers.

## 2. Goals

1. Let task authors generate atomic-skill motion from a strict, versioned
   configuration without implementing per-task motion generators or Action Bank
   node functions.
2. Give Python callers, Expert Programs, and future model-generated calls the
   same semantic validation, grounding, binding, planning, execution, effect
   verification, and recovery path.
3. Make object identity, observation, geometry, affordance, and collision data
   come from one scene registry.
4. Infer robot-resource bindings and stable runtime policies from reusable profiles;
   require explicit choices only when the request is genuinely ambiguous.
5. Preserve lazy observation and per-environment recovery for programs whose
   later goals depend on earlier physical effects.
6. Feed commands through `env.step()` during demonstration generation so normal
   managers, recorders, timing, and dataset boundaries remain authoritative.
7. Migrate Action Bank incrementally and keep legacy demonstration tasks usable
   until equivalent sequential and parallel behavior is proven.

## 3. Non-goals

- Replacing the typed contracts in `embodichain.lab.sim.atomic_actions`.
- Adding a second action registry, recovery loop, scheduler, or simulator I/O
  loop.
- Exposing raw joint commands, planner protocols, EEF/object transform math, or
  arbitrary Python expressions in the normal Expert Program schema.
- Building a general DAG/OR-Tools scheduler in the first version.
- Supporting `Parallel`, `If`, and `Retry` before sequential observed execution
  is stable.
- Claiming that every new physical interaction can be expressed without any
  shared skill implementation.
- Removing Action Bank before feature parity and a documented migration window.

## 4. Baseline on current `main`

This plan is updated against committed `main@bcccb787` after PRs #475 and #476. The
implementation series is stacked from that baseline: PR1 is complete on
`refactor/atomic-actions-phase0`, PR2A is implemented by
`feat/atomic-action-pr2a-scene-registry`, and PR2B is implemented by
`feat/atomic-action-pr2b-robot-skill-profile`. These status statements do not
imply that the stacked changes have landed on `main`.

| Capability | Current main | Design consequence |
|---|---|---|
| Typed goals, invocations, plans, state deltas, recovery, and skill descriptors (#448) | Available | Keep as the core contract; all frontends compile to it. |
| Lazy `DemoSegment` execution and legacy demo compatibility (#460) | Available | Use a thin demo adapter; do not create a second dataset executor. |
| Closed-loop `ExecutionRunner` and simulator ports (#449) | Available | `SkillRuntime` wraps/reuses the runner rather than scheduling commands itself. |
| Dynamic scene recovery and `DynamicCollisionMode` (#450) | Available | Profiles select precise collision semantics and fail early when required capabilities are unavailable. |
| Refined planning architecture (#475) | `MotionGenerator.generate()` is the single planning facade; each `ActionPlan` owns one trajectory and one recovery boundary; named `TrajectorySegment`s are metadata | Do not reintroduce `TrajectoryBuilder`, `MotionPlanningAdapter`, or trajectory-segment recovery. |
| Environment cadence through `BaseEnv.step_dt` (#472) | Available; planner and action trajectories now require explicit timing | Expert configuration, `MotionPolicy`, and the engine do not own a fallback period. Environment integrations put `BaseEnv.step_dt` on `PlanningContext` only for action-owned interpolation. |
| Adaptive dynamic-object settling (#470) | Reset/event implementation exists | Extract a reusable monitor; demo post-policies must advance through `env.step()`. |
| Authoritative scene registry (#487) | Foundation available; official environments not migrated | Reuse it from the semantic compiler and opt in task scenes explicitly. |
| Declarative robot skill profiles (#487) | Foundation available; official profiles not yet installed | Bind reusable embodiment profiles through the semantic integration layer; put optional backend compatibility in `SkillPolicyPreset.required_planner`, not `MotionPolicy`. |
| Repeated cube pick/place demo | Manually constructs invocations and transform math | First configuration-only vertical slice. |
| Open Drawer task (#473) | Manually builds approach, grasp, pull, and command trajectories | Evidence that the semantic layer needs articulation/link/affordance references and a reusable articulation skill. |
| Action Bank | Configuration plus task-specific Python node/edge functions | Keep only as a compatibility path while semantic coverage is built. |

PR #475 resolved cumulative translation/rotation publication, removed the dead
`MotionPolicy.interpolation` field, and unified strategy dispatch. It also made
`AtomicAction.plan()` framework-owned and `_plan()` the only custom-action
extension hook. Rejecting a subclass that overrides `plan()` is an intentional
hard break: the project will not provide a compatibility adapter or deprecation
window for that former extension contract. Custom actions must migrate to
`_plan()` so framework-owned scene binding cannot be bypassed.

Phase 1 closes the core identity, registry-backed collision integration, and
embodiment-owned capability/profile prerequisites. The remaining #474 work is
adoption and semantic orchestration:

- ordinary callers still see a large low-level public surface and must perform
  semantic transform and verifier plumbing;
- official environments do not yet provide authoritative registry population;
- official robot configurations do not yet install reusable skill profiles;
- named presets are not yet selected by a semantic facade/runtime; and
- effect monitoring and the configuration/demo path remain later-phase work.

One #474 finding has changed since its review branch: the ambiguous
`collision_check` switch has been replaced by `DynamicCollisionMode.OFF`,
`AUTO`, and `REQUIRED` on current main. The semantic preset layer should build
on this contract, not reintroduce a boolean collision flag.

## 5. How #474 changes #471

The following #471 decisions remain valid:

- strict, discriminated, versioned configuration;
- `Sequence`, bounded `Repeat`, `Segment`, and `Invoke` in version 1;
- lazy re-observation when later goals depend on physical effects;
- distinct action-effect verification, segment post-policy, and task-level
  validation responsibilities;
- stable named trajectory segments for tracing instead of recomputed trajectory
  indices;
- sequential execution first, then resource-aware parallel execution;
- continued Action Bank compatibility and only the explicitly documented
  direct-core fallbacks during migration. This does not include the intentional
  `plan()` to `_plan()` hard break.

The following parts must be adjusted:

| Original #471 direction | Revised decision after #474 |
|---|---|
| Expert configuration may directly contain `ActionBinding`, `ActionOptions`, `MotionPolicy`, and `RecoveryPolicy`. | The default schema contains semantic calls and named presets. Raw core contracts are allowed only in a clearly marked advanced override. |
| `InvocationCompiler` grounds configuration directly to `ActionInvocation`. | A shared `SemanticSkillCompiler` first consumes `SemanticCallSpec`; its final lowering stage produces `ActionInvocation`. Python and MLLM calls use the same compiler. |
| `AtomicDemoBridge` owns context, effect verification, timing, and post-policy behavior. | The bridge owns only Gym/demo adaptation. `SkillRuntime`, `SceneRegistry`, and built-in `EffectMonitor`s own reusable runtime behavior. |
| Object providers can be individually registered by the program. | A single `SceneRegistry` is authoritative for identity, pose source, geometry, affordance, and collision metadata. Programs reference registry IDs. |
| Callers may supply place EEF poses and pickup look-ahead options. | `Place` is object-centric; the compiler derives EEF targets from verified held state and propagates downstream targets automatically. |
| Configuration and handwritten code are separate entry paths. | Both construct the same semantic call specification and converge before binding or grounding. |

### 5.1 Segment terminology after #475

The design uses three different segment layers. Bare "segment" should be
avoided wherever the layer would be ambiguous.

| Term | Type | Meaning |
|---|---|---|
| Program segment | `SegmentCfg` | Expert Program logical transaction boundary; owns post-policies, validators, and re-observation semantics. |
| Demo segment | `DemoSegment` | Lazy Gym/demo executor carrier and dataset boundary produced from a program segment. |
| Trajectory segment | `TrajectorySegment` | Named half-open waypoint range within one `ActionPlan`; used for inspection, visualization, tracing, and terminal-effect correlation only. |

A trajectory segment is not an independent planning, recovery, effect, or
timeout boundary. One atomic action remains the recovery/effect boundary.
"Phase 0" through "Phase 8" below refer only to implementation-plan stages;
atomic motion structure is called a trajectory segment, not a phase.

## 6. Proposed architecture

### 6.1 API layers

#### Semantic user layer

The golden Python path should remain close to:

```python
skills = AtomicSkills.from_env(env, preset="safe")
cube = skills.scene.object("cube")
tray = skills.scene.object("tray")

result = skills.run(
    Pick(cube),
    Place(cube, on=tray),
)
```

The public concepts are:

- `AtomicSkills`: convenience facade and factory;
- `SkillRuntime`: synchronous and step-wise execution service;
- `SceneEntityRef` and specialized object/articulation references;
- object-centric semantic calls such as `Pick`, `Place`, and `HandOver`;
- one `SkillResult` with per-environment state, events, effects, and failures;
- stable presets such as `safe`, `fast`, and `precise`.

#### Integration layer

The integration layer contains:

- `SceneRegistry` and immutable scene snapshots;
- `RobotSkillProfile` and capability-based binding;
- `SemanticSkillCompiler` and goal grounders;
- grasp/affordance providers;
- built-in `EffectMonitor`s;
- named runtime/planning presets;
- catalog discovery and explicit engine installation.

This layer translates semantic intent to the current core contracts. It is
shared by interactive Python, configuration, demo collection, and future MLLM
callers.

#### Core/advanced layer

The current action-owned goal dataclasses, `ActionInvocation`, generic endpoint
`ActionBinding`, policies, `PlanningContext`, `ActionPlan`,
`ExecutionSession`, `ExecutionRunner`, and provider protocols remain available
for framework authors and unusual integrations. `ActionPlan.commands` is the
runtime authority; a joint-backed plan may additionally retain a
`TimedTrajectory` for joint feedback and offline compilation. These contracts
are no longer prerequisites for ordinary task authoring.

### 6.2 Proposed package ownership

The exact names can be finalized in the first API PR, but ownership should be
kept separate:

```text
embodichain/lab/sim/skills/
  calls.py              # semantic call specs and public facade
  scene.py              # entity references and SceneRegistry
  profiles.py           # RobotSkillProfile and named presets
  compiler.py           # workflow analysis, grounding, invocation lowering
  runtime.py            # SkillRuntime built on ExecutionRunner
  effects.py            # built-in EffectMonitor contracts/implementations

embodichain/lab/sim/atomic_actions/
  runtime_commands.py   # transport-neutral endpoint payloads and timed frames
  transports.py         # endpoint transport protocol and exact-ID router
  ...                   # typed core and built-in atomic planners

embodichain/lab/gym/envs/expert_program/
  cfg.py                # strict @configclass schema
  decoder.py            # versioned discriminated decoding
  bridge.py             # Gym/demo runtime ports and DemoSegment adapter
  post_policies.py      # environment-aware segment post-policies
  validators.py         # declarative segment validators
```

The semantic package is a curated entry point. Existing atomic-action exports
remain compatible during migration; removals require separate deprecation work.

## 7. Core integration contracts

### 7.1 Scene registry as the single source of truth

`SceneRegistry` owns stable semantic identity and all integration metadata for
an entity:

- observation/pose provider;
- geometry and collision representation;
- semantic type and affordances;
- articulation/link relationships;
- dynamic/static classification;
- aliases used by perception or hardware adapters.

The reference hierarchy should support at least:

```text
SceneEntityRef
  +-- SceneObjectRef
  +-- SceneArticulationRef
  +-- SceneLinkRef
  +-- SceneAffordanceRef
```

Rules:

1. The registry ID is the authoritative entity identity used by semantic calls,
   snapshots, scene dependencies, effect monitors, and planner obstacles. An
   entity is registered once under that ID.
2. A simulation object's existing `uid` may be imported as a legacy alias only.
   Aliases are resolved once at an integration boundary and normalized to the
   registry ID; they never replace the authoritative ID. Duplicate registry IDs,
   ambiguous aliases, or an alias colliding with another registry ID fail during
   registry construction.
   For links and affordances, one typed `(parent, native_name)` physical source
   may have only one canonical ID; changing the canonical spelling does not
   create a second entity.
3. Grounding reads dynamic pose/confidence from one immutable snapshot and
   static geometry/affordance metadata from the immutable registry that
   produced it. It must not mix a snapshot with a live simulation entity pose.
4. Automatic grasp selection declares a target dependency automatically.
5. Collision setup is derived from authoritative registry IDs.
   `collision_geometry_by_id()` derives planner geometry while excluding
   non-collision registrations, and `make_planning_scene_provider()` performs
   the complete provider/planner cross-validation. The registry's full
   `STATIC ∪ DYNAMIC` collision ID set must exactly equal the planner's complete
   collision-world ID set. Within it, the registry's dynamic subset, the
   provider's `collision_entity_ids`, and the planner's dynamic-obstacle IDs must
   also agree exactly. Every collision ID must have the required geometry, and
   the selected planner must support the declared dynamic update mode. Aliases
   are resolved before these contracts are constructed, never inside the
   planner. The current
   planner-local name check remains a lower-level defensive validation, not the
   integration contract.
6. The `safe` preset requests `DynamicCollisionMode.REQUIRED` when the registry
   declares dynamic collision entities and fails early if the active planner
   cannot satisfy it.
7. Environment scene configuration opts into registry population explicitly;
   it is not inferred by scanning the simulation. Explicit providers remain
   available for perception and hardware integration.

PR2A fixes three public identity and collision-world choices:

- **Authoritative planner IDs:** registry-backed cuRobo configuration passes an
  explicit `registry_id -> RigidObject` mapping derived by
  `collision_geometry_by_id()`. Mapping keys are canonical logical/source IDs
  for cache identity and the complete registry/planner collision-world
  contract. Cuboid and mesh worlds also use them unchanged as physical YAML
  obstacle names and runtime pose-update keys. A static sphere source expands
  to derived physical names such as `registry_id_0`; dynamic sphere worlds are
  rejected. A registry mapping with geometry missing for the selected
  representation fails fast instead of silently omitting that source. The list
  form remains an advanced direct-core path and continues to derive names from
  simulator UIDs or fallback names.
- **Flat reference IDs:** object, articulation, link, and affordance IDs share
  one globally unique flat namespace. Link and affordance ancestry is stored in
  `SceneEntityRegistration.parent`; callers do not encode hierarchy into an ID.
  Within one reference type, the same `(parent, native_name)` cannot be assigned
  more than one canonical ID.
- **Explicit vectorized-world semantics:** one-environment dynamic collision
  setup may infer a shared collision world. A multi-environment registry with
  dynamic collision entities must explicitly select shared or per-environment
  collision worlds, and integration validation requires the planner mode to
  match that selection.

PR1 provides the core migration bridge consumed by the registry.
`ObjectSemantics.entity_id` remains a string lowering target in the typed core;
the canonical semantic path obtains that value from a resolved
`SceneEntityRef`. `ObjectSemantics` is shallow-frozen so top-level fields,
including `entity_id`, cannot be rebound after attachment state captures the
semantics; identity changes require a new instance. Nested affordance and
metadata objects remain mutable but never establish identity.

For object identity, explicit and legacy namespaces stay separate. If either
side supplies `entity_id`, both sides must supply the same explicit ID; a
same-spelled simulation `entity.uid` is not sufficient. Only when both explicit
IDs are absent may the bridge compare non-empty legacy UIDs, requiring both UIDs
to exist and match. Only when neither side has an explicit ID or valid UID may
comparison fall back to the same semantic object or live entity handle.
Semantic labels are never identity. Arbitrary alias mapping, uniqueness
enforcement, and normalization to an authoritative registry ID belong to PR2A.

For pose grounding, an explicit `entity_id` is strict: the pose comes only from
the current versioned `PlanningContext.scene`, and a missing entry is an error.
The planner never falls back to a live entity after an explicit ID fails. A live
`ObjectSemantics.entity` read remains temporarily available, with a deprecation
warning and without a scene dependency, only when no `entity_id` was supplied.
The same boundary applies to `AssembleGoal.base_pose`: the snapshot reference is
canonical, while an omitted reference permits the deprecated direct-core
`AssembleAffordance.base_object_entity` path.

PR2A hardens `SceneSnapshot` at the public boundary. Construction owns a copy of
every dynamic `EntityState`, and every public entity lookup returns a defensive
copy, so mutating an input state or a previously returned pose cannot mutate the
published snapshot. The registry continues to own static integration metadata,
including typed identity, aliases, parent relationships, geometry, collision
role, dynamics classification, semantic type, and affordances. A snapshot owns
only versioned dynamic pose/confidence plus collision revision metadata; it does
not duplicate the registration catalog.

### 7.2 Robot skill profiles

A `RobotSkillProfile` is reusable per embodiment, but its resource model is not
an `arm + tool` schema. It contains a generic resource DAG:

- each `RobotResource` has a stable logical ID, zero or more named execution
  endpoints, and optional member resources;
- each endpoint declares open, namespaced capabilities explicitly and lowers
  through a `ResourceEndpoint` implementation; `ControlPartEndpoint` is the
  current joint/control-part declaration, while registered
  `ResourceEndpointAdapter`s resolve any endpoint kind into generic
  `EndpointResolution` metadata (a typed runtime target, command-profile key,
  physical claim tokens, and optional joint IDs) without changing the graph,
  matcher, or slot model. Adapters register by exact endpoint type; the
  built-in control-part adapter is not overrideable, and different controller
  semantics use a new endpoint subtype;
- members describe physical composition and claim closure, not capability
  inheritance. A composite must explicitly declare `motion.whole_body`; it
  does not acquire that capability because it contains a base, torso, or arms;
- semantic control commands such as `open`, `grasp`, or a future `stop` remain
  embodiment data owned by generic profile IDs selected by each endpoint
  adapter; only the current core bridge lowers applicable profiles to robot
  control-part keys;
- versioned `SkillPolicyPreset` values own motion, recovery, and runner policy,
  plus an optional required-planner compatibility constraint;
- per-skill defaults map every skill-local slot to one resource ID.

Resource and endpoint declarations are owned snapshots. A custom endpoint with
non-trivial nested payloads implements `snapshot()` to return an independent
value of its exact type, so caller-owned mutation cannot rewrite a bound
profile.

Skills own the robot-independent half of the contract. A concrete atomic action
must explicitly publish a `SkillBindingContract`; inheriting the default
`primary` role or inheriting another action's contract does not expose a new
semantic skill. The contract declares skill-local participant slots and the
endpoint requirements inside each participant. For example, `pick_up` has one
`primary` participant with a `motion` endpoint and a `grasp` endpoint. A profile
can satisfy it with `left_actor`, whose endpoints lower to `left_arm` and
`left_hand`. Selecting the participant as one unit prevents invalid cross-side
combinations such as `left_arm + right_hand`.

Endpoint names are local protocols, not global robot-part categories. A future
`navigate` skill can require `body.motion: motion.base.se2`; a
`whole_body_reach` skill can require `body.motion: motion.whole_body`. Neither
requires new `RobotSkillProfile` fields. Profile resolution lowers every
required endpoint directly into an engine-owned `ActionBinding` keyed by
`(slot_id, endpoint_id)` and carrying its typed runtime target; there is no
arm/tool-shaped intermediate binding layer.

Binding follows strict rules:

1. Filter each slot by endpoint presence, all required capabilities, typed
   semantic commands, explicit caller selection, and installed endpoint
   support.
2. Apply explicit physical-claim constraints. Built-in manipulation contracts
   declare their `motion` and `grasp` views disjoint, while coupled whole-body
   views may overlap when the skill omits that constraint. Multi-participant
   contracts such as handover use pairwise-disjoint resource claims.
3. No candidates means the skill is unsupported on this profile and is omitted
   from the profile-backed semantic catalog.
4. One complete candidate is selected automatically.
5. Multiple candidates are resolved only by a complete, still-valid per-skill
   default or enough explicit slot selections. Partial defaults, mapping order,
   and lexical order never break ambiguity.

`ResourceClaim` contains transitive leaf-resource IDs, concrete joint IDs, and
adapter-defined physical/controller claim tokens. It makes `whole_body`
conflict with `base`, `torso`, or a contained arm even when the underlying
`Robot.control_parts` names are different, and lets a non-joint base adapter
claim a controller without inventing joints. PR2B
exposes deterministic claim/conflict data only. PR2C runners emit endpoint
command frames and transports own target-scoped safe holds, but claims still do
not imply safe parallel execution. Parallel scheduling still requires one
coordinator, deterministic command arbitration/merge, planner serialization or
isolation, cancellation semantics, and inter-trajectory collision checks.

`AtomicActionEngine.actions` remains the direct-core implementation registry.
`engine.skills` contains only installed, agent-visible actions whose concrete
class explicitly declares a binding contract. A bound profile filters that
catalog again by the current robot resources. Constructing an engine with
`skill_profile=...` installs the profile's command snapshots as the single
authoritative source and binds the validated profile after built-ins load.
Known FK/IK capabilities on the control-part adapter are checked against the
selected control part's configured solver; Cartesian motion is not equated with
solver presence because native planners may provide it directly. Profile joint
commands must be one-dimensional and broadcastable; per-environment values
belong in invocation overrides.

### 7.3 Semantic call specification

Version 1 should provide first-class calls for:

- `Pick(object, grasp?, resources?)`;
- `Place(object, pose?|on?|in?, resources?)`;
- `HandOver(object, final_target?, resources?)`;
- a registered semantic call for shared extensions.

`resources`, when present, is a mapping from the selected skill's local slot
IDs to profile resource IDs (for example, `{"primary": "left_actor"}` or
`{"body": "mobile_base"}`). It is an explicit ambiguity override, not a
fixed arm/tool field. Ordinary calls omit it and use unique or profile-default
resolution. Within one analyzed workflow, an omitted `Place.primary` or
`HandOver.source` inherits the known resource holding that object. Explicit
consumer selections remain authoritative constraints and fail if they conflict
with the known holder; inference never crosses a registered-call boundary.

`Place` consumes verified held-object state. The compiler computes the release
EEF pose from the requested object-space target and the verified
`object_to_eef` relation. Task code and configuration never perform
`desired_object_pose @ object_to_eef`.

As the core migration path for assembly, `AssembleGoal` gains
`base_pose: SceneEntityPose | None`. The semantic compiler always supplies a
`SceneEntityPose` containing the authoritative base-object registry ID, so the
base pose is resolved from the same immutable snapshot and automatically becomes
a scene dependency. `None` preserves the existing live
`AssembleAffordance.base_object_entity` lookup only for legacy direct-core
callers; the semantic facade and Expert Program never emit that fallback.

The workflow compiler inspects later calls and propagates downstream object
targets to pickup/grasp selection. The caller does not repeat later goals in
`PickUpOptions`.

### 7.4 Semantic skill compiler

Compilation has two stages.

1. **Static workflow analysis**
   - validate references, presets, capabilities, resources, and bounded loops;
   - infer ordering and data/effect dependencies;
   - propagate downstream object goals for grasp selection;
   - identify every call boundary that requires fresh observation or verified
     effects, without coalescing calls in Version 1;
   - reject ambiguous bindings and unsupported semantic relations before
     execution.
2. **Runtime grounding and lowering**
   - capture the latest registry snapshot for active environment IDs;
   - resolve the next semantic goal and binding;
   - consume verified task state such as held-object relations;
   - lower to a typed `ActionInvocation`;
   - dispatch through the canonical `SkillRuntime`.

Version 1 executes exactly one semantic call per `ExecutionSession`. The runtime
captures a fresh registry snapshot, lowers one call to one `ActionInvocation`,
constructs a one-invocation session, drives it through terminal effect
verification, commits the verified per-environment task state, and only then
advances to the next call. It never places multiple semantic calls in one
`ExecutionSession`.

Static `engine.compile()` remains an advanced core API for explicitly
observation-independent offline planning. The Version 1 semantic runtime does
not coalesce calls into static stages; such an optimization requires a later
design proving that it preserves the call, effect, and re-observation
boundaries.

### 7.5 Skill runtime

`SkillRuntime` is an application-facing owner around the existing
`ExecutionRunner`, not a replacement scheduler. It provides:

- synchronous `run(...)` and non-blocking `step()` entry points;
- planning-context refresh through registered observation ports;
- JIT lowering of the next semantic call;
- exactly one semantic call and one invocation per `ExecutionSession`;
- persistent, per-environment verified `TaskState`;
- built-in effect-monitor selection and feedback to `ExecutionSession`;
- uniform `SkillResult`, cancellation, timeout, and safe-stop behavior;
- semantic action events and optional trajectory-segment trace metadata.

Catalog discovery and runtime installation should have distinct names. For
example, a catalog can `discover` a descriptor while an engine explicitly
`install`s an implementation. The final naming is an API-review item, but one
verb must not imply both operations.

### 7.6 Effect monitors, validators, and post-policies

These responsibilities remain separate:

| Component | Scope | Runtime effect |
|---|---|---|
| `EffectMonitor` | One semantic/atomic call | Reports grasp, release, handover, or articulation effect success to the execution session; participates in recovery. |
| Segment post-policy | Between motion completion and segment validation | Advances environment behavior such as settling; does not duplicate atomic-action recovery. |
| Segment validator | Dataset/task boundary | Decides whether the completed segment is acceptable and records task metrics. |

Simulation integrations should provide standard monitors for grasp, release,
handover, and articulation-joint progress. Hardware can implement the same
contract with perception, force, or controller feedback. Custom monitors stay
an advanced extension point.

## 8. Expert Program configuration

### 8.1 Version 1 schema

The top-level model is:

```text
ExpertProgramCfg
  schema_version
  program_id
  integration
    robot_profile
    scene_registry
    runtime_preset
  targets
  program: ProgramNodeCfg
```

The initial discriminated unions are:

```text
ProgramNodeCfg = SequenceCfg | RepeatCfg | SegmentCfg | InvokeCfg
SemanticCallCfg = PickCfg | PlaceCfg | HandOverCfg | RegisteredSemanticCallCfg
```

`Parallel`, `Barrier`, `If`, and `Retry` are reserved for later schema versions.
`Repeat` is bounded and has a validated maximum. Version 1 contains no
unbounded loops.

Use `@configclass` for the typed configuration objects and a dedicated explicit
decoder for YAML/JSON. The decoder must:

- reject unknown fields and unknown discriminators;
- resolve only stable registry/catalog IDs;
- reject imports, `eval`, executable expressions, and dotted environment
  attribute traversal;
- validate all entity, target, profile, skill, and validator references before
  simulation starts;
- report errors with a complete configuration path;
- support explicit schema migrations rather than silently changing meaning.

### 8.2 Illustrative repeated-cube program

This example is intentionally semantic. Pose values describe the task goal;
there are no robot bindings, planner settings, EEF transforms, trajectory
indices, control periods, or custom verifier functions.

```yaml
schema_version: 1
program_id: repeated_cube_pick_place

integration:
  robot_profile: auto
  scene_registry: env
  runtime_preset: safe

targets:
  drop_pose:
    kind: cyclic_pose
    values:
      - position: [0.45, -0.20, 0.20]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
      - position: [0.45, 0.00, 0.20]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
      - position: [0.45, 0.20, 0.20]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]

program:
  kind: repeat
  count: 3
  body:
    kind: segment
    name: move_cube
    steps:
      kind: sequence
      items:
        - kind: invoke
          call:
            kind: pick
            object: cube
        - kind: invoke
          call:
            kind: place
            object: cube
            at:
              kind: target_ref
              target: drop_pose
    post:
      - kind: wait_stable
        entity: cube
        preset: rigid_object
    validators:
      - kind: object_near_target
        object: cube
        target: drop_pose
        position_tolerance: 0.03
```

Each repeat iteration resolves the next typed target value. A target reference
is a schema object and registry lookup, not Python string interpolation.

### 8.3 Advanced overrides

An advanced section may expose typed, validated policy overrides after the
semantic path is stable. It must be opt-in and must not be required by built-in
examples. Stable names should be preferred over internal fields:

```yaml
advanced:
  call_presets:
    pick: precise
  recovery_preset: dynamic_scene
```

Raw planner instances, callables, arbitrary imports, and environment paths are
never serializable configuration values. Version 1 does not attach motion or
recovery policies to individual `TrajectorySegment`s.

## 9. Demonstration execution semantics

### 9.1 Thin Gym/demo bridge

`AtomicDemoBridge` adapts `SkillRuntime` to lazy `DemoSegment`s. It should use
Gym-aware runtime ports:

- observation provider: captures a current planning context from the
  environment and scene registry;
- command sink: buffers the next transport-neutral endpoint-command frame for
  the environment action manager;
- clock: advances only when the demo executor calls `env.step()`;
- metadata sink: records compiler decisions, action trajectory segments,
  effects, recovery, scene revisions, and post-policy results.

The existing `SimulationExecutionAdapter` is not the demo execution loop
because direct simulator updates can bypass environment managers and recorders.
The bridge should remain a generator: yield a command, let the environment step
normally, then resume with a fresh observation.

### 9.2 Timing

`BaseEnv.step_dt` is the authoritative control cadence. Semantic task
configuration, `MotionPolicy`, and `AtomicActionEngine` do not expose or own a
fallback `control_dt`. Environment integrations copy `BaseEnv.step_dt` into
`PlanningContext.control_dt` when an action performs deterministic interpolation.

Timing is a strict producer contract. A planner result with positions includes
per-waypoint `dt` and derives its per-environment `duration`; an atomic action
passes a complete `TimedTrajectory` to `build_plan()`. Missing or inconsistent
timing is rejected at construction. No layer repairs an untimed planner result
or raw action position tensor with a default period.

Version 1 should require every emitted
`RuntimeCommandFrame.hold_duration` to be representable by an integer number
of environment steps, preferably one step per yielded frame. An incompatible
frame is rejected with a clear timing error; it is not silently resampled.
Explicit timed-command resampling can be a later, separately tested feature.

Recovery timeout and retry budgets are scoped to the enclosing action attempt.
A `TrajectorySegment` does not start an independent timer or own a recovery
policy. Program-segment settling and validation use separate post-policy
deadlines.

### 9.3 Named atomic trajectory segments

Version 1 freezes the trajectory-segment names already emitted by current
built-ins. A successful non-empty plan exposes the following ordered names;
zero-length optional segments are omitted:

| Atomic skill ID | Ordered trajectory-segment names |
|---|---|
| `move_joints` | `move_joints` |
| `move_end_effector` | `move_end_effector` |
| `move_held_object` | `transport` |
| `pick_up` | `approach`, `close`, `lift` |
| `place` (including `AssembleGoal`) | `approach`, `release`, `retract` |
| `press` | `close`, `press`, `retract` |
| `hand_over` | `transfer`, `approach`, `close`, optional `hold`, `release`, `deliver` |
| `coordinated_pickment` | `approach`, `close`, `lift`, `move`, optional `hold` |
| `coordinated_placement` | `approach`, optional `hold`, optional `release`, `retreat` |

These spellings are a trace/metadata contract. Renaming or removing one requires
an explicit API review and migration rather than a silent change in a primitive.

Names are validated by `ActionPlan`; ranges may change after replanning when a
backend returns a different sample count. Effect monitors run at the action
effect boundary and may use `EffectVerificationRequest.terminal_segment` for
correlation. Program post-policies and validators subscribe to program/demo
segment boundaries, not trajectory segments. Articulation segment names should
be stabilized with the reusable articulation skill rather than predeclared in
the configuration schema.

### 9.4 Dynamic settling

The current reset event `wait_for_dynamic_objects_to_settle` should be refactored
around reusable state:

```text
DynamicSettleMonitorCfg
DynamicSettleMonitor.observe(snapshot, env_ids) -> per-env settle state
```

The reset functor can keep its current behavior by using the monitor internally.
The Expert Program post-policy uses the same monitor but emits hold actions via
`env.step()` so observations and dataset frames are recorded. It records at
least elapsed steps, final velocities, per-environment settled masks, timeout
masks, and threshold preset in segment metadata. Settling does not implicitly
clear object dynamics.

### 9.5 Per-environment behavior

All runtime state is indexed by stable environment IDs:

- scene revisions and active collision dependencies;
- current program segment, semantic call, action waypoint, and command deadline;
- recovery budgets and failure masks;
- verified held-object/effect state;
- post-policy progress and segment validation;
- result and metadata.

Version 1 uses a shared program/call barrier for the environment batch; it does
not maintain a divergent AST program counter or a separate `ExecutionSession`
per environment. The runtime advances to the next semantic call or program
segment only when every still-eligible active row reaches the current boundary.
A slower or recovering active row therefore keeps the batch at that boundary.

Within the shared barrier, task state, effects, recovery budgets, eligibility,
success, and failure remain independent per environment. Completed, failed, or
otherwise inactive rows emit hold behavior and cannot overwrite another row's
state while the active cohort catches up.

## 10. Action Bank migration

Expert Program replaces task-specific Action Bank authoring only through proven
capability parity.

| Action Bank concept | Expert Program / semantic runtime |
|---|---|
| scope | Program `SegmentCfg` or nested `SequenceCfg` |
| custom node function | registered semantic call and shared compiler |
| custom edge/target function | typed target provider or goal grounder |
| graph edge | explicit sequence/effect dependency inferred by compiler |
| synchronization constraint | later `Parallel`/`Barrier` resource contract |
| task subclass | scene registry/profile plus declarative program |
| precomputed target/trajectory | JIT grounding from the latest snapshot |
| Gantt scheduling | later resource-aware parallel scheduling over the canonical runtime |

Migration rules:

1. Keep `create_demo_action_list` and current Action Bank configuration paths
   working during the transition.
2. Add `EmbodiedEnvCfg.expert_program` and a CLI input such as
   `--expert_program`; reject simultaneous legacy and new program inputs.
3. Do not require official-task migration in PR1. Start opt-in sequential-task
   migration with the repeated-cube vertical slice after the registry, compiler,
   runtime, and demo bridge contracts are available, then compare generated
   metadata and outcomes.
4. Add `Parallel` only with deterministic resource conflict checks, trajectory
   alignment, synchronization barriers, and per-environment `StateDelta`
   merging.
5. Migrate PourWater only after those parallel contracts are tested.
6. Announce deprecation only after documented feature parity, examples, and a
   compatibility window. Remove legacy code in a separate change.

## 11. Articulated interactions and Open Drawer

The Open Drawer task on current main demonstrates that configuration alone
cannot replace missing reusable semantics: it manually derives a handle
approach, gripper close, pull trajectory, and command assembly.

The integration model should therefore support articulation, link, and
affordance references from the beginning, while implementation remains phased.
A reusable semantic call can be shaped as:

```text
OperateArticulation(
  articulation=drawer,
  affordance=handle,
  target={joint_position or semantic state},
  operation=pull,
)
```

Its compiler selects an affordance pose, resolves one participant resource and
its required motion/interaction endpoints, builds the approach and constrained
operation, and installs an articulation effect monitor. Once implemented once
in the shared layer, Open Drawer variants should differ only in
scene/affordance data, target state, resource defaults, presets, and validators.

This is the precise meaning of "almost no action-layer code": task expansion is
configuration-only when a compatible semantic capability already exists; new
interaction physics extends the shared skill library once, never each task.

## 12. Implementation plan

Each item below should remain a focused PR with its own public-API review and
tests. The dependency order is:

```text
Phase 0 correctness (complete)
        |
        v
PR1 snapshot/identity bridge (complete)
        |
        +-----------------------+
        v                       v
PR2A SceneRegistry      PR2B RobotSkillProfile
   (implemented)          (implemented)
        |                       |
        |                       v
        |              PR2C Runtime Endpoints
        |                    (implemented)
        +-----------+-----------+
                    v
Semantic calls/compiler --> SkillRuntime/effect monitors
        |                              |
        +---------------+--------------+
                        v
            Expert Program + demo bridge
                        |
                        v
              repeated-cube vertical slice
                        |
             +----------+-----------+
             v                      v
    articulation/Open Drawer   Parallel/PourWater
             +----------+-----------+
                        v
              Action Bank deprecation
```

### Phase 0: correctness and core-contract decisions (complete)

Landed on `main` through #475:

- cumulative sub-threshold translation and rotation compare against the last
  published pose;
- target/general-scene and per-environment collision revisions have regression
  coverage;
- the dead `MotionPolicy.interpolation` field is removed and strategy dispatch
  is unified;
- one action owns one trajectory and one recovery/effect boundary, while named
  `TrajectorySegment`s remain metadata;
- `_plan()` is the only supported custom-action extension hook. The immediate
  class-definition failure for a legacy `plan()` override is a documented,
  tested, intentional hard break with no compatibility adapter;
- planner-local dynamic-obstacle name validation remains in place as a defensive
  core check. Complete provider/planner cross-validation is deliberately owned
  by the authoritative `SceneRegistry` integration in Phase 1.

Exit criteria were met by PR #475 and remain the foundation for the completed
Phase 1 bridge. That bridge adds neither a legacy `plan()` adapter nor a
pre-registry duplicate of the integration-level obstacle validator.

### PR1: core snapshot and identity bridge (complete)

PR1 is deliberately smaller than Phase 1. It establishes the core seams that
the later registry and profile integrations consume:

- add optional, validated `ObjectSemantics.entity_id` as the stable
  `SceneSnapshot` key for canonical object grounding;
- resolve explicit IDs only from `PlanningContext.scene`, with a hard error and
  no live fallback when the snapshot entry is missing;
- keep `ObjectSemantics.entity` only as a deprecated no-ID compatibility path;
- shallow-freeze `ObjectSemantics` fields so captured `entity_id` values cannot
  be rebound without constructing a new semantic value;
- define stable held-object identity and partial-batch `StateDelta` merging:
  if either side has an explicit `entity_id`, both explicit IDs must exist and
  match; only two explicit-ID-less values may compare matching legacy
  `entity.uid` strings, and only values with neither ID form may fall back to the
  same semantic object or live handle;
- preserve scalar semantics during same-identity partial `StateDelta` merges:
  while any previously active row remains, retain `previous.semantics` and
  merge only per-environment masks, transforms, and grasp poses; adopt
  `candidate.semantics` only when all previously active rows are replaced;
- add an action-owned scene-dependency hook. `PickUp` declares its semantic
  object ID, coordinated pickup declares it only for the implicit initial-pose
  path, and goal-owned `SceneEntityPose` values remain automatic dependencies;
- resolve each pickup object pose once per planning attempt and reuse that
  tensor for grasp sampling, upright adjustment, and `object_to_eef`;
- derive held-object pose for `MoveHeldObject` and `HandOver` from the observed
  EEF pose and verified `object_to_eef` instead of a live entity read;
- add `AssembleGoal.base_pose: SceneEntityPose | None`; the explicit reference
  is snapshot-backed and dependency-tracked, while `None` retains the deprecated
  `AssembleAffordance.base_object_entity` fallback;
- add focused tests, documentation, and one canonical snapshot-grounded moving
  target tutorial. Keep `scripts/tutorials/atomic_action/assemble.py` explicitly
  documented as a legacy fallback example until its later registry migration.

PR1 does not add a `SceneRegistry`, a `SceneEntityRef` hierarchy, alias maps,
cross-source uniqueness or collision validation, a `RobotSkillProfile`, or
semantic presets. It does not require official task environments to migrate;
they remain on the compatibility path until a later opt-in vertical slice.

Exit criteria are met on `main` through PR #487: canonical object grounding
never mixes snapshot and live poses; explicit missing IDs fail; dependency
metadata matches the poses actually consumed; stable-identity merges are
deterministic; and existing direct-core callers remain usable only through the
documented deprecated fallbacks.

### Phase 1: unified integration data

Phase 1 is implemented as three focused follow-up PRs. PR2A and PR2B branch
from the PR1 foundation; PR2C follows PR2B and joins PR2A before the semantic
facade/compiler work.

#### PR2A: SceneRegistry (landed in PR #487)

Deliverables:

- `SceneEntityRef` hierarchy and `SceneRegistry`;
- authoritative registry IDs with simulation `uid` values accepted only as
  normalized legacy aliases;
- immutable snapshots as the only grounding pose authority for the canonical
  semantic/compiler path;
- opt-in environment-to-registry population and collision/provider derivation;
- complete construction-time agreement checks between registry and planner
  full collision-world IDs, plus registry/provider/planner dynamic subsets,
  geometry, mode, and planner capability;
- explicit catalog-discovery versus engine-installation terminology.

The implemented scope also records the A+C+E decisions from the PR2A API
review:

- registry-backed cuRobo worlds use a canonical-ID mapping, while the list form
  remains an advanced direct-core escape hatch;
- all reference IDs are globally unique and flat, with link/affordance parent
  relations retained only by their registrations;
- one-environment dynamic worlds may default to shared, while vectorized
  dynamic worlds require an explicit shared/per-environment choice.

`ObjectSemantics.entity_id` and `AssembleGoal.base_pose` already provide the
lowering targets from PR1. PR2A replaces manually coordinated IDs/providers with
one authoritative registration and performs alias normalization exactly once at
the integration boundary.

PR2A exit criteria are met by the feature change: registry-ID and alias
collisions fail at construction, typed lookups cannot silently change entity
kind, one typed parent/native physical source cannot be registered twice,
registry-derived snapshots contain only canonical keys, independent providers
keep independent revision state, full registry/planner collision-world IDs and
dynamic registry/provider/planner subsets are validated before execution, the
collision-world batch mode agrees, and cuRobo uses canonical mapping keys end
to end as logical source IDs (and as physical keys for cuboid/mesh worlds).

#### PR2B: RobotSkillProfile (landed in PR #487)

Deliverables:

- a generic `RobotResource` DAG whose named `ResourceEndpoint`s are not tied to
  arm/tool categories, plus a formal `ResourceEndpointAdapter` registry and
  `EndpointResolution` protocol; `ControlPartEndpointAdapter` is the first
  implementation;
- action-owned `SkillBindingContract`s with participant-local endpoint,
  capability, typed-command, and disjoint-claim requirements;
- capability-based candidate filtering, complete per-skill defaults, explicit
  selection overrides, and deterministic ambiguity/unsupported diagnostics;
- profile-owned semantic commands plus immutable, versioned planning/recovery/
  runner presets;
- validation against installed agent-visible engine skills, robot control
  parts, joint ownership, endpoint overlap, configured solvers, commands, and
  presets;
- immutable leaf/joint/adapter-token `ResourceClaim` data and explicit
  same-slot endpoint disjointness for future conflict analysis, without
  claiming safe parallel execution.

The profile and endpoint-runtime APIs can represent mobile-base and whole-body
resources today, and the generic paths are covered by whole-body joint and
custom planar-velocity tests. They are extension seams, not built-in navigation
or whole-body behavior: no current curated semantic skill consumes the example
`motion.base.*` or `motion.whole_body` capabilities. A production shared
capability still needs its semantic descriptor/lowerer, atomic skill, payload,
endpoint adapter, transport, and effect integration as applicable. Once that
reusable bundle exists, another task supplies an Expert Program plus typed
scene/profile integration declarations without task-specific motion code.

The standard Gym bridge currently composes every custom transport action over
a full-qpos hold and the standard simulation factory owns a
`MotionGenerator`. This supports robots without named control parts, but a
truly jointless or natively structured mobile controller still needs a reusable
base-action composition/provider integration. That extension must not add
base- or whole-body-shaped fields to the generic resource, binding, runner, or
router contracts.

The current task vertical slices still construct their typed profile bindings
from task modules. Promoting stable bindings into an embodiment-owned profile
catalog is rollout packaging needed for cross-task reuse; it does not require a
new resource or runtime contract.

PR2A and PR2B landed together through PR #487. That foundation does not migrate
official tasks; the repeated-cube vertical slice opts in only after the
compiler, runtime, and demo bridge are available.

#### PR2C: generic runtime endpoints (implemented on the feature branch)

PR2C removes the temporary arm/tool lowering seam and makes the profile's
generic endpoint model executable end to end:

- `ActionBinding` is an engine-owned collection keyed only by
  `(slot_id, endpoint_id)`; `ActionBindingRoute`, arm/tool role maps, and the
  intermediate resolved-control-part binding types are removed as an
  intentional clean break;
- every resolved profile endpoint owns a typed immutable
  `RuntimeEndpointTarget`, while `EndpointCommand` combines that destination
  with a transport-specific `RuntimeCommandPayload`;
- `RuntimeCommandFrame` synchronizes per-environment endpoint commands and
  timing, and `TimedCommandSequence` becomes the authoritative runtime content
  of `ActionPlan`;
- `EndpointCommandTransport` and `EndpointCommandRouter` perform exact-ID
  registration, preflight payload validation, transport grouping,
  acknowledgement aggregation, cancellation, and transport-owned safe holds;
- the framework authorizes planned commands against binding-owned targets and
  physical claims, requires stable destinations across frames and recovery
  replans, and retains previously active targets when a failed plan is empty;
- transports actively neutralize inactive environment rows for every addressed
  target instead of treating an omitted write as a safe state;
- `SimulationExecutionAdapter` implements the built-in joint-position
  transport and writes or holds only the joints claimed by each addressed
  endpoint;
- joint-backed planners retain an optional full-robot `TimedTrajectory` for
  existing joint feedback and `engine.compile()`, while non-joint plans use
  timed completion plus the existing semantic-effect verification boundary;
- full-body joint control and a custom planar-velocity endpoint are exercised
  from binding/profile resolution through planning, session execution, routing,
  completion, and safe hold without arm/tool-shaped fields.
- an explicit invocation revision declares the same non-empty runtime
  destination set and preserves each target's address/safe-hold fingerprint.
  The runner keeps the active frame deadline and replans from a fresh due-time
  observation; a pending physical effect must be verified first. Changing a
  base, arm, whole-body, controller destination, or hold footprint starts a new
  invocation rather than hot-switching controller ownership in place.

PR2C does not add parallel scheduling, claim merging, transport rollback, or a
generic endpoint-feedback evaluator. It also does not add cross-destination
hot revision. Those require separate contracts.

PR2C exit criteria: an installed custom endpoint kind needs one reusable
endpoint declaration/adapter, payload, transport, and shared atomic skill, but
no core binding or runner changes; whole-body joint endpoints use the same
path; unknown transports and incompatible payloads fail before dispatch; and
cancel/hold behavior remains transport-owned and auditable.

Combined Phase 1 exit criteria: an object is registered once under an
authoritative ID, aliases cannot introduce ambiguity, dynamic-object
configuration mismatches fail before execution with an entity-centric
diagnostic, robot capabilities resolve bindings/presets without task-owned
motion code, and generic resolved endpoints can reach their registered runtime
transports without adding arm/tool-specific core paths.

Implementation status: when `safe` is reachable and the registry declares
dynamic collision entities, binding rejects an unsupported active planner
before observation or planning. Linking produces an effective
`DynamicCollisionMode.REQUIRED` preset snapshot without mutating the profile's
source preset. This preflight coverage does not replace the remaining
end-to-end dynamic-obstacle recovery simulation.

### Phase 2: semantic facade and compiler

Implementation status: the semantic facade, provider-free linking, canonical
compiler, bounded program preflight, and cross-segment sequential look-ahead are
implemented. Relation placement remains an exact typed integration capability;
a reusable production support-surface/container affordance and grounder are
follow-up work rather than inferred behavior.

Deliverables:

- `SemanticCallSpec`, object-centric `Pick`, `Place`, and `HandOver`;
- static workflow analysis and downstream-goal look-ahead;
- JIT grounding and lowering to existing `ActionInvocation`s;
- verified held-object state used for object-to-EEF conversion;
- `AtomicSkills` facade and a pick/place quickstart of no more than 15 lines,
  excluding scene construction;
- curated semantic exports while retaining advanced contracts.

Exit criteria: the quickstart performs pick/place without raw qpos, 4x4
transform math, planner selection, context/session construction, or a custom
effect verifier.

### Phase 3: canonical runtime and effects

Implementation status: core contracts are implemented in the current stack.
The backend-neutral typed state expectations, evidence addresses and sources,
pose/binary/scalar/joint evidence clauses, versioned monitor registry,
profile-owned monitor selection, grounded Pick/Place/HandOver/articulation
effects, row-local composite hysteresis kernel, canonical `SkillRuntime`, and
production simulation evidence ports are wired end to end. Physical simulation
acceptance is partial: Open Drawer and one cube Pick/Place/settle/validator
cycle have completed, while the full repeated-cube run and embodiment-owned
HandOver pose integration remain validation work.

Deliverables:

- `SkillRuntime` wrapping `ExecutionRunner` for sync and step-wise use;
- exactly one semantic call lowered to one invocation in one
  `ExecutionSession`;
- built-in simulation effect monitors for grasp, release, and handover;
- uniform per-environment `SkillResult` and persistent verified `TaskState`;
- a shared Version 1 program/call barrier with independent per-environment task,
  effect, recovery, eligibility, and result state;
- safe cancellation, timeout, and hold behavior inherited from the runner.

Exit criteria: Python calls and a programmatic `SemanticCallSpec` use identical
compiler/runtime code and produce equivalent results.

### Phase 4: demo integration primitives

Implementation status: implemented. The bridge uses buffered runtime commands and
an environment-step clock, dynamic settling is shared with reset behavior, and
JSON-safe lifecycle metadata covers every installed plan attempt, named
trajectory segment, effect decision/evidence, recovery event, scene/collision
revision, post-policy outcome, and validator result.

Deliverables:

- expose the existing named plan trajectory segments through optional demo
  trace metadata without adding segment-level recovery;
- reusable `DynamicSettleMonitor` shared by reset and demo paths;
- Gym observation, buffered command, and environment-clock ports;
- thin `AtomicDemoBridge` yielding lazy `DemoSegment`s;
- exact `BaseEnv.step_dt` timing validation;
- runtime metadata for calls, trajectory segments, effects, recovery, scene
  revisions, settling, and validation.

Exit criteria: no demo command bypasses `env.step()`, and no post-policy,
effect, or trace integration contains a hard-coded trajectory index.

### Phase 5: Expert Program version 1 and repeated-cube vertical slice

Implementation status: configuration and task migration are implemented. The
strict decoder/loader, lazy compiler, environment/CLI integration, shared
simulation factory, and three-segment cube program are implemented. The task
combines declarative program configuration with typed scene/profile integration
declarations and installs the shared adapter without overriding task motion
generation. A supported-simulation run has completed the first physical
Pick/Place/settle/validator cycle; completing all three cycles remains an
acceptance item while thresholds are calibrated.

Deliverables:

- strict `@configclass` schema and versioned decoder;
- `Sequence`, bounded `Repeat`, `Segment`, and `Invoke`;
- registered targets, post-policies, and validators;
- `EmbodiedEnvCfg` and CLI integration with legacy fallback;
- motion-code-free migration of repeated cube pick/place using a declarative
  program and typed scene/profile integration declarations.

Exit criteria:

- three lazy program/demo segments complete in supported simulation;
- each program/demo segment re-observes the cube after free-fall settling;
- grasp and release effects are verified;
- placement uses verified held-object state;
- settle and validation data are present in metadata;
- the environment batch advances through the shared call barrier while success,
  failure, effect, recovery, and eligibility masks remain independent;
- the task contains no task-specific motion-generation code.

### Phase 6: sequential skill coverage and articulated interaction

Implementation status: the articulation path and task migration are
implemented. Articulation/link/operation-affordance registration,
`OperateArticulation`, typed joint-state effects/evidence, and the declarative
Open Drawer program with typed integration declarations use the same
compiler/runtime path as pick/place. Its supported-simulation physical run now
completes and reaches the configured drawer joint target.

Deliverables:

- articulation/link/affordance registry integration;
- reusable articulation-operation semantic call, compiler, effect monitor, and
  named trajectory segments;
- configuration-based Open Drawer migration;
- migrate additional sequential tasks to reveal missing reusable grounders,
  monitors, and validators.

Exit criteria: Open Drawer task variants no longer assemble approach/pull
trajectories in task code.

### Phase 7: parallel execution and PourWater

Implementation status: the schema/runtime contracts and fail-closed safety
boundary are implemented. Schema
version 2 provides explicit parallel branches and barriers; static resource
conflict analysis, shared-clock lane coordination, deterministic hold padding,
transport/safety validation, row-local failure and cancellation, timeouts, and
deterministic state merge are covered by tests. A production simulation safety
validator and parallel physical integration remain pending. The PourWater task
migration is outside the current scope because it would require modifying
Action Bank code.

Deliverables:

- `Parallel` and explicit `Barrier` nodes in a new schema version;
- robot-resource conflict analysis;
- deterministic strict-step-grid alignment with hold padding; fractional frame
  durations are rejected rather than implicitly resampled;
- synchronization and timeout behavior;
- deterministic per-environment `StateDelta` merge rules;
- PourWater migration from its Action Bank subclass.

Exit criteria: conflict, timing, cancellation, partial failure, and state-merge
tests pass before the legacy task is switched.

### Phase 8: rollout, documentation, and deprecation

Implementation status: partial. The canonical semantic/Expert Program documentation,
project-development context, task vertical slices, and public integration
guidance are included in this stack. Metrics, migrations that touch Action
Bank, and any deprecation proposal remain explicitly separate follow-up work.

Deliverables:

- semantic quickstart and advanced-core integration guide;
- migration guide from Action Bank and direct invocation construction;
- capability coverage matrix for robots, semantic skills, effects, and
  execution modes;
- metrics comparing task code/config size and demo success;
- formal Action Bank deprecation proposal after parity.

Exit criteria: there is one documented canonical workflow and legacy removal is
independent of adoption of the new path.

## 13. Validation strategy

### Unit tests

- strict decoder, unknown fields, schema versioning, bounded repeats, and
  registry reference errors;
- authoritative registry-ID normalization, legacy-`uid` alias collisions,
  typed parent/native-source collisions, complete registry/planner collision-
  world agreement, and registry/provider/planner dynamic-subset agreement;
- cumulative scene movement and collision dependency revision behavior;
- profile capability matching, deterministic binding, and ambiguity errors;
- `AssembleGoal.base_pose` snapshot resolution and its automatic scene
  dependency, with the `None` fallback isolated to legacy direct-core use;
- same-identity partial `StateDelta` merges retain previous scalar semantics
  until every previously active row is replaced, for both individual and
  coordinated attachments;
- exactly one semantic call and one invocation per `ExecutionSession`;
- downstream target propagation for grasp selection;
- object-centric place conversion from one immutable snapshot and verified
  held state;
- effect monitor state transitions and timeout/recovery feedback;
- trajectory-segment coverage/name validation and exact step-duration
  conversion;
- Action Bank compatibility adapters where introduced.

### Integration tests with fake ports

- Python facade and Expert Program lower to equivalent invocations;
- runner scheduling, acknowledgement, safe stop, and cancellation are reused;
- the Version 1 shared call barrier holds active rows together while completed,
  recovering, and failed rows retain independent masks and state;
- command buffering advances only through the environment clock;
- program/demo-segment and trajectory-segment metadata are deterministic and
  serializable.

### Simulation tests

- three-program/demo-segment repeated cube pick/place with free-fall
  re-observation;
- moving target and dynamic collision recovery with the `safe` preset;
- grasp/release/handover effect monitors;
- settling success and timeout metadata;
- Open Drawer articulation effect;
- GPU-backed dynamic cuRobo coverage where supported;
- parallel PourWater only after Phase 7 contracts land.

## 14. Acceptance criteria

The design is complete when all of the following hold:

- [x] A reachable `safe` preset in a dynamic-collision scene resolves to
      `DynamicCollisionMode.REQUIRED` and rejects an unsupported active planner
      before observation, planning, or command emission without mutating the
      profile configuration.
- [x] A versioned Expert Program is fully validated before execution and cannot
      evaluate arbitrary code or traverse environment attributes by string.
- [x] Python, configuration, and MLLM calls share one semantic compiler,
      typed atomic-action core, and runtime.
- [x] A common new task using existing semantic skills needs no task-specific
      motion-generation code.
- [x] Robot capability binding is expressed through generic participant
      resources and endpoints, so mobile-base and whole-body skills do not
      require new arm/tool-shaped profile fields.
- [x] Runtime binding, command framing, routing, and safe stop are endpoint
      generic; joint trajectories remain an optional planning/feedback artifact
      rather than the only runtime carrier.
- [x] Each scene entity is registered once under an authoritative registry ID
      across semantics, observation, affordance, and collision handling;
      simulation `uid` values are legacy aliases only.
- [x] The default pick/place path does not expose raw qpos, grasp/EEF matrix
      math, planner construction, session plumbing, or custom verification.
- [x] Automatic grasping tracks target revisions and receives downstream object
      goals without caller duplication.
- [x] `Place` is object-centric and consumes verified held-object state.
- [ ] Built-in grasp, release, handover, and supported articulation effect
      monitors work in simulation.
- [x] Repeated sub-threshold motion eventually publishes the correct scene
      revision.
- [x] Custom actions have a documented and tested intentional hard-break
      migration from overriding `plan()` to implementing `_plan()`; no
      compatibility adapter is required.
- [x] Version 1 creates exactly one one-invocation `ExecutionSession` for each
      semantic call and re-observes before lowering the next call.
- [x] Demonstration timing is derived from `BaseEnv.step_dt` and commands pass
      through `env.step()`.
- [x] No program post-policy, effect, or tracing integration depends on
      hard-coded waypoint indices.
- [ ] Repeated cube pick/place completes at least three lazy, independently
      observed program/demo segments with settle/effect/validation metadata.
- [x] Version 1 uses one shared program/call barrier while per-environment task
      state, effects, recovery, eligibility, success, and failure remain
      independent.
- [x] Advanced users retain typed goals, invocations, policies, providers,
      sessions, and planners as escape hatches.
- [x] Parallel resource conflicts, synchronization, timing, cancellation, and
      state merging are tested before PourWater migration.
- [ ] Action Bank remains usable until feature parity and a deprecation window
      are documented.

## 15. Risks and design checkpoints

| Risk | Mitigation / checkpoint |
|---|---|
| Semantic facade merely renames low-level fields | Enforce the 15-line quickstart and configuration example as API acceptance tests. |
| Scene registry duplicates environment configuration | Generate it from environment scene definitions where possible; require one explicit adapter only for external perception/hardware. |
| Automatic binding makes surprising choices | Use capability validation and deterministic profile preferences; surface semantic ambiguity rather than silently selecting. |
| Presets become opaque or unstable | Version preset semantics, emit the resolved core policies in runtime metadata, and keep typed overrides available to advanced users. |
| Built-in effect monitors overfit simulation | Keep the contract backend-neutral and provide replaceable hardware implementations; record monitor evidence and thresholds. |
| Static compilation uses stale state | Version 1 never coalesces semantic calls into one session or static stage; keep `engine.compile()` as an explicit advanced-core API until a later optimization proves equivalent observation/effect boundaries. |
| Demo bridge duplicates runner logic | Keep scheduling, acknowledgement, recovery, timeout, and safe stop in `ExecutionRunner`; bridge only the Gym step boundary. |
| Configuration grows into a programming language | Keep version 1 bounded and discriminated; add only registered nodes and no expressions or arbitrary DAG scheduler. |
| Articulation and parallel work delay useful delivery | Ship the sequential cube vertical slice first; add reusable capabilities independently. |
| Premature Action Bank removal breaks tasks | Maintain compatibility until sequential and parallel parity are demonstrated and measured. |

The most important review checkpoints are after Phase 1 (ownership of scene and
robot integration data), Phase 2 (semantic API shape), Phase 5 (configuration
and demo vertical slice), and Phase 7 (parallel semantics). These checkpoints
should approve public contracts before broad task migration.
