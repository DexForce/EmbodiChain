# Declarative Expert Programs and Unified Semantic Skill Runtime

- Status: design plan
- Baseline: `main@26b69c22d7efbf96cb35f5487f6922c8645f91d7`
- Last updated: 2026-08-10
- Related issues: [#471](https://github.com/DexForce/EmbodiChain/issues/471),
  [#474](https://github.com/DexForce/EmbodiChain/issues/474)

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

- a new task that uses existing semantic capabilities: scene configuration,
  Expert Program configuration, and optionally a declarative validator;
- a new robot: one reusable `RobotSkillProfile`, not task-specific motion code;
- a genuinely new physical interaction: one reusable semantic skill/compiler/
  monitor implementation, after which tasks use it from configuration.

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
4. Infer robot-part bindings and stable runtime policies from reusable profiles;
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

This plan is based on commit `26b69c22` rather than uncommitted working-tree
changes.

| Capability | Current main | Design consequence |
|---|---|---|
| Typed goals, invocations, plans, state deltas, recovery, and skill descriptors (#448) | Available | Keep as the core contract; all frontends compile to it. |
| Lazy `DemoSegment` execution and legacy demo compatibility (#460) | Available | Use a thin demo adapter; do not create a second dataset executor. |
| Closed-loop `ExecutionRunner` and simulator ports (#449) | Available | `SkillRuntime` wraps/reuses the runner rather than scheduling commands itself. |
| Dynamic scene recovery and `DynamicCollisionMode` (#450) | Available | Profiles select precise collision semantics and fail early when required capabilities are unavailable. |
| Environment cadence through `BaseEnv.step_dt` (#472) | Available | Expert configuration does not expose a separate control period. |
| Adaptive dynamic-object settling (#470) | Reset/event implementation exists | Extract a reusable monitor; demo post-policies must advance through `env.step()`. |
| Repeated cube pick/place demo | Manually constructs invocations and transform math | First configuration-only vertical slice. |
| Open Drawer task (#473) | Manually builds approach, grasp, pull, and command trajectories | Evidence that the semantic layer needs articulation/link/affordance references and a reusable articulation skill. |
| Action Bank | Configuration plus task-specific Python node/edge functions | Keep only as a compatibility path while semantic coverage is built. |

Several #474 findings remain prerequisites on this baseline:

- `RigidObjectSceneProvider` still updates its pose baseline on every snapshot,
  so repeated sub-threshold movement may never publish a revision.
- `AtomicAction` rejects the formerly documented `plan()` extension override
  and requires `_plan()` without a compatibility window.
- scene pose, semantics, affordance, and collision registration still have
  multiple sources of truth;
- ordinary callers still see a large low-level public surface and must perform
  semantic transform and verifier plumbing;
- `MotionPolicy` still exposes implementation-level tuning, including an
  unused/misleading interpolation option.

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
- named phases instead of trajectory indices;
- sequential execution first, then resource-aware parallel execution;
- continued legacy compatibility during migration.

The following parts must be adjusted:

| Original #471 direction | Revised decision after #474 |
|---|---|
| Expert configuration may directly contain `ActionBinding`, `ActionOptions`, `MotionPolicy`, and `RecoveryPolicy`. | The default schema contains semantic calls and named presets. Raw core contracts are allowed only in a clearly marked advanced override. |
| `InvocationCompiler` grounds configuration directly to `ActionInvocation`. | A shared `SemanticSkillCompiler` first consumes `SemanticCallSpec`; its final lowering stage produces `ActionInvocation`. Python and MLLM calls use the same compiler. |
| `AtomicDemoBridge` owns context, effect verification, timing, and post-policy behavior. | The bridge owns only Gym/demo adaptation. `SkillRuntime`, `SceneRegistry`, and built-in `EffectMonitor`s own reusable runtime behavior. |
| Object providers can be individually registered by the program. | A single `SceneRegistry` is authoritative for identity, pose source, geometry, affordance, and collision metadata. Programs reference registry IDs. |
| Callers may supply place EEF poses and pickup look-ahead options. | `Place` is object-centric; the compiler derives EEF targets from verified held state and propagates downstream targets automatically. |
| Configuration and handwritten code are separate entry paths. | Both construct the same semantic call specification and converge before binding or grounding. |

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

The current `ActionGoal`, `ActionInvocation`, `ActionBinding`, policies,
`PlanningContext`, `ActionPlan`, `ExecutionSession`, `ExecutionRunner`, and
provider protocols remain available for framework authors and unusual
integrations. They are no longer prerequisites for ordinary task authoring.

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
  ...                   # existing typed core and built-in atomic planners

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

1. An entity is registered once. Planner obstacles, scene dependencies, effect
   monitors, and semantic calls consume that registration.
2. Grounding reads pose and geometry from one immutable snapshot. It must not
   mix a snapshot with a live simulation entity pose.
3. Automatic grasp selection declares a target dependency automatically.
4. Dynamic collision setup is derived and cross-validated at construction
   time. The `safe` preset requests `DynamicCollisionMode.REQUIRED` when the
   registry declares dynamic collision entities and fails early if the active
   planner cannot satisfy it.
5. Environment scene configuration should populate the registry automatically;
   explicit providers are reserved for perception and hardware integration.

### 7.2 Robot skill profiles

A `RobotSkillProfile` is reusable per embodiment and contains:

- capability declarations for arms, grippers, hands, and tools;
- mappings from semantic roles to compatible control parts;
- semantic commands such as `open`, `grasp`, `release`, and `ready`;
- available planners/motion strategies and their constraints;
- default grasp, effect-monitor, and runtime preset selections;
- optional preference rules when more than one binding is valid.

The compiler resolves the only valid binding automatically. If two arms are
equally valid and the profile has no deterministic preference, validation asks
for a semantic choice such as `arm: left`; it never asks the task to construct
an `ActionBinding`.

### 7.3 Semantic call specification

Version 1 should provide first-class calls for:

- `Pick(object, grasp?, arm?)`;
- `Place(object, pose?|on?|in?, arm?)`;
- `HandOver(object, receiver?, final_target?)`;
- a registered semantic call for shared extensions.

`Place` consumes verified held-object state. The compiler computes the release
EEF pose from the requested object-space target and the verified
`object_to_eef` relation. Task code and configuration never perform
`desired_object_pose @ object_to_eef`.

The workflow compiler inspects later calls and propagates downstream object
targets to pickup/grasp selection. The caller does not repeat later goals in
`PickUpOptions`.

### 7.4 Semantic skill compiler

Compilation has two stages.

1. **Static workflow analysis**
   - validate references, presets, capabilities, resources, and bounded loops;
   - infer ordering and data/effect dependencies;
   - propagate downstream object goals for grasp selection;
   - identify static stages versus observation-dependent boundaries;
   - reject ambiguous bindings and unsupported semantic relations before
     execution.
2. **Runtime grounding and lowering**
   - capture the latest registry snapshot for active environment IDs;
   - resolve the next semantic goal and binding;
   - consume verified task state such as held-object relations;
   - lower to a typed `ActionInvocation`;
   - dispatch through the canonical `SkillRuntime`.

Static `engine.compile()` is valid only when later goals do not depend on
observations or effects produced by earlier calls. `engine.start()` and observed
execution are required for grasp/release verification, moving targets,
recovery, post-settling, or any JIT-grounded goal. The default mode is `auto`:
the compiler partitions safe static stages and inserts observed boundaries.

### 7.5 Skill runtime

`SkillRuntime` is an application-facing owner around the existing
`ExecutionRunner`, not a replacement scheduler. It provides:

- synchronous `run(...)` and non-blocking `step()` entry points;
- planning-context refresh through registered observation ports;
- JIT lowering of the next semantic call;
- persistent, per-environment verified `TaskState`;
- built-in effect-monitor selection and feedback to `ExecutionSession`;
- uniform `SkillResult`, cancellation, timeout, and safe-stop behavior;
- semantic and named-phase events.

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
  phase_presets:
    secure_grasp: precise
  recovery_preset: dynamic_scene
```

Raw planner instances, callables, arbitrary imports, and environment paths are
never serializable configuration values.

## 9. Demonstration execution semantics

### 9.1 Thin Gym/demo bridge

`AtomicDemoBridge` adapts `SkillRuntime` to lazy `DemoSegment`s. It should use
Gym-aware runtime ports:

- observation provider: captures a current planning context from the
  environment and scene registry;
- command sink: buffers the next full-robot command for the environment action
  manager;
- clock: advances only when the demo executor calls `env.step()`;
- metadata sink: records compiler decisions, phases, effects, recovery, scene
  revisions, and post-policy results.

The existing `SimulationExecutionAdapter` is not the demo execution loop
because direct simulator updates can bypass environment managers and recorders.
The bridge should remain a generator: yield a command, let the environment step
normally, then resume with a fresh observation.

### 9.2 Timing

`BaseEnv.step_dt` is the authoritative control cadence. Semantic task
configuration does not expose `control_dt`.

Version 1 should require every emitted `JointCommand.hold_duration` to be
representable by an integer number of environment steps, preferably one step
per yielded command. An incompatible command is rejected with a clear timing
error; it is not silently resampled. Explicit timed-command resampling can be a
later, separately tested feature.

Timeout for a named phase starts when its first command is dispatched, not when
an earlier phase or the whole segment is compiled.

### 9.3 Named phases

Plans and execution events need stable semantic phase names. Initial built-ins
should expose at least:

- pick: `approach`, `grasp_close`, `lift`;
- place: `lower`, `release`, `retract`;
- handover: role-specific approach, transfer, release, and retreat phases;
- articulation operation: `approach`, `grasp_close`, `operate`, `release`,
  `retract`.

Post-policies and effect monitors subscribe to names, not trajectory sample
indices. The runtime validates requested phase names against the active skill
descriptor before execution.

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
- current call/phase and command deadline;
- recovery budgets and failure masks;
- verified held-object/effect state;
- post-policy progress and segment validation;
- result and metadata.

One environment may finish, recover, settle, or fail without blocking or
overwriting another. Program structure is shared, but runtime progress is
masked per environment.

## 10. Action Bank migration

Expert Program replaces task-specific Action Bank authoring only through proven
capability parity.

| Action Bank concept | Expert Program / semantic runtime |
|---|---|
| scope | `Segment` or nested `Sequence` |
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
3. Migrate sequential tasks first and compare generated metadata and outcomes.
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

Its compiler selects an affordance pose, binds an arm/tool, builds the approach
and constrained operation, and installs an articulation effect monitor. Once
implemented once in the shared layer, Open Drawer variants should differ only
in scene/affordance data, target state, presets, and validators.

This is the precise meaning of "almost no action-layer code": task expansion is
configuration-only when a compatible semantic capability already exists; new
interaction physics extends the shared skill library once, never each task.

## 12. Implementation plan

Each item below should remain a focused PR with its own public-API review and
tests. The dependency order is:

```text
Phase 0 correctness
        |
        v
SceneRegistry + RobotSkillProfile
        |
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

### Phase 0: correctness and compatibility prerequisites

Deliverables:

- fix cumulative sub-threshold translation and rotation publication in
  `RigidObjectSceneProvider` by comparing with the last published/significant
  pose;
- add regression tests for target and collision-world revisions;
- decide the supported `plan()`/`_plan()` custom-action extension contract and
  provide a compatibility/deprecation path before enforcing a break;
- remove or implement misleading `MotionPolicy` fields, keeping collision
  semantics expressed by `DynamicCollisionMode`;
- add early cross-validation for registry/provider/planner obstacle names.

Exit criteria: all #474 P0 items are resolved on main and custom actions have a
documented, tested upgrade path.

### Phase 1: unified integration data

Deliverables:

- `SceneEntityRef` hierarchy and `SceneRegistry`;
- immutable snapshot as the only grounding pose authority;
- environment-to-registry population and collision/provider derivation;
- `RobotSkillProfile`, capability-based binding, semantic tool commands, and
  stable presets;
- explicit catalog-discovery versus engine-installation terminology.

Exit criteria: an object is registered once and a dynamic-object configuration
error fails before execution with an entity-centric diagnostic.

### Phase 2: semantic facade and compiler

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

Deliverables:

- `SkillRuntime` wrapping `ExecutionRunner` for sync and step-wise use;
- built-in simulation effect monitors for grasp, release, and handover;
- uniform per-environment `SkillResult` and persistent verified `TaskState`;
- automatic static/observed stage selection;
- safe cancellation, timeout, and hold behavior inherited from the runner.

Exit criteria: Python calls and a programmatic `SemanticCallSpec` use identical
compiler/runtime code and produce equivalent results.

### Phase 4: demo integration primitives

Deliverables:

- stable named phases in plans/descriptors/events;
- reusable `DynamicSettleMonitor` shared by reset and demo paths;
- Gym observation, buffered command, and environment-clock ports;
- thin `AtomicDemoBridge` yielding lazy `DemoSegment`s;
- exact `BaseEnv.step_dt` timing validation;
- runtime metadata for calls, phases, effects, recovery, scene revisions,
  settling, and validation.

Exit criteria: no demo command bypasses `env.step()`, and phase/post-policy
behavior contains no hard-coded trajectory index.

### Phase 5: Expert Program version 1 and repeated-cube vertical slice

Deliverables:

- strict `@configclass` schema and versioned decoder;
- `Sequence`, bounded `Repeat`, `Segment`, and `Invoke`;
- registered targets, post-policies, and validators;
- `EmbodiedEnvCfg` and CLI integration with legacy fallback;
- configuration-only migration of repeated cube pick/place.

Exit criteria:

- three lazy segments complete in supported simulation;
- each segment re-observes the cube after free-fall settling;
- grasp and release effects are verified;
- placement uses verified held-object state;
- settle and validation data are present in metadata;
- multi-environment success, failure, and recovery masks remain independent;
- the task contains no task-specific motion-generation code.

### Phase 6: sequential skill coverage and articulated interaction

Deliverables:

- articulation/link/affordance registry integration;
- reusable articulation-operation semantic call, compiler, effect monitor, and
  named phases;
- configuration-based Open Drawer migration;
- migrate additional sequential tasks to reveal missing reusable grounders,
  monitors, and validators.

Exit criteria: Open Drawer task variants no longer assemble approach/pull
trajectories in task code.

### Phase 7: parallel execution and PourWater

Deliverables:

- `Parallel` and explicit `Barrier` nodes in a new schema version;
- robot-resource conflict analysis;
- deterministic trajectory alignment/resampling policy;
- synchronization and timeout behavior;
- deterministic per-environment `StateDelta` merge rules;
- PourWater migration from its Action Bank subclass.

Exit criteria: conflict, timing, cancellation, partial failure, and state-merge
tests pass before the legacy task is switched.

### Phase 8: rollout, documentation, and deprecation

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
- cumulative scene movement and collision dependency revision behavior;
- profile capability matching, deterministic binding, and ambiguity errors;
- static versus observed stage partitioning;
- downstream target propagation for grasp selection;
- object-centric place conversion from one immutable snapshot and verified
  held state;
- effect monitor state transitions and timeout/recovery feedback;
- named phase validation and exact step-duration conversion;
- Action Bank compatibility adapters where introduced.

### Integration tests with fake ports

- Python facade and Expert Program lower to equivalent invocations;
- runner scheduling, acknowledgement, safe stop, and cancellation are reused;
- one environment can complete while another recovers or fails;
- command buffering advances only through the environment clock;
- segment metadata is deterministic and serializable.

### Simulation tests

- three-segment repeated cube pick/place with free-fall re-observation;
- moving target and dynamic collision recovery with the `safe` preset;
- grasp/release/handover effect monitors;
- settling success and timeout metadata;
- Open Drawer articulation effect;
- GPU-backed dynamic cuRobo coverage where supported;
- parallel PourWater only after Phase 7 contracts land.

## 14. Acceptance criteria

The design is complete when all of the following hold:

- [ ] A versioned Expert Program is fully validated before execution and cannot
      evaluate arbitrary code or traverse environment attributes by string.
- [ ] Python, configuration, and future MLLM calls share one semantic compiler,
      typed atomic-action core, and runtime.
- [ ] A common new task using existing semantic skills needs no task-specific
      motion-generation code.
- [ ] Each scene entity is registered once across semantics, observation,
      affordance, and collision handling.
- [ ] The default pick/place path does not expose raw qpos, grasp/EEF matrix
      math, planner construction, session plumbing, or custom verification.
- [ ] Automatic grasping tracks target revisions and receives downstream object
      goals without caller duplication.
- [ ] `Place` is object-centric and consumes verified held-object state.
- [ ] Built-in grasp, release, handover, and supported articulation effect
      monitors work in simulation.
- [ ] Repeated sub-threshold motion eventually publishes the correct scene
      revision.
- [ ] Custom actions have a documented and tested compatibility path.
- [ ] Demonstration timing is derived from `BaseEnv.step_dt` and commands pass
      through `env.step()`.
- [ ] Phase hooks use stable names rather than trajectory indices.
- [ ] Repeated cube pick/place completes at least three lazy, independently
      observed segments with settle/effect/validation metadata.
- [ ] Multi-environment progress, effects, recovery, and failures remain
      independent.
- [ ] Advanced users retain typed goals, invocations, policies, providers,
      sessions, and planners as escape hatches.
- [ ] Parallel resource conflicts, synchronization, timing, cancellation, and
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
| Static compilation uses stale state | Default to dependency-driven `auto` partitioning and force observed boundaries after external effects or dynamic post-policies. |
| Demo bridge duplicates runner logic | Keep scheduling, acknowledgement, recovery, timeout, and safe stop in `ExecutionRunner`; bridge only the Gym step boundary. |
| Configuration grows into a programming language | Keep version 1 bounded and discriminated; add only registered nodes and no expressions or arbitrary DAG scheduler. |
| Articulation and parallel work delay useful delivery | Ship the sequential cube vertical slice first; add reusable capabilities independently. |
| Premature Action Bank removal breaks tasks | Maintain compatibility until sequential and parallel parity are demonstrated and measured. |

The most important review checkpoints are after Phase 1 (ownership of scene and
robot integration data), Phase 2 (semantic API shape), Phase 5 (configuration
and demo vertical slice), and Phase 7 (parallel semantics). These checkpoints
should approve public contracts before broad task migration.
