# Declarative Expert Programs and the Unified Semantic Skill Runtime

- Status: Expert Program registration and runtime catalogs are integrated; this
  stack layer adds physical-effect reconciliation, segment gates, held-object
  guards, bounded workflow reacquisition, and a dual-UR5 HandOver reference.
- Stack baseline: `feat/expert-program-registration-runtime-catalog`
  ([PR #504](https://github.com/DexForce/EmbodiChain/pull/504))
- Current branch: `feat/workflow-reacquisition`
  ([PR #480](https://github.com/DexForce/EmbodiChain/pull/480))
- Last updated: 2026-08-24
- Related issues: [#471](https://github.com/DexForce/EmbodiChain/issues/471),
  [#474](https://github.com/DexForce/EmbodiChain/issues/474)

## 1. Purpose

An Expert Program is a strict declarative frontend for the existing semantic
skill system. It is not a second action engine, scheduler, effect system, or
simulation loop.

```text
JSON / YAML
    |
    v
strict schema-v2 decoder
    |
    v
ExpertProgramCfg
    |
    v
ExpertProgramCompiler -----> SceneManifest
    |
    v
CompiledProgram
    |
    v
ExpertProgramEnvironmentAdapter
    |
    +---- static preflight ----> SemanticSkillCompiler
    |
    v
AtomicDemoBridge
    |
    v
SkillRuntime -> ExecutionRunner -> AtomicActionEngine
    |
    v
DemoSegment actions -> normal env.step()
```

Python semantic calls, Expert Programs, and future model-generated calls must
converge at `SemanticCallSpec` and use the same `SemanticSkillCompiler`,
`SkillRuntime`, effect monitors, and typed atomic-action core.

## 2. Goals

1. Let task authors express object-centric intent without task-local motion
   generators, controller routing, or EEF transform math.
2. Validate the complete serialized structure and all static integration
   requirements before any physical command is emitted.
3. Ground each semantic call from a fresh observation after prior verified
   effects.
4. Keep scene identity in `SceneRegistry`/`SceneManifest` and embodiment choices
   in `RobotSkillProfile`.
5. Route all demonstration actions through `env.step()` so managers, recording,
   rewards, and dataset boundaries remain authoritative.
6. Preserve row-local effect, recovery, eligibility, and result state while
   maintaining deterministic batch barriers.
7. Support fail-closed parallel execution only when resource, timing, symbolic
   state, and physical-safety checks all succeed.

## 3. Non-goals

- Replacing `AtomicActionEngine`, `ExecutionSession`, or `ExecutionRunner`.
- Serializing planners, callables, raw qpos, controller handles, or environment
  attribute paths.
- Adding another process-wide action registry or task-specific runtime wrapper.
- Treating resource disjointness as proof of collision-free parallel motion.
- Treating an accepted controller command as evidence that a physical effect
  occurred.
- Providing schema-v1 compatibility or maintaining pre-merge experimental APIs.
- Adding a dedicated drawer or `OperateArticulation` semantic primitive when
  the existing `Slide` primitive plus application verification is sufficient.
- Claiming task-level physical success from unit and fake-port coverage alone.

## 4. Ownership model

| Responsibility | Canonical owner |
|---|---|
| Serialized structure, bounds, discriminators | Expert Program decoder and config values |
| Static scene identity and aliases | `SceneManifest` projected from `SceneRegistry` |
| Semantic workflow analysis and lowering | `SemanticSkillCompiler` |
| Robot resources, endpoint capabilities, presets | `RobotSkillProfile` |
| Atomic planning and recovery | `AtomicActionEngine` / `ExecutionRunner` |
| Semantic call lifecycle and verified task state | `SkillRuntime` |
| Physical evidence and effect decisions | typed evidence providers and `EffectMonitor` |
| Gym action buffering and step handshake | `AtomicDemoBridge` and the demo executor |
| Settling between calls and task acceptance | segment post-policy and validator ports |
| Final Expert Program task success | completed bridge acceptance mask exposed by `EmbodiedEnv` |
| Dataset segment metadata | `DemoSegment` / `DemoSegmentResult` |

The layers are intentionally one-way. Gym frontend types must not be imported
into `embodichain.lab.sim.skills`, and the bridge must not reproduce runner
scheduling, recovery, verification, or safe-stop logic.

## 5. Expert Program schema

### 5.1 One supported top-level version

`EXPERT_PROGRAM_SCHEMA_VERSION` is `2`, and version 2 is the only accepted
top-level schema. There is no decoder branch, compatibility constant, or public
version-1 AST.

The registered semantic-call payload has its own independent revision:
`REGISTERED_SEMANTIC_CALL_SCHEMA_VERSION == 1`. This version describes an opaque
catalog call's arguments; it must not be coupled to the top-level program
schema.

### 5.2 Top-level shape

```text
ExpertProgramCfg
  schema_version: 2
  program_id
  integration
    robot_profile
    scene_registry
    runtime_preset
  targets
  program
```

Supported program nodes are:

```text
SequenceCfg | RepeatCfg | SegmentCfg | InvokeCfg | ParallelCfg
```

`BarrierCfg` is synchronization metadata owned only by `ParallelCfg`; it is not
a standalone `ProgramNodeCfg`. A parallel block:

- has at least two branches;
- forbids nested parallel blocks;
- permits `Invoke`, `Sequence`, and bounded `Repeat` inside branches;
- places post-policies and validators on one enclosing `SegmentCfg`;
- owns one named, positive-timeout, fail-fast barrier.

Supported built-in call configs are `PickCfg`, `PlaceCfg`, and `HandOverCfg`.
`RegisteredSemanticCallCfg` is the explicit catalog extension boundary.

Supported segment validators are:

- `ObjectNearTargetValidatorCfg`, which compares a bound rigid object's
  measured position with a resolved program target; and
- `ArticulationJointPositionValidatorCfg`, which compares one explicitly named
  articulation joint with an inclusive lower and/or upper position bound.

### 5.3 Safety and boundedness

The decoder and config constructors enforce:

- exact mappings/lists and discriminated unions;
- unknown-field and unknown-discriminator rejection;
- finite numeric values and valid UTF-8/JSON/YAML input;
- duplicate-key rejection;
- bounded input bytes, AST depth, node count, repeat count, and expanded calls;
- acyclic, executable-free registered-call arguments;
- rejection of imports, evaluation expressions, callables, modules, and dotted
  environment traversal;
- complete pathful diagnostics.

Structural decoding and provider-backed validation remain separate on purpose.
Programmatic config construction needs the same invariants as serialized input,
while an optional `ExpertProgramValidationContext` resolves external catalog,
scene, profile, policy, and validator IDs without touching live state.

### 5.4 Example

```yaml
schema_version: 2
program_id: repeated_cube_pick_place
integration:
  robot_profile: expert_program_ur5_pick_place
  scene_registry: expert_program_repeated_pick_place
  runtime_preset: safe
targets:
  drop_pose:
    kind: cyclic_pose
    values:
      - position: [-0.40, 0.48, 0.10]
        quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
      - position: [-0.42, -0.08, 0.10]
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
            resources: {primary: manipulator}
        - kind: invoke
          call:
            kind: place
            object: cube
            at: {kind: target_ref, target: drop_pose}
            resources: {primary: manipulator}
    post:
      - {kind: wait_stable, entity: cube, preset: rigid_object}
    validators:
      - kind: object_near_target
        object: cube
        target: drop_pose
        position_tolerance: 0.12
```

This is the packaged program at
`embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/expert/program.yaml`; its
integration IDs are matched exactly by the reference environment rather than
being inferred from simulator names.

## 6. Compilation

`ExpertProgramCompiler` accepts one canonical `SceneManifest`. The convenience
constructor projects that manifest from a `SceneRegistry` without retaining or
observing live providers.

Compilation:

1. resolves aliases and exact scene-reference types;
2. converts configuration calls to canonical `SemanticCallSpec` values;
3. expands bounded repeats and cyclic targets;
4. assigns contiguous call and program-segment indices;
5. preserves parallel branches and their barrier rather than flattening them;
6. returns one immutable, already materialized `CompiledProgram`.

There is no public scene-resolver protocol, registry-specific resolver wrapper,
lazy compiled-program subtype, or second `materialize()` step. The internal AST
templates exist only during the compiler call.

`CompiledProgram.preflight_analyses()` combines consecutive sequential segments
for downstream-goal look-ahead and splits analysis at each parallel barrier.
`sequential_execution_analysis()` executes only the selected segment prefix but
allows the later sequential suffix to influence static target propagation.

## 7. Runtime and environment integration

### 7.1 One semantic runtime

`SkillRuntime` is the only semantic runtime implementation. It owns:

- static workflow analysis;
- fresh observation and JIT grounding per call;
- one invocation/session/runner per semantic call;
- persistent verified `TaskState` and shrinking row eligibility;
- typed evidence collection and effect-monitor feedback;
- structured call, plan, recovery, scene, effect, and failure traces;
- synchronous `run()`, nonblocking `start()`/`step()`, lane `fork()`, and
  explicit cancellation.

`SkillRuntime.from_simulation()` is the standard simulation factory.
`SkillRuntime.from_components()` is the explicit hardware/custom-port path.
There is no `SemanticSkillRuntime` compatibility subclass.

### 7.2 Environment ownership

`EmbodiedEnv` directly owns the Expert Program delegation hooks:

- `expert_program_adapter`;
- `compile_expert_program()`;
- `create_expert_program_bridge()`.

An environment that enables `EmbodiedEnvCfg.expert_program` supplies one exact
`ExpertProgramEnvironmentAdapter`. A separate mixin would duplicate the base
environment's responsibility and is not part of the API.

The adapter validates integration IDs, compiles against a provider-free scene
manifest, recreates fresh live providers for execution, performs semantic and
parallel preflight, and constructs one `AtomicDemoBridge`.

For an enabled Expert Program, `EmbodiedEnv.is_task_success()` is false until
the bridge's segment iterator finishes normally. Normal completion publishes
the bridge's final row-local eligibility mask after runtime results,
post-policies, and every segment validator have been combined. Reset first lets
`BaseEnv` consume that result for dataset saving and then clears the completed
bridge, so a stale program result cannot leak into the next episode.

### 7.3 Timing

The Gym control cadence belongs to the live `PlanningContext.control_dt`, which
the simulation observation provider sets from `BaseEnv.step_dt`. Motion presets
describe planning behavior; the factory does not clone and rewrite every
`MotionPolicy` merely to inject cadence.

Every runtime frame must map exactly to the Gym step grid. Off-grid durations
fail instead of being silently resampled.

### 7.4 Physical effects

Accepted command state is not a physical sensor. The production path does not
maintain an accepted-command evidence tracker or a command-observer side
channel. Simulation integrations must supply real contact, constraint, force,
wrench, articulation, and pose observations as applicable. Missing channels
remain invalid and fail closed.

The standard registration may own a `ControlPartEvidenceProviderFactory` for
the exact built-in control-part evidence route. Its immutable declaration is
fingerprinted before simulation startup, and every runtime assembly receives a
fresh live provider bound to the exact robot, scene registry, engine, and shared
scene provider. The repeated-cube integration uses this extension with a
`ContactSensor`: a grasp constraint is true only when the cube has valid contact
with both configured gripper finger links in the same environment row.

The following boundaries remain distinct:

| Boundary | Meaning |
|---|---|
| Atomic/semantic effect monitor | Decides whether one physical call succeeded and participates in recovery |
| Program-segment post-policy | Advances environment behavior such as settling after motion completion |
| Program-segment validator | Decides whether the dataset/task segment is acceptable |

Combining these would either duplicate atomic recovery or let a dataset metric
fabricate verified task state.

### 7.5 Physical gates, reconciliation, and workflow reacquisition

`EffectVerificationResult` carries row-local `invalidation_mask` and
`retry_mask` values, both constrained to the terminal `failure_mask`.
Invalidation applies only an invocation-owned, removal-only
`failure_invalidation`; effect providers cannot inject replacement symbolic
state. Retry is selected only when per-expectation evidence proves that replay
of the same invocation remains physically valid. Unresolved terminal evidence
is reconciled fail-closed at the action deadline.

Curated semantic calls also install two kinds of in-flight physical boundary:

- `HeldObjectGuardRequest` observes a held relation before commands in named
  segments. Proven loss removes only action-authorized held-object relations
  before retry or `RECOVERY_REQUIRED`.
- `PhaseEffectGateRequest` blocks entry to a named segment until the preceding
  physical transition succeeds. While evidence is unresolved, execution
  replays the preceding command for the synchronized active cohort; success
  unlocks motion without committing terminal `TaskState`.

Pick gates destination attachment before `lift`, and Place gates source release
before `retract`. The unified HandOver primitive owns pickup, transfer,
placement, and final release. It gates source pickup before
`pickup_transport`, destination pickup before `handover_release`, and source
release before `place`; source- and destination-held guards cover the segments
that physically depend on those relations. Every gate and guard owns a fresh
monitor instance and correlated request ID. None may create a simulator joint,
managed attachment, frozen body, or pose override.

`SkillPolicyPreset` schema version 3 adds `WorkflowRecoveryPolicy`, whose
per-row attempt budget defaults to zero. After terminal reconciliation, a row
that still has the required verified source relation retries the failed call
from a fresh observation. If the relation was invalidated, the runtime executes
a real semantic Pick on the failed call's resolved source resource and then
retries the original call. Recovery calls use ordinary analysis, grounding,
planning, dispatch, physical verification, and trace recording; they are not
symbolic edits or a second workflow executor. Successful peer rows remain at
the existing shared call barrier.

Timing remains owned by `PlanningContext.control_dt`. The factory does not
rewrite `MotionPolicy` cadence. A joint-position task may disable runner holds
during effect polling or after successful completion so the bridge can replay
the last accepted environment action and preserve gripper preload. Failure and
cancellation retain the normal cancel-then-safe-hold path.

### 7.6 Demo execution

`AtomicDemoBridge` never calls `env.step()` itself. It yields owned
`ControllerAction` values through lazy `DemoSegment` iterables. The shared
demo executor performs the environment step, advances the bridge clock only
after consumption, and records completion metadata.

The buffered sink owns only unconsumed Gym actions and safe-stop handshakes. It
does not publish evidence or maintain a redundant wrapper around each buffered
action.

### 7.7 Official reference task integrations

The current branch keeps the canonical task examples under
`embodichain_tasks.manipulation`. The repeated cube task has one Gym ID and
implementation; its former compatibility package and ID are removed. Open
Drawer likewise has one canonical integration, `ExpertProgramOpenDrawer-v1`,
and the dual-UR5 reference adds `HandOver-v1`.

| Environment ID | Declarative path | Atomic path | Application acceptance |
|---|---|---|---|
| `ExpertProgramRepeatedPickPlace-v1` | schema-v2 `Repeat(Segment(Sequence(Pick, Place)))` with a cyclic pose target | built-in `PickUp` and `Place` through the semantic compiler; a registration-owned provider derives constraint evidence from cube contact with both physical finger links | standard `object_near_target` validator checks the measured cube position against the selected cyclic target; seed-0 three-cycle and physical-loss/reacquisition slow gates pass |
| `ExpertProgramOpenDrawer-v1` | registered `embodichain_tasks.open_drawer` call with a strict executable-free payload | a task-owned `RegisteredSemanticLowerer` produces the built-in `SlideGoal` and `SlideOptions` for the live drawer-handle link | standard `articulation_joint_position` validator checks the measured passive drawer joint against the configured threshold |
| `HandOver-v1` | one `Segment(HandOver)` with a final object target | unified built-in `HandOver` over disjoint left/right arm-and-gripper resources, standalone grasp-pose generators, and physical gates/guards | rigid-object settling plus `object_near_target`; supported-simulation recovery qualification remains a later stack layer |

Both configurations load their Expert Program through the top-level
`expert_program_path`, bind the same UR5 parallel-gripper embodiment explicitly,
and consume commands through the normal `env.step()` demo path. The Open Drawer
example deliberately does not add an atomic articulation effect to `Slide`:
`Slide` completion means motion completion, while drawer opening remains an
application-level observation. Both examples use shared built-in settling
presets (`rigid_object` and `articulation` respectively), and neither task class
reimplements `is_task_success()`.

A fully generic `DeclarativeExpertProgramEnv` is intentionally deferred. The
two task classes still own concrete scene assets, sensors/evidence providers,
robot-profile assembly, and Open Drawer's registered `Slide` lowerer. This step
standardizes only lifecycle and policy behavior that is independent of those
composition choices.

## 8. Parallel execution

`ParallelCfg` lowers to one `CompiledParallelBlock`. At execution, the bridge
creates a `ParallelSkillRuntime` whose lanes are forks of the same canonical
`SkillRuntime`.

The parallel boundary requires:

1. statically disjoint resource claims;
2. disjoint runtime destinations;
3. exact shared-clock step alignment;
4. deterministic hold padding for shorter lanes;
5. conflict-free symbolic state writes;
6. a mandatory `ParallelCommandSafetyValidator` for every merged command;
7. bounded fail-fast timeout, cancellation, and safe stop.

Resource disjointness proves controller ownership only. Missing or inconclusive
physical-safety evidence prevents the block from starting or dispatching.
Nested parallel blocks and alternate failure policies are intentionally absent.

## 9. Simulation declarations

`SimulationSceneBinding` builds the authoritative registry from explicit rigid
object, articulation, link, and antipodal-grasp declarations. It does not scan
arbitrary environment attributes.

`SimulationRobotSkillProfileBinding` accepts:

- `ControlPartResourceBinding` for robot-control-part-backed resources; and
- the core `RobotResource` directly for generic mobile, whole-body, or custom
  endpoints.

There is no duplicate generic `RobotResourceBinding` or protocol hierarchy.
Custom endpoint kinds extend the core `ResourceEndpoint`/adapter contract and
provide a matching Gym runtime transport.

Grasp-pose generation is a planning service, not scene data and not a
`MotionGenerator` feature. The shared hierarchy is:

```text
GraspPoseGenerator
└── ParallelJawGraspPoseGenerator
    └── AntipodalGraspPoseGenerator
```

`ParallelJawGripperModelCfg` owns physical two-finger geometry;
`AntipodalGraspPoseGeneratorCfg` owns sampling/ranking behavior;
`ParallelJawGraspCollisionCfg` owns collision policy; and
`GraspAnnotationCfg` owns region-selection/cache refresh. A concrete product
name such as `dh_pgi_140_80` appears only as a gripper-model `model_id`, never
in a public class name.

The service accepts target-local mesh tensors per call, so a handwritten
environment can call it directly before using `MotionGenerator`. Atomic-action
and Expert Program paths install the same instance on
`ActionPlanningServices`, keyed by the grasp endpoint's runtime `target_id`.
`AntipodalAffordance` and `AntipodalGraspAffordanceBinding` therefore retain
only target geometry. The reference Gym configurations no longer carry grasp
sampling or annotation settings in `env.extensions`. A handwritten Gym task
may instead accept a generator through its constructor. The repeated cube
reference installs the shared generator through the production Expert Program
adapter and has no second handwritten planning path.

Articulation/link registry data and joint evidence remain reusable. Drawer-like
tasks should lower semantic intent through `Slide` and verify the observed
articulation result through a segment validator at the application boundary.
The removed
`OperateArticulation` experiment duplicated an existing motion primitive and
incorrectly bundled task completion into that primitive.

## 10. Consolidation decisions

The current branch intentionally makes breaking changes because none of these
pre-merge experimental surfaces requires compatibility.

| Removed redundancy | Canonical replacement |
|---|---|
| Top-level schema versions 1 and 2 | schema version 2 only |
| Standalone `Barrier` program node | `ParallelCfg.barrier` value owned by its parallel block |
| Dedicated `OperateArticulation` config/call/binding/test path | `Slide` plus application-level articulation verification |
| `ExpertProgramSceneResolver` and `SceneRegistryProgramResolver` | core `SceneManifest` |
| Lazy `CompiledProgram` plus `MaterializedCompiledProgram` | one bounded `CompiledProgram` returned directly by `compile()` |
| Temporary internal compiled-program object immediately materialized | direct internal expansion function |
| `ExpertProgramEnvironmentMixin` | delegation methods on `EmbodiedEnv` |
| `SemanticSkillRuntime` compatibility subclass | canonical `SkillRuntime`, including `from_simulation()` |
| Accepted-command tracker/observer as grasp evidence | explicit typed physical evidence providers |
| Generic simulation resource-binding protocols and wrapper | core `RobotResource`; retain only control-part convenience binding |
| Factory-wide cloning of motion presets to inject `control_dt` | `PlanningContext.control_dt` from the environment clock |
| Duplicate JSON-safe metadata copier | one private Gym JSON ownership helper |
| Single-field buffered-action wrapper | `deque[ControllerAction]` directly |
| Unused observation/provider properties and settling predicate | direct owning APIs and shared `DynamicSettleMonitor` |
| Optional post-policy/result/metadata protocol variants | one complete `SegmentPostPolicyPort` and one complete `SegmentValidatorPort` |
| Task-local articulation settling configuration | shared built-in `articulation` settling preset |
| Task-local success thresholds, joint indices, and `is_task_success()` methods | declarative segment validators plus completed bridge acceptance on `EmbodiedEnv` |
| Repeated public exports from every implementation submodule | curated `embodichain.lab.gym.envs.expert_program` entry point |
| Grasp generator/config/runtime state on scene affordances | endpoint-owned `GraspPoseGenerator` service plus target-local affordance mesh |
| Grasp sampling and annotation flags in Gym `extensions` | typed generator/model/collision/annotation configuration at planning-service assembly |

The following similar-looking layers are retained because they have different
trust or lifecycle ownership:

- config `__post_init__` validation versus untrusted JSON/YAML decoding;
- provider-free `SceneManifest` versus live `SceneRegistry` providers;
- Expert Program compilation versus semantic call analysis/grounding;
- atomic effect monitor versus segment post-policy versus segment validator;
- `SkillRuntime` scheduling versus Gym/demo step adaptation;
- control-part convenience bindings versus generic core resources.

## 11. Public API direction

The canonical import path is:

```python
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramCfg,
    ExpertProgramCompiler,
    ExpertProgramEnvironmentAdapter,
    SimulationSceneBinding,
    SimulationRobotSkillProfileBinding,
    decode_expert_program,
    load_expert_program,
)
```

Implementation submodules keep explicit internal imports but no longer repeat
the same declarations as independent public API contracts. Compiled branch,
call, target-selection, clock, and provider records remain implementation
details unless a later use case proves that they need a stable public contract.

## 12. Current implementation status

Implemented on the current branch:

- strict schema-v2 config, JSON/YAML loader, decoder, and optional static
  validation context;
- bounded sequential, repeat, segment, invoke, parallel, and barrier handling;
- provider-free scene-manifest compilation to one materialized program;
- cross-segment look-ahead and branch-aware semantic preflight;
- direct `EmbodiedEnv` adapter integration and CLI/config loading;
- lazy `env.step()` demo bridge, completion metadata, abort/safe-hold handshake;
- completed-bridge acceptance exposed as the standard Expert Program task
  success state on `EmbodiedEnv`;
- shared `rigid_object` and `articulation` dynamic-settling presets;
- shared object-near-target and articulation-joint-position validation;
- simulation scene/profile/evidence factories;
- canonical `SkillRuntime` integration and fail-closed parallel coordinator;
- per-expectation terminal outcomes, removal-only failure reconciliation,
  segment-scoped held-object guards, blocking physical-effect gates, and
  bounded workflow retry/reacquisition;
- official `ExpertProgramRepeatedPickPlace-v1` integration with a packaged
  three-cycle program, cyclic targets, rigid-object settling, segment
  validation, a registration-owned dual-finger contact-evidence provider, and
  supported-simulation three-cycle plus physical-loss/reacquisition gates;
- official `ExpertProgramOpenDrawer-v1` integration with a strict registered
  call lowered to `Slide`, shared articulation settling, and declarative
  measured passive-joint application acceptance;
- official `HandOver-v1` dual-UR5/PGI integration using the unified HandOver
  primitive, standalone grasp-planning services, settling, and declarative
  target validation;
- focused unit and fake-port coverage for schema, compiler, bridge, environment,
  evidence, timing, recovery metadata, parallel failure cases, and both task
  reference integrations;
- one-episode Viser simulator smoke coverage for Open Drawer, reaching its
  declarative acceptance boundary and committing the episode.

The registration follow-up makes
`SimulationExpertProgramRegistration` the sole extension owner for the
standard runtime path. `SkillPolicyPreset` schema version 3 requires an exact
typed action-option template for every reachable semantic call; lowering may
replace only explicitly compiler-owned dynamic target fields. Endpoint
adapters, ordered Gym transports, a parallel-safety factory, and an optional
control-part evidence factory enter the provider-free registration fingerprint
and are checked again against live endpoint resolution. Runtime assembly
consumes those same registered objects, freezes the command encoder, takes
runner policy from the selected preset, and creates fresh safety/evidence
providers for every runtime. Helper arguments cannot replace registered
components after preflight. Stateful extension declarations must be frozen
dataclasses with recursively immutable configuration so nested mutable values
cannot become a post-registration runtime side channel.

This slice covers command transport, not arbitrary closed-loop backend
injection. Custom endpoint adapters on the standard path must declare empty
tracking and effect-evidence routes, and therefore support only timed/open-loop
completion. The built-in `ControlPartEndpoint` retains its built-in routes.
Non-joint feedback providers, desired-state projectors, metric evaluators, and
non-control-part effect-evidence backends require separate registration-owned
provider-factory contracts before they can be advertised as standard mobile or
whole-body closed-loop support. Each transport owns a trusted `hold()`
primitive; the parallel safety validator authorizes active merged command
frames before dispatch.

Still required before broader task-level qualification:

- repeat the repeated-cube physical gates across controlled seeds and the
  intended randomization envelope;
- repeat the Open Drawer simulator qualification across controlled seeds and
  the intended randomization envelope, then inspect persisted metadata;
- an environment-qualified parallel physical-safety validator before migrating
  PourWater or any other concurrent task;
- separate frontend and task-migration work for model-produced programs;
- measured capability parity before any Action Bank removal decision.

`DeclarativeExpertProgramEnv` construction is also outside this step. Before it
can replace these task composition roots, scene assets, sensors/evidence,
robot-profile selection, affordance extraction, and registered lowerers need
their own complete declarative declarations and factories.

## 13. Validation surface

Focused validation for changes in this design should include:

- `tests/gym/envs/expert_program/`;
- `tests/gym/envs/test_demo.py` and
  `tests/gym/envs/test_embodied_env_expert_program.py`;
- `tests/gym/envs/test_settling.py`;
- `tests/gym/envs/expert_program/test_task_vertical_slices.py` and
  `tests/gym/envs/expert_program/test_task_hand_over.py`;
- `tests/sim/skills/test_runtime.py` and parallel-runtime tests;
- semantic tutorial tests;
- CLI/config-path tests;
- Black, API documentation coverage, and a Sphinx dummy build.

Acceptance requires more than passing fake-port tests: a real environment must
demonstrate command consumption through `env.step()`, live effect evidence,
settling, validation, row-local outcomes, safe cancellation, and deterministic
metadata.

The retained supported-simulation smoke surface is:

```bash
embodichain run-env \
  --gym_config embodichain_tasks/configs/tasks/manipulation/open_drawer/env.json \
  --viser --max_episodes 1
```

The deterministic capability matrix, migration-size snapshot, and demo-success
artifact contract are maintained in
[`expert_program_rollout_report.md`](expert_program_rollout_report.md).

The Open Drawer command completed with exit status 0 and committed one episode.
Repeated cube completes a seed-0 three-cycle physical run, and a separate
bounded gripper-open fault proves held-object loss, real semantic
reacquisition, retried placement, and final segment acceptance.
