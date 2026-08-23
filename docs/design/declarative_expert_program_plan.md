# Declarative Expert Programs and the Unified Semantic Skill Runtime

- Status: Expert Program runtime implemented on the current branch; integration
  and physical task qualification remain in progress
- Main baseline: `origin/main@df06818a` (semantic runtime and parallel foundation,
  merged through [PR #496](https://github.com/DexForce/EmbodiChain/pull/496))
- Current branch: `feat/declarative-expert-program-runtime`
  ([PR #497](https://github.com/DexForce/EmbodiChain/pull/497))
- Last updated: 2026-08-22
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
  robot_profile: ur5_parallel_gripper_v1
  scene_registry: multi_segments_cube_v1
  runtime_preset: safe
targets:
  drop_pose:
    kind: cyclic_pose
    values:
      - position: [-0.40, 0.48, 0.10]
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
          call: {kind: pick, object: cube}
        - kind: invoke
          call:
            kind: place
            object: cube
            at: {kind: target_ref, target: drop_pose}
    post:
      - {kind: wait_stable, entity: cube, preset: rigid_object}
    validators:
      - kind: object_near_target
        object: cube
        target: drop_pose
        position_tolerance: 0.12
```

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

The following boundaries remain distinct:

| Boundary | Meaning |
|---|---|
| Atomic/semantic effect monitor | Decides whether one physical call succeeded and participates in recovery |
| Program-segment post-policy | Advances environment behavior such as settling after motion completion |
| Program-segment validator | Decides whether the dataset/task segment is acceptable |

Combining these would either duplicate atomic recovery or let a dataset metric
fabricate verified task state.

### 7.5 Demo execution

`AtomicDemoBridge` never calls `env.step()` itself. It yields owned
`ProcessedEnvAction` values through lazy `DemoSegment` iterables. The shared
demo executor performs the environment step, advances the bridge clock only
after consumption, and records completion metadata.

The buffered sink owns only unconsumed Gym actions and safe-stop handshakes. It
does not publish evidence or maintain a redundant wrapper around each buffered
action.

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

Articulation/link registry data and joint evidence remain reusable. Drawer-like
tasks should lower semantic intent through `Slide` and verify the observed
articulation result at the application boundary. The removed
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
| Single-field buffered-action wrapper | `deque[ProcessedEnvAction]` directly |
| Unused observation/provider properties and settling predicate | direct owning APIs and shared `DynamicSettleMonitor` |
| Optional post-policy/result/metadata protocol variants | one complete `SegmentPostPolicyPort` and one complete `SegmentValidatorPort` |
| Repeated public exports from every implementation submodule | curated `embodichain.lab.gym.envs.expert_program` entry point |

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
- shared dynamic settling and object-near-target validation;
- simulation scene/profile/evidence factories;
- canonical `SkillRuntime` integration and fail-closed parallel coordinator;
- focused unit and fake-port coverage for schema, compiler, bridge, environment,
  evidence, timing, recovery metadata, and parallel failure cases.

Still required before claiming task-level completion:

- one supported simulation integration with authoritative physical evidence for
  a complete repeated pick/place program;
- full three-cycle repeated-cube success and dataset metadata inspection;
- reusable semantic `Slide` lowering plus Open Drawer application verification;
- an environment-qualified parallel physical-safety validator before migrating
  PourWater or any other concurrent task;
- separate frontend and task-migration work for model-produced programs;
- measured capability parity before any Action Bank removal decision.

## 13. Validation surface

Focused validation for changes in this design should include:

- `tests/gym/envs/expert_program/`;
- `tests/gym/envs/test_demo.py` and
  `tests/gym/envs/test_embodied_env_expert_program.py`;
- `tests/gym/envs/test_settling.py`;
- `tests/sim/skills/test_runtime.py` and parallel-runtime tests;
- semantic tutorial tests;
- CLI/config-path tests;
- Black, API documentation coverage, and a Sphinx dummy build.

Acceptance requires more than passing fake-port tests: a real environment must
demonstrate command consumption through `env.step()`, live effect evidence,
settling, validation, row-local outcomes, safe cancellation, and deterministic
metadata.
