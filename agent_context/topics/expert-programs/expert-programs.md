# Expert Programs

## Scope and public entry point

Expert Program is the task-level action-generation entry point for declarative
semantic sequences. The provider-independent public package is:

```text
embodichain.lab.expert_program
```

It exports schema values, strict loading/decoding/validation, the
`ExpertProgramCompiler`, and `CompiledProgram`.

Gym and simulation adapters live in:

```text
embodichain.lab.gym.envs.expert_program
```

That package exports environment assembly, registration, bridge, transport,
and simulation binding APIs. It does not re-export the core schema or compiler.

`embodichain.lab.semantic_skills` supplies declarative contracts only. It
is not another execution entry point.

## Resolution path

```text
program.yaml / program.json / ExpertProgramCfg
  -> load_expert_program / decode_expert_program
  -> exact validation and bounded expansion checks
  -> ExpertProgramCompiler(SceneManifest)
  -> immutable CompiledProgram
  -> ExpertProgramEnvironmentAdapter provider-aware preflight
  -> live semantic grounding and AtomicAction invocation
  -> AtomicDemoBridge
  -> lazy DemoSegment actions
  -> normal env.step() consumption
```

The provider-independent compiler observes no simulation state and generates no
controller actions. Live grounding occurs only after the environment adapter
has matched the compiled scene/profile/catalog snapshot to its immutable
registration.

## Core package map

| Concern | Source of truth |
|---|---|
| Public schema | `embodichain/lab/expert_program/cfg.py` |
| Strict decoding and validation | `embodichain/lab/expert_program/decoder.py` |
| JSON/YAML file loading and size bounds | `embodichain/lab/expert_program/loader.py` |
| Provider-independent compilation | `embodichain/lab/expert_program/compiler.py` |
| Public exports | `embodichain/lab/expert_program/__init__.py` |
| Internal semantic lowering | `_semantic_compiler.py` |
| Internal sequential execution | `_semantic_executor.py`, `_semantic_results.py` |
| Internal parallel scheduling/execution | `_parallel.py`, `_parallel_executor.py` |
| Internal component assembly | `_semantic_assembly.py` |

Underscore-prefixed modules are private implementation details and have empty
`__all__`. Do not document or import them as application APIs.

## Gym and simulation package map

| Concern | Source of truth |
|---|---|
| Environment adapter and preflight | `environment.py` |
| Lazy Gym action/segment lifecycle | `bridge.py` |
| Immutable task integration and fingerprint | `catalog.py`, `extensions.py` |
| Provider-free simulation declarations | `simulation.py` |
| Live simulation assembly | `simulation_environment.py` |
| Segment settling/validation ports | `simulation_policies.py` |
| Callable-free Gym runtime decode | `_configured_runtime_decoder.py` |
| Allowlisted configured services | `_configured_runtime_services.py` |
| Dynamic Gym ID registration | `configured_runtime.py` |
| Environment binding and episode selection | `../embodied_env.py`, `../../utils/registration.py` |

The configured runtime decoder is intentionally separate from Gym registration.
Do not grow `configured_runtime.py` into a second schema/assembly module.

## Schema contract

Top-level `ExpertProgramCfg` owns:

- `program_id`;
- exact `ExpertProgramIntegrationCfg` IDs;
- named targets; and
- one program node.

Supported nodes:

- `SequenceCfg`
- `RepeatCfg`
- `SegmentCfg`
- `InvokeCfg`
- `ParallelCfg` with owned `BarrierCfg`

Built-in calls:

- `PickCfg`
- `PlaceCfg`
- `HandOverCfg`
- `RegisteredSemanticCallCfg` for allowlisted extensions

External inputs have no `schema_version`. Unknown fields, duplicate keys,
invalid exact types, non-finite values, excessive depth/nodes/repeats, cyclic or
executable registered payloads, and unresolved references fail early.

## Compilation contract

`ExpertProgramCompiler` resolves canonical scene references, bounded repeats,
cyclic targets, stable segment/call indices, sequential look-ahead, segment
post-policies/validators, and parallel structure.

It returns a fully materialized immutable `CompiledProgram`. It does not own a
live engine or observe providers.

MLLM input uses the same strict decoder through
`embodichain.agents.mllm.expert_program`. The host still owns integration IDs
and executable extension registration.

## Simulation registration

`SimulationExpertProgramRegistration` is the standard composition root. Its
fingerprint covers:

- scene and robot-profile declarations;
- semantic call catalog;
- settling presets;
- relation grounders and hand-over pose providers;
- endpoint adapters and runtime transports;
- evidence and parallel-safety factories; and
- registered semantic-lowerer factories.

Adapter creation calls `assert_unchanged()` and revalidates live scene/profile,
engine catalog, endpoint routes, and extension declarations. Do not add
task-side live overrides around a standard registration.

`config_to_cfg()` may decode `expert_program_runtime`, create this registration,
and register the common `EmbodiedEnv` under the JSON-owned ID. Reloading an
identical declaration is idempotent; ID or fingerprint drift fails.

## Explicit effect assurance

Each `SkillPolicyPreset` requires `effect_assurance`:

- `verified`: state advances from measured evidence. Curated Pick, Place, and
  HandOver calls require explicit monitor mappings.
- `projected`: monitors are forbidden and the expected plan effect is projected
  after command completion.

There is no implicit built-in monitor installation. Missing assurance is a
decode error. A verified curated call with no monitor fails static analysis as
`missing_effect_monitor`.

Projected execution is a trajectory demonstration and must not be described as
physical task success.

## Runtime and acceptance

`EmbodiedEnv` owns adapter binding and action generation. It accepts its static
`cfg.expert_program`, an episode-local `ExpertProgramCfg`, or a trusted
`CompiledProgram`.

`AtomicDemoBridge` never calls `env.step()`. It yields lazy `DemoSegment`
actions so the shared demo executor remains the owner of stepping, recording,
rewards, reset, and persistence.

Acceptance boundaries are separate:

1. semantic effect monitor: one call's physical postcondition and recovery;
2. segment post-policy: environment advancement such as settling; and
3. segment validator: task/dataset acceptance.

Final success is published only after all lazy segment lifecycles finish
normally.

Held-object guards and phase-effect gates are observational and may only remove
action-authorized invalid symbolic relations. They never create attachments,
freeze objects, or override poses.

## Parallel blocks

Parallel execution requires:

- disjoint resource claims and runtime targets;
- shared control-grid timing;
- conflict-free symbolic writes;
- an authoritative `ParallelCommandSafetyValidator`; and
- explicit fail-fast barrier semantics.

Missing or inconclusive physical-safety evidence fails closed. Resource
disjointness alone is insufficient. Nested parallel blocks are rejected.

## Registered semantic calls

Registered call payloads are declarative and executable-free. A standard
registration must provide exactly one matching factory, fingerprint the call
ID/revision/target descriptor, and create a fresh lowerer for every live
assembly.

Use this extension to expose a reusable Atomic Action through Expert Program.
Do not implement task-local motion generators or a duplicate runtime in a
lowerer.

## Reference integrations

| Environment | Assurance | Source |
|---|---|---|
| `ExpertProgramRepeatedPickPlace-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/` |
| `ExpertProgramOpenDrawer-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/open_drawer/` |
| `HandOver-v1` | verified | `embodichain_tasks/configs/tasks/manipulation/hand_over/` |
| `PourWater-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/` |

All four use configuration-owned `expert_program_runtime` declarations and the
common `EmbodiedEnv`; none requires a task-local environment subclass.

## Recommended change sites

| Change | Owning location |
|---|---|
| Schema shape or bounds | `embodichain/lab/expert_program/cfg.py` |
| Untrusted input behavior | `decoder.py`, `loader.py` |
| Static expansion / compiled structure | `compiler.py` |
| Scene/profile/runtime config format | `_configured_runtime_decoder.py` |
| Allowlisted configured service implementation | `_configured_runtime_services.py` |
| Registration fingerprint or extensions | `catalog.py`, `extensions.py` |
| Live simulation assembly | `simulation_environment.py` |
| Gym action and acceptance lifecycle | `bridge.py` |
| Environment program selection | `embodied_env.py` |
| Semantic call/profile/effect declarations | `embodichain/lab/semantic_skills/` |
| Atomic motion behavior | `embodichain/lab/sim/atomic_actions/` |

## Focused validation

```bash
pytest -q tests/lab/expert_program
pytest -q tests/gym/envs/expert_program
pytest -q tests/gym/envs/test_embodied_env_expert_program.py
pytest -q tests/agents/mllm/test_expert_program.py
pytest -q tests/lab/semantic_skills tests/sim/atomic_actions
python docs/scripts/check_api_docs.py
```

Public API changes also require `tests/docs/test_check_api_docs.py` and a Sphinx
dummy build. Physical acceptance requires real environment runs with measured
evidence, multiple seeds/randomization, segment validators, and persisted
completion metadata.
