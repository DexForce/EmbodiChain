# Embodied Task Program

## Scope and terminology

Task Program is EmbodiChain's declarative orchestration and execution DSL for
embodied tasks. Its provider-independent entry point is:

```text
embodichain.lab.task_program
```

Use these terms consistently:

- **Atomic Skill**: an executable low-level capability implemented under
  `embodichain.lab.sim.atomic_actions`.
- **Semantic Call**: a high-level Task Program instruction such as `Pick`,
  `Place`, `HandOver`, or an allowlisted registered call.
- **Task Program**: a bounded declarative program that orchestrates Semantic
  Calls and lowers them to Atomic Skills.

Semantic Calls and their scene/profile/effect contracts live inside
`task_program.semantics`. They are not an independent action entry point.

## Ownership and package map

| Concern | Source of truth |
|---|---|
| Stable public language API | `embodichain/lab/task_program/__init__.py` |
| Schema and AST values | `embodichain/lab/task_program/language/schema.py` |
| Strict decoding and reference validation | `embodichain/lab/task_program/language/decoder.py` |
| JSON/YAML loading and input bounds | `embodichain/lab/task_program/language/loader.py` |
| Compiled program model and AST expansion | `embodichain/lab/task_program/compiler/program.py` |
| Semantic Call analysis and lowering | `embodichain/lab/task_program/compiler/lowering.py` |
| Semantic Call execution | `embodichain/lab/task_program/runtime/executor.py`, `results.py` |
| Parallel scheduling and safety boundary | `embodichain/lab/task_program/runtime/parallel.py`, `parallel_executor.py` |
| Semantic Calls, scene, profile, effects, evidence | `embodichain/lab/task_program/semantics/` |
| Immutable catalog and extension declarations | `embodichain/lab/task_program/integrations/catalog.py`, `extensions.py` |
| Environment adapter and runtime assembly | `embodichain/lab/task_program/integrations/environment.py` |
| Physical Gym component composition | `embodichain/lab/gym/utils/_component_composition.py` |
| Configured semantic/policy composition | `embodichain/lab/task_program/integrations/_configured_composition.py` |
| Callable-free configured runtime decode | `embodichain/lab/task_program/integrations/configured.py` |
| Simulation bindings and live assembly | `embodichain/lab/task_program/integrations/simulation/` |
| Gym `DemoSegment` / `env.step()` bridge | `embodichain/lab/gym/envs/task_program/bridge.py` |
| Dynamic Gym ID registration | `embodichain/lab/gym/envs/task_program/registration.py` |
| Episode selection, recording, final success | `embodichain/lab/gym/envs/embodied_env.py`, `demo.py` |
| MLLM untrusted-JSON frontend | `embodichain/agents/mllm/task_program.py` |

The core `language`, `semantics`, `compiler`, and `runtime` packages must not
import Gym. Provider-specific dependencies are confined to `integrations`.
Gym retains only lifecycle coupling; it does not own schemas, compilers,
catalogs, lowerers, or simulation assembly.

## Resolution path

```text
Human / MLLM / Agent
  -> JSON, YAML, or TaskProgramCfg
  -> load / decode / validate
  -> TaskProgramCompiler
  -> immutable CompiledTaskProgram
  -> Semantic Call lowering
  -> Atomic Skills
  -> TaskProgramDemoBridge
  -> DemoSegment
  -> env.step / recording / acceptance
```

Compilation is provider-independent: it observes no live simulator state and
generates no controller action. Live grounding occurs only after
`TaskProgramEnvironmentAdapter` matches the compiled scene/profile/catalog
snapshot to the exact integration registration.

## Read on demand

- [Language and semantic integration](configuration.md): strict decoding, scene/profile contracts, component ownership and service allowlists.
- [Assurance and execution](execution.md): measured/projected effects, acceptance, parallel calls and MLLM boundaries.
- [Dataset persistence](../data-pipeline/data-pipeline.md): recorder commits, fragments, failures and finalization.

## Reference integrations

| Environment | Assurance | Source |
|---|---|---|
| `TaskProgramRepeatedPickPlace-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/` |
| `TaskProgramRepeatedPickPlace-Newton-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.newton.yaml` |
| `TaskProgramRepeatedPickPlace-Franka-Newton-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.newton.yaml` |
| `TaskProgramOpenDrawer-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/open_drawer/` |
| `TaskProgramOpenDrawer-Newton-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/open_drawer/task.ur5.newton.yaml` |
| `HandOver-v1` | verified | `embodichain_tasks/configs/tasks/manipulation/hand_over/` |
| `PourWater-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/` |

## Recommended change sites

| Requested change | Start here |
|---|---|
| Language shape or limits | `language/schema.py`, then decoder/loader tests |
| Untrusted input behavior | `language/decoder.py`, `language/loader.py` |
| AST expansion or compiled structure | `compiler/program.py` |
| Call-to-Atomic-Skill lowering | `compiler/lowering.py` |
| Semantic Call, scene, robot, effect contracts | `semantics/` |
| Runtime sequencing or parallel behavior | `runtime/` |
| Registration fingerprint/extensions | `integrations/catalog.py`, `extensions.py` |
| Configured integration format | `integrations/configured.py` |
| Live simulation binding | `integrations/simulation/` |
| Gym action/segment lifecycle | `gym/envs/task_program/bridge.py` |
| Episode program selection/final success | `gym/envs/embodied_env.py` |
| Outcome annotations and persistence mode | `gym/envs/demo.py`, `gym/envs/embodied_env.py` |
| LeRobot fragment slicing/idempotency | `gym/envs/managers/datasets.py`, `async_datasets.py` |

## Focused validation

```bash
pytest -q tests/lab/task_program
pytest -q tests/gym/envs/task_program
pytest -q tests/gym/envs/test_embodied_env_task_program.py
pytest -q tests/gym/envs/test_demo.py tests/gym/envs/managers
pytest -q tests/agents/mllm/test_task_program.py
pytest -q tests/sim/atomic_actions
python docs/scripts/check_api_docs.py
```

Public API changes also require the docs checker tests and a Sphinx dummy
build. Physical qualification requires real environment runs with measured
evidence, controlled seeds/randomization, explicit validators, and persisted
completion metadata.
