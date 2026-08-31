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

## Language contract

`TaskProgramCfg` owns:

- one `program_id`;
- exact `TaskProgramIntegrationCfg` IDs;
- optional named targets; and
- one bounded program tree.

Supported nodes are `SequenceCfg`, `RepeatCfg`, `SegmentCfg`, `InvokeCfg`, and
`ParallelCfg` with an owned `BarrierCfg`. Built-in call configs are `PickCfg`,
`PlaceCfg`, and `HandOverCfg`; `RegisteredSemanticCallCfg` is the allowlisted
extension form.

Unknown fields, duplicate keys, non-finite values, invalid exact types,
excessive depth/nodes/repeats, cyclic or executable registered payloads, and
unresolved references fail before live providers are touched.

`TaskProgramCompiler` resolves canonical scene references, expands bounded
repeats and cyclic targets, assigns stable segment/call indices, preserves
parallel branches, and returns an immutable `CompiledTaskProgram`.

## Semantic integration

`task_program.semantics` contains:

- `calls.py`: Semantic Calls and the call catalog;
- `scene.py`: canonical references, registry, affordances, and collision roles;
- `profiles.py`: robot resources, endpoints, Atomic Skill bindings, policy
  presets, and `EffectAssurance`;
- `effects.py` / `evidence.py`: typed effects and measured evidence; and
- `integration.py`: provider-free manifests, static binding, and diagnostics.

`SimulationTaskProgramRegistration` is the standard composition root. Its
fingerprint covers scene/profile declarations, the Semantic Call catalog,
settling presets, grounders, endpoint transports, evidence and safety
factories, and registered-call lowerers. Adapter creation calls
`assert_unchanged()` and revalidates all live bindings.

Configured environments use runnable `task.<embodiment>.yaml` deployments with
three typed selections: `environment.component`,
`task_program.{program,integration,execution_policy}`, and
`embodiment.component`. The reusable `env.yaml` owns only
embodiment-independent Gym values and physical simulation entities. All
deployment component paths resolve from `task.<embodiment>.yaml`.
`config_to_cfg()` checks that semantic binding targets exist in the physical
scene, validates the task's required scene/embodiment contracts, composes the
immutable integration catalog, injects its trusted profile/scene/preset
selection into the unbound program, and registers the common `EmbodiedEnv`.
The CLI may override only the program. Components have closed fields and
intentionally omit compatibility `version` values.

Ownership is explicit rather than a generic deep merge:

- `program.yaml` owns task flow, targets, post-policies, and validators; it
  contains neither robot IDs nor an `integration` selection;
- task-local `integration.yaml` owns required contracts, its nested semantic
  `scene_binding`, semantic defaults, action options, effect monitors, and
  task-specific runtime services;
- task-local `env.yaml` owns only physical simulation entities and ordinary
  Gym environment values, so it can also be reused by handwritten trajectories;
- `configs/components/embodiments/*.yaml` owns simulation robot construction,
  the sensor suite, and an optional `skill_profile` containing logical
  resources/endpoints, command presets, and embodiment-specific services;
- `configs/components/execution_policies/*.yaml` owns motion, tracking,
  recovery, runner, and effect-assurance policy.

The reference embodiment `skill_profile.contract_id` and `profile_id` values
are unversioned. Versioned Gym, task-integration, or scene-registry IDs are
separate identity domains and do not imply a skill-profile version.
For `joint_position_constraint` evidence, `object_ids` is an optional explicit
narrowing. When omitted, runtime assembly scopes the embodiment service to the
selected scene's graspable rigid objects, so reusable embodiments do not name
task-local objects.

Deployment embodiment overrides affect only the robot simulation fields and
are restricted to `uid`, `init_pos`, `init_rot`, and `init_qpos`. Sensor lists
are selected atomically with the embodiment. Task grasp-generator overrides
are similarly allowlisted. This gives one embodiment to many tasks and one
task to many compatible embodiments without copying either declaration or
admitting arbitrary merge semantics. `repeated_pick_place` and `open_drawer`
each provide UR5 and Franka deployments as reference compositions.

Physical environment, embodiment, and standalone scene expansion is owned by
`gym/utils/_component_composition.py` and also runs for ordinary handwritten
Gym tasks. Task Program integration and nested scene-binding composition is
owned by `task_program/integrations/_configured_composition.py`. An embodiment
component may omit `skill_profile`, while a scene component never owns Task
Program metadata.
Selecting a physical component does not by itself parameterize hard-coded
Python control-part names or trajectory dimensions. A configured Task Program
must select an embodiment `skill_profile`; its integration must declare a
`scene_binding` satisfying the scene and execution-policy contracts. The
removed embodiment-level and scene-level `task_program` metadata keys are not
accepted.

Within `integration.yaml.scene_binding`, every affordance is nested in an
`affordances` list under its owning `rigid_objects`, `articulations`, or `links`
entry. The child keeps a globally unique `entity_id` and a closed `kind`
discriminator (`antipodal_grasp`, `support_surface`, or `container`); the YAML
does not repeat ownership with `object_id` or `parent_id`. The configured
decoder derives that relation and normalizes the authoring hierarchy into the
flat `SimulationSceneBinding` / `SceneRegistry` index. Scene-level affordance
collections are not accepted.

## Effect assurance and acceptance

Every `SkillPolicyPreset` explicitly selects one authority:

- `verified`: semantic state advances only from measured effect evidence;
  curated Pick/Place/HandOver calls require monitor mappings.
- `projected`: monitor mappings are forbidden and the action plan's expected
  symbolic effect is projected after command completion.

Projected execution proves the intended trajectory path completed; it does not
prove physical task success.

Keep these boundaries separate:

1. effect monitor: one Semantic Call's physical postcondition and recovery;
2. segment post-policy: environment advancement such as settling; and
3. segment validator: task or dataset acceptance.

`TaskProgramDemoBridge` never calls `env.step()`. It yields lazy
`DemoSegment` values so the ordinary Gym executor retains stepping, recording,
reward, reset, and persistence. Final success is published only after every
segment lifecycle completes normally.

## Parallel and registered calls

Parallel execution requires disjoint resource claims and runtime targets,
control-grid alignment, conflict-free symbolic writes, an authoritative
`ParallelCommandSafetyValidator`, and an explicit fail-fast barrier. Resource
disjointness alone is not physical safety evidence. Nested parallel blocks are
rejected.

A registered call payload is declarative and executable-free. The immutable
integration must provide exactly one matching lowerer factory and fingerprint
its call ID, revision, and target Atomic Skill descriptor. Use this extension
to expose shared Atomic Skills; do not place task-local motion generators in a
lowerer.

## MLLM boundary

`embodichain.agents.mllm.task_program` accepts untrusted JSON, reuses the
canonical decoder/validator/compiler, and requires a trusted host adapter. The
model cannot choose integrations, robot resources, live providers, or
executable lowerers. The current frontend permits the constrained node/call
subset enforced by its decoder; Task Program itself supports the complete
language.

## Reference integrations

| Environment | Assurance | Source |
|---|---|---|
| `TaskProgramRepeatedPickPlace-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/` |
| `TaskProgramOpenDrawer-v1` | projected | `embodichain_tasks/configs/tasks/manipulation/open_drawer/` |
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

## Focused validation

```bash
pytest -q tests/lab/task_program
pytest -q tests/gym/envs/task_program
pytest -q tests/gym/envs/test_embodied_env_task_program.py
pytest -q tests/agents/mllm/test_task_program.py
pytest -q tests/sim/atomic_actions
python docs/scripts/check_api_docs.py
```

Public API changes also require the docs checker tests and a Sphinx dummy
build. Physical qualification requires real environment runs with measured
evidence, controlled seeds/randomization, explicit validators, and persisted
completion metadata.
