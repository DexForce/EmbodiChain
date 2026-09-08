(task-programs)=

# Embodied Task Program

```{toctree}
:hidden:

scene_registry
robot_profiles
```

```{currentmodule} embodichain.lab.task_program
```

Embodied Task Program is **a declarative orchestration and execution DSL for
embodied tasks**. It accepts bounded source programs, validates them against
provider-free scene and robot contracts, compiles Semantic Calls, and lowers
those calls to executable Atomic Skills through an explicit integration.

Semantic Calls are the language's internal high-level instructions. They are
not a separate product, package entry point, or independently executable API.

```text
Human / MLLM / Agent
          |
          v
JSON/YAML/Python TaskProgramCfg
          |
          v
language -> decode / validate -> compiler
          |
          v
Semantic Calls -> semantic lowering -> Atomic Skills
          |
          v
simulation / robot -> Gym lifecycle / recording
```

## Ownership boundaries

The public API is split deliberately:

| Package | Owns | Does not own |
|---|---|---|
| `embodichain.lab.task_program.language` | Schema/AST values, strict decoding, validation, and loading | Live providers or execution |
| `embodichain.lab.task_program.semantics` | Semantic Calls plus scene, robot-profile, effect, and evidence contracts | An independent action API |
| `embodichain.lab.task_program.compiler` | AST compilation and Semantic Call lowering | Gym stepping or simulator ownership |
| `embodichain.lab.task_program.runtime` | Semantic Call execution state and parallel scheduling | Episode lifecycle or recording |
| `embodichain.lab.task_program.integrations` | Immutable registrations and explicit environment/simulation assembly | Core language rules |
| `embodichain.lab.gym.envs.task_program` | `DemoSegment` adaptation, `env.step()` handshake, and configured Gym registration | Compiler, catalog, or simulation assembly |
| `embodichain.lab.sim.atomic_actions` | Executable Atomic Skills, planning, commands, recovery, and verification requests | Task-language sequencing |

The stable language API is re-exported from `embodichain.lab.task_program`.
Provider-specific applications import `task_program.integrations`; Gym bridge
types are intentionally kept under `gym.envs.task_program`.

## Semantic layer

The semantics subpackage is internal to Task Program but its typed contracts
are public for integration authors:

| Contract family | Main types | Responsibility |
|---|---|---|
| Semantic Calls | {class}`~embodichain.lab.task_program.semantics.Pick`, {class}`~embodichain.lab.task_program.semantics.Place`, {class}`~embodichain.lab.task_program.semantics.HandOver`, {class}`~embodichain.lab.task_program.semantics.RegisteredSemanticCall` | Immutable object-centric task intent. |
| Scene semantics | {class}`~embodichain.lab.task_program.semantics.SceneRegistry`, {class}`~embodichain.lab.task_program.semantics.SceneManifest` | Canonical identity, topology, affordances, and provider-free snapshots. |
| Robot semantics | {class}`~embodichain.lab.task_program.semantics.RobotSkillProfile`, {class}`~embodichain.lab.task_program.semantics.SkillPolicyPreset` | Embodiment resources, Atomic Skill bindings, policies, and effect assurance. |
| Effects and evidence | {class}`~embodichain.lab.task_program.semantics.SemanticEffectSpec`, {class}`~embodichain.lab.task_program.semantics.EffectMonitorRef`, {class}`~embodichain.lab.task_program.semantics.EffectEvidenceCollector` | Typed postconditions and measured-evidence routing. |

The built-in call catalog contains `Pick`, `Place`, and `HandOver`. Extensions
use a declarative `RegisteredSemanticCall`; the trusted integration, never the
program payload, owns the corresponding lowerer. See {doc}`scene_registry` and
{doc}`robot_profiles` for the two main integration contracts.

## Program schema

An in-memory {class}`TaskProgramCfg` contains one `program_id`, an exact trusted
{class}`TaskProgramIntegrationCfg` selection, optional named targets, and one
bounded program tree. A program authored for a componentized deployment omits
the integration selection from its serialized YAML. The deployment injects
the selected scene-registry, robot-profile, and policy-preset IDs before strict
decoding, which keeps the source program portable across compatible
embodiments.

Supported program nodes are {class}`SequenceCfg`, {class}`RepeatCfg`,
{class}`SegmentCfg`, {class}`InvokeCfg`, and {class}`ParallelCfg`. A
{class}`BarrierCfg` belongs to its enclosing parallel node. Built-in call
configs are {class}`PickCfg`, {class}`PlaceCfg`, and {class}`HandOverCfg`;
{class}`RegisteredSemanticCallCfg` is the allowlisted extension form.

Example:

```yaml
program_id: repeated_cube_pick_place
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
        - kind: invoke
          call:
            kind: place
            object: cube
            at: {kind: target_ref, target: drop_pose}
```

Serialized input is strict: unknown fields, duplicate keys, non-finite values,
invalid discriminators, excessive depth or expansion, executable values, and
unresolved references fail before live providers are touched. External formats
do not carry a `schema_version` field.

## Load and compile

Use the provider-independent package for both file and programmatic input:

```python
from embodichain.lab.task_program import (
    TaskProgramIntegrationCfg,
    TaskProgramCompiler,
    load_task_program,
)

program = load_task_program(
    "task_program/program.yaml",
    integration=TaskProgramIntegrationCfg(
        robot_profile="ur5_dh_pgi_140_80",
        scene_registry="task_program_repeated_pick_place",
        runtime_preset="trajectory",
    ),
)
compiled = TaskProgramCompiler(scene_manifest).compile(program)
```

Compilation observes no live simulator state. It resolves canonical scene
references, expands bounded repeats and cyclic targets, assigns stable segment
and call indices, preserves parallel branches, and returns an immutable
{class}`CompiledTaskProgram`.

The compiler does not generate controller actions. Live grounding and atomic
planning begin only after an environment adapter has verified that the
registration still matches the compiled scene/profile/catalog snapshots.

## Configure a simulation environment

Configuration-defined Task Programs separate the reusable physical environment
from the runnable task deployment:

```text
repeated_pick_place/
├── env.yaml
├── task.franka.yaml
├── task.ur5.yaml
└── task_program/
    ├── integration.yaml
    └── program.yaml
```

`env.yaml` owns the physical scene and ordinary environment values. It has an
`environment_id` for component identity but deliberately has no runnable `id`,
robot, sensor, or Task Program selection:

```yaml
environment_id: repeated_pick_place
max_episode_steps: 1200
simulation:
  rigid_object:
    - uid: cube
      # Physical shape, dynamics, and initial pose.
env:
  sim_steps_per_control: 4
  events: {}
  dataset: {}
```

A runnable `task.<embodiment>.yaml` selects the environment, all three Task
Program components, and one reusable embodiment:

```yaml
id: TaskProgramRepeatedPickPlace-v1
environment:
  component: env.yaml
task_program:
  program: task_program/program.yaml
  integration: task_program/integration.yaml
  execution_policy: ../../../components/execution_policies/trajectory_open_loop.yaml
embodiment:
  component: ../../../components/embodiments/ur5_dh_pgi_140_80.yaml
```

The callable-free `integration.yaml` owns the semantic scene binding and
task-specific profile additions. Its canonical identities map explicitly to
UIDs in the physical environment:

```yaml
integration_id: repeated_pick_place_v1
program_id: repeated_cube_pick_place
requires:
  scene_contract: repeated_pick_place_scene_v1
  embodiment_contract: single_arm_parallel_gripper
scene_binding:
  contract_id: repeated_pick_place_scene_v1
  registry_id: task_program_repeated_pick_place
  rigid_objects:
    - entity_id: cube
      simulation_uid: cube
      dynamics: dynamic
      semantic_type: cube
      affordances:
        - entity_id: cube_grasp
          kind: antipodal_grasp
profile:
  defaults: {}
  action_options: {}
  effect_monitors: {}
```

The abbreviated profile mappings above must be populated for the calls used by
the program. See `env.yaml`, `task.ur5.yaml`, and
`task_program/integration.yaml` under
`embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/` for the
complete reference composition.

Scene affordances are authored as children of their owning entry in
`rigid_objects`, `articulations`, or `links`. The parent's `entity_id` names the
physical scene entity; the child's `entity_id` is a separate, globally unique
semantic feature that can be selected explicitly by a program or parent
default. Nesting is the only way to declare ownership: scene-level
`antipodal_grasps`, `support_surfaces`, and `containers`, and child
`object_id`/`parent_id` fields, are rejected. The loader derives that parent
relation from the YAML structure and normalizes the declarations into the flat,
globally indexed Scene Registry.

The task integration accepts only closed scene bindings, profile additions,
monitor mappings, and allowlisted runtime-service kinds. Robot resources come
from the embodiment component, while motion and runner policy come from the
execution-policy component. Their composed integration does not accept dotted
imports or arbitrary callables. The configured integration loader lives in
`task_program/integrations/configured.py`; Gym registration remains the small
concern of `gym/envs/task_program/registration.py`.

`config_to_cfg()` resolves every component reference relative to the runnable
deployment, expands the physical environment and embodiment, validates each
semantic `simulation_uid` against the physical scene, composes the immutable
integration catalog, validates the program, and registers the common
`EmbodiedEnv` under the deployment-owned ID. A task-specific Python environment
module is not required. The pure `env.yaml` can also be selected by a
handwritten task because it contains no Task Program fields.

Run a selected program with:

```bash
embodichain run-env \
  --gym_config path/to/task.ur5.yaml \
  --task-program path/to/program.yaml
```

The CLI override changes the program only. It cannot replace the integration,
execution policy, environment, or embodiment selected by the runnable task
deployment.

## Runtime lifecycle

The environment adapter compiles or accepts a trusted {class}`CompiledTaskProgram`,
assembles fresh live providers, and creates one `TaskProgramDemoBridge`.
`EmbodiedEnv.create_demo_segments()` exposes lazy actions; the bridge never
calls `env.step()` itself. The normal demo executor owns the step handshake,
recording, rewards, and reset lifecycle.

```text
compile program
    -> provider-aware preflight
    -> prepare next semantic call from fresh observation
    -> lower to ActionInvocation
    -> AtomicActionEngine + ExecutionRunner
    -> verify or project the declared effect
    -> post-policies
    -> segment validators
    -> publish final completion mask
```

Keep three acceptance boundaries separate:

| Boundary | Meaning |
|---|---|
| Semantic effect monitor | Establishes whether one physical call achieved its postcondition and participates in recovery. |
| Segment post-policy | Advances environment behavior after motion, for example waiting for an object to settle. |
| Segment validator | Determines whether the resulting task or dataset segment is acceptable. |

## Effect assurance is mandatory

Every selected robot policy preset explicitly chooses
`effect_assurance: verified` or `effect_assurance: projected`.

- `verified` advances semantic state only from typed measured evidence. Curated
  Pick, Place, and HandOver calls require an explicit monitor mapping.
- `projected` advances the action plan's expected symbolic state after command
  completion and forbids monitor mappings. It is suitable for trajectory-only
  demonstrations, not proof of physical success.

No monitor set is installed implicitly. A missing assurance field is a decode
error, and a verified curated call without a monitor fails static compilation.

Held-object guards and phase-effect gates remain observational. They may block
a named atomic segment or invalidate an action-authorized symbolic relation;
they never attach objects, freeze bodies, or overwrite poses.

## Parallel programs

Parallel blocks require:

- disjoint resource claims and runtime targets;
- aligned command frames on the environment control grid;
- conflict-free symbolic writes;
- an authoritative `ParallelCommandSafetyValidator`; and
- an explicit fail-fast barrier.

Resource disjointness alone is not physical-safety evidence. Missing or
inconclusive safety validation fails closed. Nested parallel blocks are
rejected.

## Registered calls

{class}`RegisteredSemanticCallCfg` carries only declarative data. The immutable
simulation registration must declare exactly one matching lowerer factory and
include that declaration in its integration fingerprint. Each live assembly
receives a fresh lowerer instance and revalidates its call ID, revision, and
target descriptor.

Use registered calls to reuse a shared Atomic Skill from configuration. Do not
put task-local motion generation in a lowerer or create another semantic
runtime.

## Reference tasks

| Environment | Assurance | What it demonstrates |
|---|---|---|
| `TaskProgramRepeatedPickPlace-v1` | projected | Bounded repeat and cyclic targets over Pick and Place. |
| `TaskProgramOpenDrawer-v1` | projected | An allowlisted registered call lowered to atomic Slide. |
| `HandOver-v1` | verified | Coordinated dual-resource HandOver with measured evidence, settling, and final-target validation. |
| `PourWater-v1` | projected | Registered held-object transport and Pour calls composed after Pick. |

Projected examples demonstrate planning and command generation only. They must
not be presented as physically qualified task-success integrations.

## Recommended change sites

| Change | Owning location |
|---|---|
| Schema and AST dataclasses | `embodichain/lab/task_program/language/schema.py` |
| Strict decoding and validation | `embodichain/lab/task_program/language/decoder.py` |
| File loading and resource bounds | `embodichain/lab/task_program/language/loader.py` |
| Provider-free compilation | `embodichain/lab/task_program/compiler/program.py` |
| Semantic Call lowering | `embodichain/lab/task_program/compiler/lowering.py` |
| Sequential and parallel execution | `embodichain/lab/task_program/runtime/` |
| Semantic declarations | `embodichain/lab/task_program/semantics/` |
| Registration and environment assembly | `embodichain/lab/task_program/integrations/` |
| Gym bridge and segment lifecycle | `embodichain/lab/gym/envs/task_program/bridge.py` |
| Dynamic Gym ID registration | `embodichain/lab/gym/envs/task_program/registration.py` |
| Atomic planning and execution | `embodichain/lab/sim/atomic_actions/` |

## Focused validation

Run:

```bash
pytest -q tests/lab/task_program
pytest -q tests/gym/envs/task_program
pytest -q tests/sim/atomic_actions
python docs/scripts/check_api_docs.py
```

Physical qualification additionally requires real-environment runs with
measured evidence, controlled seeds/randomization, task validators, and saved
completion metadata.

## Further reading

- {doc}`scene_registry` — canonical scene identities, affordances, and live providers
- {doc}`robot_profiles` — robot resources, Atomic Skill bindings, and assurance
- {doc}`../sim/atomic_actions/index` — direct typed atomic-action planning and
  execution
- {doc}`/tutorial/task_program_python` — author and compile a Task Program in Python
- {doc}`/tutorial/task_program` — configure and run an embodied Task Program
