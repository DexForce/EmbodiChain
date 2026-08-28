(expert-programs)=

# Expert Programs

```{currentmodule} embodichain.lab.expert_program
```

Expert Program is EmbodiChain's task-level action-generation entry point. It
loads a bounded declarative program, validates it against provider-free scene
and robot contracts, and delegates execution to the existing Atomic Actions
engine through an environment adapter.

Semantic skills provide the vocabulary consumed here; they are not an
independent execution entry point.

```text
JSON/YAML or ExpertProgramCfg
          |
          v
embodichain.lab.expert_program
  strict decode -> validation -> provider-free compilation
          |
          v
embodichain.lab.gym.envs.expert_program
  immutable registration -> live scene/profile binding -> Gym bridge
          |
          v
AtomicActionEngine -> ExecutionRunner -> DemoSegment -> env.step()
```

## Ownership boundaries

The public API is split deliberately:

| Package | Owns | Does not own |
|---|---|---|
| `embodichain.lab.expert_program` | Config dataclasses, strict JSON/YAML decoding, validation, provider-free compilation, `CompiledProgram` | Simulator objects, Gym stepping, controller transports |
| `embodichain.lab.semantic_skills` | Semantic calls, scene/profile/effect/evidence declarations | Public compilation or execution APIs |
| `embodichain.lab.gym.envs.expert_program` | Gym bridge, simulation bindings, immutable integration registration, runtime-service factories | A duplicate schema, decoder, or compiler |
| `embodichain.lab.sim.atomic_actions` | Typed planning, commands, execution, recovery, verification requests | Task sequencing or semantic JSON/YAML |

Do not import underscore-prefixed Expert Program modules. Their compiler,
semantic-call executor, and parallel scheduler are implementation details
behind the public `ExpertProgramCompiler` and environment adapter.

## Program schema

An {class}`ExpertProgramCfg` contains:

- one `program_id`;
- exact scene, robot-profile, and policy-preset IDs;
- optional named targets; and
- one bounded program tree.

Supported program nodes are {class}`SequenceCfg`, {class}`RepeatCfg`,
{class}`SegmentCfg`, {class}`InvokeCfg`, and {class}`ParallelCfg`. A
{class}`BarrierCfg` belongs to its enclosing parallel node. Built-in call
configs are {class}`PickCfg`, {class}`PlaceCfg`, and {class}`HandOverCfg`;
{class}`RegisteredSemanticCallCfg` is the allowlisted extension form.

Example:

```yaml
program_id: repeated_cube_pick_place
integration:
  robot_profile: expert_program_ur5_pick_place
  scene_registry: expert_program_repeated_pick_place
  runtime_preset: trajectory
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
```

Serialized input is strict: unknown fields, duplicate keys, non-finite values,
invalid discriminators, excessive depth or expansion, executable values, and
unresolved references fail before live providers are touched. External formats
do not carry a `schema_version` field.

## Load and compile

Use the provider-independent package for both file and programmatic input:

```python
from embodichain.lab.expert_program import (
    ExpertProgramCompiler,
    load_expert_program,
)

program = load_expert_program("expert/program.yaml")
compiled = ExpertProgramCompiler(scene_manifest).compile(program)
```

Compilation observes no live simulator state. It resolves canonical scene
references, expands bounded repeats and cyclic targets, assigns stable segment
and call indices, preserves parallel branches, and returns an immutable
{class}`CompiledProgram`.

The compiler does not generate controller actions. Live grounding and atomic
planning begin only after an environment adapter has verified that the
registration still matches the compiled scene/profile/catalog snapshots.

## Configure a simulation environment

A task-local Gym configuration selects its program and declares the supported
runtime without Python callbacks:

```json
{
  "id": "ExpertProgramRepeatedPickPlace-v1",
  "expert_program_path": "expert/program.yaml",
  "expert_program_runtime": {
    "scene": {
      "registry_id": "expert_program_repeated_pick_place",
      "rigid_objects": [
        {"entity_id": "cube", "dynamics": "dynamic", "semantic_type": "cube"}
      ]
    },
    "robot_profile": {
      "profile_id": "expert_program_ur5_pick_place",
      "resources": [],
      "presets": []
    }
  }
}
```

The abbreviated arrays above must contain the real resource and preset
declarations. See
`embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json` for
a complete configuration.

`expert_program_runtime` accepts only allowlisted scene bindings, robot
resources, policies, monitors, and runtime-service kinds. It does not accept
dotted imports or arbitrary callables. The configured runtime decoder lives in
`_configured_runtime_decoder.py`; Gym registration remains the small concern of
`configured_runtime.py`.

When a supported config defines `expert_program_runtime`, `config_to_cfg()` can
register the common `EmbodiedEnv` under the config-owned ID. A task-specific
Python environment module is not required.

Run a selected program with:

```bash
python -m embodichain.lab.scripts.run_env \
  --gym_config path/to/env.json \
  --expert-program path/to/program.yaml
```

The CLI override changes the program only. The trusted Gym configuration still
owns the scene, robot profile, extension factories, and executable integration.

## Runtime lifecycle

The environment adapter compiles or accepts a trusted {class}`CompiledProgram`,
assembles fresh live providers, and creates one `AtomicDemoBridge`.
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

Use registered calls to reuse a shared Atomic Action from configuration. Do not
put task-local motion generation in a lowerer or create another semantic
runtime.

## Reference tasks

| Environment | Assurance | What it demonstrates |
|---|---|---|
| `ExpertProgramRepeatedPickPlace-v1` | projected | Bounded repeat and cyclic targets over Pick and Place. |
| `ExpertProgramOpenDrawer-v1` | projected | An allowlisted registered call lowered to atomic Slide. |
| `HandOver-v1` | verified | Coordinated dual-resource HandOver with measured evidence, settling, and final-target validation. |
| `PourWater-v1` | projected | Registered held-object transport and Pour calls composed after Pick. |

Projected examples demonstrate planning and command generation only. They must
not be presented as physically qualified task-success integrations.

## Recommended change sites

| Change | Owning location |
|---|---|
| Schema and config dataclasses | `embodichain/lab/expert_program/cfg.py` |
| Strict decoding and validation | `embodichain/lab/expert_program/decoder.py` |
| File loading and resource bounds | `embodichain/lab/expert_program/loader.py` |
| Provider-free compilation | `embodichain/lab/expert_program/compiler.py` |
| Gym bridge and segment lifecycle | `embodichain/lab/gym/envs/expert_program/bridge.py` |
| Environment adapter assembly | `embodichain/lab/gym/envs/expert_program/environment.py` |
| Simulation bindings and registration | `embodichain/lab/gym/envs/expert_program/simulation.py`, `catalog.py`, and `simulation_environment.py` |
| Callable-free runtime decoding | `embodichain/lab/gym/envs/expert_program/_configured_runtime_decoder.py` |
| Dynamic Gym ID registration | `embodichain/lab/gym/envs/expert_program/configured_runtime.py` |
| Semantic declarations | `embodichain/lab/semantic_skills/` |
| Atomic planning and execution | `embodichain/lab/sim/atomic_actions/` |

## Focused validation

Run:

```bash
pytest -q tests/lab/expert_program
pytest -q tests/gym/envs/expert_program
pytest -q tests/lab/semantic_skills tests/sim/atomic_actions
python docs/scripts/check_api_docs.py
```

Physical qualification additionally requires real-environment runs with
measured evidence, controlled seeds/randomization, task validators, and saved
completion metadata.
