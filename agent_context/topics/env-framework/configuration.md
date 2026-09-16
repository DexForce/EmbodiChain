# Environment configuration and registration

Read this when the request needs these details. [Topic overview](env-framework.md).

## Task Registration

### Decorator

```python
from embodichain.lab.gym.utils.registration import register_env

@register_env("MyTask-v1", max_episode_steps=600)
class MyTaskEnv(EmbodiedEnv):
    ...
```

### Mechanics

1. `register_env(uid)` is a class decorator defined in `registration.py`.
2. It calls `register()` which stores an `EnvSpec` in the module-level
   `REGISTERED_ENVS` dict, keyed by `uid`.
3. It also calls `gym.register()` so the env is available via
   `gym.make(uid)`.
4. Simulator tasks with a supported RL training config declare
   `supports_rl=True`; this is stored on `EnvSpec` and is not forwarded to the
   environment constructor.
5. `kwargs` passed to `@register_env` must be **JSON-serialisable** (no
   classes/types). A `RuntimeError` is raised otherwise.
6. Use `override=True` to re-register an existing uid (useful in scripts/tests).

### Gym ID convention

Format: `<TaskName>-v<N>` (e.g. `PourWater-v1`, `PushCubeRL`).
RL tasks sometimes drop the `-v<N>` suffix (`CartPoleRL`, `PushCubeRL`).

### Component ownership

| File / component | Owns | Must not own |
|---|---|---|
| Reusable physical `env.yaml` | Exactly one `physics: default|newton` backend, its optional matching `physics_config`, scene entities, and ordinary environment values; `environment_id` identifies the environment | Runnable `id`, robot, sensor or Task Program fields |
| Runnable `task.<embodiment>.yaml` | Gym `id`, component selections, task-local run/deployment settings | Duplicate inline fields owned by selected components |
| `configs/components/embodiments/*.yaml` | One robot, its sensors, optional Task Program-facing `skill_profile` | Task-local semantic scene binding |
| `task_program/program.yaml` | Embodiment-independent program | Trusted runtime provider implementations |
| `task_program/integration.yaml` | Nested semantic `scene_binding` and allowlisted service declarations | Physical scene assets |
| `configs/components/execution_policies/*.yaml` | Reusable execution policy | Physical environment/embodiment ownership |
| `<task>/agents/<algorithm>.{json,yaml}` | Optional RL training configuration | Task identity or Python registration ownership |

Component files do not use compatibility `version` fields. Inline runnable
Gym configs remain supported when no conflicting component selector is used.

### Reusable Gym deployment components

`config_to_cfg()` resolves optional `environment.component`,
`embodiment.component`, and `scene.component` selections before ordinary Gym
parsing. This is not
coupled to Task Program: an import-registered handwritten-demo task can reuse
an embodiment's simulation robot and sensor suite while keeping its events,
observations, objects, and Python demo logic task-local. All deployment-owned
component paths resolve relative to the runnable config that declares them
(conventionally `task.<embodiment>.yaml`). Component-owned fields and their
inline counterparts are mutually exclusive; without a selector, the original
inline `robot`, `sensor`, and scene fields continue to parse unchanged.
`build_env_cfg_from_args()` expands `environment.component` before applying
launcher arguments so environment-owned run controls such as `max_episodes`
remain visible to the run loop while explicit CLI values retain precedence.

An inline runnable config must declare exactly one
`physics: default|newton` backend. A reusable environment component also owns
exactly one backend and its optional `physics_config`; the thin deployment
cannot repeat either field. `config_to_cfg()` constructs the backend-specific
typed physics config and rejects fields from the other backend. Launcher
`--physics` may confirm the declared value but cannot switch the file-owned
backend. Use separate environment files when one logical task needs both.

Device selection is one shared runtime value. The typed physics config supplies
the backend default (`cpu` for Default, `cuda:0` for Newton), an optional
top-level Gym `device` overrides it, and an explicitly supplied CLI `--device`
wins last. Config-backed launchers leave `--device` unset by default, so
omission preserves the authored/backend value. `BaseEnv` tensors use the
manager's resolved device; there is no separate environment-device setting.

The component boundary is implemented in
`gym/utils/_component_composition.py`. An embodiment's optional `skill_profile`
is consumed only by a configured Task Program deployment. Scene components are
always physical-only; semantic entity mappings and affordances live in the
task integration's nested `scene_binding`. The shared
`cobotmagic.yaml` component owns a top-view RGB camera, two wrist
RGB cameras, and an optional right-arm skill profile. Tableware handwritten and
configured Task Program deployments reuse that same embodiment.

### Configuration-owned Task Program environment

A simple supported Task Program does not require a task subclass. Its thin Gym
deployment selects a reusable environment and embodiment and declares all
three Task Program component paths. The environment component owns the
physical scene, one physics backend and its settings, and ordinary environment
values. After the generic resolver
lowers the environment, robot, and sensors into the existing
`EmbodiedEnvCfg` fields, it checks every semantic root's `simulation_uid`
against the physical scene. The Task Program layer then checks
scene/embodiment contracts, composes the immutable catalog, preflights the
deployment-bound program, and calls
`register_env_function(EmbodiedEnv, config["id"], ...)`.

The ID is selected by the config and may be any free valid Gym ID. Loading the
same ID with the same integration and episode limit is idempotent. Reusing it
for a different class, integration declaration, or limit fails closed; the
loader does not use `override=True`. Registration is process-local, so callers
must load the Gym config before calling `gym.make(id)`. Such an ID is not
present merely because task-package discovery ran.

The integration has no task-level kind. It composes a typed scene and robot
profile with optional allowlisted live-service declarations. The built-in service
leaves currently cover antipodal parallel-jaw grasp generation, configured
hand-over poses, articulation-link Slide lowering, and joint-position
constraint evidence. New executable provider families require a core
allowlisted implementation and decoder entry; never serialize dotted imports
or arbitrary callables into this config boundary. The official Task Program
examples use this path and have no task environment subclass.

### Instantiation

```python
from embodichain.lab.gym.utils.registration import make
env = make("MyTask-v1", cfg=my_cfg)
```

Or via gymnasium: `gym.make("MyTask-v1")`.

### Listing registered tasks

`embodichain list-task` calls `discover_task_packages()` and prints a stable
table whose `Task` column is a directory tree derived from task-first modules
and packaged `configs/tasks/` paths. Deployments for the same logical task are
kept together, with one divider between task groups; the title reports both
logical-task and environment counts. The other columns show the environment ID,
selected embodiment, supported use, and runnable config filename:

- the embodiment is the selected component filename without its extension, or
  an inline robot's `robot_type` with `uid` as the fallback;
- `Config` lists every top-level runnable config that declares the environment
  ID; `-` means that metadata does not come from a Gym deployment config;

- `[Expert Demo: Task Program]` comes from a task-local Gym config declaring
  the `task_program` component mapping;
- `[Expert Demo: Handwritten Trajectory]` means the registered task class
  overrides `create_demo_segments()` or `create_demo_action_list()`;
- `[RL]` comes from explicit simulator `supports_rl`, a task-local agents
  directory, or a registered lightweight learning environment;
- `[Environment Only]` means none of those supported execution paths is
  currently declared.

Configuration-owned Task Program IDs are included from their packaged task
configs without eagerly building or registering the integration. Duplicate
JSON/YAML variants and registry entries merge case-insensitively into one task
leaf. Discovery is schema-driven: it scans top-level JSON/YAML resources in a
task directory and treats only mappings with a non-empty `id` as runnable
deployments. A pure `env.yaml` component has `environment_id` but no `id`, so
it is not listed and deployment filenames do not need an `env*` prefix. The
framework-level `EmbodiedEnv-v1` registration is omitted because it
is a reusable base environment rather than an installed task-package entry.

---

## Creating a New Task

Use the `/add-task-env` skill. It first selects one of two registration paths:

1. Import-backed handwritten or RL tasks add a task-named module at
   `embodichain_tasks/embodichain_tasks/<category-path>/<task>.py`, keep
   `@register_env("<GymId>")` and `__all__` there, and add a runnable config.
2. Supported configuration-defined Task Programs omit the Python module. They
   add a reusable physical `env.yaml`, one or more runnable
   `task.<embodiment>.yaml` deployments, and
   `task_program/{program,integration}.yaml`; semantic scene bindings are
   nested under `integration.yaml.scene_binding`.

Both paths add focused tests. A componentized import-backed task may also reuse
the same environment and embodiment owners while omitting `task_program`.

The category path starts with a top-level task family and may include a
subdomain. Tableware tasks use `manipulation/tableware`; general manipulation
tasks can stay directly under `manipulation`.

Do not organize task ownership around a solution method such as `rl` or
`task_program`. Keep registration in the task-named module and do not create
a same-named Python package for a task that has only one Python entry point.
