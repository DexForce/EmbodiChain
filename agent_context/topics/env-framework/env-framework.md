# env-framework

> Topic: Environment framework — BaseEnv / EmbodiedEnv class hierarchy,
> task registration, manager wiring, and lifecycle.

---

## Entry Points

| File | Role |
|---|---|
| `embodichain/__main__.py` | Unified CLI dispatch, including `list-task` discovery and the `run-task` alias |
| `embodichain/lab/gym/envs/base_env.py` | `BaseEnv(gym.Env)` + `EnvCfg` — low-level env loop |
| `embodichain/lab/gym/envs/types.py` | `ControllerAction` — owned controller-ready action boundary |
| `embodichain/lab/gym/envs/embodied_env.py` | `EmbodiedEnv(BaseEnv)` + `EmbodiedEnvCfg` — modular task base class |
| `embodichain/lab/gym/envs/demo.py` | Segment-aware demonstration execution, result, and persistence-mode contracts |
| `embodichain/lab/scripts/run_env.py` | Offline collection retries and explicit dataset commit/abort boundaries |
| `embodichain/lab/gym/utils/registration.py` | `@register_env` decorator + `REGISTERED_ENVS` registry + `make()` |
| `embodichain/lab/gym/utils/gym_utils.py` | Gym config parsing and config-owned runtime registration |
| `embodichain/lab/gym/utils/_component_composition.py` | Reusable physical environment, embodiment, and standalone scene resolution |
| `embodichain/lab/task_program/integrations/configured.py` | Strict built-in Task Program integration decoder |
| `embodichain/lab/task_program/integrations/_configured_composition.py` | Task Program semantic component and contract composition |
| `embodichain/lab/gym/envs/task_program/registration.py` | Config-owned Gym ID registration |
| `embodichain_tasks/embodichain_tasks/__init__.py` | Recursively imports official tasks to trigger registration |
| `embodichain/lab/gym/envs/managers/__init__.py` | Manager re-exports: `EventManager`, `ObservationManager`, `RewardManager`, `ActionManager`, `DatasetManager` |
| `embodichain/lab/gym/envs/wrapper/no_fail.py` | `NoFailWrapper` — forces `is_task_success() → True` |
| `embodichain/lab/gym/envs/wrapper/replay.py` | `ReplayWrapper` — record-and-replay trajectories (kinematic/dynamic/control) |

---

## Overview

The env framework provides a Gymnasium-compatible simulation loop for
embodied manipulation tasks. All tasks inherit from **EmbodiedEnv**, which
itself extends **BaseEnv(gym.Env)**. Managers (event, observation, reward,
action, dataset) are optionally wired into the env via config fields and
follow the Functor/FunctorCfg pattern.

---

## Architecture

```
gym.Env
  └── BaseEnv            (EnvCfg)
        └── EmbodiedEnv   (EmbodiedEnvCfg)
              └── <YourTask>  (YourTaskCfg)
```

### BaseEnv (`base_env.py`)

- Owns: `SimulationManager`, `Robot`, sensors dict, action/observation spaces.
- Implements the full `step()` / `reset()` loop (see Lifecycle below).
- Defines hook points subclasses override:
  - `_setup_robot()` — load robot, set `single_action_space`. **Must** return `Robot`.
  - `_prepare_scene()` — add scene assets.
  - `_setup_sensors()` → `Dict[str, BaseSensor]`.
  - `_init_sim_state()` — one-time post-scene init.
  - `_initialize_episode(env_ids)` — per-episode reset / randomization.
  - `_update_sim_state()` — called each step after physics.
  - `evaluate()` → `{"success": ..., "fail": ...}`.
  - `get_reward(obs, action, info)` → `torch.Tensor`.
  - `_preprocess_action(action)` / `_postprocess_action(action)`.
  - `_hook_after_sim_step(obs, action, rewards, dones, info)`.

### EmbodiedEnv (`embodied_env.py`)

- Adds declarative config fields: `robot`, `sensor`, `light`, `background`,
  `rigid_object`, `rigid_object_group`, `articulation`, manager configs
  (`events`, `observations`, `rewards`, `actions`, `dataset`), `extensions`.
- Creates managers in `_init_sim_state()` from config.
- Overrides `_extend_obs()` to run `ObservationManager.compute()`.
- Overrides `_extend_reward()` to run `RewardManager.compute()` and add to
  base reward.
- Overrides `_initialize_episode()` to run event-manager `reset` mode,
  dataset save, and manager resets.
- Overrides `_update_sim_state()` to run event-manager `interval` mode.
- Manages rollout buffer (expert or RL mode) via `_hook_after_sim_step()`.
- Owns per-row demonstration metadata and dense segment acceptance, attempt,
  and causal-continuity annotations.
- `extensions` dict entries are set as attributes on both cfg and env instance.

### Action boundary (`types.py`, `embodied_env.py`)

- Raw policy actions run through `ActionManager` terms in `pre` mode.
- `ControllerAction` wraps a command that already completed raw-policy
  preprocessing. `EmbodiedEnv` unwraps it and skips only `pre` terms.
- Both paths converge at `_prepare_controller_action()`, which validates the
  vector batch, control keys (`qpos`, `qvel`, `qf`), active/full joint width,
  floating dtype, and environment device before robot control.
- `ControllerAction` does not bypass `env.step()` or `ActionManager` terms in
  `post` mode. Task Program uses this boundary so runtime commands, wait
  holds, and abort-safe holds retain the normal Gym lifecycle.
- A structured controller `TensorDict` may carry auxiliary fields such as
  `ik_success`, but it must contain at least one supported control key.

### Task Program completion (`embodied_env.py`, `task_program/bridge.py`)

- `EmbodiedEnvCfg.task_program` remains opt-in. A registered task may attach
  an `TaskProgramAdapterFactory` to its `EnvSpec`; `EmbodiedEnv` binds the
  exact adapter after `BaseEnv` has initialized the live scene and robot.
- A standard simulation adapter factory exposes its immutable registration, so
  `EnvSpec` derives the catalog used by `config_to_cfg()` preflight from the
  same declaration that later creates the live adapter.
- `create_demo_segments(task_program=...)` accepts a config or trusted
  `CompiledTaskProgram` for one episode, allowing an MLLM frontend to use the same
  bridge without mutating the environment's static config.
- `create_demo_segments()` retains the active `TaskProgramDemoBridge` while its lazy
  segments execute.
- For an enabled program, `is_task_success()` returns all false until the bridge
  iterator completes normally. It then returns the bridge's final row-local
  acceptance mask, which already combines runtime, post-policy, and validator
  results.
- `reset()` lets `BaseEnv` read that final mask before clearing the active
  bridge, preventing completed state from leaking into the next episode.
- `DemoSegmentResult.successes` is the persisted segment acceptance authority.
  `DemoExecutionCfg` keeps continuous episodes as the default and can instead
  make each eligible natural segment an independent dataset fragment. This
  mode does not restore state or resume after failure.
- Environments without a Task Program keep the ordinary `BaseEnv.is_task_success()`
  behavior and task-specific overrides.

---

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
`physics: default|newton` backend. A reusable environment component must also
declare exactly one backend and owns its optional `physics_config`; the thin
deployment cannot repeat either field. `config_to_cfg()` constructs the
backend-specific typed physics config and rejects fields from the other
backend. Launcher `--physics` may confirm the declared value but cannot switch
the file-owned backend. Use separate environment files when the same logical
task needs both backends.

Device selection is one shared runtime value. The selected typed physics config
provides the backend default (`cpu` for Default and `cuda:0` for Newton), an
optional top-level Gym `device` overrides it, and an explicitly supplied CLI
`--device` wins last. A config-backed launcher leaves `--device` unset by
default, so omission preserves the authored/backend value. `config_to_cfg()`
passes the resolved value through `SimulationManagerCfg`, and `BaseEnv` tensors
use the manager's resulting device; there is no separate environment-device
setting.

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

## EmbodiedEnv Lifecycle

### Construction (`__init__`)

```
EmbodiedEnv.__init__(cfg)
  ├── bind extensions → cfg + self
  ├── init manager slots to None
  ├── super().__init__(cfg)  →  BaseEnv.__init__
  │     ├── set seed, compute frequencies
  │     ├── _setup_scene()
  │     │     ├── create SimulationManager
  │     │     ├── _setup_robot()  → Robot + single_action_space
  │     │     ├── _prepare_scene()  → lights, background, objects
  │     │     └── _setup_sensors() → sensors dict
  │     ├── SimulationManager.prepare() (backend-neutral readiness boundary)
  │     ├── open window (if not headless)
  │     └── _init_sim_state()
  │           ├── _apply_functor_filter() (strip visual rand if configured)
  │           ├── create EventManager   (if cfg.events)
  │           │     └── apply "startup" mode
  │           ├── create ObservationManager (if cfg.observations)
  │           ├── create RewardManager      (if cfg.rewards)
  │           └── create ActionManager      (if cfg.actions)
  │                 └── override single_action_space
  ├── create DatasetManager (if cfg.dataset and not filter_dataset_saving)
  └── init rollout buffer (if cfg.init_rollout_buffer)
```

### `reset(seed, options)`

```
reset(options)
  ├── is_task_success() → save status before resetting
  ├── sim.reset_objects_state(env_ids, excluded_uids)
  ├── _initialize_episode(env_ids)
  │     ├── dataset_manager.apply("save") for successful episodes, eligible
  │     │   segment-fragment rows, or an explicit dataset-only
  │     │   ``commit_env_ids`` subset during a final vector batch
  │     ├── event_manager.apply("reset", env_ids)
  │     ├── observation_manager.reset(env_ids)
  │     └── reward_manager.reset(env_ids)
  ├── _elapsed_steps[env_ids] = 0
  └── return get_obs(), get_info()
```

### `step(action)`

```
step(action)
  ├── _preprocess_action(action)
  │     ├── raw action → ActionManager "pre"
  │     ├── ControllerAction → unwrap and skip "pre"
  │     └── _prepare_controller_action() validation
  ├── _step_action(action)           # subclass sends control to sim
  ├── sim.update(dt, sim_steps_per_control)
  ├── _update_sim_state()            # event_manager "interval" mode
  ├── get_obs()
  │     ├── robot.get_proprioception()[:, active_joint_ids]
  │     ├── _get_sensor_obs()
  │     └── _extend_obs()            # ObservationManager.compute()
  ├── get_info() → evaluate()
  ├── get_reward() + _extend_reward()  # RewardManager.compute()
  ├── _postprocess_action(action)
  ├── elapsed_steps += 1
  ├── compute terminateds (success | fail), truncateds (time limit)
  ├── _hook_after_sim_step()         # rollout buffer write
  └── auto-reset done envs → reset(reset_ids)
```

---

## Manager Integration

Managers are **optional** — set the corresponding `EmbodiedEnvCfg` field to
wire one in. Each manager follows the Functor/FunctorCfg pattern (see
`manager-functor` topic).

| Manager | Config field | Created in | Called during |
|---|---|---|---|
| `EventManager` | `cfg.events` | `_init_sim_state()` | startup, reset, interval (each step) |
| `ObservationManager` | `cfg.observations` | `_init_sim_state()` | `_extend_obs()` on every `get_obs()` |
| `RewardManager` | `cfg.rewards` | `_init_sim_state()` | `_extend_reward()` on every step |
| `ActionManager` | `cfg.actions` | `_init_sim_state()` | overrides `single_action_space` |
| `DatasetManager` | `cfg.dataset` | `__init__` (after super) | `_initialize_episode()` save mode |

### Event manager modes

- `startup` — runs once after `_init_sim_state()`.
- `reset` — runs in `_initialize_episode()` for the reset env_ids.
- `interval` — runs every step in `_update_sim_state()`.

### Functor filter

`cfg.filter_visual_rand = True` strips all visual randomization functors
from the event config before the event manager is created.

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

### Minimal manual skeleton

```python
from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

@register_env("MyTask-v1", max_episode_steps=300)
class MyTaskEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)

    # Keep only behavior that cannot be expressed by task-local config here.
```

---

## Wrappers

| Wrapper | Location | Purpose |
|---|---|---|
| `NoFailWrapper` | `envs/wrapper/no_fail.py` | Forces `is_task_success() → True` |
| `TimeLimitWrapper` | `utils/registration.py` | Batched truncation via `elapsed_steps >= max_episode_steps` |
| `ReplayWrapper` | `envs/wrapper/replay.py` | Replays a recorded trajectory: `kinematic` (physics off, set states), `dynamic` (feed recorded actions, physics on), `control` (interactive scrubber via `go_to_step`) |

---

## Recording & Replay

`EmbodiedEnv` can record per-object kinematic trajectories and replay them later.

- **Recording**: set `cfg.record_trajectory = True`. A dedicated per-env
  `self._traj_buffer` (TensorDict: `states` = robot root_pose+qpos, articulations,
  rigid objects; `actions` = the **pre-process** action) is written each step via
  `_write_trajectory_step` (called from `_hook_after_sim_step`). A per-env
  `self._traj_steps` counter means **async parallel envs** (different reset times)
  don't corrupt each other. `cfg.trajectory_uids` restricts which non-robot objects
  are recorded.
- **Persistence**: `env.save_trajectory(path, env_ids=None)` writes a `.pt` with
  `states`, `actions`, and `meta` (incl. per-env `lengths`). With
  `cfg.trajectory_auto_save = True` (default), trajectories auto-save to
  `<EMBODICHAIN_DEFAULT_DATA_ROOT>/trajectories/<run_id>/` at episode end and on
  `close()` (best-effort: IO errors warn + skip, never crash the episode).
- **Replay**: `ReplayWrapper(env, trajectory, mode)` wraps any `EmbodiedEnv`.
  `kinematic` disables physics and writes recorded states (obs only, exact
  reproduction); `dynamic` feeds recorded actions through `env.step` so the
  `ActionManager` re-applies the transform (faithful even with delta/eef_pose
  actions); `control` exposes `go_to_step(step)` for O(1) scrubbing. The replay
  env must use the same robot/objects/`actions` config as the recording env.
- **Decoupled from `rollout_buffer`**: the trajectory buffer is separate from the
  shared `rollout_buffer` (obs/actions/rewards) used by LeRobot/RL.
  `current_rollout_step`, LeRobot recorder, and RL mode are untouched.
- **CLI**: `run-env --replay --replay_trajectory <path> --replay_mode {kinematic,dynamic,control}`.

---

## Profiling

`BaseEnv` carries an `EnvProfiler` (`self._profiler`) that records per-section
**wall time** of the reset/step pipeline. It is a no-op (zero overhead) when
disabled, so the instrumentation stays in place unconditionally.

.. note::
   Only wall time is profiled. GPU-memory profiling has been temporarily
   removed (entry parameter, recording, and report output).

### Config

`EnvCfg.profiler: EnvProfilerCfg | None = None` (None = off). Fields:

| Field | Default | Purpose |
|---|---|---|
| `enable_time` | `True` | Per-section mean/min/max/std wall time |
| `sync_cuda` | `False` | `torch.cuda.synchronize()` at section boundaries for accurate GPU time (higher overhead) |
| `warmup_steps` | `5` | Discard first N top-level step/reset samples |
| `nvtx` | `False` | Also push NVTX ranges (named in `nsys` timelines) |
| `output_path` | `None` | Dump JSON report to this path on `report()` |

CLI: `run-env --profile [--profile_output prof.json]`. In code:
`cfg.profiler = EnvProfilerCfg(enable_time=True, ...)`.

### Instrumented sections

Section names are hierarchical (built from the active call stack); a parent's
time includes its children.

- **step** → `preprocess_action`, `step_action`, `sim_update`, `update_sim_state`
  (`event_interval`), `get_obs` (`proprio`, `sensor` (`render_camera_group`,
  `sensor_fetch`), `extend` (`obs_compute`)), `get_info`, `reward`
  (`reward_compute`), `postprocess_action`, `hook_after` (`rollout_write`,
  `trajectory_write`), `auto_reset`
- **reset** → `is_task_success`, `reset_objects_state`, `initialize_episode`
  (`dataset_save`, `record_camera_save`, `trajectory_save`, `event_reset`,
  `obs_reset`, `reward_reset`, `dataset_reset`), `get_obs`

`reset()` called during step's auto-reset does **not** open a duplicate root;
its children attribute to `step.auto_reset.*`.

### Per-functor breakdown

Every registered **event** and **observation** functor is additionally timed by
name via `ManagerBase._call_functor` (centralized in the base manager; the
event/obs `apply`/`compute` loops call through it). Each functor's section is
its config attribute name, nesting under the manager call site:

- `step.update_sim_state.event_interval.<functor>` (e.g. `record_camera`)
- `reset.initialize_episode.event_reset.<functor>` (e.g. `init_bottle_pose`)
- `step.get_obs.extend.obs_compute.<functor>` (e.g. `norm_robot_eef_joint`)

`calls` reflects the firing count -- interval event functors fire every
`interval_step` (so a `interval_step=10` functor shows `calls = num_steps / 10`).
The same obs functor appears under both `step.get_obs.*` and `reset.get_obs.*`.
Zero overhead when profiling is disabled. Reward/action/dataset functors are not
yet wired through the helper.

### Report

`env._profiler.report()` prints a tree (calls / mean / min / max / std / total /
`%par` of parent, sorted by total within each parent; `(other)` = parent total
minus measured children). It is also flushed automatically in `close()` **before
`sim.destroy()`** (which exits the process). `%par` is relative to the immediate
parent; the first `warmup_steps` samples are discarded.

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: "Env X not found in registry"` | Task entry-point package not imported → `@register_env` never ran | Check the `embodichain.tasks` entry point and package import |
| Config-owned Task Program ID is missing | `gym.make(id)` ran before its Gym config passed through `config_to_cfg()` | Load/build the environment config first, then construct the registered ID |
| Config-owned ID reports a different integration | The same process already registered that ID with different integration data or episode limit | Choose a new ID or keep the declaration identical; do not override it |
| `RuntimeError: non json dumpable kwargs` | Passing class/type objects to `@register_env(…, kwarg=SomeClass)` | Use string keys + lookup mapping instead |
| `single_action_space is None` | `_setup_robot()` didn't set `self.single_action_space` | Set it before returning the Robot |
| `_setup_robot()` returns `None` | Forgot to return the Robot instance | Ensure `return robot` |
| Observation/reward manager has no effect | `cfg.observations` / `cfg.rewards` left as `None` | Set the manager config in your `EmbodiedEnvCfg` subclass |
| Visual randomization still active during debug | `filter_visual_rand` not set | Set `cfg.filter_visual_rand = True` |
| Dataset not saving | `filter_dataset_saving = True` or no `cfg.dataset` | Check both flags |
| Rollout buffer overflow warning | `max_episode_steps` < actual episode length | Increase `max_episode_steps` or check termination logic |
| Controller-ready action is transformed twice | Runtime output was passed as a raw tensor | Wrap it in `ControllerAction`; the environment skips only `pre` terms |
| Controller action rejected before robot control | Batch, control key, dtype, or active/full joint width is invalid | Fix the producer at the controller boundary; do not bypass validation |
| `Env X already registered` warning | Duplicate import or re-registration | Use `override=True` in tests/scripts |
