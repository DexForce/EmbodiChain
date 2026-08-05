# env-framework

> Topic: Environment framework — BaseEnv / EmbodiedEnv class hierarchy,
> task registration, manager wiring, and lifecycle.

---

## Entry Points

| File | Role |
|---|---|
| `embodichain/lab/gym/envs/base_env.py` | `BaseEnv(gym.Env)` + `EnvCfg` — low-level env loop |
| `embodichain/lab/gym/envs/embodied_env.py` | `EmbodiedEnv(BaseEnv)` + `EmbodiedEnvCfg` — modular task base class |
| `embodichain/lab/gym/utils/registration.py` | `@register_env` decorator + `REGISTERED_ENVS` registry + `make()` |
| `embodichain/lab/gym/envs/tasks/__init__.py` | All concrete task imports (forces registration on import) |
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
- `extensions` dict entries are set as attributes on both cfg and env instance.

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
4. `kwargs` passed to `@register_env` must be **JSON-serialisable** (no
   classes/types). A `RuntimeError` is raised otherwise.
5. Use `override=True` to re-register an existing uid (useful in scripts/tests).

### Gym ID convention

Format: `<TaskName>-v<N>` (e.g. `PourWater-v3`, `PushCubeRL`).
RL tasks sometimes drop the `-v<N>` suffix (`CartPoleRL`, `PushCubeRL`).

### Instantiation

```python
from embodichain.lab.gym.utils.registration import make
env = make("MyTask-v1", cfg=my_cfg)
```

Or via gymnasium: `gym.make("MyTask-v1")`.

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
  │     ├── init GPU physics (if CUDA)
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
  │     ├── dataset_manager.apply("save") for successful episodes
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

Use the `/add-task-env` skill. It scaffolds:

1. A new file under `embodichain/lab/gym/envs/tasks/<category>/`.
2. `@register_env("<GymId>")` decorator on the class.
3. `EmbodiedEnvCfg` subclass with robot, sensor, object configs.
4. Stub implementations of `_setup_robot()`, `evaluate()`, `get_reward()`.
5. Import entry in `tasks/__init__.py`.
6. Test stub.

### Minimal manual skeleton

```python
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

@configclass
class MyTaskCfg(EmbodiedEnvCfg):
    robot: RobotCfg = MISSING

@register_env("MyTask-v1", max_episode_steps=300)
class MyTaskEnv(EmbodiedEnv):
    def __init__(self, cfg: MyTaskCfg = MyTaskCfg(), **kwargs):
        super().__init__(cfg, **kwargs)

    def _setup_robot(self, **kwargs) -> Robot:
        # load robot, set self.single_action_space
        ...

    def evaluate(self, **kwargs) -> dict:
        return {"success": ..., "fail": ...}

    def get_reward(self, obs, action, info) -> torch.Tensor:
        ...
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
| `KeyError: "Env X not found in registry"` | Task module not imported → `@register_env` never ran | Add import to `tasks/__init__.py` |
| `RuntimeError: non json dumpable kwargs` | Passing class/type objects to `@register_env(…, kwarg=SomeClass)` | Use string keys + lookup mapping instead |
| `single_action_space is None` | `_setup_robot()` didn't set `self.single_action_space` | Set it before returning the Robot |
| `_setup_robot()` returns `None` | Forgot to return the Robot instance | Ensure `return robot` |
| Observation/reward manager has no effect | `cfg.observations` / `cfg.rewards` left as `None` | Set the manager config in your `EmbodiedEnvCfg` subclass |
| Visual randomization still active during debug | `filter_visual_rand` not set | Set `cfg.filter_visual_rand = True` |
| Dataset not saving | `filter_dataset_saving = True` or no `cfg.dataset` | Check both flags |
| Rollout buffer overflow warning | `max_episode_steps` < actual episode length | Increase `max_episode_steps` or check termination logic |
| `Env X already registered` warning | Duplicate import or re-registration | Use `override=True` in tests/scripts |
