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
