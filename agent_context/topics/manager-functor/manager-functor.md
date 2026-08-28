# Manager / Functor Pattern

## Entry Points

| What | Path |
|------|------|
| Base classes (`ManagerBase`, `Functor`) | `embodichain/lab/gym/envs/managers/manager_base.py` |
| All config classes (`FunctorCfg`, `EventCfg`, `ObservationCfg`, `RewardCfg`, `ActionTermCfg`, `DatasetFunctorCfg`, `SceneEntityCfg`) | `embodichain/lab/gym/envs/managers/cfg.py` |
| Observation manager + built-in functors | `managers/observation_manager.py`, `managers/observations.py` |
| Reward manager + built-in functors | `managers/reward_manager.py`, `managers/rewards.py` |
| Event manager + built-in functors | `managers/event_manager.py`, `managers/events.py` |
| Action manager + built-in terms | `managers/action_manager.py`, `managers/actions.py` |
| Dataset manager + built-in recorders | `managers/dataset_manager.py`, `managers/datasets.py`, `managers/async_datasets.py` |
| Randomization functors (event sub-type) | `managers/randomization/` (spatial, visual, physics, geometry) |

All paths relative to `embodichain/lab/gym/envs/`.

---

## Overview

Managers orchestrate collections of **functors** that run at specific points in the environment step loop.
Each manager owns a typed config (`@configclass`) whose attributes are `FunctorCfg` (or subclass) instances.
At init, the manager resolves every `FunctorCfg.func` (string → callable or class → instance), validates argument signatures against `FunctorCfg.params`, resolves `SceneEntityCfg` objects to scene indices, and groups functors by `mode`.

**Key invariant**: The config attribute name becomes the functor's unique identifier within that manager.

---

## Manager Types

| Manager | Config class per functor | Modes | `compute` / `apply` signature (beyond `self`) |
|---------|------------------------|-------|-----------------------------------------------|
| `ObservationManager` | `ObservationCfg` | `modify`, `add` | `compute(obs) → EnvObs` |
| `RewardManager` | `RewardCfg` | `add`, `replace` | `compute(obs, action, info) → (reward, info_dict)` |
| `EventManager` | `EventCfg` | `startup`, `reset`, `interval`, user-defined | `apply(mode, env_ids)` |
| `ActionManager` | `ActionTermCfg` | `pre`, `post` | `process_action(action, mode) → EnvAction` |
| `DatasetManager` | `DatasetFunctorCfg` | `save` | `step(obs, action, done, info)` |

---

## Dataset Saving (LeRobot)

`DatasetManager` runs in the `save` mode on `env.reset()` (`_initialize_episode`) and persists completed episodes via a recorder functor. Two recorders ship:

| Recorder | File | Behavior |
|----------|------|----------|
| `LeRobotRecorder` | `managers/datasets.py` | Synchronous. `__call__` runs convert + `add_frame` + `save_episode` inline, blocking `env.reset()`. Default; base class for the async variant. |
| `AsyncLeRobotRecorder` | `managers/async_datasets.py` | Subclass. `__call__` clones the rollout-buffer slice (obs+actions) to CPU, enqueues it, and returns immediately. A single daemon worker thread drains the queue and runs the same `_save_single_episode` path. `finalize()` drains then calls `dataset.finalize()`. |

**Save flow**: `env.step` writes each frame into `rollout_buffer` (`_hook_after_sim_step`). On truncation the caller does `env.reset(options={"save_data": True})` -> `_initialize_episode` -> `DatasetManager.apply("save", env_ids)`. For a final partial vector batch, `run-env` performs one full reset with `save_data=False` and `commit_env_ids`, so only selected **dataset** rows are persisted while whole-world reset events remain safe; camera and trajectory recorders retain their normal discard behavior. `DatasetFunctorCfg.save_failed_episodes=True` saves every env on every ordinary reset (not only successes). `env.close()` -> `dataset_manager.finalize()` flushes any remaining buffer.

**Two independent speed levers** (both honor `image_writer_threads` / `image_writer_processes` in `params`, wired through to `LeRobotDataset.create()` -> lerobot `AsyncImageWriter`):
- Opt A: `LeRobotRecorder` + `image_writer_threads=4` - per-frame PNG writes offloaded to a thread pool. ~2.5x faster, no background thread, bounded memory.
- Opt B: `AsyncLeRobotRecorder` - whole-episode save offloaded to a worker; sim never blocks. Use for `num_envs > 1` collection.

**When to use which**:
- Single env / debug / memory-constrained -> `LeRobotRecorder` (sync).
- Parallel collection (`num_envs > 1`) -> `AsyncLeRobotRecorder` + `image_writer_threads=4`.

**Correctness invariants for the async recorder** (do not break these when editing):
- The buffer slice is **cloned in the caller thread** before enqueue - the worker must not hold a view into `rollout_buffer` (it is cleared/reused on reset).
- **Single worker** only - `LeRobotDataset` is not thread-safe and FIFO order must be preserved for `episode_index`.
- `finalize()` must drain the queue before `dataset.finalize()`.
- `__call__` accepts `**kwargs` because `DatasetManager.apply` passes `**functor_cfg.params` (includes construction-only params like `image_writer_threads`); `manager_base._resolve_common_functor_cfg` tolerates `**kwargs`.

**Gotcha**: `env.close()` calls `sim.destroy()`, which exits the process without returning to Python. To finalize the dataset without killing the process, call `env.dataset_manager.finalize()` directly. Scripts running multiple envs must use one subprocess per env.

Benchmark: `scripts/benchmark/data_pipeline/benchmark_lerobot_save.py` (uses the `StayStillSave-v1` env). At 4 envs x 2 eps x 100 steps / 480x640 / 800 frames: sync 57.4s -> Opt A 22.0s (2.6x) -> Opt B 56.6s total but sim unblocked -> Opt A+B 20.8s (2.8x) + sim unblocked. Sync sim-stall grows linearly with `num_envs`; async sim-stall stays near zero.

---

## FunctorCfg Pattern

```python
from embodichain.lab.gym.envs.managers.cfg import FunctorCfg, SceneEntityCfg

FunctorCfg(
    func=my_function_or_class,   # Callable | str (dot-path) | Functor subclass
    params={                      # kwargs forwarded to func after positional args
        "entity_cfg": SceneEntityCfg(uid="cube"),
        "scale": 1.0,
    },
    extra={"shape": (3,)},        # metadata (e.g. observation output shape)
)
```

- `func` can be a **string** (resolved via `string_to_callable` at init) or a direct reference.
- `params` values of type `SceneEntityCfg` are auto-resolved to joint/body indices when the sim starts.
- Subclass configs add fields: `EventCfg.mode`, `EventCfg.interval_step`, `RewardCfg.weight`, `ObservationCfg.name`, `ActionTermCfg.mode`.

---

## Two Functor Styles

### Function-style

Plain function. The manager calls it directly with positional env args + `**params`.

**Observation functor** (mode `"add"`):
```python
def get_object_pose(
    env: EmbodiedEnv,
    obs: EnvObs,            # positional: current obs dict
    entity_cfg: SceneEntityCfg = None,
    to_matrix: bool = True,
) -> torch.Tensor:
    ...
```

**Reward functor**:
```python
def distance_between_objects(
    env: EmbodiedEnv,
    obs: dict,
    action: EnvAction,
    info: dict,
    source_entity_cfg: SceneEntityCfg = None,
    target_entity_cfg: SceneEntityCfg = None,
    exponential: bool = False,
    sigma: float = 1.0,
) -> torch.Tensor:
    ...
```

**Event functor**:
```python
def randomize_mass(
    env: EmbodiedEnv,
    env_ids: Sequence[int] | None,
    entity_cfg: SceneEntityCfg = None,
    mass_range: tuple[float, float] = (0.5, 2.0),
) -> None:
    ...
```

### Class-style

Inherit from `Functor`. Manager instantiates the class at init (`func=MyClass` → `MyClass(cfg, env)`), then calls the instance on each step.

```python
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg

class compute_exteroception(Functor):
    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        # allocate persistent buffers here

    def __call__(self, env: EmbodiedEnv, obs: EnvObs, **params) -> torch.Tensor:
        # return observation tensor
        ...

    def reset(self, env_ids=None) -> None:
        # optional: reset internal state
        ...
```

**When to use class-style**: functor needs persistent state, buffers, or expensive one-time setup.

**`ActionTerm`** is a special class-style functor (inherits `Functor`) that must implement `process_action(action) → EnvAction`, `input_key` (property), and `action_dim` (property).

---

## Manager Lifecycle

### Initialization (env `__init__`)
1. Task config instantiates manager configs (e.g., `ObservationCfg(...)` per functor).
2. Manager `__init__` calls `_prepare_functors()`:
   - Iterates config attributes, calls `_resolve_common_functor_cfg()` per functor.
   - Validates `func` is callable, checks param signatures match (`min_argc` varies by manager).
   - Resolves `SceneEntityCfg` → joint/body indices (deferred until sim starts via callback).
   - If `func` is a class, instantiates it: `func = func(cfg=functor_cfg, env=env)`.
   - Groups functors by `mode` into `_mode_functor_names` / `_mode_functor_cfgs`.

### Per-step execution (env `step`)
1. **Actions**: raw actions use `ActionManager.process_action(..., mode="pre")`;
   explicit `ControllerAction` values skip `pre`. Both paths then pass through
   `EmbodiedEnv._prepare_controller_action()` before robot control.
2. **Sim step**: physics advances.
3. **Observations**: `ObservationManager.compute(obs)` → updated obs dict.
4. **Rewards**: `RewardManager.compute(obs, action, info)` → `(total_reward, reward_info)`.
5. **Events**: `EventManager.apply("interval")` for interval-mode functors (step counter checked internally).

After robot control and simulation, configured action terms in `post` mode run
for both raw and `ControllerAction` inputs. `ControllerAction` is therefore not
an alternate action manager; it marks that only the raw-policy preprocessing
stage has already completed.

### On reset
1. `EventManager.apply("reset", env_ids)` — domain randomization etc.
2. All managers' `.reset(env_ids)` — resets class-style functors via `functor.reset(env_ids)`.

### Per-functor profiling

Event and observation functors are timed individually and automatically:
`EventManager.apply` / `ObservationManager.compute` invoke each functor through
`ManagerBase._call_functor(name, cfg, *args)`, which opens an
`env._profiler.section(name)` around `cfg.func(*args, **cfg.params)`. The
section name is the functor's config attribute name and nests under the active
call site (e.g. `step.update_sim_state.event_interval.record_camera`). No-op
when env profiling is disabled. See the `env-framework` topic's *Profiling*
section. Reward/action/dataset functors are not yet wired through the helper.

---

## Adding New Functors

Use the **`/add-functor`** skill. It scaffolds:
- Correct function/class signature for the target manager type.
- Proper imports and `__all__` export.
- Placement in the right module (`observations.py`, `rewards.py`, `events.py`, `randomization/`, etc.).

Manual checklist if not using the skill:
1. Write function or `Functor` subclass in the appropriate module.
2. Add to `__all__` in that module.
3. Register in the task's config class as a `FunctorCfg` / `ObservationCfg` / `RewardCfg` / `EventCfg` attribute.
4. Ensure `params` keys match the function's keyword arguments (excluding positional env args).

---

## Common Failure Modes

| Symptom | Cause |
|---------|-------|
| `TypeError: ... is not of type FunctorCfg` | Config attribute is not a `FunctorCfg` subclass (e.g., raw dict or wrong type). |
| `AttributeError: ... is not callable` | `func` is a string that failed to resolve or points to a non-callable. |
| `ValueError: expects mandatory parameters ...` | `params` dict keys don't match the functor's non-default kwargs (after the positional env args). |
| `ValueError: scene entity '...' does not exist` | `SceneEntityCfg.uid` doesn't match any asset in `SimulationManager`. Check spelling / scene setup. |
| `TypeError: ... is not of type ManagerTermBase` | Class-style functor doesn't inherit from `Functor`. |
| Stale tensor references / data mutation | Function-style functor returns un-cloned mutable tensor. Clone before returning. |
| `interval` event fires for wrong envs | `EventCfg.is_global=False` (default) means per-env counters; set `True` for global interval. |
| Observation shape mismatch | `extra={"shape": ...}` doesn't match actual returned tensor shape. |
