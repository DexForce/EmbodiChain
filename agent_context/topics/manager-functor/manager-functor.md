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
| Dataset manager + built-in recorders | `managers/dataset_manager.py`, `managers/datasets.py` |
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
| `ActionManager` | `ActionTermCfg` | `pre`, `post` | `process_actions(actions) → EnvAction` |
| `DatasetManager` | `DatasetFunctorCfg` | `save` | `step(obs, action, done, info)` |

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
1. **Actions**: `ActionManager.process_actions(raw_actions)` → robot control commands.
2. **Sim step**: physics advances.
3. **Observations**: `ObservationManager.compute(obs)` → updated obs dict.
4. **Rewards**: `RewardManager.compute(obs, action, info)` → `(total_reward, reward_info)`.
5. **Events**: `EventManager.apply("interval")` for interval-mode functors (step counter checked internally).

### On reset
1. `EventManager.apply("reset", env_ids)` — domain randomization etc.
2. All managers' `.reset(env_ids)` — resets class-style functors via `functor.reset(env_ids)`.

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
