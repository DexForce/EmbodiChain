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
At init, the manager resolves every `FunctorCfg.func` (string → callable or class → instance), validates argument signatures against `FunctorCfg.params`, and resolves `SceneEntityCfg` objects to scene indices. Event, observation, reward, and dataset managers group functors by `mode`; `ActionManager` instead preserves configuration order as one flat policy layout.

**Key invariant**: The config attribute name becomes the functor's unique identifier within that manager.

---

## Manager Types

| Manager | Config class per functor | Modes | `compute` / `apply` signature (beyond `self`) |
|---------|------------------------|-------|-----------------------------------------------|
| `ObservationManager` | `ObservationCfg` | `modify`, `add` | `compute(obs) → EnvObs` |
| `RewardManager` | `RewardCfg` | `add`, `replace` | `compute(obs, action, info) → (reward, info_dict)` |
| `EventManager` | `EventCfg` | `startup`, `reset`, `interval`, user-defined | `apply(mode, env_ids)` |
| `ActionManager` | `ActionTermCfg` | Ordered flat slices | `process_action(action)`, then `apply_action()` |
| `DatasetManager` | `DatasetFunctorCfg` | `save` | `apply(mode, env_ids)` |

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
- Subclass configs add fields such as `EventCfg.mode`, `EventCfg.interval_step`, `RewardCfg.weight`, and `ObservationCfg.name`. `ActionTermCfg` has no mode field and may declare a versioned semantic `contract` resolved by the current action implementation registry.

---

## Callable contracts

Function-style functors receive the arguments for their manager, followed by
`**params`. Class-style functors inherit `Functor`, initialize with `(cfg, env)`,
and implement the same call contract; use them for persistent state or buffers.

| Functor | Positional inputs after `self`, if any | Result |
|---|---|---|
| Observation | `env, obs` | Added tensor or modified observations, according to mode |
| Reward | `env, obs, action, info` | Reward tensor |
| Event / randomization | `env, env_ids` | Side effects; normally `None` |
| Dataset | `env, env_ids` | Persistence side effects |
| Action term | `process_actions(actions)`, then `apply_actions()` | Stores one raw slice, then writes its owned robot resources |

An `ActionTerm` exposes its dimension and Box space, current raw and processed
buffers, command type, controlled joint IDs, and immutable descriptor. The
descriptor carries a versioned semantic contract ID when one is registered;
the concrete Python class is an implementation detail. Use
`/add-functor` for the precise scaffold; do not apply the event signature to
observations, rewards or action terms.

## Lifecycle

Managers resolve callable strings, validate parameters, resolve `SceneEntityCfg`
UID/joint/body selectors, and instantiate class functors during preparation.
The config attribute name is the stable functor identity.

For a flat policy tensor, the environment calls manager processing, manager
application, physics, event `interval`, observation, and reward in that order.
`ControllerAction` bypasses the manager and uses the direct controller boundary;
the full loop and reset ordering are owned by
[environment execution](../env-framework/execution.md).

`DatasetManager.apply("save", env_ids)` runs from episode initialization/reset,
not as a `step(obs, action, done, info)` method. Recorder queues, explicit commit
and abort, fragment idempotency and finalization are owned by
[data pipeline](../data-pipeline/data-pipeline.md).

Event RNG scopes and interval-counter reset behavior are owned by
[randomization](../randomization/randomization.md). Renaming a functor changes its
seed stream because the name is part of the stream identity.

Event and observation calls have per-functor profiling through
`ManagerBase._call_functor()`; reward/action/dataset calls do not yet use that
helper. See [profiling](../env-framework/profiling.md).

## Change sites and validation

Change the selected manager's dispatch for scheduling changes; change its
functor module for one operation; change `cfg.py` only for shared configuration
contracts. Test with the matching cases under `tests/gym/envs/managers/` and a
focused environment integration case when invocation order changes.

## Common failure modes

- Unresolved callable or unexpected parameters: inspect the manager-specific
  positional contract and the callable signature before editing `params`.
- Missing entity: check `SceneEntityCfg.uid` against the live simulation registry.
- Observation shape mismatch: align the returned tensor with `extra.shape`.
- State leaks across resets: implement the stateful functor's reset behavior.
- Incorrect interval rows: distinguish global and per-row interval counters.
- Corrupted asynchronous recordings: snapshot owned CPU payloads before rollout
  buffers are cleared; follow the recorder's persistence contract.

### Flat action ownership

`ActionManager` concatenates term Box bounds in configuration order, validates
the whole `(num_envs, action_dim)` floating tensor before changing state, and
binds each term to an `ActionDescriptor` slice. Terms that write the same
command type cannot own overlapping joint IDs. Each term applies only its
resolved non-mimic selection; this makes arm and gripper terms composable and
leaves a separate resource namespace available for future tendon/synergy terms.
Joint-command descriptors must name every controlled ID in the same order, so
third-party terms cannot hide overlap. During sticky vectorized demos, the
manager replaces inactive qpos rows with measured holds and inactive qvel/qf
rows with zero before applying terms.

`DefaultJointPositionAction` implements the generic
`joint_position.default_offset@1` contract and exposes `raw_actions`,
`previous_raw_actions`, and `position_bias` for locomotion state construction.
`EefPoseAction` owns selected
arm IK and `ParallelGripperAction` maps one scalar to explicit independent
gripper joints. ActionManager dispatches selected-row reset to every term so
untouched rows retain their action history.
