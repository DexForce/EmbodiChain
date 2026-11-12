# Manager / Functor Pattern

## Entry Points

| What | Path |
|------|------|
| Base classes (`ManagerBase`, `Functor`) | `embodichain/lab/gym/envs/managers/manager_base.py` |
| All config classes (`FunctorCfg`, `EventCfg`, `ObservationCfg`, `RewardCfg`, `ActionTermCfg`, `DatasetFunctorCfg`, `SceneEntityCfg`) | `embodichain/lab/gym/envs/managers/cfg.py` |
| Observation manager + built-in functors | `managers/observation_manager.py`, `managers/observations.py` |
| Reward manager + built-in functors | `managers/reward_manager.py`, `managers/rewards.py` |
| Event manager + built-in functors | `managers/event_manager.py`, `managers/events.py` |
| Episode camera recording events | `managers/record.py` |
| Action manager + built-in terms | `managers/action_manager.py`, `managers/actions.py` |
| RL termination manager + condition functors | `managers/termination_manager.py`, `managers/conditions.py` |
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
| `TerminationManager` | `ConditionCfg` in `RLTaskCfg` | success/failure/truncation groups | `compute(time_out, ignore_terminations=...)` then cached flags/snapshots |
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
| Condition | `env, **params` | `ConditionResult` with truth, validity and measurements |
| Event / randomization | `env, env_ids` | Side effects; normally `None` |
| Dataset | `env, env_ids` | Persistence side effects |
| Action term | `process_actions(actions)`, then `apply_actions()` | Stores one raw slice, then writes its owned robot resources |

An `ActionTerm` exposes its dimension and Box space, current raw and processed
buffers, command type, controlled joint IDs, and immutable descriptor. The
descriptor carries a versioned semantic contract ID when one is registered;
the concrete Python class is an implementation detail. Use
`/add-functor` for the precise scaffold; do not apply the event signature to
observations, rewards or action terms.

`normalize_robot_joint_data` selects either observation-column `joint_ids` or
named `control_parts`. Named selection resolves current robot joint IDs and
maps them into `env.active_joint_ids`; observed mimic joints are included.
Use this selector for reusable arm/gripper observations across robot DOFs and
state orders. Physical joint IDs are not observation-column indices.

## Raw observations and consumer views

`managers/observation_types.py` contains the shared raw value/frame types.
`ObservationManager` owns acquisition scopes, the single semantic schema and
alias-to-field-ID binding. `ObservationCfg.consumers` defaults to policy.
Recording add terms must produce described `ObservationSample` values;
plain Tensor policy outputs retain their existing contract. The two consumers
are policy and recording; evidence adapters read qualified recording fields.
`ObservationManager` resolves both configured and required replay terms before
matching their source functions/parameters, including callable strings and
omitted default arguments. Env only supplies the declarations. Recording-only
fields never enlarge policy observations or policy rollout allocation.

`observations.get_entity_state` derives native UID, quantity, arena/joint frame,
representation, shape, dtype and joint order from its source. Immutable
schema descriptors are reused by semantic value; no simulation handles enter
the bounded cache. Pose acquisition uses the resolved UID without constructing
another `SceneEntityCfg` during each step. Explicit boundaries
share its original acquisition across aliases and existing pose functors.
Policy transformations receive separate tensors; zero-filled missing sources
retain false validity and reasons. Preserve configured policy functor/RNG order.
The host rollout writer stores raw values, availability, clocks, acquisition
errors and cursors directly in the existing TensorDict. There is no history
instance or separate bind/seed/append manager. `ObservationManager` binds read
access with `bind_rollout_storage`, keeps only the semantic schema and source
aliases, and serves `read_history`/`snapshot_history` as indexed snapshot reads.
`read_history_window` copies exact inclusive per-environment intervals from that
same store, including validity, acquisition errors, revisions and timestamps.
Intervals retain their native row order and lengths; no padding, new schema,
physical capture or loop of single-slot queries is introduced. Missing slots,
closed owners or stale/uninitialized episodes reject the read.
`read_history_step_counts` returns only qualified completed boundaries from the
same store, without copying values or invoking acquisition. Aliases resolve
through the same schema and returned tensors cannot mutate the stored history.
Evidence adapters bind roles through `manager.schema`. Rollout raw writes and
indexing helpers live beside the existing buffer utilities in `utils/gym_utils.py`.
`export_observation_history` projects the same schema for compatible output.
Contact windows remain sensor-owned and cannot be replaced with a control-rate
pose/Boolean. History timing and host lifecycle belong to
[environment recording](../env-framework/env-framework.md#expert-trajectory-contract).

## Lifecycle

Managers resolve callable strings, validate parameters, resolve `SceneEntityCfg`
UID/joint/body selectors, and instantiate class functors during preparation.
The config attribute name is the stable functor identity.

For a flat policy tensor, the environment calls manager processing, manager
application, physics, event `interval`, observation, and reward in that order.
A term can opt into physics-substep refresh.
`ActionManager.get_substep_callback(num_substeps)` creates a callback scoped to
one control interval, or returns None when no refresh is needed. The manager
counts remaining steps, refreshes opted-in terms only before another physics
step, and preserves inactive demo rows' measured holds. The first command is
processed and applied normally. `EmbodiedEnv` connects the callback after sensor
sampling; simulation advancement and publication remain environment-owned.
`expert_trajectory.use_action_manager` routes position-only expert commands
through `ActionManager.process_joint_targets`. Relative substep DLS and binary
gripper terms support that route, retaining exact FK targets and the expert
storage schema while using the policy controller. Completed rows are masked
after target conversion so binary grippers also retain measured joint holds.
`ControllerAction` otherwise bypasses the manager for direct controller writes;
the full loop and reset ordering are owned by
[environment execution](../env-framework/execution.md).

Condition functors receive `env, **params` and return a `ConditionResult` from
`envs/objectives`. Stateful conditions use `Functor.reset(env_ids)` like other
manager terms. Built-in conditions bind rigid objects and validate geometric
parameters in their `Functor` constructors, before reset targets exist; the
manager has no task-specific condition checks. Metrics retain their schema;
snapshots do not invoke functors.
The host's termination and replay rules belong to
[configured RL execution](../env-framework/execution.md#configured-manipulation-rl-outcomes).

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

### Camera recording

`record_camera_data` is an interval event, independent of policy dataset
recording. It renders its camera group before reading the buffers because
interval events precede observation rendering. Views use arena-relative camera
extrinsics. It retains separate owned CPU frames and episode counters per row.
It captures only the event's selected `env_ids` within `max_env_num`, preserving
their original arena IDs; `None` captures all views within that limit. Selected
save/discard only touches those rows. `fps=None` follows control frequency for
an every-step recording event; an explicit fps fixes the output cadence.

Reset commits or discards frames via `save_and_clear` / `discard_and_clear`.
Close discards uncommitted frames, so a host ending without reset must explicitly
commit its last episode. Source and regression tests live in `record.py` and
`tests/gym/envs/managers/test_record.py`.

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
- Source-bound Real2Sim recordings need the synchronous receipt and typed
  action-trace contract in [Real2Sim motion](../data-pipeline/real2sim.md),
  not the ordinary reset-time recorder alone.

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

Prepared semantic asset combinations use the existing
`events.replace_assets_from_group` event's `deployment` mode at a cleared batch
boundary. It shares the original directory replacement entry point; preparation,
Task Program adaptation and reused-environment collection are owned by
[asset generalization](../asset-generalization/asset-generalization.md).
