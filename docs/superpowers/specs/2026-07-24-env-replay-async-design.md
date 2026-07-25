# Environment Task Replay - Async + Dedicated Buffer Design Spec

- **Date**: 2026-07-24
- **Status**: Draft (for review)
- **Supersedes**: the trajectory-recording/replay portions of `2026-07-24-env-replay-design.md` (PR #425). The original spec assumed synchronized parallel envs and stored `states` in the shared `rollout_buffer` with `actions` taken from the global-step `rollout_buffer["actions"]`. This spec evolves it to support **async parallel envs**, a **dedicated per-env trajectory buffer**, **pre-process action recording**, and a **default auto-save** mechanism.

## 1. Overview

Evolve the record-and-replay feature so that:

1. **Async parallel envs** - envs that terminate/reset at different times no longer corrupt each other's recording. Each env has its own step counter and length.
2. **Dedicated trajectory buffer** - trajectory data (`states` + `action`) lives in its own `self._traj_buffer`, fully decoupled from the shared `rollout_buffer` (which stays global-sync for LeRobot/RL).
3. **Pre-process action recording** - the recorded `action` is the **raw action passed to `env.step`** (before `_preprocess_action`), so dynamic replay feeds it back through `env.step` and the `ActionManager` re-applies the same transform - making dynamic replay faithful **even with an ActionManager** (previously deferred).
4. **Default auto-save** - trajectories auto-save to `~/.cache/embodichain_data/trajectories/` at episode end and on `close()`, with an explicit `save_trajectory(path)` still available.
5. **Multi-env replay** - `ReplayWrapper` replays per-env lengths (envs finish at different times).

Two replay modes unchanged in intent:
- **Pure kinematic** - physics off, write recorded `states` directly each step; obs only.
- **Dynamic** - restore initial scene, feed recorded `action` through `env.step`; physics on, full pipeline.

## 2. Scope

### In scope
- Dedicated `self._traj_buffer` (per-env `states` + `action`) + per-env `self._traj_steps`.
- Recording `states` (post-step) and `action` (pre-process) per-env.
- Auto-save to `EMBODICHAIN_DEFAULT_DATA_ROOT/trajectories/` at episode end + `close()`.
- `ReplayWrapper` with per-env-length replay (kinematic + dynamic).
- Pre-process action capture via `_preprocess_action` override (enables ActionManager in dynamic replay).

### Out of scope (unchanged from original)
- RL `"rl"`-mode rollout buffer / `set_rollout_buffer` injection path.
- `qvel` recording.
- `current_rollout_step` and the shared `rollout_buffer` (`obs/actions/rewards`) - **untouched**. LeRobot recorder, RL collector, reward functors, RL benchmark scripts all unchanged.

## 3. Background - what changes from PR #425

| Aspect | PR #425 (current) | This spec |
|---|---|---|
| `states` storage | `rollout_buffer["states"]` | dedicated `self._traj_buffer["states"]` |
| `action` source | `rollout_buffer["actions"]` (global `current_rollout_step`) | `self._traj_buffer["action"]` (per-env `_traj_steps`), **pre-process** action |
| Step counter | global `current_rollout_step` (resets on ANY env reset) | per-env `self._traj_steps` (resets only the reset envs) |
| Parallel envs | synchronized only (async corrupts) | async supported |
| Auto-save | none (explicit `save_trajectory` only) | default auto-save to `~/.cache/embodichain_data/trajectories/` |
| Dynamic replay action | post-process qpos target, no preprocessing | pre-process action, re-preprocessed by `env.step` (ActionManager works) |
| Replay | uniform `idx` (all envs same length) | per-env lengths |

PR #425's `current_rollout_step`, `rollout_buffer`, `_write_episode_rollout_step`, LeRobot recorder, RL mode, and reward functors are NOT touched.

## 4. Architecture

```
EmbodiedEnv
  ├── rollout_buffer (obs/actions/rewards)   # UNCHANGED - global-sync, for LeRobot/RL
  ├── self._traj_buffer  (states + action)   # NEW - per-env, async, dedicated
  └── self._traj_steps   [num_envs]          # NEW - per-env step counter

ReplayWrapper(gym.Wrapper)   # per-env-length replay, both modes
```

## 5. Components and file placement

| Component | File | Responsibility |
|---|---|---|
| Trajectory buffer builder | `embodichain/lab/gym/utils/gym_utils.py` - extend `build_trajectory_states_buffer` (or new `build_trajectory_buffer`) | Build `self._traj_buffer` (nested `states` + `action` tensor from `single_action_space`), `[num_envs, max_steps, ...]`. |
| Recording | `embodichain/lab/gym/envs/embodied_env.py` | cfg fields; create `_traj_buffer`+`_traj_steps` in `__init__`; stash raw action in `_preprocess_action`; `_write_trajectory_step()` per-env from `_hook_after_sim_step`; reset `_traj_steps[env_ids]=0` in `_initialize_episode`. |
| Auto-save + persistence | `embodichain/lab/gym/envs/embodied_env.py` | `save_trajectory(path, env_ids=None)`; `_save_trajectory_for_env(env_id)` (auto-save); auto-save at episode end + `close()`. |
| Auto-reset guard | `embodichain/lab/gym/envs/base_env.py` | UNCHANGED from PR #425 (`_replay_no_auto_reset`). |
| Replay | `embodichain/lab/gym/envs/wrapper/replay.py` | `ReplayWrapper` with per-env-length replay. |
| Exports | `wrapper/__init__.py` | UNCHANGED. |

## 6. Data model

### `self._traj_buffer` (TensorDict, batch `[num_envs, max_steps]`)
```
{
  "states": {                                    # nested, per object
    "robot":          {"root_pose": [N,T,7], "qpos": [N,T,dof]},
    "articulations":  {"<uid>": {"root_pose":[N,T,7], "qpos":[N,T,dof]}, ...},
    "rigid_objects":  {"<uid>": {"pose":[N,T,7]}, ...},
  },
  "action": [N, T, action_dim],                  # pre-process action (matches single_action_space)
}
```
`self._traj_steps: torch.Tensor` shape `[num_envs]`, dtype long - each env's current write index.

### `.pt` trajectory file
```
{
  "states": <_traj_buffer["states"] sliced to [:, :max_len]>,   # padded
  "action": <_traj_buffer["action"] sliced to [:, :max_len]>,   # padded
  "meta": {
      "lengths": [int, ...],          # per-env actual recorded length (NEW)
      "num_steps": int,               # max(lengths)
      "num_envs": int,
      "dt": float,
      "active_joint_ids": [int],
      "robot_uid": str, "robot_dof": int,
      "articulation_uids": [...], "articulation_dofs": {...},
      "rigid_object_uids": [...],
      "env_ids": [int],               # which envs (for per-env auto-save files)
  },
}
```
Single-env auto-save files are `[1, length, ...]` with `meta["lengths"]=[length]`.

## 7. Data flow

### 7.1 Recording (per-env, pre-process action)
```
env.step(raw_action)
  ├─ _preprocess_action(raw_action)        # stash self._traj_raw_action = raw_action (if record_trajectory)
  │     └─ ActionManager.process_action (if any) -> resolved action
  ├─ _step_action(resolved) -> robot.set_qpos(target=True)
  ├─ sim.update (physics)
  ├─ get_obs / get_reward / ...
  └─ _hook_after_sim_step
       └─ _write_trajectory_step()         # NEW (replaces _write_trajectory_states)
            ├─ idx = arange(num_envs); st = _traj_steps; mask = st < max_steps
            ├─ _traj_buffer["states"][idx[mask], st[mask]].copy_(read_states()[mask])
            ├─ _traj_buffer["action"][idx[mask], st[mask]].copy_(_traj_raw_action[mask])
            └─ _traj_steps += 1
```
`read_states()` reads robot root_pose+qpos, articulations, rigid objects (all envs); `[mask]` selects non-overflow envs.

### 7.2 Reset (per-env)
```
_initialize_episode(env_ids):
  (auto-save for env_ids - see 7.3, BEFORE clearing)
  self._traj_steps[env_ids] = 0            # only the reset envs; others keep their count
  (existing current_rollout_step = 0 unchanged)
```

### 7.3 Auto-save (default mechanism)
- Default dir: `cfg.trajectory_save_dir` or `EMBODICHAIN_DEFAULT_DATA_ROOT/trajectories/{run_id}/`, where `run_id` is set once at env init (e.g. `datetime.now().strftime("%Y%m%d_%H%M%S")`) to avoid cross-run filename collisions in the shared cache dir.
- Triggered when `cfg.record_trajectory and cfg.trajectory_auto_save`:
  - At episode end (`_initialize_episode(env_ids)`): for each `env_id` in `env_ids` with `_traj_steps[env_id] > 0`, call `_save_trajectory_for_env(env_id)` -> writes `{dir}/traj_env{env_id}_{count:06d}.pt`, increments `self._traj_save_count`.
  - On `close()`: for each env with `_traj_steps > 0`, auto-save (finalize), mirroring LeRobot's `finalize()`.
- `_save_trajectory_for_env(env_id)`: picks the filename, calls `save_trajectory(path, env_ids=[env_id])`.
- Explicit `save_trajectory(path, env_ids=None)`: saves env_ids (default all envs) into one `.pt` (padded + `meta["lengths"]`). Returns `path`. The user chooses the path (no run-id).

### 7.4 Replay - kinematic (`mode="kinematic"`)
```
reset():  env.reset() -> enable_physics(False) -> _set_all_states(states[:, 0]) -> _replay_steps=0
step():
  st = _replay_steps.clamp(max=_lengths-1)        # finished envs hold last state
  _set_all_states(states[arange(N), st])
  sim.update (physics off)
  obs = get_obs()
  _replay_steps = (_replay_steps + 1).clamp(max=_lengths)
  trunc = _replay_steps >= _lengths               # per-env
  return (obs, zeros, zeros, trunc, {})
```

### 7.5 Replay - dynamic (`mode="dynamic"`)
```
reset():  env.reset() -> _set_all_states(states[:, 0]) -> enable_physics(True) -> _replay_no_auto_reset=True -> _replay_steps=0
step():
  st = _replay_steps.clamp(max=_lengths-1)        # finished envs re-apply last action (hold)
  action = traj["action"][arange(N), st]          # pre-process action
  obs, reward, term, trunc, info = env.step(action)   # ActionManager re-preprocesses
  _replay_steps = (_replay_steps + 1).clamp(max=_lengths)
  trunc = trunc | (_replay_steps >= _lengths)     # per-env
  return (obs, reward, term, trunc, info)
```
`_lengths = meta["lengths"]` broadcast to `num_envs` (single-env trajectory -> all envs use that length).

## 8. API

### `EmbodiedEnvCfg` (new/changed fields)
- `record_trajectory: bool = False` (existing).
- `trajectory_uids: list[str] | None = None` (existing).
- `trajectory_save_dir: str | None = None` (NEW) - default `<EMBODICHAIN_DEFAULT_DATA_ROOT>/trajectories/{run_id}/` (run_id per-run, avoids collisions).
- `trajectory_auto_save: bool = True` (NEW) - auto-save at episode end + close.

### `EmbodiedEnv` (new/changed methods)
- `_preprocess_action(action)` - stash `self._traj_raw_action = action` before transforming (when `record_trajectory`).
- `_write_trajectory_step()` - per-env write of `states` + `action` (replaces `_write_trajectory_states`).
- `save_trajectory(path: str, env_ids: Sequence[int] | None = None) -> str` - save env_ids (default all) to `path`; returns `path`.
- `_save_trajectory_for_env(env_id) -> str` - auto-save helper.
- `_initialize_episode(env_ids)` - auto-save resetting envs, then `self._traj_steps[env_ids] = 0`.
- `close()` - auto-save in-flight envs (finalize), then existing close.

### `ReplayWrapper(gym.Wrapper)`
- `ReplayWrapper(env, trajectory: str | dict, mode="kinematic"|"dynamic")`.
- Per-env `_replay_steps`, `_lengths` (from `meta["lengths"]`, broadcast for single-env trajectory).
- `reset` / `step` / `close` as in 7.4/7.5.
- `_set_all_states(states)` - UNCHANGED from PR #425 (writes robot root_pose+qpos non-mimic, articulations, rigid objects).
- Helper `load_trajectory(path)` - UNCHANGED (validates + returns dict; now also reads `meta["lengths"]`).

## 9. Key design decisions

1. **Pre-process action enables ActionManager.** Recording the raw (pre-`_preprocess_action`) action and feeding it back through `env.step` lets the `ActionManager` re-apply the same transform (delta->qpos, eef_pose->qpos, etc.), so dynamic replay is faithful for any action config - provided the replay env uses the same `ActionManager` config as the recording env. This removes the original spec's "ActionManager deferred" limitation. Requirement: replay env's `actions` cfg must match the recording env's.
2. **Dedicated buffer decouples sync semantics.** The shared `rollout_buffer` is global-sync (for LeRobot/RL); mixing per-env async trajectory data into it would create two sync semantics in one buffer. `self._traj_buffer` keeps them clean.
3. **Per-env `_traj_steps` fixes async.** Only the reset envs' counters clear; non-reset envs keep recording. This is the core fix - no more global reset corrupting other envs.
4. **Action != `states["robot"]["qpos"]`.** `action` is the commanded target (pre-physics); `states["robot"]["qpos"]` is the actual position (post-physics, with PD error). Dynamic replay re-commands the target; kinematic replay sets the actual position.
5. **Auto-save mirrors LeRobot.** Save at episode end (per resetting env) + finalize on close, to the data cache dir, with an opt-out (`trajectory_auto_save=False`) and custom dir (`trajectory_save_dir`). Per-env files for async; explicit `save_trajectory` can bundle envs.
6. **Auto-reset guard retained.** Dynamic replay still uses `_replay_no_auto_reset` so mid-trajectory success/fail doesn't auto-reset. Per-env truncation is handled by `_replay_steps >= _lengths`.

## 10. Error handling

- `_traj_steps[env]` reaching `max_steps`: that env stops recording (skipped via mask), warning once.
- Auto-save dir not writable / cannot create: warn and skip auto-save (do not crash the episode); explicit `save_trajectory` still raises on IO error.
- `num_envs` mismatch on replay: single-env trajectory broadcasts to N; multi-env trajectory must match `env.num_envs` (else raise).
- Missing uid in replay scene: warn + skip. Extra object: warn + leave.
- `meta["lengths"]` missing (old PR #425 files): treat as uniform `num_steps` (backward compat).
- Physics/flag restored in `close()` (try/finally).
- Replay env `ActionManager`/`active_joint_ids` mismatch with recording: warn (and dynamic replay may diverge).

## 11. Testing (`/add-test`, pytest)

- **Unit - builder**: `build_trajectory_buffer` produces correct shapes for `states` + `action` (incl. uids filter).
- **Async recording (core)**: 2 envs, force env0 to terminate early (patch `compute_task_state` like PR #425's guard test); step further; assert env1's recorded states are NOT overwritten, `_traj_steps` differ per env, and `states[env1]` matches actual.
- **Pre-process action recorded**: with a delta `ActionManager`, assert `_traj_buffer["action"]` equals the raw delta (not the resolved qpos).
- **Auto-save**: record, trigger reset, assert a `.pt` appears under `trajectory_save_dir` (use a tmp dir) with correct `meta["lengths"]`; assert `close()` finalizes in-flight envs.
- **save_trajectory(env_ids)**: explicit save of a subset env produces a file with only that env's data.
- **Kinematic multi-env replay**: async-recorded trajectory (different lengths), `ReplayWrapper(kinematic)` replays; assert per-env states match recorded up to each env's length; finished envs hold last state.
- **Dynamic multi-env replay**: assert robot qpos tracks recorded (within tol) per env; ActionManager re-preprocessing verified (delta action config).
- **Backward compat**: load a PR #425-style file (no `lengths`); replay treats as uniform length.
- **ActionManager dynamic replay**: record with a delta action manager, dynamic-replay on a matching-config env, assert robot qpos tracks recorded.
- Existing PR #425 tests updated to the new buffer/API (states now in `_traj_buffer`, action pre-process).

## 12. Relation to PR #425 (migration)

PR #425 is a draft. This spec refactors its trajectory code:
- `rollout_buffer["states"]` -> `self._traj_buffer["states"]`.
- `_write_trajectory_states` -> `_write_trajectory_step` (per-env + action).
- `save_trajectory` -> reads `_traj_buffer`, adds `action` + `lengths`.
- `ReplayWrapper` -> per-env lengths, pre-process action for dynamic.
- New: `_traj_steps`, `_traj_raw_action` stash, auto-save, default path.
- `current_rollout_step` / `rollout_buffer["actions"]` no longer used by the trajectory feature.
- Auto-reset guard (`base_env.py`) unchanged.

## 13. Future work (deferred)

- `qvel` recording.
- Per-env termination semantics during dynamic replay (currently finished envs hold last action while others continue; a "freeze finished envs" mode could be added).
- Async `rollout_buffer` (obs/actions/rewards) for LeRobot - separate, larger effort.
