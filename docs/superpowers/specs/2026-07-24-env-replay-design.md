# Environment Task Replay — Design Spec

- **Date**: 2026-07-24
- **Status**: Approved (brainstorming complete)
- **Topic**: Record and replay environment tasks in two modes — pure kinematic and dynamic

## 1. Overview

Add a **record-and-replay** capability to the environment framework so that a
run episode can be saved as a trajectory file and replayed later in one of two
modes:

- **Pure kinematic replay** — deterministically reproduce the recorded motion of
  *all* objects (robot + interactive objects) by writing their state directly
  each step, with physics disabled. Produces observations (camera images /
  proprioception) but no reward / success / action.
- **Dynamic replay** — restore the recorded initial scene, then feed only the
  recorded robot trajectory (joint targets) back through the normal `env.step`
  pipeline so physics re-simulates object interaction. Produces the full
  `obs / reward / terminated / truncated / info` output.

The feature reuses the existing in-env `rollout_buffer`, the
`_hook_after_sim_step` write hook, the `gym.Wrapper` pattern, and the existing
per-object state read/write APIs. Core env changes are minimal.

## 2. Scope

### In scope
- The **expert-mode** `rollout_buffer` owned by `EmbodiedEnv` (the one that
  records `obs / actions / rewards` per step), extended with a `states` field.
- A new `ReplayWrapper(gym.Wrapper)` driving both replay modes.
- Assumption: the env's action is a **raw joint-position target** (no
  `ActionManager` transformation). This is the typical expert / demo scenario.

### Out of scope (deferred)
- RL training integration: the `set_rollout_buffer` injection path, the `"rl"`
  buffer mode, and `ActionManager` (delta / normalized action terms). These can
  be layered on later when RL replay is needed.
- Soft / cloth object trajectories (different state APIs; not required by the
  stated use case).
- Online streaming / distributed replay.

## 3. Background — reused components and gaps

### Reused
- `env.rollout_buffer` (`TensorDict`, shape `[num_envs, max_episode_steps]`),
  created in `EmbodiedEnv.__init__` when `cfg.init_rollout_buffer` is True, and
  written each step via `_hook_after_sim_step` → `_write_episode_rollout_step`.
  The code already flags extensibility
  (`embodied_env.py` `# TODO: we may add more keys and make the buffer extensible`).
- `_hook_after_sim_step` / `_write_episode_rollout_step` — the per-step write
  path.
- `gym.Wrapper` pattern — `NoFailWrapper` already follows it.
- **State read**: `Robot/Articulation.get_local_pose()`, `get_qpos()`;
  `RigidObject.get_local_pose()`. Object registries on `sim._robots /
  _articulations / _rigid_objects` (dicts keyed by uid).
- **State write (kinematic, bypass physics)**: `Articulation.set_local_pose()`,
  `set_qpos(target=False)`; `RigidObject.set_local_pose()`,
  `set_body_type("kinematic")`.
- **State write (dynamic)**: `robot.set_qpos(target=True)` — the normal
  `env.step` path via `_step_action`.
- `sim.enable_physics(bool)` — global physics toggle.
- Sensor rendering is independent of physics: `_get_sensor_obs`
  (`base_env.py:369`) calls `sim.render_camera_group(...)` then
  `sensor.update(...)`, so cameras render the currently-set scene state
  regardless of whether physics is running.

### Gaps this feature fills
- `rollout_buffer` stores only `obs / actions / rewards` — **not** object
  kinematic state (trajectories).
- **No runtime state snapshot/restore** — `sim.reset_objects_state()` only
  restores config defaults (`init_pos / init_rot / init_qpos`). `ReplayWrapper`
  restores the recorded step-0 state instead.
- **No env-level replay** — only sim-level `replay_trajectory` (in
  `scripts/tutorials/atomic_action/tutorial_utils.py`) which does not go through
  `env.step`.

## 4. Architecture

```
gym.Env
  └── BaseEnv            (EnvCfg)
        └── EmbodiedEnv   (EmbodiedEnvCfg)   <-- + record_trajectory cfg, + states field, + save_trajectory()
              └── <Task>                       <-- unchanged

ReplayWrapper(gym.Wrapper)  wraps any EmbodiedEnv for replay
```

Recording lives in the env (extends the existing buffer + hook). Replay lives in
a wrapper, leaving the core env loop untouched.

## 5. Components and file placement

| Component | File | Responsibility |
|---|---|---|
| Trajectory schema builder | `embodichain/lab/gym/utils/gym_utils.py` — new `build_trajectory_states_buffer(env, max_steps, num_envs, device)` | Scan `sim._robots/_articulations/_rigid_objects`; build nested `TensorDict` for `states`; return metadata (uids, dims). |
| Recording config + write | `embodichain/lab/gym/envs/embodied_env.py` | Add 3 cfg fields; attach `rollout_buffer["states"]` in `__init__`; append `_write_trajectory_states()` to `_write_episode_rollout_step`. |
| Auto-reset guard | `embodichain/lab/gym/envs/base_env.py` | Guard the auto-reset block in `step()` with `if not getattr(self, "_replay_no_auto_reset", False)` so dynamic replay is not disrupted by mid-trajectory resets. |
| Persistence | `embodichain/lab/gym/envs/embodied_env.py` — new `save_trajectory(path)` | Slice `[:current_rollout_step]`, bundle `states + actions + meta`, `torch.save` to `.pt`. |
| Replay | `embodichain/lab/gym/envs/wrapper/replay.py` — new `ReplayWrapper(gym.Wrapper)` | Load trajectory; `reset` restores initial state; `step` drives per mode. |
| Exports | `embodichain/lab/gym/envs/wrapper/__init__.py` | Export `ReplayWrapper`. |

## 6. Data model

### `rollout_buffer["states"]` (shape `[num_envs, max_episode_steps, ...]`)
```
states = {
    "robot": {
        "root_pose": [N, T, 7],   # xyz + wxyz
        "qpos":      [N, T, dof], # ALL joints (active + passive), for faithful kinematic replay
    },
    "articulations": {            # non-robot articulations
        "<uid>": {"root_pose": [N, T, 7], "qpos": [N, T, dof]},
        ...
    },
    "rigid_objects": {
        "<uid>": {"pose": [N, T, 7]},
        ...
    },
}
```
`qvel` / `vel` are recorded only when `trajectory_record_qvel=True` (optional,
default off — not needed for the two core modes).

### `.pt` trajectory file
```
{
  "states": <states TensorDict sliced to [:num_steps]>,
  "actions": <actions tensor sliced to [:num_steps]>,   # recorded robot qpos targets, for dynamic replay
  "meta": {
      "uids": {"robot": str, "articulations": [...], "rigid_objects": [...]},
      "dims": {"robot": {"qpos": dof}, "articulations": {"<uid>": dof}, ...},
      "num_steps": int,
      "num_envs": int,
      "dt": float,
      "active_joint_ids": <list[int]>,   # for sanity check against replay env
  },
}
```

## 7. Data flow

### 7.1 Recording
```
env.step(action)
  └─ _hook_after_sim_step
       ├─ _write_episode_rollout_step   # writes obs/actions/rewards  (existing)
       └─ _write_trajectory_states      # writes states[:, t]          (new)
episode end → env.save_trajectory(path) → .pt
```
`_write_trajectory_states` reads, for the current step `t`:
- robot: `robot.get_local_pose()`, `robot.get_qpos()` (all joints)
- each articulation: `art.get_local_pose()`, `art.get_qpos()`
- each rigid object: `obj.get_local_pose()`
and `copy_()` into `rollout_buffer["states"][:, t, ...]`.

### 7.2 Pure kinematic replay (`mode="kinematic"`)
```
reset():
  env.reset()
  _set_all_states(states[:, 0])          # restore recorded initial scene
  sim.enable_physics(False)
  obs = env.get_obs(); return obs, info

step(action):                            # action is ignored
  _set_all_states(states[:, idx])        # set every object's pose/qpos directly
  sim.update(physics_dt, sim_steps_per_control)   # physics OFF: commits FK/scene, no integration
  obs = env.get_obs()
  idx += 1
  trunc = idx >= num_steps
  return obs, zeros, False, trunc, {}    # no reward / success / action
```

### 7.3 Dynamic replay (`mode="dynamic"`)
```
reset():
  env.reset()
  _set_all_states(states[:, 0])          # restore recorded initial scene
  sim.enable_physics(True)
  env._replay_no_auto_reset = True       # prevent mid-trajectory auto-reset (see 9.5)
  obs = env.get_obs(); return obs, info

step(action):                            # action is ignored
  obs, reward, term, trunc, info = env.step(actions[:, idx])   # feed recorded qpos target; physics re-simulates
  idx += 1
  trunc = trunc | (idx >= num_steps)
  return obs, reward, term, trunc, info  # success/fail still observable in info; env does NOT auto-reset
```
Because the env is assumed to have no `ActionManager`, `buffer["actions"]`
(the resolved qpos target) is fed straight back through `env.step`, which
applies it via `_step_action` → `robot.set_qpos(target=True)` and runs the full
pipeline.

### `_set_all_states(states)`
```
robot:          robot.set_local_pose(states["robot"]["root_pose"])
                robot.set_qpos(states["robot"]["qpos"], target=False)
per articulation (uid): art.set_local_pose(s["root_pose"]); art.set_qpos(s["qpos"], target=False)
per rigid object (uid): obj.set_local_pose(s["pose"])
```
`target=False` writes the current joint position directly, bypassing the drive —
the kinematic-write path. `env_ids` defaults to all envs.

**Mimic joints**: record all joints (`get_qpos()` with no part filter), but on
replay set only non-mimic joints (mimic joints are coupled and must not be
written directly). The active-joint id list in `meta` is used to sanity-check
the replay env against the recording.

## 8. API

### `EmbodiedEnvCfg` (new fields)
- `record_trajectory: bool = False` — when True, forces
  `init_rollout_buffer=True` and writes `states` each step.
- `trajectory_uids: list[str] | None = None` — restricts which non-robot objects
  are recorded; `None` = all rigid objects + all articulations. The robot is
  always recorded.
- `trajectory_record_qvel: bool = False` — optionally record `qvel` / `vel`.

### `EmbodiedEnv` (new method)
- `save_trajectory(path: str) -> None` — persist the current episode's
  `states` + `actions` + `meta` to a `.pt` file.

### `ReplayWrapper(gym.Wrapper)`
- `ReplayWrapper(env, trajectory: str | dict, mode: Literal["kinematic","dynamic"] = "dynamic")`
  — `trajectory` is a `.pt` path or a loaded dict.
- `reset(*, seed=None, options=None)`
- `step(action)` — `action` is ignored in both modes; the trajectory drives
  playback.
- `close()` — restore `sim.enable_physics(True)` and clear
  `_replay_no_auto_reset` (try/finally).
- Helper `load_trajectory(path: str) -> dict` — load `.pt`, validate `meta`.

## 9. Key design decisions (non-obvious)

1. **`buffer["actions"]` holds the resolved qpos target.** In `base_env.step`
   the action passes through `_preprocess_action` → `_step_action` →
   `_postprocess_action` before reaching `_hook_after_sim_step`
   (`base_env.py:623→637→658`). With no `ActionManager`, preprocess/postprocess
   are identity, so the stored action equals the applied qpos target and can be
   fed straight back through `env.step`. (ActionManager support is deferred —
   see Scope.)
2. **Initial-state restore closes the snapshot gap.** `env.reset()` only returns
   objects to config defaults; `ReplayWrapper.reset()` overwrites every object
   with `states[:, 0]` so both modes start from the recorded scene.
3. **Kinematic `sim.update` with physics off.** `sim.enable_physics(False)`
  disables dynamics integration, so set states hold exactly. `sim.update` is
  still called to commit FK / scene-graph updates so rendering reflects the set
  joint positions. **To verify during implementation:** confirm FK/scene commit
  happens with physics off; if not, fall back to keeping physics on with rigid
  objects set to `body_type="kinematic"` and the robot written via
  `set_qpos(target=False)` (plus `fix_base` if gravity perturbs free chains).
4. **Event randomization.** Kinematic replay bypasses `env.step`, so interval
  event functors do not fire (good — pure reproduction). Dynamic replay runs the
  full `env.step`, so interval events fire; if they include stochastic
  randomization, use a matching seed or `filter_*` flags for reproducibility.
5. **Auto-reset suppression in dynamic replay.** `base_env.step` auto-resets
  done envs (`base_env.py:669-671`), which would disrupt trajectory playback if
  the task reaches success/fail mid-replay. `ReplayWrapper` sets
  `env._replay_no_auto_reset = True` for dynamic mode; the guarded block in
  `base_env.step` then skips the auto-reset. `terminated` / `truncated` /
  `info["success"]` are still computed and returned (so reproducibility can be
  observed), only the reset is suppressed. The flag is cleared in `close()`.
  Kinematic mode is unaffected (it never calls `env.step`).

## 10. Error handling

- Trajectory `num_steps > max_episode_steps`: replay truncates at
  `max_episode_steps` with a warning.
- `num_envs` mismatch: trajectory with 1 env is broadcast to `N` envs; a
  trajectory whose env count is neither 1 nor `env.num_envs` raises.
- Missing uid in the replay scene (object absent): warn and skip that object.
  Extra object in the scene (not in trajectory): leave as-is with a warning.
- Physics is always restored to enabled in `close()` / `__exit__` (try/finally),
  even on error.
- Malformed `.pt` / missing `meta`: raise a clear error.
- `record_trajectory=True` implicitly forces `init_rollout_buffer=True` so the
  buffer (with `states`) always exists.

## 11. Testing (`/add-test`, pytest)

- **Unit — schema builder**: on a simple task env with a robot + a rigid object,
  assert `build_trajectory_states_buffer` produces correct shapes/dims per uid.
- **Kinematic round-trip (core)**: record N steps → save → load →
  `ReplayWrapper(mode="kinematic")` → replay N steps → assert every object's
  `root_pose` / `qpos` at each step equals the recorded values (within
  tolerance). Validates exact reproduction.
- **Dynamic replay**: record → `ReplayWrapper(mode="dynamic")` → assert the
  robot `qpos` tracks the recorded action (within tolerance); assert key objects'
  end states are close to the recorded ones (looser tolerance — physics
  reproducibility).
- **Physics state**: assert physics is `False` during kinematic replay, `True`
  during dynamic replay, and restored to `True` after `close()`.
- **Initial-state restore**: after `ReplayWrapper.reset()`, assert the scene
  state equals `trajectory[:, 0]`.
- **num_envs broadcast**: record 1 env, replay on N envs, assert broadcast is
  correct.

## 12. Usage

```python
from embodichain.lab.gym.utils.registration import make
from embodichain.lab.gym.envs.wrapper import ReplayWrapper

# --- Record ---
cfg = MyTaskCfg(...)
cfg.init_rollout_buffer = True
cfg.record_trajectory = True
env = make("MyTask-v1", cfg=cfg)
for _ in range(episode_len):
    obs, reward, term, trunc, info = env.step(action)
env.save_trajectory("episode.pt")

# --- Replay: pure kinematic ---
env_k = make("MyTask-v1", cfg=MyTaskCfg(...))
env_k = ReplayWrapper(env_k, "episode.pt", mode="kinematic")
env_k.reset()
for _ in range(num_steps):
    obs, _, _, trunc, _ = env_k.step(None)     # action ignored; obs = reproduced motion

# --- Replay: dynamic ---
env_d = make("MyTask-v1", cfg=MyTaskCfg(...))
env_d = ReplayWrapper(env_d, "episode.pt", mode="dynamic")
env_d.reset()
for _ in range(num_steps):
    obs, reward, term, trunc, info = env_d.step(None)   # robot trajectory drives; physics re-simulates
```

## 13. Future work (deferred)

- `ActionManager` support for dynamic replay (skip-preprocess flag or raw-action
  capture) for RL-trained policies with delta / normalized action terms.
- Replay of the RL `"rl"`-mode rollout buffer / `set_rollout_buffer` injection
  path.
- Soft / cloth object trajectories.
- A `TrajectoryRecorder` functor for `DatasetManager` to auto-save `.pt` at
  episode end alongside `LeRobotRecorder`.
