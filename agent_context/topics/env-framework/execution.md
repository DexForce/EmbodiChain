# Environment execution and demonstration lifecycle

Read this when the request needs these details. [Topic overview](env-framework.md).

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

### Explicit demonstration candidates (`demo.py`)

- `execute_demo_episode(env, segments=...)` accepts one `DemoSegment` or a lazy
  iterable of segments. Explicit segments bypass both task planning factories;
  combining them with planning keyword arguments raises `ValueError` before
  recording begins. An empty iterable executes no candidate and does not fall
  back to task planning.
- `segments=None` preserves `resolve_demo_segments()` and its legacy
  `create_demo_action_list()` fallback. Both paths validate segment types lazily
  and fill missing instructions from dataset metadata without changing the
  supplied segment.
- Explicit candidates use the same normal action processing, per-row masks,
  validators, cancellation/abort handshake, and recording lifecycle as task
  plans. Auto-reset remains suspended during execution; the caller owns the
  later commit or discard boundary. Supplying a candidate does not restore its
  initial state.
- Focused coverage: `tests/gym/envs/test_demo.py` and the Task Program bridge
  and completion tests under `tests/gym/envs/task_program/`.

### Fixed-scene generation preparation (`base_env.py`, `embodied_env.py`)

- `BaseEnv.acquire_generation_lease(owner)` reserves the complete environment
  batch by owner identity. Same-owner acquisition is idempotent; other owners
  are rejected. The generation flag disables auto-reset independently of demo
  and replay flags. Ordinary `reset()` fails before seeding or state mutation
  while the lease is held.
- `EmbodiedEnv.prepare_generation_episode(owner, *, prepare, restore, settle,
  verify)` is the full-batch discard-and-prepare boundary. Freeze prior episode
  evidence first: this method clears camera buffers, expert/trajectory counts,
  demo annotations, success state, and the active Task Program bridge without
  saving pending episodes.
- Preparation runs deterministic task/controller initialization, physical
  restoration, settling, standard manager reset, and nonempty all-passed
  `ValidationResult` verification
  through trusted callbacks. It does not run startup/reset/interval events or
  rewind environment/event RNGs. The host profile owns required initialization
  and certifies any interval events that remain active during later rollouts.
- Observation/history, reward, and dataset managers reset before verification;
  `get_obs()` and `get_info()` then refresh the initial observation/task state,
  and recording seeds from that settled state. Preparation does not use Gym
  `step()` or write a training transition. Failure leaves stepping disabled.
- `generation_epoch` advances on lease acquisition/release, every preparation
  attempt, and normal resets. Failed preparation invalidates previous epochs.
  `FixedSceneHost` in `embodichain/lab/trajectory_generation/initial_state.py`
  matches candidate bindings against this epoch and verifies physical/task
  initial state before publishing a `PreparedBatch` or Gym first frame.
- `BaseEnv.observe_generation_commands(owner, callback)` observes a successfully
  submitted `_step_action` command before physics. The observer requires the
  same generation lease and does not issue a second command or step.
  `execute_demo_episode` exposes `step_observer(result, active_before_step)` and
  `row_step_limits` so the qpos executor records real returned observations and
  stops each short row's recording before batch padding/holds.
- `FixedSceneHost.initial_observation(binding)` copies the prepared Gym first
  frame without a second `get_obs()` or history update.
- `release_generation_lease(owner)` restores normal reset behavior without
  saving, resetting, or modifying demo/replay flags. Callers serialize access;
  the host remains the sole simulation stepper while the lease is active.
- Focused coverage: `tests/gym/envs/test_fixed_scene_preparation.py`, including
  real Gym lifecycle methods wired to `FixedSceneHost` through fake physical
  ports; physical adapters have separate tests under
  `tests/lab/trajectory_generation/`.

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

### Segment terminal progress (`demo.py`, `run_env.py`)

Every demo path reaches the same terminal `tqdm` wrapper through
`execute_demo_episode()`. A `DemoSegment` can declare an exact
`progress_total_steps` without materializing its lazy action iterable; the
executor exposes that total through `__len__` only to the progress wrapper.
Task Program bridges and handwritten trajectory tasks must set it only when
the emitted action count is fixed. Dynamic recovery, feedback settling, or
data-dependent waits retain an unknown total rather than reporting a false
percentage. Segment metadata `segment_count` selects the `segment N/M` label
for both execution styles; legacy action-list tasks receive
`segment_count=1` automatically.

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
