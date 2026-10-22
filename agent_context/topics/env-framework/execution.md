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
- Owns `expert_trajectory.joint_command_mode`: position-only by default or
  position-plus-velocity for expert execution and persistence.
- Creates managers in `_init_sim_state()` from config.
- Overrides `_extend_obs()` to run `ObservationManager.compute()`.
- Overrides `_extend_reward()` to run `RewardManager.compute()` and add to
  base reward.
- Uses `_before_episode_reset()` to save/discard old records before physical reset,
  `_initialize_episode()` for reset events and managers, and `_after_episode_reset()`
  to seed the authoritative first recording frame.
- Overrides `_update_sim_state()` to run event-manager `interval` mode.
- Manages rollout buffer (expert or RL mode) via `_hook_after_sim_step()`.
- Owns per-row demonstration metadata and dense segment acceptance, attempt,
  and causal-continuity annotations.
- `extensions` dict entries are set as attributes on both cfg and env instance.

### Task language (`demo.py`, `embodied_env.py`)

`EmbodiedEnvCfg.task_instruction` owns handwritten and direct-control goals;
Task Programs own the root `instruction` in their program. The common executor
resolves explicit episode language first, then the selected program, task
configuration, and legacy recorder configuration. Resolution reads legacy config
without constructing DatasetManager, so online generation retains compatibility.
Conflicting legacy recorders need an authoritative task-level description.

The executor freezes `DemoEpisodeResult.instruction` and its source before
planning and restores outer snapshot state even on hook/planning exceptions.
Recorded metadata preserves an unknown dynamic-program snapshot rather than
borrowing the static program's goal. Segment fallback labels use the resolved
episode goal and record their fallback source; they are not fine-grained labels.
The [data pipeline](../data-pipeline/data-pipeline.md) owns disk mappings and
stable online language IDs. Language intent does not establish physical success.

### Action boundary (`types.py`, `embodied_env.py`)

- Raw policy actions are flat floating tensors. `ActionManager.process_action()`
  validates and slices them in configuration order; `_step_action()` then calls
  `ActionManager.apply_action()` once before physics.
- `ControllerAction` wraps a controller-ready qpos/qvel/qf command.
  `EmbodiedEnv` keeps the envelope through preprocessing so `_step_action()` can
  bypass `ActionManager` and apply it through the direct controller boundary.
- `_prepare_controller_action()` validates controller batch, keys, active/full
  joint width, floating dtype, and device. Policy tensors use the manager's
  flat shape/device validation instead.
- Task Program runtime commands, waits, and abort-safe holds use
  `ControllerAction`; they retain the ordinary Gym lifecycle without becoming
  policy actions.
- Descriptor-driven policy recording and policy-width trajectory recording do
  not accept `ControllerAction`; mixing those fixed layouts fails before action
  processing. Expert recording continues to use `ExpertActionSpec`.
- For raw-policy vector demos with staggered completion, ActionManager masks
  inactive rows after processing: selected qpos resources hold measured values,
  while qvel/qf resources receive zero commands.
- A structured controller `TensorDict` may carry auxiliary fields such as
  `ik_success`, but it must contain at least one supported control key.
- Explicit controller actions and position-velocity expert actions are encoded
  only after validation and demo row masking, so recorded targets match the
  commands actually sent to the controller. A missing qvel in position-velocity
  mode is an error before physics advances.

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

### Segment terminal progress (`demo.py`, `run_task.py`)

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
  │     ├── EmbodiedEnv._setup_scene()
  │     │     ├── filter functors and resolve ground visibility
  │     │     └── BaseEnv._setup_scene()
  │     │           ├── resolve render demand → create SimulationManager
  │     │           ├── _setup_robot() → Robot + single_action_space
  │     │           ├── _prepare_scene() → lights, background, objects
  │     │           └── _setup_sensors() → sensors dict
  │     ├── SimulationManager.prepare() (backend-neutral readiness boundary)
  │     ├── open window (if not headless)
  │     └── _init_sim_state()
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
  ├── validate selections and episode_prepare callback
  ├── is_task_success() → capture old task outcome
  ├── _before_episode_reset() → existing save policy × EpisodeContext eligibility
  ├── begin selected context episodes only after old buffers are consumed
  ├── sim.reset_objects_state(env_ids, excluded_uids)
  ├── _initialize_episode() → event/manager reset
  ├── episode_prepare() → optional trusted host preparation
  ├── physical objective reset → final observation → _after_episode_reset()
  └── complete initialization → return observations and context info

```

### `step(action)`

```
step(action)
  ├── _preprocess_action(action)
  │     ├── flat policy action → ActionManager.process_action()
  │     └── ControllerAction → validate and preserve envelope
  ├── _step_action(action)
  │     ├── policy → ActionManager.apply_action()
  │     └── ControllerAction → direct qpos/qvel/qf application
  ├── sim.update(dt, sim_steps_per_control)
  ├── _update_sim_state()            # event_manager "interval" mode
  ├── _update_physical_objective()   # optional, once per control step
  ├── get_obs()
  │     ├── robot.get_proprioception()[:, active_joint_ids]
  │     ├── _get_sensor_obs()
  │     └── _extend_obs()            # ObservationManager.compute()
  ├── get_info() → evaluate()
  ├── get_reward() + _extend_reward()  # RewardManager.compute()
  ├── _postprocess_action(action)     # BaseEnv no-op; no action post mode
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
| `DatasetManager` | `cfg.dataset` | `__init__` (after super) | `_before_episode_reset()`; see the reset lifecycle above |

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

### Independent measured objectives

`EmbodiedEnv.physical_objective` is optional. Its per-row ordered milestone
history advances once after interval events, while final-region truth can become
false again. `get_info()["physical_objective"]` clones tensor snapshots;
`get_demo_episode_metadata()` includes a JSON-compatible row snapshot. Querying
either does not advance progress. Reset clears only selected rows after existing
recording consumption and episode reset logic. These results never overwrite
`compute_task_state()`, `is_task_success()`, segment validation or dataset decisions.
Stable pose/velocity does not prove gripper detachment.

`embodichain.lab.scripts.evaluate_task_objective` runs one explicit deployment
through the expert bridge and dynamic replay, recording independent outcomes,
resolved config/component snapshots and actual initial pose variation. It does
not orchestrate multiple experiments or qualify kinematic playback.

`EmbodiedEnv` can record per-object kinematic trajectories and replay them later.

- **Recording**: set `cfg.record_trajectory = True`. A dedicated per-env
  `self._traj_buffer` (TensorDict: `states` = robot root_pose+qpos, articulations,
  rigid objects; `actions` = the flat raw policy action when ActionManager is
  configured, active qpos for position-mode experts, or effective flat
  `[qpos, qvel]` targets in position-velocity expert mode) is written each step via
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
  reproduction); `dynamic` feeds raw-policy actions through `env.step` so the
  `ActionManager` re-applies the transform. An expert trajectory with
  `action_kind: expert` validates its schema, joint layout and cadence and
  decodes `ControllerAction`; the legacy `expert_controller` marker remains
  accepted. Objective-enabled dynamic replay requires one row and a full reset.
  `control` exposes `go_to_step(step)` for O(1) scrubbing. The replay env must
  use the same robot/objects/`actions` config as the recording env.
- **Cable observations**: `states.cables[uid].segment_state_w` records world
  segment pose/velocity, not solver history. `ReplayWrapper` accepts dynamic
  replay only when every recorded initial state matches the full reset; it
  rejects scrubbing, partial reset and broadcasting world-space cable states.
  `trajectory_state.restore_trajectory_state()` rejects cable entries before
  any write. Use action replay to validate a new episode, not a visible-state
  restore as a history-preserving checkpoint.
- **Decoupled from `rollout_buffer`**: the trajectory buffer is separate from the
  shared `rollout_buffer` (obs/actions/rewards) used by LeRobot/RL.
  `current_rollout_step`, LeRobot recorder, and RL mode are untouched.
- **Fixed action layout**: an ActionManager-backed trajectory uses the batched
  policy action space; a controller/expert trajectory uses the private
  `ExpertActionSpec` layout. One trajectory cannot mix these schemas.
- **CLI**: `run-task --replay --replay_trajectory <path> --replay_mode {kinematic,dynamic,control}`.

---

### Generic recording eligibility

`EpisodeContext` has three component interfaces: `check`, `filter_recordable`
and `snapshot`. Construction is uninitialized. Checks are synchronous owner-thread
scopes; pending checks survive internal resets and exceptions latch rejection in
current buffers. Only the host's successful buffer-consumption boundary opens a
new episode; demo annotation replacement and reseeding do not. Context snapshots
are strict JSON copies for async writers. Neither eligibility nor geometry evidence
changes task success. Save/discard failure leaves the old episode and scene intact;
initialization/first-frame failure blocks the selected new rows.

`commit_env_ids` retains dataset-only selection, including with `save_data=False`.
All normal output selections also intersect context eligibility. Explicit
`purpose="diagnostic"` trajectory/camera exports retain failure evidence separately
and are never dataset commits. Closing only drains committed writes and discards
live buffers. Camera recordings use independent row buffers and files, with owned
context sidecars; asynchronous jobs do not wait for another row or a mosaic.
