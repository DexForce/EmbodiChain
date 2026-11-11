# Environment framework

`BaseEnv` owns the Gymnasium loop; `EmbodiedEnv` adds configured scene entities,
managers, demonstration collection, and the opt-in Task Program bridge.
Lightweight RL environments have a separate owner in
[rl-learning](../rl-learning/rl-learning.md).

## Entry points and resolution

| Request | Owning file / resolution path |
|---|---|
| Command dispatch | `embodichain/cli/main.py`; `embodichain/__main__.py` is the module entry wrapper |
| Example discovery / launch | `embodichain/cli/run_example.py` → script path in the ordinary `examples/` tree → `_example_runner.py`; see [running and adding examples](../../../examples/README.md) |
| Task discovery / gallery / `show-task` | `embodichain/cli/_task_catalog.py` → shared logical-task/deployment records; `list_task.py` and `show_task.py` render them |
| Launch / CLI overrides | `embodichain/lab/scripts/run_task.py` → `build_env_cfg_from_args()` → `config_to_cfg()` |
| Public execution-free component expansion | `embodichain/lab/gym/utils/composition.py` → shared strict decoders |
| Config decoding | `embodichain/lab/gym/utils/gym_utils.py` |
| Robot selector decoding | `robot.class_type` → `embodichain/robot_registry.py`; catalog ownership is in [robot-system](../robot-system/robot-system.md) |
| Import registration | `embodichain/lab/gym/utils/registration.py`; official task imports under `extensions/embodichain_tasks/embodichain_tasks/` |
| Environment cadence resolution | `embodichain/lab/gym/utils/control_timing.py`: `resolve_control_timing`, `ControlTiming` |
| Environment loop | `embodichain/lab/gym/envs/base_env.py`: `EnvCfg`, `BaseEnv` |
| Scene, managers, RL outcomes and demonstration integration | `embodichain/lab/gym/envs/embodied_env.py` |
| Expert trajectory control and stored-action schema | `embodichain/lab/gym/envs/expert_trajectory.py` |
| Raw schema, optional history and exact historical queries | `managers/observation_manager.py`; Env dispatches capture/reset/close boundaries |
| Optional measured physical objectives | `embodichain/lab/gym/envs/objectives/`; Gym owns observation/reset hooks independently of expert acceptance |
| Controller-ready commands | `embodichain/lab/gym/envs/types.py`: `ControllerAction` |
| Demo segment execution / outcomes | `embodichain/lab/gym/envs/demo.py` |
| Configured Task Program registration | `embodichain/lab/gym/envs/task_program/registration.py` |
| Timing instrumentation | `embodichain/lab/gym/utils/profiler.py` |

Read [configuration and registration](configuration.md) for task-first layout,
component ownership, path resolution, config-owned IDs, and task listing.
Read [fluid Gym integration](fluid-replay.md) for online Task Program execution
with rigid/SPH dynamics, fluid configuration and coordinate conventions.
Read [execution](execution.md) for hooks, bridge acceptance, reset ordering,
wrappers, and replay. Read [profiling](profiling.md) only for instrumentation.

Velocity locomotion deployments select catalog robot IDs in both backends.
Their `contracts/_config.py` expands the task JSON's matching model selection
using the backend-free robot profile reader. Task-specific actions, observations,
rewards and contact termination rules remain in the task package; robot defaults
and the distinct asset/contract joint orders are owned by
[robot-system](../robot-system/robot-system.md).

Reference-motion tracking is owned by
`extensions/embodichain_tasks/embodichain_tasks/locomotion/tracking/`.
`g1_motion.py` owns simulator state, reference-frame progress, selected-row resets
and pre-action/terminal snapshots; unusable nonfinite successors hard-terminate
and are excluded from WM supervision. `_task.py` owns tensor objectives shared
with model-driven rollouts. The task's `ReferenceJointPositionAction` records
policy outputs separately from the reference-combined setpoint and executed
joint target. Read the task configuration's README for data preparation,
training and per-clip evaluation commands.

## Launcher argument ownership

`embodichain run-task` is the standalone launcher for existing task deployments,
dispatched to `embodichain.lab.scripts.run_task:cli`. The cross-engine
[`gensim run` workflow](../gen-sim/gen-sim.md) generates and binds a deployment
before reusing configured Gym/Task Program execution. The former `run-env` command and
`embodichain.lab.scripts.run_env` module are removed.

`add_env_launcher_args_to_parser()` composes the lightweight simulation options
from `embodichain.cli.sim` with Gym config, seed and recording options. When
`require_gym_config=True`, omitted runtime overrides stay `None`: `num_envs`,
`device`, `headless`, `renderer`, `physics`, `arena_space`, and `gpu_id` preserve
file-owned values. Explicit values override runtime settings (`--headless` / `--no-headless`
select either mode); `--physics` can only confirm the file-owned backend. An omitted seed preserves the config seed.
Standalone examples use `add_sim_args_to_parser()` and opt into seed separately.
`--disable_sensor` (also accepted as `--disable-sensor`) sets the resolved Gym
config's `enable_sensor` switch to false; this keeps reusable embodiment sensor
declarations available to the config while skipping their instantiation and
image acquisition.

Environment construction predicts native render demand before creating the
World, so auto can use NoRender for headless tasks without visual consumers.
BaseEnv owns a copy of the configuration; filtering and renderer resolution do
not change the caller's reusable configuration. Filters run before the demand
hook, including when a subclass replaces that hook without calling super.
Custom camera/visual code selects an explicit native renderer or reserves
rendering through `BaseEnv._requires_native_renderer()`.
See [render selection](../simulation-system/rendering.md#norender-initialization)
for precedence, manager cameras, Viser and filter ordering.

## Timing contract

`resolve_control_timing()` resolves the file-owned physics interval and control
declaration without creating an environment. `BaseEnv._configure_timing()` uses
this same resolver before constructing the scene:

1. `physics_dt` comes from `cfg.sim_cfg.physics_dt` and must be finite and positive.
2. Without `target_control_frequency`, `sim_steps_per_control` must be a
   positive integer (default `4`).
3. A provided finite, positive `target_control_frequency` takes precedence
   over `sim_steps_per_control`. `1 / (physics_dt * target_control_frequency)`
   must equal a positive integer within absolute `1e-9` tolerance (on the
   step count, not on seconds).
4. This resolution never changes `physics_dt` or silently rounds an
   unrepresentable control period.

`step_dt = physics_dt * sim_steps_per_control`; `physics_frequency = 1 / physics_dt`
and `control_frequency = 1 / step_dt`. `control_freq` / `sim_freq` are compatibility
aliases; `render_fps` follows the control frequency. For example, `physics_dt=0.01`
and `target_control_frequency=25` resolve to four physics steps per control step.
Source trajectory timing does not alter this contract: expert paths are retimed
to `step_dt` before their commands reach `env.step()`.

## Expert trajectory contract

`EmbodiedEnvCfg.expert_trajectory` owns source-neutral expert generation
behavior. `joint_command_mode="position"` is the compatibility default;
`"position_velocity"` requires both effective qpos and qvel controller targets.
This setting applies equally to handwritten, MotionGenerator, and Task Program
experts and does not change the policy `action_space`.

Expert recording uses its own immutable action layout. Position mode stores
active-joint qpos. Position-velocity mode stores a flat
`[active qpos, active qvel]` vector and publishes joint names, slices, mode,
schema version, and `step_dt` in trajectory/demo metadata. Missing qvel fails
before stepping in position-velocity mode; inactive and hold targets use zero
qvel.

`ObservationManager` owns raw acquisition, the single field schema, alias
resolution and read-only access to raw rollout observations. Standard replay fields are declared by
`trajectory_observation_cfgs`; Env supplies those declarations and the owned
buffers, without interpreting source functions or maintaining another schema.
See [manager functors](../manager-functor/manager-functor.md#raw-observations-and-consumer-views).

The host's `_rollout_storage` is the common TensorDict owner. Its leading batch
axis is environment-only so policy state slots, trajectory state slots and
learner windows can retain their own time axes and devices without implicit
copies. Internal expert policy storage keeps T+1 observations; the public
`rollout_buffer` remains its T-transition view. With trajectory recording, that
view also exposes raw observation/validity/time views from the same storage.
`_traj_buffer` and indexed observation reads reference the shared raw tensors.
Raw writes are part of the host rollout callbacks, using the existing buffer
utilities. Episode IDs, availability, acquisition clocks and closed state are
rollout metadata; no history class owns another lifecycle or cursor.

An explicit observation boundary follows physics and interval events, before
reward/action-postprocessing hooks and autoreset. Reset seeds S0 after
`episode_prepare`; T transitions retain T+1 raw states, including terminal ST.
Ordinary `get_obs()` queries do not advance history; explicit captures in the
same tick have distinct revisions. Policy rollouts retain the actual transformed
and noisy values. `step(..., recording_active_mask=...)` independently selects
history rows for a diagnostic settling tail.

Historical consumers bind fields using `observation_manager.schema`, then use
`read_history` with exact row-local frame indices and episode IDs. The indexed
rollout read checks actual episode ownership, initialized rows and lifecycle
metadata before/after copying. It never queries
live state or substitutes a previous frame. Results own values, validity,
reasons and temporal identities, without another descriptor model.

`export_observation_history` projects the manager's schema onto one owned numeric
snapshot. Legacy trajectory state views and raw history share that snapshot's
storage. The default snapshot is CPU; file export can retain its device for
existing replay compatibility. `_traj_buffer`/`_traj_steps` remain projection
aliases for existing persistence/replay adapters. Attaching/reusing an external
RL rollout never truncates or clears episode history. Policy acceptance and raw
diagnostic recording retain separate named cursors/masks in the same owner;
normalization, noise, projection and different lifetimes prevent arbitrary
aliasing of raw values with transformed policy values.

## Lifecycle and boundaries

`EmbodiedEnv._setup_robot()` creates an instance-owned `active_joint_ids` list
on each setup, preserving control-part order or copying explicit configured
IDs. Repeated setup and other environment instances cannot append to that list.
The isolation cases are in `tests/gym/envs/test_embodied_env_joint_setup.py`.

Construction seeds before scene setup, constructs the robot and enabled sensors, then
initializes simulation state and configured managers. An explicit effective
seed on reset also reaches the event manager; scoped randomization behavior is
owned by [randomization](../randomization/randomization.md). Robot-mounted cameras
use the same acquisition switch and participate in the environment sensor
registry; see [configuration](configuration.md) for declaration ownership.

The policy step order is:

```text
flat action validation/slicing → term processing → term application
→ physics update → interval events → physical objective
→ observations → evaluation/info → rewards
→ elapsed steps and termination → rollout hook → optional reset
```

Raw policy input is one ordered flat floating tensor. `ActionManager` validates
it before state changes, processes every slice, and applies each term's owned
resources. `ControllerAction` stays wrapped through preprocessing and normally validates
qpos/qvel/qf keys, batch, joint width, floating dtype and device at the direct
controller boundary. The position-only `expert_trajectory.use_action_manager`
opt-in instead processes joint targets through the configured policy terms,
retaining exact FK targets and expert recording schema. Both paths retain the
ordinary observation, reward, termination, and rollout lifecycle.

Read final task success before resetting scene/bridge state. A Task Program
reports success only after its bridge completes normally and combines runtime,
post-policy and validator acceptance. Segment acceptance and append-only
dataset persistence are separate contracts; see
[Task Programs](../task-programs/task-programs.md) and
[data pipeline](../data-pipeline/data-pipeline.md).

Source-bound Real2Sim episodes require call-level action tracing and confirmed
artifact receipts. The ordinary `run-task` recorder rejects them; a deployment's
`real2sim.profile` or explicit `--real2sim-profile` selects the tabletop owner,
which uses the qualified synchronous sink and
retains a staged payload if confirmation fails. See
[Real2Sim motion](../data-pipeline/real2sim.md).

Optional physical objectives publish separate measured results; they do not
change termination, program acceptance or persistence. Read [execution](execution.md)
for snapshot/reset ordering and expert-versus-replay qualification.

A trusted B=1 host can pass a zero-argument `episode_prepare` reset
callback. `BaseEnv.reset` consumes it after ordinary reset events and before
physical-objective initialization and final observations; it is not forwarded
to functors. `BaseEnv` owns `EpisodeContext`; successful initialization seeds recording from
the final observation. `EmbodiedEnv` consumes old records before physical reset.
Failed checks remain attached to buffers until the selected rows are consumed
and a new episode begins. Info and recorder metadata carry owned context snapshots.
`EnvCfg.episode_context_in_info` defaults to true. Batched RL deployments may
disable automatic info snapshots while retaining context lifecycle, recorder
metadata and explicit snapshots. Reset retains each row's seed and shared
provenance snapshot even without a recorder. Demo recording materializes
independent metadata rows on access; row UUIDs and mutable
segment lists are owned by each episode. Reset reuses the snapshot from `get_info()`;
overrides that omit it receive one when the option is enabled. The concrete preparation/restore host and
snapshot limitations belong to [scene expansion](../scene-expansion/scene-expansion.md).

## Change sites and focused validation

| Change | Validation surface |
|---|---|
| Cadence resolution in `control_timing.py` and BaseEnv timing | `tests/gym/envs/test_env_timing.py` |
| Seeding and reset | `tests/gym/envs/test_env_seed.py`, `tests/gym/envs/managers/test_event_manager_seed.py` |
| Scene preparation ordering and saved initial-state restore | `tests/gym/envs/task_program/test_simulation_scene_expansion.py` |
| Config, registration, or sensor acquisition switch | Relevant tests under `tests/gym/`; use `rg --files tests` to select the component/registration case |
| Controller or demo bridge | Relevant action/demo/Task Program tests under `tests/gym/envs/` |
| Expert trajectory mode, action layout, or retiming | `tests/gym/envs/test_expert_trajectory.py`, `tests/gym/envs/test_demo.py`, `tests/gym/envs/test_replay.py` |
| CLI discovery | `tests/test_main.py` |

Use `/add-task-env` to select the task implementation route; use `/add-task-program`
for program authoring. Keep runnable deployments task-first and use the
configuration detail page as the owner of physical/semantic composition rules.

## Common failure modes

- Missing Gym ID: package discovery registers Python tasks; a configuration-owned
  Task Program ID requires loading its runnable config first.
- Wrong config values: trace CLI expansion and generic components before inspecting
  Task Program semantic composition. Component selectors and owned inline fields
  cannot coexist.
- Wrong control rate: inspect resolved `step_dt`, not only the requested frequency.
- Controller width/device errors: inspect `_prepare_controller_action()` and the
  active/full joint mapping before changing an action term.
- Early or stale success: preserve bridge completion and reset ordering.

## Initialization failure cleanup

`BaseEnv` queues teardown of its owned simulation when scene declaration, preparation or
later base initialization fails. `EmbodiedEnv` applies the same boundary to
post-base manager/buffer initialization, finalizing an initialized dataset
manager first. Cleanup disables process exit and retains the original error
if resource cleanup also fails. The caller drains `SimulationManager.flush_cleanup_queue()`
after the failed constructor has unwound and its traceback is released, as for
ordinary deferred destruction. Draining inside the exception handler is unsafe:
the traceback can retain native resources not yet registered with the manager.

### Stateful sensor and action lifecycle

`BaseEnv._advance_physics()` starts one simulation-manager update per control
interval and samples participating sensors through its `after_substep` observer.
`EmbodiedEnv._advance_physics()` connects an ActionManager control-interval callback
to the batched simulation update after sensor sampling, and delegates ordinary
actions to the base implementation. ActionManager owns the substep count, term
refresh and inactive-row holds. Intermediate
recording and visualization frames retain the simulation manager's publication cadence.
Direct controller-ready commands bypass
this refresh path; the explicit expert ActionManager route participates. See the sensor-owned [contact history contract](../sensor-system/contact-history.md)
for interval accumulation, counterpart filtering and selective reset.
EmbodiedEnv resets action and event managers before applying reset-mode events.
