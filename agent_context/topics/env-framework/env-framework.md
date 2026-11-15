# Environment framework

`BaseEnv` owns the Gymnasium loop. `EmbodiedEnv` composes scene entities,
managers, demonstration collection and the opt-in Task Program bridge.
[Lightweight RL](../rl-learning/rl-learning.md) has a separate environment owner.

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
| Cable scene declarations and raw recording | `gym_utils.py` → `EmbodiedEnv._setup_interactive_objects()`; `trajectory_state.py` → `observations.get_entity_state` |
| Demo segment execution / outcomes / collection qualification | `embodichain/lab/gym/envs/demo.py` |
| Configured Task Program registration | `embodichain/lab/gym/envs/task_program/registration.py` |
| Timing instrumentation | `embodichain/lab/gym/utils/profiler.py` |

Read [configuration and registration](configuration.md) for task-first layout,
physical/semantic component ownership, config IDs and paths; [execution](execution.md)
for reset/hooks, bridge acceptance, storage and replay; [profiling](profiling.md)
for instrumentation. Read [fluid Gym integration](fluid-replay.md) for online
Task Program execution, DexUni rigid/DFSPH dynamics and fluid coordinates.
Official model defaults stay in the robot catalog; task
contracts own locomotion actions, observations, rewards and termination.
Reference-motion task state/snapshots live under
`embodichain_tasks/locomotion/tracking/`, with model rollout boundaries in
[robot world model](../robot-world-model/robot-world-model.md).

## Launcher argument ownership

`run-task` loads existing deployments; `gensim run` creates/binds a deployment
before using the same Gym/Task Program execution. `run-env` is removed.
Gym launcher arguments compose lightweight `cli.sim` options with config, seed
and recording options. Omitted runtime overrides stay `None` and preserve file
values; explicit runtime values override them. `--physics` only confirms the
file-owned backend. Omitted seed preserves the config seed.

`--disable_sensor`/`--disable-sensor` disables acquisition without removing
reusable declarations. BaseEnv copies config, applies filters, then determines
render demand before World construction, including overridden demand hooks.
Headless tasks without visual consumers may use NoRender. Custom visual code
reserves demand through `_requires_native_renderer()` or selects a renderer;
see [render selection](../simulation-system/rendering.md#norender-initialization).

## Timing contract

`resolve_control_timing()` is the single execution-free resolver used before
scene construction. `physics_dt` is finite/positive. Without target frequency,
`sim_steps_per_control` is a positive integer. A finite positive
`target_control_frequency` takes precedence and must represent an integer number
of physics steps within the resolver's tolerance; the source owns the exact
check. Resolution never changes physics dt or silently rounds an impossible rate.

`step_dt = physics_dt * sim_steps_per_control`. Render cadence and compatibility
frequency aliases derive from it. Source trajectories are retimed to this command
period before `env.step()`; their source timing cannot redefine the physics loop.

## Expert trajectory contract

`expert_trajectory` is source-neutral and leaves the policy action space alone.
Position mode stores active qpos. Position-velocity mode requires effective
qpos/qvel targets and stores flat `[active qpos, active qvel]` with ordered names,
slices, schema and `step_dt`. Missing qvel rejects before stepping; hold/inactive
targets use zero qvel.

ObservationManager owns raw acquisition and the field schema. Environment
rollout callbacks own one backing storage, episode identity and lifecycle;
policy transforms/noise remain distinct from raw history. T transitions retain
T+1 raw states including the terminal state. Exact historical reads validate
row/frame/episode ownership and return owned values/validity, without querying
live state or substituting older frames. See
[recording and replay](execution.md#recording--replay) and
[manager views](../manager-functor/manager-functor.md#raw-observations-and-consumer-views).

Use `advance_observation` to feed a compiled `RuleSession` from explicit
control slots and qualified host captures; `evaluate_recorded_rules` evaluates
the same graph in bounded recorded chunks. Results remain row-local at each
recorded boundary. Missing or foreign evidence cannot satisfy a predicate.

## Lifecycle and boundaries

Construction seeds before scene setup, acquires enabled sensors, prepares
simulation and initializes dependent managers. Robot joint selections are
instance-owned and preserve control-part order across repeated setup.

Policy steps validate/process/apply actions, advance physics and interval
observers, capture objective/observations, evaluate rewards/termination, append
rollout state, then optionally reset. ActionManager refreshes participating
terms at substeps; direct ControllerAction commands bypass that refresh path.
Reset action/event state before reset-mode events. [Sensors](../sensor-system/contact-history.md)
own interval contact accumulation and selected-row history reset.

Read final success before resetting scene/bridge. Task Program bridges combine
runtime, post-policy and segment acceptance. With final predicates,
`is_task_success()` also requires fresh raw physical measurements. The demo
executor seals each row's terminal measurements/outcome; reset recording consumes
them before clearing row state. Dataset persistence and optional physical
objectives remain separate. Source-bound Real2Sim episodes
use the qualified synchronous receipt owner, not the ordinary reset recorder.

`EpisodeContext` follows reset and recorder eligibility. A trusted B=1
`episode_prepare` callback runs after reset events and before physical objectives
and final observations; it is not a functor parameter. Recording starts from
that final observation. Optional info snapshots may be disabled for batched RL
without disabling context lifecycle or recorder metadata. Row UUIDs/segments
remain independently owned; see [scene expansion](../scene-expansion/scene-expansion.md).

Failed constructors defer native cleanup and preserve the original error.
Finalize initialized dataset managers first, release scene locals/tracebacks,
then drain the simulation cleanup queue. Never flush while a failed constructor's
traceback can still retain unregistered native children.

## Change sites and focused validation

| Boundary | Coverage |
|---|---|
| Cadence, seeds and selected resets | `tests/gym/envs/test_env_timing.py`, `test_env_seed.py`, `tests/gym/envs/managers/test_event_manager_seed.py` |
| Expert commands, recording and replay | `tests/gym/envs/test_expert_trajectory.py`, `test_demo.py`, `test_replay.py` |
| Config/registration, manager scheduling and scene preparation | `tests/gym/utils/`, `tests/gym/envs/managers/`, `tests/gym/envs/task_program/` |
| CLI discovery | `tests/test_main.py` |

Missing IDs require package discovery or config-owned registration; incorrect
config requires tracing generic components before semantic composition. Diagnose
rate errors from resolved `step_dt`, controller errors from active/full joint
mapping, and stale success from bridge/reset order. Use the owning add-task skill
when creating a new deployment rather than duplicating composition rules here.
