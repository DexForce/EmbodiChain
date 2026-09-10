# Environment framework

`BaseEnv` owns the Gymnasium loop; `EmbodiedEnv` adds configured scene entities,
managers, demonstration collection, and the opt-in Task Program bridge.
Lightweight RL environments have a separate owner in
[rl-learning](../rl-learning/rl-learning.md).

## Entry points and resolution

| Request | Owning file / resolution path |
|---|---|
| Command dispatch | `embodichain/cli/main.py`; `embodichain/__main__.py` is the module entry wrapper |
| Task discovery / `list-task` | `embodichain/cli/list_task.py` → package discovery and runnable config scan |
| Launch / CLI overrides | `embodichain/lab/scripts/run_env.py` → `build_env_cfg_from_args()` → `config_to_cfg()` |
| Generic component expansion | `embodichain/lab/gym/utils/_component_composition.py` |
| Config decoding | `embodichain/lab/gym/utils/gym_utils.py` |
| Import registration | `embodichain/lab/gym/utils/registration.py`; official task imports under `embodichain_tasks/embodichain_tasks/` |
| Environment timing and loop | `embodichain/lab/gym/envs/base_env.py`: `EnvCfg`, `BaseEnv` |
| Scene, manager and demonstration integration | `embodichain/lab/gym/envs/embodied_env.py` |
| Controller-ready commands | `embodichain/lab/gym/envs/types.py`: `ControllerAction` |
| Demo segment execution / outcomes | `embodichain/lab/gym/envs/demo.py` |
| Configured Task Program registration | `embodichain/lab/gym/envs/task_program/registration.py` |
| Timing instrumentation | `embodichain/lab/gym/utils/profiler.py` |

Read [configuration and registration](configuration.md) for task-first layout,
component ownership, path resolution, config-owned IDs, and task listing.
Read [execution](execution.md) for hooks, bridge acceptance, reset ordering,
wrappers, and replay. Read [profiling](profiling.md) only for instrumentation.

## Launcher argument ownership

`add_env_launcher_args_to_parser()` composes the lightweight simulation options
from `embodichain.cli.sim` with Gym config, seed and recording options. When
`require_gym_config=True`, omitted runtime overrides stay `None`: `num_envs`,
`device`, `headless`, `renderer`, `physics`, `arena_space`, and `gpu_id` preserve
file-owned values. Explicit values override runtime settings (`--headless` / `--no-headless`
select either mode); `--physics` can only confirm the file-owned backend. An omitted seed preserves the config seed.
Standalone examples use `add_sim_args_to_parser()` and opt into seed separately.

## Timing contract

`BaseEnv._configure_timing()` resolves `EnvCfg` before constructing the scene:

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

## Lifecycle and boundaries

Construction seeds before scene setup, constructs the robot and sensors, then
initializes simulation state and configured managers. An explicit effective
seed on reset also reaches the event manager; scoped randomization behavior is
owned by [randomization](../randomization/randomization.md).

The step order is:

```text
preprocess → robot control → physics update → interval events
→ observations → evaluation/info → rewards → postprocess
→ elapsed steps and termination → rollout hook → optional reset
```

Raw policy input goes through action `pre` terms. `ControllerAction` skips only
that preprocessing; both paths validate controller keys, batch, joint width,
floating dtype and device, then retain `post` terms and the ordinary Gym loop.

Read final task success before resetting scene/bridge state. A Task Program
reports success only after its bridge completes normally and combines runtime,
post-policy and validator acceptance. Segment acceptance and append-only
dataset persistence are separate contracts; see
[Task Programs](../task-programs/task-programs.md) and
[data pipeline](../data-pipeline/data-pipeline.md).

## Change sites and focused validation

| Change | Validation surface |
|---|---|
| Timing in `base_env.py` | `tests/gym/envs/test_env_timing.py` |
| Seeding and reset | `tests/gym/envs/test_env_seed.py`, `tests/gym/envs/managers/test_event_manager_seed.py` |
| Config or registration | Relevant tests under `tests/gym/`; use `rg --files tests` to select the component/registration case |
| Controller or demo bridge | Relevant action/demo/Task Program tests under `tests/gym/envs/` |
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
