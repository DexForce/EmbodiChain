# Running Environments with `run-env`

`embodichain run-env` is the common entry point for loading a configured
Gymnasium environment. It supports three primary workflows: inspecting an
environment interactively, running an expert rollout while recording data, and
replaying a previously recorded trajectory.

```{list-table} Execution modes
:header-rows: 1
:widths: 18 27 28 27

* - Mode
  - Command switch
  - What it does
  - Typical use
* - Rollout
  - No mode switch
  - Generates task actions, steps the environment, and lets configured
    dataset or video recorders save each episode.
  - Expert demonstration generation and task smoke tests.
* - Preview
  - `--preview`
  - Resets the environment and opens an interactive IPython session on
    request. Dataset saving is disabled automatically.
  - Inspecting scene state, stepping manually, and checking observations.
* - Replay
  - `--replay --replay_trajectory <file>`
  - Restores recorded states, re-applies recorded actions, or provides an
    interactive trajectory scrubber, depending on `--replay_mode`.
  - Reproducing a rollout and diagnosing dynamics or task behavior.
```

`--preview` and `--replay` are mutually exclusive. Browser visualization with
`--viser` is a display backend rather than a fourth execution mode, so it can
be used while running or replaying an environment.

## Inputs and startup behavior

Every invocation requires a gym config:

```bash
embodichain run-env --gym_config path/to/gym_config.yaml
```

JSON, YAML, and YML files are supported. The config's `id` selects the
registered environment, while the rest of the file describes its simulation,
robot, sensors, managers, episode limits, and optional dataset recorders. If a
task uses the action bank, pass its action graph separately:

```bash
embodichain run-env \
    --gym_config embodichain_tasks/configs/gym/pour_water/gym_config.json \
    --action_config embodichain_tasks/configs/gym/pour_water/action_config.json
```

At startup, `run-env`:

1. discovers installed task packages through the `embodichain.tasks` entry
   point and executes their initialization hooks;
2. loads the gym and optional action configs;
3. applies CLI overrides such as `--num_envs`, `--device`, `--renderer`, and
   `--max_episodes`;
4. creates the environment selected by the gym config's `id`; and
5. enters rollout, preview, or replay mode.

Use `embodichain run-env --help` for the complete option list. The
{ref}`CLI Reference <cli-run-environment>` also lists defaults and
visualization arguments.

## Preview an environment

Preview mode is intended for inspection before an expensive data-generation
run:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --preview
```

After constructing and resetting the environment, the terminal accepts:

- `p`: enter an IPython session with `env` in scope;
- `q`: close the preview.

IPython is required only when entering the embedded session. Install it with
`pip install ipython` if the `p` command reports that it is unavailable.

Inside the session, use the regular Gymnasium interface and EmbodiChain helpers
to inspect state or advance the simulation. For example:

```python
# Inspect the current robot configuration.
env.unwrapped.robot.get_qpos()

# Advance with an action that is valid for this task.
obs, reward, terminated, truncated, info = env.step(action)

# Display an RGB sensor observation.
env.unwrapped.preview_sensor_data("camera")
```

Preview mode does not execute the task's expert policy. It also sets
`filter_dataset_saving=True`, so configured structured dataset recorders do not
write debugging episodes. Visual randomization remains enabled unless
`--filter_visual_rand` is supplied.

### Native window and Viser

`--preview` controls the interactive terminal workflow. `--viser` instead
selects the browser visualization backend and makes the simulation headless:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --viser \
    --viser-host 127.0.0.1 \
    --viser-port 8080
```

Use Viser to inspect selected environments, camera frustums, RGB previews, and
overlays locally or remotely. Limit the published batch with
`--viser-env-ids`, and tune scene, image, or soft-body publication rates with
the corresponding `--viser-*-fps` options. See
{doc}`/overview/sim/viser_visualization` for browser controls and remote-access
details.

## Run and record

Without `--preview` or `--replay`, `run-env` enters offline rollout mode. For
each episode, it asks the task for a demonstration action list, applies the
actions through `env.step()`, discards invalid generations, and resets the
environment. `--max_episodes` overrides the value in the gym config:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --action_config path/to/action_config.yaml \
    --headless \
    --device cuda \
    --max_episodes 10
```

Headless execution is normally preferred for throughput. Use
`--filter_dataset_saving` for a rollout smoke test that should not create a
structured dataset.

### Choose the recording output you need

EmbodiChain uses "recording" for three related but distinct outputs:

```{list-table} Recording outputs
:header-rows: 1
:widths: 22 28 25 25

* - Output
  - How it is enabled
  - Contents
  - Intended consumer
* - Structured dataset
  - A dataset manager in the gym config.
  - Observations, actions, task metadata, and optionally sensor videos.
  - Imitation-learning or data-processing pipelines.
* - Debug or demo video
  - `record_camera_data` or `record_camera_data_async` as an interval event
    functor in the gym config.
  - Human-viewable RGB video from a configured camera pose.
  - Visual debugging, reports, and demonstrations.
* - Replay trajectory
  - `--record_trajectory`.
  - Per-step robot and object kinematic state, raw task action, and replay
    metadata in a PyTorch `.pt` file.
  - `run-env --replay` and `ReplayWrapper`.
```

Dataset video is still structured training data; it is not interchangeable
with a replay trajectory. Conversely, `--record_trajectory` does not configure
a LeRobot dataset or export an MP4.

For structured datasets, see {doc}`/tutorial/data_generation` and
{doc}`/overview/gym/dataset_functors`. For human-viewable video, see
{doc}`/overview/gym/event_functors`.

### Record replayable trajectories

Add `--record_trajectory` to a normal rollout:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --action_config path/to/action_config.yaml \
    --record_trajectory \
    --trajectory_save_dir outputs/trajectories
```

The recorder stores the robot root pose and complete joint position, the raw
action before ActionManager preprocessing, and the pose or joint state of scene
rigid objects and articulations. Each environment in a vectorized rollout is
tracked independently. A trajectory is auto-saved at episode reset and again
on close if an unfinished buffer still contains steps.

Files are named like `traj_env0_000000.pt`. When
`--trajectory_save_dir` is omitted, they are written below:

```text
${EMBODICHAIN_DATA_ROOT:-~/.cache/embodichain_data}/trajectories/<run_id>/
```

Each file contains `states`, `actions`, and `meta`. Metadata includes the actual
per-environment lengths, timestep, robot identity and DOF, active joint IDs,
recorded object IDs, and original environment IDs. The final directory is also
printed when the run finishes.

## Replay a trajectory

Replay requires the trajectory file and the gym config used to record it:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --replay \
    --replay_trajectory outputs/trajectories/traj_env0_000000.pt
```

Keep the robot, active joint selection, scene object IDs, action processing,
and episode horizon compatible with the recording run. Robot DOF or active
joint mismatches are rejected. Missing or additional scene objects produce
warnings. A single-environment trajectory can be broadcast with
`--num_envs`; a multi-environment trajectory requires the replay environment
count to match. Recorded lengths that exceed the current episode horizon are
clamped.

If the task needs an action config to construct its environment or
ActionManager, pass the same `--action_config` during replay as during
recording.

If the gym config contains dataset recorders, add `--filter_dataset_saving`
when the replay is for inspection only. Control replay applies this filter
automatically.

### Kinematic replay

Kinematic replay is the default and the best first check:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --replay \
    --replay_trajectory path/to/trajectory.pt \
    --replay_mode kinematic
```

Physics is disabled and recorded poses and joint positions are written into the
scene at every step. This gives an exact visual/state reproduction and renders
new observations, but it does not recompute meaningful task rewards or
successes from the recorded actions. At completion, `run-env` reports the
maximum robot joint-state error against the recording.

### Dynamic replay

Dynamic replay sends each recorded raw action back through `env.step()`:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --replay \
    --replay_trajectory path/to/trajectory.pt \
    --replay_mode dynamic
```

Physics, ActionManager preprocessing, observations, rewards, and termination
logic all run normally. Use this mode to test whether the behavior can be
re-simulated rather than merely restored. Results may diverge if the replay
configuration changes physics, control, randomization, assets, or timestep
settings that are not stored in the trajectory file.

### Interactive control replay

Control mode uses kinematic state restoration but lets you scrub the trajectory
from the terminal:

```bash
embodichain run-env \
    --gym_config path/to/gym_config.yaml \
    --replay \
    --replay_trajectory path/to/trajectory.pt \
    --replay_mode control
```

The commands are:

- `n`: next step;
- `p` or `b`: previous step;
- `<N>` followed by Enter: jump to step `N`;
- `a`: auto-play, then press any key to pause;
- `r`: return to step 0;
- `q`: quit.

Run control mode with a native render window. `--headless` leaves no window in
which to see the scrubbed state.

## Recommended workflow

Use the modes in this order when bringing up a task:

1. Run `--preview --filter_visual_rand` to inspect the initial scene and sensor
   output without saving a dataset.
2. Run one headless episode with `--filter_dataset_saving` and
   `--record_trajectory` to validate the expert policy and create a compact
   replay artifact.
3. Replay the artifact in `kinematic` mode to inspect the exact recorded
   motion, then use `dynamic` mode if physics reproducibility matters.
4. Remove `--filter_dataset_saving`, choose the desired episode count, and run
   the full data-generation job.

For rollout profiling, renderer selection, and every available CLI option, see
the {ref}`CLI Reference <cli-run-environment>`.
