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

Use Viser to inspect selected environments, camera frustums, and all camera RGB
previews locally or remotely. The expanded preview panel separates cameras
created by `record_camera_data` under **Record cameras** from configured
observation sensors under **Sensor cameras**.
Limit the published batch with `--viser-env-ids`, and tune scene, image, or
soft-body publication rates with the corresponding `--viser-*-fps` options. See
{doc}`/overview/sim/viser_visualization` for browser controls and remote-access
details.

## Run and record

Without `--preview` or `--replay`, `run-env` enters offline rollout mode. For
each episode, it asks the task for its demonstration segments, applies every
action through `env.step()`, and commits the complete episode with one reset.
An empty plan, truncation, or failed final task validation discards the whole
attempt. Attempts are bounded by `demo_max_attempts` in the gym config
(default: 3). `--max_episodes` overrides the value in the gym config:

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

### Multi-segment episodes

An episode is the complete task; a segment is one semantic subtask inside it.
For example, moving three objects is one episode containing three pick/place
segments, even if each segment has its own motion trajectory. The task owns
the number, order, and targets of those segments. The runner only manages the
episode lifecycle, termination checks, retry, and commit/discard boundary.

Existing tasks implementing `create_demo_action_list()` remain compatible and
are recorded as a single `legacy` segment. A multi-object task can instead
implement a lazy segment planner:

```python
from embodichain.lab.gym.envs import DemoSegment


def create_demo_segments(self):
    for object_uid in self.object_order:
        # This runs after the preceding segment, so planning sees the latest
        # scene state.
        actions = self.plan_pick_and_place(object_uid)
        yield DemoSegment(
            actions=actions,
            name="pick_and_place",
            target_uid=object_uid,
            instruction=f"Place {object_uid} in its target bin",
            # Segment validation is separate from Gym episode termination.
            validator=lambda uid=object_uid: self.is_object_placed(uid),
        )
```

Gym `terminated` and `truncated` always describe the whole episode, never an
individual segment. A segment normally ends when its action iterable is
exhausted; its optional zero-argument `validator` then returns one boolean per
parallel environment. A failed validator aborts the batch. Episode-level
success termination stops the remaining lazy plan without requesting another
segment.

The executor checks terminal signals after every action and temporarily
disables Gym auto-reset. Dataset recording is transactional: only the explicit
reset after successful final validation saves the episode. Exceptions,
interrupts, failed attempts, and closing an environment with a live rollout
abort pending structured data, videos, and trajectories.

Segment actions pass through the same action-dimension normalization used by
legacy `create_demo_action_list()` tasks. A time-limit truncation is always an
invalid expert rollout, including when it occurs on the planner's final
action, so a task's `max_episode_steps` must be greater than the longest valid
expert plan.

In a vectorized environment, segments and actions remain on one shared planner
clock, but completion is tracked independently. When one environment reports
success, its terminal result and recording cursor become sticky; subsequent
shared actions use a safe hold/no-op command for that row while unfinished rows
continue. Consequently, rollout and trajectory lengths may differ by row.
The commit remains batch-atomic: every row must eventually succeed, while any
failure or truncation aborts the whole attempt.

Failed attempts use `reset(options={"save_data": False})`. This discards
structured datasets, replay trajectories, and camera-video buffers; recorder
options such as `save_failed_episodes` do not override the runner's expert
transaction policy.

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
  - Observations, actions, episode/segment annotations, task metadata, and
    optionally sensor videos.
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
tracked independently. A trajectory is saved only at an explicit successful
episode reset. `close()` is a durability barrier for already committed writes,
not an implicit commit, so an unfinished trajectory is discarded.

Files are named like `traj_env0_000000.pt`. When
`--trajectory_save_dir` is omitted, they are written below:

```text
${EMBODICHAIN_DATA_ROOT:-~/.cache/embodichain_data}/trajectories/<run_id>/
```

Each file contains `states`, `actions`, and `meta`. Metadata includes the actual
per-environment lengths, segment ranges and targets, timestep, robot identity
and DOF, active joint IDs, recorded object IDs, and original environment IDs.
LeRobot exports also contain per-frame `annotation.segment_*` fields and a
`meta/embodichain_episodes.jsonl` sidecar with the complete segment records.
The final directory is also printed when the run finishes.

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

Control mode uses kinematic state restoration and lets you scrub the trajectory
from the terminal or Viser:

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
which to see the scrubbed state. Alternatively, add `--viser` to open an
expanded **Replay control** panel in the browser. Its integer **Frame** slider
jumps directly to any recorded frame, stays synchronized with terminal and
auto-play commands, and pauses auto-play when dragged.

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
