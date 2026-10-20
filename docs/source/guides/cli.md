# CLI Reference

EmbodiChain provides a unified CLI through the ``embodichain`` console
command. ``python -m embodichain <command>`` is an equivalent fallback.
Run ``embodichain --help`` to list commands or
``embodichain <command> --help`` for complete command arguments.

---

## Asset Download

List and download simulation assets (robots, objects, scenes, etc.).

```bash
# List all available assets
embodichain data list

# List assets in a category
embodichain data list --category robot

# Download a specific asset
embodichain data download --name CobotMagicArm

# Download all assets in a category
embodichain data download --category robot

# Download everything
embodichain data download --all
```

---

## SimReady Asset Pipeline

Convert a raw mesh asset directory into sim_ready assets for simulation.

```bash
# Run the full SimReady pipeline on a single asset directory
embodichain simready \
    --input_dir /path/to/raw_mesh_folder \
    --output_root /path/to/output_folder \
    --category YourCategory
```

Select the source preparation strategy in
``embodichain/gen_sim/simready_pipeline/configs/gen_config.json`` via
``ingest.source_preparation.mode``. Supported modes are ``blender`` and
``trimesh``.

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--input_dir`` | *(required)* | Directory containing the raw asset files |
| ``--output_root`` | *(required)* | Directory where processed assets are written |
| ``--category`` | *(required)* | Category hint passed into the pipeline |

The generated output contains the canonical source mesh under ``asset_source/``, the final SimReady mesh under ``asset_simready/``, and USD export files under ``asset_usd/`` when export succeeds.

---

(cli-preview-asset)=
## Preview Asset

Load one or more USD, mesh, or URDF assets without writing a simulation script:

```bash
embodichain preview-asset --asset_path /path/to/robot.urdf --viser
```

Run `embodichain preview-asset --help` for the authoritative option list. See
{doc}`preview_asset` for visualization modes, multi-asset placement, Viser
joint controls, the interactive terminal, and worked examples.

---

(cli-preview-robot)=
## Preview Robot

List exact robot and variant IDs, then preview the selected configuration:

```bash
embodichain list-robot
embodichain preview-robot --robot FrankaPanda
embodichain preview-robot --robot DexforceW1 --variant v025/DH_PGC --viser
```

The native window uses I to activate or toggle configured control-part IK
Gizmos. Viser supports Gizmo dragging, joint controls, and pose reset.
See {doc}`preview_robot` for robot files, overrides, and bounded headless runs.

---

(cli-list-tasks)=
## List Tasks

Discover installed tasks and show their environment IDs and supported expert
demo or RL capabilities:

```bash
embodichain list-task
embodichain list-task --category manipulation
embodichain show-task embodichain_tasks:repeated_pick_place
```

`show-task` displays a logical task's named deployments, launch commands and
available validation records. `list-task --export-html task-gallery.html`
exports the same catalog as a static gallery. For simulator-free discovery,
pass `--config-root PACKAGE=PATH` with a `configs/tasks` directory.
See {doc}`task_catalog` for catalog authoring and gallery export.

---

(cli-list-robots)=
## List Robots

Discover installed robot families and their supported configurations:

```bash
embodichain list-robot
python -m embodichain list-robot
embodichain list-robot --help
```

The command reads lightweight catalog metadata without constructing robot
configurations, initializing simulation, or downloading assets.

| Column | Meaning |
|---|---|
| `Robot ID` | Registered family ID, such as `FrankaPanda`, `URRobot`, or `DexforceW1` |
| `Variants` | Supported models or configuration profiles; `*` marks the default |
| `Control DOF` | Unique independent joints in the default control parts; mimic followers are excluded and overlapping groups count once |
| `Control Parts` | Default group names and joint counts; `eef (1+1m)` means one independent joint and one mimic follower |
| `Mimic Joints` | Follower joint names from the default URDF, including passive followers outside the control parts |

`-` means unavailable metadata. `None` in the mimic column means that the
catalog declares no mimic joints. A count marked `total` includes all joints
in that part when mimic membership is unknown.

UR models share one family row. Dual-arm variants show the DOFs of each base
robot, and W1 variants show body releases and hand combinations. Equivalent
profiles are grouped together. Long values wrap to fit the terminal, and
the table width is bounded between 80 and 120 columns.
`NO_COLOR=1 embodichain list-robot` disables terminal colors.

See {doc}`robot_catalog` for variant selection, DOF examples, third-party
catalog registration, and the current import paths.

---

(cli-run-taskironment)=
## Run Task

Launch a Gymnasium environment for data generation, interactive preview, or trajectory replay.
``embodichain run-task`` is the sole task execution command.

For an end-to-end explanation of mode selection, preview, the differences
between dataset/video/trajectory recording, and all three replay modes, see
{doc}`run_task`.

Task environments are **auto-discovered**: any installed package that declares
an ``embodichain.tasks`` entry point is imported at startup, registering its
environments via ``@register_env``. The main ``embodichain`` distribution
already includes and registers the official ``embodichain_tasks`` import
package, so no separate task installation is needed. Repository-style task
config paths resolve from the source checkout or installed wheel. The task to
launch is selected by the ``"id"`` field of the gym config. Pass a runnable
config (for example ``task.ur5.yaml``), not a pure reusable environment
component that has only ``environment_id``. A componentized task may select one
backend with ``environment.component`` or expose ``environment.default`` and
``environment.newton`` variants in the same deployment.

```bash
# Run an environment with a gym config file
embodichain run-task --gym_config path/to/config.yaml

# Run with multiple environments on GPU
embodichain run-task \
    --gym_config config.yaml \
    --num_envs 4 \
    --device cuda \
    --gpu_id 0

# Preview mode for interactive development
embodichain run-task --gym_config config.yaml --preview

# Headless execution
embodichain run-task --gym_config config.yaml --headless

# Headless browser visualization
embodichain run-task --gym_config config.yaml --viser

# Publish selected environments at controlled rates
embodichain run-task --gym_config config.yaml \
    --viser \
    --viser-env-ids 0 2 \
    --viser-fps 15 \
    --viser-image-fps 2 \
    --viser-soft-body-fps 5

# Generate data AND record trajectories for later replay
embodichain run-task --gym_config config.yaml --record_trajectory
# trajectories auto-save to ~/.cache/embodichain_data/trajectories/<run_id>/

# Replay a recorded trajectory (kinematic - exact reproduction, default)
embodichain run-task --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt

# Replay with physics re-simulation (dynamic)
embodichain run-task --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt --replay_mode dynamic

# Interactive scrubber (kinematic; step forward/back/jump via terminal)
embodichain run-task --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt --replay_mode control
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--gym_config`` | *(required)* | Path to a runnable gym config with ``id`` (``.json``, ``.yaml``, or ``.yml``) |
| ``--action_config`` | ``None`` | Path to action config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--num_envs`` | ``1`` | Number of parallel environments |
| ``--device`` | *(config/backend default)* | Explicit device override (for example ``cpu`` or ``cuda:0``); omission preserves the configured or selected-backend default |
| ``--physics`` | *(from config)* | File-owned physics backend (``default`` or ``newton``); this option may confirm the same value but cannot switch a Gym config to another backend |
| ``--headless`` | ``False`` | Run in headless mode |
| ``--renderer`` | *(config; ``auto`` if absent)* | Renderer backend: ``auto``, ``hybrid``, ``fast-rt`` or ``rt`` |
| ``--arena_space`` | ``5.0`` | Arena space size |
| ``--gpu_id`` | ``0`` | GPU ID to use |
| ``--preview`` | ``False`` | Enter interactive preview mode |
| ``--filter_visual_rand`` | ``False`` | Filter out visual randomization |
| ``--filter_dataset_saving`` | ``False`` | Filter out dataset saving |
| ``--max_episodes`` | *(from config)* | Override the exact number of persisted per-environment episodes; vector batches are trimmed to this count |
| ``--record_trajectory`` | ``False`` | Record per-object kinematic trajectories during generation (for replay). Episodes auto-save to ``--trajectory_save_dir`` (or ``~/.cache/embodichain_data/trajectories/<run_id>/``) |
| ``--trajectory_save_dir`` | ``None`` | Directory for auto-saved trajectories (default: ``~/.cache/embodichain_data/trajectories/<run_id>/``) |
| ``--replay`` | ``False`` | Replay a recorded trajectory (``--replay_trajectory`` required; mutually exclusive with ``--preview``) |
| ``--replay_trajectory`` | ``None`` | Path to the ``.pt`` trajectory file to replay |
| ``--replay_mode`` | ``kinematic`` | Replay mode: ``kinematic`` (exact, physics off), ``dynamic`` (feed recorded actions, physics on), ``control`` (interactive scrubber) |
| ``--profile`` | ``False`` | Enable per-section wall-time profiling of reset/step; prints a breakdown report on ``env.close()`` |
| ``--profile_output`` | ``None`` | Dump the profiling report as JSON to this path on ``env.close()`` |
| ``--viser`` | ``False`` | Enable headless Viser and allow trusted clients to drag configured Gizmos |
| ``--viser-host`` | ``127.0.0.1`` | Viser bind interface |
| ``--viser-port`` | ``8080`` | Viser HTTP/WebSocket port |
| ``--viser-fps`` | ``15.0`` | Maximum rigid pose and overlay update rate |
| ``--viser-image-fps`` | every environment step | Maximum camera RGB preview rate when explicitly supplied; otherwise `run-task` captures after every environment step |
| ``--viser-soft-body-fps`` | ``5.0`` | Maximum cloth and soft-body vertex rate |
| ``--viser-env-ids`` | ``0`` | Space-separated environment IDs published to Viser, or ``all`` |

The Viser panel supports environment visibility, camera-frustum selection, RGB
preview, and overlay visibility. For supported object types, programmatic
configuration, remote access, and performance details, see
:doc:`../overview/sim/viser_visualization`.

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``env`` in scope
- **``q``** — quit

See {doc}`run_task` for examples of inspecting and stepping ``env`` from the
embedded session, and for the distinction between interactive preview and the
Viser browser backend.

### Replay Mode

When ``--replay`` is enabled (with ``--replay_trajectory <path>``), the env loads a recorded ``.pt`` trajectory and drives it via ``ReplayWrapper``. The replay env must use the same gym config (robot/objects/ActionManager) as the recording env.

Trajectories are recorded by passing ``--record_trajectory`` (or setting ``record_trajectory: true`` in the gym config); recorded episodes auto-save to ``~/.cache/embodichain_data/trajectories/<run_id>/`` (or ``--trajectory_save_dir``) at episode end, and the save path is logged at the end of the run. Point ``--replay_trajectory`` at one of these files (or any ``.pt`` produced by ``env.save_trajectory(path)``).

``--replay_mode`` selects how the trajectory is replayed:

- **``kinematic``** (default) - disable physics and write the recorded object states directly each step. Exact reproduction; produces observations only.
- **``dynamic``** - feed the recorded robot actions back through ``env.step`` so physics re-simulates the scene. Produces the full ``obs/reward/terminated/truncated/info``. Faithful even with an ``ActionManager`` (the raw action is re-preprocessed).
- **``control``** - interactive kinematic scrubber. Terminal commands:

  - **``n``** - next step immediately (no Enter required)
  - **``p``** - previous step immediately (no Enter required)
  - **``<N>``** - jump to step N
  - **``a``** - start auto-play; press any key to pause
  - **``r``** - reset to step 0
  - **``q``** - quit

  ``control`` mode needs a render window (re-run without ``--headless``).
  Dataset saving is disabled automatically in this mode.

``--replay`` and ``--preview`` are mutually exclusive.

See {doc}`run_task` for the recorded file contents, environment compatibility
requirements, vectorized replay behavior, and a complete record/replay
workflow.

### Profiling

Pass ``--profile`` to record per-section wall time of the reset/step pipeline
and print a breakdown on ``env.close()``. Add ``--profile_output prof.json`` to
also dump the report as JSON.

```bash
embodichain run-task --gym_config config.yaml --headless --device cuda \
    --profile --profile_output prof.json --max_episodes 2
```

The profiler instruments the full step/reset chain with hierarchical, nested
section names (a parent's time includes its children). Example report:

```
section                     calls   mean(ms)      min      max     std  total(s)     %par
-------------------------------------------------------------------------------------------
step                          196     33.214    30.012   45.330   2.841     6.510   100.0%
  step.sim_update              196     12.410    11.802   18.321   0.902     2.432    37.3%
  step.get_obs                 196      8.230     7.510   10.612   0.512     1.613    24.8%
    step.get_obs.sensor          196      7.510     6.800    9.401   0.480     1.472    91.3%
      step.get_obs.sensor.render_camera_group   196   7.388  ...   100.0%
      step.get_obs.sensor.sensor_fetch          196   0.121  ...     1.6%
    step.get_obs.proprio        196      0.538     0.429    0.966   0.082     0.105     6.5%
    step.get_obs.extend         196      0.535     0.342    5.475   0.396     0.105     6.5%
  step.update_sim_state        196      1.342     0.736    4.256   0.808     0.263     4.0%
  ...
reset                           1   1075.858  ...   100.0%
  reset.initialize_episode      1    999.485  ...    92.9%
    reset.initialize_episode.event_reset      1   682.162  ...
    reset.initialize_episode.record_camera_save  1 315.435 ...
```

Notes:

- Only **wall time** is profiled. GPU-memory profiling is not available in this
  release.
- Every registered **event** and **observation** functor is timed individually
  and automatically (via ``ManagerBase._call_functor``), nesting under its
  manager call site -- e.g. ``step.update_sim_state.event_interval.record_camera``
  or ``step.get_obs.extend.obs_compute.norm_robot_eef_joint``. ``calls``
  reflects the firing count (interval event functors fire every
  ``interval_step``).
- For GPU workloads run with ``--device cuda``. The default
  ``sync_cuda=False`` keeps overhead low and reflects CPU-side cost (including
  any syncs the sim performs internally); set ``sync_cuda=True`` on
  ``EnvProfilerCfg`` for accurate absolute GPU timings (it forces
  ``torch.cuda.synchronize()`` at section boundaries).
- The first ``warmup_steps`` (default 5) step/reset samples are discarded so
  JIT/cuDNN autotune setup does not skew the averages.
- ``%par`` is the share of the immediate parent section's total; ``(other)``
  is the parent total minus its measured children (inter-section overhead).
- Set ``nvtx=True`` on ``EnvProfilerCfg`` to also emit NVTX ranges, which show
  up named in an Nsight Systems timeline when running under ``nsys profile``.

In environment code, set
``cfg.profiler = EnvProfilerCfg(enable_time=True, ...)``
(``cfg.profiler is None`` leaves profiling disabled unless
``cfg.sim_cfg.profiler`` is configured directly). The profiler remains
available as ``env._profiler``; call
``env._profiler.report()`` to print mid-run. The report is flushed in
``close()`` **before** ``sim.destroy()`` (which exits the process).

The profiler is owned by the simulation layer and can also be used without a
Gym environment:

```python
from embodichain.lab.sim import ProfilerCfg, SimulationManager, SimulationManagerCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        profiler=ProfilerCfg(enable_time=True, warmup_steps=0),
    )
)
sim.update(step=10)
sim.profiler.report()
```

Standalone updates are reported below ``sim_update``. When an environment owns
the manager, it reuses the same instance and simulation sections stay below the
existing ``step.sim_update`` path. The legacy ``EnvProfiler`` and
``EnvProfilerCfg`` imports remain aliases of ``Profiler`` and ``ProfilerCfg``.
Within ``physics_steps``, ``gizmo_update`` and ``world_update`` are sampled once
per physics substep. Optional window recording and Viser publication are
reported separately as ``window_record_capture`` and
``visualization_capture``, so they are not attributed to physics time.

---

(cli-preview-lerobot-data)=
## Preview LeRobot Data

Print and validate one recorded LeRobot episode without launching the
simulator:

```bash
embodichain preview_lerobot_data \
    outputs/lerobot/task_program \
    --latest \
    --episode 0 \
    --expect-segments 3
```

The positional path must be an exact dataset root containing
`meta/info.json`, unless `--latest` is used to select the newest direct child.
`--expect-segments` is an optional exact-count assertion; it does not select,
split, or modify segments.

| Argument | Default | Description |
|---|---|---|
| ``dataset_root`` | *(required)* | Dataset root, or parent directory with ``--latest`` |
| ``--episode`` | ``0`` | Episode index to inspect |
| ``--expect-segments`` | *(unchecked)* | Fail unless the episode has exactly this many segments |
| ``--latest`` | ``False`` | Select the newest direct child dataset |

The command prints dataset format, robot, FPS, state/action shapes and ranges,
task text, segment frame ranges, subtask descriptions, and sidecar success. It
returns status 0 when all checks pass, 1 for a validation mismatch, and 2 when
the path, episode, or dataset cannot be loaded. For the complete validation
contract and a comparison with LeRobot's official Rerun visualization, see
{ref}`Inspect Recorded LeRobot Data <tutorial_data_expansion_preview>`.

---

(cli-dataset-tools)=
## Offline Dataset Tools

Use an exact finalized dataset root containing `meta/info.json`:

```bash
embodichain dataset validate /absolute/path/to/dataset \
    --media-frame-counts --output /tmp/validation.json
embodichain dataset split /absolute/path/to/dataset /tmp/split.json \
    --fractions 0.8 0.1 0.1 --seed 42 --success-only
embodichain dataset recover /absolute/path/to/dataset
embodichain dataset recover /absolute/path/to/dataset --repair
```

| Command | Purpose |
| --- | --- |
| `dataset validate` | Check frames, timing, action/language references, segments, provenance and optional media |
| `dataset split` | Write a quality-filtered manifest that keeps source episodes and derivatives together |
| `dataset recover` | Diagnose commit evidence; `--repair` can restore missing sidecars after disk verification |

These commands operate without simulation. Stop writers before inspection or
repair. They use an exact root and do not support preview's `--latest` option.
Validation/recovery returns `1` for errors or unresolved evidence; successful
operations return `0`, and invalid requests return `2`. Run each subcommand
with `--help` for arguments. See {doc}`offline_expert_datasets` for complete
parameters, report fields, quality/grouping rules, and recovery boundaries.

The separate `embodichain-lerobot-export SOURCE DESTINATION` command runs in a
Python 3.12 environment pinned to LeRobot 0.6.1. Installation, official language
recipes, and depth units are covered in {doc}`lerobot_export`.

---

## Train RL

Launch reinforcement learning training from a JSON or YAML config file.

```bash
# Train with a config file (JSON or YAML)
embodichain train-rl --config embodichain_tasks/configs/tasks/classic_control/cart_pole/agents/ppo.yaml

# JSON configs remain supported
embodichain train-rl --config embodichain_tasks/configs/tasks/manipulation/push_cube/agents/ppo.json

# Multi-GPU distributed training
torchrun --nproc_per_node=2 -m embodichain train-rl \
    --config embodichain_tasks/configs/tasks/manipulation/push_cube/agents/ppo.json \
    --distributed
```

The module entry point remains available for compatibility:

```bash
python -m embodichain.learning.rl.train --config embodichain_tasks/configs/tasks/classic_control/cart_pole/agents/ppo.yaml
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--config`` | *(required)* | Path to the RL training config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--distributed`` | ``None`` | Enable multi-GPU distributed training. If omitted, uses ``trainer.distributed`` from the config. Use ``--no-distributed`` to force single-process training. |
| ``--profile`` | ``False`` | Same as ``run-task --profile``; profiles the training gym env during rollouts. Requires ``trainer.gym_config``. |
| ``--profile_output`` | ``None`` | Dump the profiling report as JSON on ``env.close()`` (requires ``--profile``). |
| ``--resume`` | ``None`` | Resume an explicit checkpoint file, or select ``latest`` / ``best`` from ``--run``. |
| ``--run`` | ``None`` | Source training run for ``--resume latest`` or ``--resume best``. |
| ``--resume-scheduler`` | ``restore`` | Continue the saved learning-rate curve. ``restart`` keeps optimizer momentum and starts the curve from the current config's learning rate. |

### Resume training

PPO, GRPO, and APG use the same resume arguments:

```bash
embodichain train-rl --config train.yaml --resume outputs/source/checkpoints/policy.pt
embodichain train-rl --config train.yaml --resume latest --run outputs/source
embodichain train-rl --config train.yaml --resume best --run outputs/source
embodichain train-rl --config train.yaml --resume latest --run outputs/source --resume-scheduler restart
```

Training resumes into a new output run. Budgets remain cumulative: a checkpoint
with 100 completed updates and a target of 300 continues for 200 updates. A
target already reached performs zero updates. APG complete-rollout mode uses
``iterations`` as its update target unless ``total_timesteps`` is supplied.

Resume restores policy and normalization statistics, optimizer, scheduler,
counters, best evaluation state, and Python/NumPy/Torch RNG. The environment
starts a new episode. Compatibility checks run before collection and report
the differing algorithm, policy, environment, input/output layout, or optimizer
fields. Existing standard checkpoints and APG schema-v1 checkpoints load their
available state; missing metadata and RNG are reported. GRPO with KL requires
its saved reference policy.

CLI and programmatic trainers can exchange checkpoints. When either trainer
lacks CLI configuration context, loading reports that limitation and checks
the runtime policy, algorithm, spaces, and available trainer metadata.
APG additionally checks its effective rollout, gradient-accumulation, and
action settings. Older schema-v2 files without APG settings load with a warning.

``restore`` preserves the original scheduler horizon. Extending a zero-ending
linear curve past that horizon requires ``restart``; an unspecified new horizon
is bound to the remaining update budget. Subsequent checkpoints preserve the
restarted curve. ``best`` falls back to ``latest`` when no best checkpoint exists
and prints the selected file.

Variable-horizon APG accepts an update budget or an explicit scheduler horizon.
With a transition budget, the restored linear horizon is checked before each
actual optimizer update.

Periodic and best saves update ``run-manifest.json`` after atomically writing
a completed-update checkpoint. Configuration snapshots are frozen from the
loaded configuration on the first save and reused by subsequent saves. A final
save reuses a periodic or best checkpoint from the same update. An interrupted
update retains the preceding checkpoint. Distributed resume requires the same
process count; every rank
participates in saving and loading, while rank zero owns file writes. Each rank
restores its own RNG and observation-normalization buffers.

See the Profiling section under Run Env for report format. Outputs are written to ``./outputs/<exp_name>_<timestamp>/`` (TensorBoard logs and checkpoints). See the :doc:`../tutorial/rl` tutorial for config structure and training workflow.

---

## Policy Evaluation

### Pretrained policies

Download an official policy and open its simulator task:

```bash
embodichain eval-policy --pretrained g1-flat-ppo-newton \
    --viewer --command 0.3 0 0 --device cuda:0 --sim-device gpu
```

Model IDs are directory names under `policies/` in
[DexForceAI/embodichain_model](https://huggingface.co/DexForceAI/embodichain_model).
The shipped default selects a tested HF commit. `--revision <commit-or-tag>`
can select another repository snapshot; named revisions resolve to one commit
before the index and model are downloaded. That snapshot must use a supported
model format and an installed task package.

The first run downloads only the selected model bundle, then resolves robot
assets through the existing asset downloader. Later runs reuse downloaded
files. The published locomotion configs default to `fast-rt`; `--duration 20`
bounds a Viewer run. Omit `--viewer` and use `--episodes 1 --num-envs 1` for
Headless evaluation.

`--cache-dir` changes the model cache root, which defaults to
`~/.cache/embodichain/policies`. Robot assets still use `EMBODICHAIN_DATA_ROOT`.
Reports default to `./outputs/policy_eval/<model-id>/` and record the actual HF
repository, commit and model ID. `--output` overrides that output location.

`--pretrained` cannot be combined with RUN, `--profile`, `--checkpoint`,
`--config` or `--gym-config`: each published bundle supplies its matching
weights and configuration. `--offline` remains an external Profile option.

### Local checkpoints

Evaluate the latest checkpoint from an EmbodiChain training run:

```bash
embodichain eval-policy outputs/my_policy_<timestamp>
```

Open a simulator task in the Viewer:

```bash
embodichain eval-policy outputs/my_policy_<timestamp> \
    --checkpoint best \
    --viewer \
    --renderer hybrid
```

Evaluate an explicit EmbodiChain checkpoint:

```bash
embodichain eval-policy \
    --checkpoint /path/to/policy.pt \
    --config /path/to/train.yaml \
    --gym-config /path/to/gym.yaml
```

### Main arguments

| Argument | Default | Description |
|---|---|---|
| ``RUN`` | *(optional)* | Training run containing ``run-manifest.json`` |
| ``--pretrained`` | *(optional)* | Official pretrained model ID; supplies a complete RUN |
| ``--revision`` | Tested HF commit | Repository revision for ``--pretrained`` |
| ``--cache-dir`` | Model or Profile cache | Cache root for ``--pretrained`` or external Profiles |
| ``--checkpoint`` | ``latest`` with RUN | ``latest``, ``best``, or a checkpoint path |
| ``--config`` | RUN manifest | Training configuration override |
| ``--gym-config`` | RUN manifest | Simulator task configuration override |
| ``--episodes`` | Training configuration | Number of completed task episodes |
| ``--num-envs`` | Training configuration | Number of parallel Headless environments |
| ``--viewer`` | Headless | Open the original simulator task in the DexSim Viewer |
| ``--control-steps`` | Viewer runs continuously | Exact number of Policy actions |
| ``--duration`` | *(optional)* | Duration converted to integer control steps |
| ``--command`` | Task default | Native velocity command ``vx vy yaw_rate`` or external Profile command |
| ``--keymap`` | ``wasd`` | Viewer command keys: ``wasd`` or ``arrows`` |
| ``--renderer`` | Training configuration or ``hybrid`` | Viewer renderer |
| ``--device`` | Training configuration | PyTorch inference device |
| ``--sim-device`` | Inference device | Simulation device |
| ``--output`` | RUN/checkpoint evaluations; ``outputs/policy_eval/<model-id>`` for pretrained | Evaluation output parent directory |

External Motion Profiles use the same command with `--profile`. See
{doc}`policy_evaluation` for training-run layout, execution paths, Viewer
controls, output reports, and the complete ANYmal-C example.

---

## Annotate Grasp

Launch the browser-based grasp-region annotation tool.

```bash
embodichain annotate-grasp --mesh_path /path/to/object.ply
```

Run ``embodichain annotate-grasp --help`` for sampling, gripper-length, port,
and device options.

---

## URDF Convex Decomposition

Generate convex collision meshes and an updated URDF.

```bash
embodichain decompose-urdf \
    --urdf_path ./assets/robot.urdf \
    --output_urdf_name robot_convex.urdf
```

Run ``embodichain decompose-urdf --help`` for hull-count, inertia, and scaling
options.

---

## Benchmarks

Run the packaged benchmark suites through the same CLI:

```bash
# RL train/evaluate/report workflow
embodichain benchmark rl --tasks push_cube --algorithms ppo

# Kinematic solver and motion-generation benchmarks
embodichain benchmark robotics-kinematic-solver --solvers all
embodichain benchmark motion-generation --suite smoke

# Atomic actions, grasp generation, and workspace analysis
embodichain benchmark atomic-action --smoke
embodichain benchmark grasp-pose-generator --device auto
embodichain benchmark workspace-analyzer
```

Use ``embodichain benchmark --help`` to list benchmark suites and
``embodichain benchmark <suite> --help`` for suite-specific arguments.

---

## Workspace Cache

Inspect disk usage and manage workspace analyzer cache sessions:

```bash
embodichain workspace-cache list
embodichain workspace-cache info <session>
embodichain workspace-cache size
embodichain workspace-cache clean <session>
```

Use ``embodichain workspace-cache clean --all`` to remove every workspace
analyzer cache session; the command asks for confirmation before deletion.
