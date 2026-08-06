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

## Preview Asset

Preview a USD or mesh asset in the simulation without writing code.

```bash
# Preview a rigid object
embodichain preview-asset \
    --asset_path /path/to/sugar_box.usda \
    --asset_type rigid \
    --preview

# Preview an articulation
embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation \
    --preview

# Headless check (no render window)
embodichain preview-asset \
    --asset_path /path/to/asset.usda \
    --headless

# Control articulation joints in Viser
embodichain preview-asset \
    --asset_path /path/to/robot.urdf \
    --viser
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--asset_path`` | *(required)* | One or more asset paths (``.usd``/``.usda``/``.usdc``/``.obj``/``.stl``/``.glb``/``.urdf``) |
| ``--asset_type`` | ``rigid`` | Asset type: ``rigid`` or ``articulation``. URDF files are auto-detected as articulation. |
| ``--uid`` | *(from filename)* | Unique identifier for the asset in the scene |
| ``--init_pos X Y Z`` | ``0 0 0.5`` | Initial position |
| ``--init_rot RX RY RZ`` | ``0 0 0`` | Initial rotation in degrees |
| ``--body_type`` | ``kinematic`` | Body type for rigid objects: ``dynamic``, ``kinematic``, or ``static`` |
| ``--use_usd_properties`` | ``False`` | Use physical properties from the USD file |
| ``--fix_base`` | ``True`` | Fix the base of articulations |
| ``--sim_device`` | ``cpu`` | Simulation device |
| ``--headless`` | ``False`` | Run without rendering window |
| ``--renderer`` | ``hybrid`` | Renderer backend: ``hybrid``, ``fast-rt``, or ``rt`` |
| ``--preview`` | ``False`` | Enter interactive embed mode after loading |
| ``--joint-control`` / ``--no-joint-control`` | ``True`` | Enable or disable articulation joint controls in Viser previews |

The Viser articulation panel displays rotational joints in degrees and
prismatic joints in meters. It excludes mimic joints, leaves articulations with
unsupported multi-DOF mappings read-only, and provides per-articulation reset
buttons. The native DexSim window does not yet expose these controls.

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``sim`` and ``asset`` in scope
- **``s <N>``** — step the simulation *N* times (default 10)
- **``q``** — quit

---

(cli-run-environment)=
## Run Environment

Launch a Gymnasium environment for data generation, interactive preview, or trajectory replay.

For an end-to-end explanation of mode selection, preview, the differences
between dataset/video/trajectory recording, and all three replay modes, see
{doc}`run_env`.

Task environments are **auto-discovered**: any installed package that declares
an ``embodichain.tasks`` entry point is imported at startup, registering its
environments via ``@register_env``. The main ``embodichain`` distribution
already includes and registers the official ``embodichain_tasks`` import
package, so no separate task installation is needed. Repository-style task
config paths resolve from the source checkout or installed wheel. The task to
launch is selected by the ``"id"`` field of the gym config.

```bash
# Run an environment with a gym config file
embodichain run-env --gym_config path/to/config.yaml

# Run with multiple environments on GPU
embodichain run-env \
    --gym_config config.yaml \
    --num_envs 4 \
    --device cuda \
    --gpu_id 0

# Preview mode for interactive development
embodichain run-env --gym_config config.yaml --preview

# Headless execution
embodichain run-env --gym_config config.yaml --headless

# Headless browser visualization
embodichain run-env --gym_config config.yaml --viser

# Publish selected environments at controlled rates
embodichain run-env --gym_config config.yaml \
    --viser \
    --viser-env-ids 0 2 \
    --viser-fps 15 \
    --viser-image-fps 2 \
    --viser-soft-body-fps 5

# Generate data AND record trajectories for later replay
embodichain run-env --gym_config config.yaml --record_trajectory
# trajectories auto-save to ~/.cache/embodichain_data/trajectories/<run_id>/

# Replay a recorded trajectory (kinematic - exact reproduction, default)
embodichain run-env --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt

# Replay with physics re-simulation (dynamic)
embodichain run-env --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt --replay_mode dynamic

# Interactive scrubber (kinematic; step forward/back/jump via terminal)
embodichain run-env --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt --replay_mode control
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--gym_config`` | *(required)* | Path to gym config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--action_config`` | ``None`` | Path to action config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--num_envs`` | ``1`` | Number of parallel environments |
| ``--device`` | ``cpu`` | Device (``cpu`` or ``cuda``) |
| ``--headless`` | ``False`` | Run in headless mode |
| ``--renderer`` | ``auto`` | Renderer backend: ``auto``, ``hybrid``, ``fast-rt`` or ``rt`` |
| ``--arena_space`` | ``5.0`` | Arena space size |
| ``--gpu_id`` | ``0`` | GPU ID to use |
| ``--preview`` | ``False`` | Enter interactive preview mode |
| ``--filter_visual_rand`` | ``False`` | Filter out visual randomization |
| ``--filter_dataset_saving`` | ``False`` | Filter out dataset saving |
| ``--max_episodes`` | *(from config)* | Override the maximum number of rollout episodes |
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
| ``--viser-image-fps`` | every environment step | Maximum camera RGB preview rate when explicitly supplied; otherwise `run-env` captures after every environment step |
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

See {doc}`run_env` for examples of inspecting and stepping ``env`` from the
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

See {doc}`run_env` for the recorded file contents, environment compatibility
requirements, vectorized replay behavior, and a complete record/replay
workflow.

### Profiling

Pass ``--profile`` to record per-section wall time of the reset/step pipeline
and print a breakdown on ``env.close()``. Add ``--profile_output prof.json`` to
also dump the report as JSON.

```bash
embodichain run-env --gym_config config.yaml --headless --device cuda \
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
Within ``manual_update``, ``gizmo_update`` and ``world_update`` are sampled once
per physics substep. Optional window recording and Viser publication are
reported separately as ``window_record_capture`` and
``visualization_capture``, so they are not attributed to physics time.

---

## Train RL

Launch reinforcement learning training from a JSON or YAML config file.

```bash
# Train with a config file (JSON or YAML)
embodichain train-rl --config embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.yaml

# JSON configs remain supported
embodichain train-rl --config embodichain_tasks/configs/agents/rl/push_cube/train_config.json

# Multi-GPU distributed training
torchrun --nproc_per_node=2 -m embodichain train-rl \
    --config embodichain_tasks/configs/agents/rl/push_cube/train_config.yaml \
    --distributed
```

The module entry point remains available for compatibility:

```bash
python -m embodichain.learning.rl.train --config embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.yaml
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--config`` | *(required)* | Path to the RL training config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--distributed`` | ``None`` | Enable multi-GPU distributed training. If omitted, uses ``trainer.distributed`` from the config. Use ``--no-distributed`` to force single-process training. |
| ``--profile`` | ``False`` | Same as ``run-env --profile``; profiles the training gym env during rollouts. Requires ``trainer.gym_config``. |
| ``--profile_output`` | ``None`` | Dump the profiling report as JSON on ``env.close()`` (requires ``--profile``). |

See the Profiling section under Run Env for report format. Outputs are written to ``./outputs/<exp_name>_<timestamp>/`` (TensorBoard logs and checkpoints). See the :doc:`../tutorial/rl` tutorial for config structure and training workflow.

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

# Kinematic solver and neural planner benchmarks
embodichain benchmark robotics-kinematic-solver --solvers all
embodichain benchmark planners-neural-planner --num-waypoints 1 3 5

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
