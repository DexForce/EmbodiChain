# CLI Reference

EmbodiChain provides a unified CLI, available both as the ``embodichain``
console command and via ``python -m embodichain <subcommand>``. The two are
equivalent; this guide uses the ``python -m embodichain`` form.

---

## Asset Download

List and download simulation assets (robots, objects, scenes, etc.).

```bash
# List all available assets
python -m embodichain.data list

# List assets in a category
python -m embodichain.data list --category robot

# Download a specific asset
python -m embodichain.data download --name CobotMagicArm

# Download all assets in a category
python -m embodichain.data download --category robot

# Download everything
python -m embodichain.data download --all
```

---

## SimReady Asset Pipeline

Convert a raw mesh asset directory into sim_ready assets for simulation.

```bash
# Run the full SimReady pipeline on a single asset directory
python -m embodichain.gen_sim.simready_pipeline.cli.start \
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
python -m embodichain preview-asset \
    --asset_path /path/to/sugar_box.usda \
    --asset_type rigid \
    --preview

# Preview an articulation
python -m embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation \
    --preview

# Headless check (no render window)
python -m embodichain preview-asset \
    --asset_path /path/to/asset.usda \
    --headless
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--asset_path`` | *(required)* | Path to the asset file (``.usd``/``.usda``/``.usdc``/``.obj``/``.stl``/``.glb``) |
| ``--asset_type`` | ``rigid`` | Asset type: ``rigid`` or ``articulation``. URDF files are auto-detected as articulation. |
| ``--uid`` | *(from filename)* | Unique identifier for the asset in the scene |
| ``--init_pos X Y Z`` | ``0 0 0.5`` | Initial position |
| ``--init_rot RX RY RZ`` | ``0 0 0`` | Initial rotation in degrees |
| ``--body_type`` | ``kinematic`` | Body type for rigid objects: ``dynamic``, ``kinematic``, or ``static`` |
| ``--use_usd_properties`` | ``False`` | Use physical properties from the USD file |
| ``--fix_base`` | ``True`` | Fix the base of articulations |
| ``--sim_device`` | ``cpu`` | Simulation device |
| ``--headless`` | ``False`` | Run without rendering window |
| ``--renderer`` | ``hybrid`` | Renderer backend: ``legacy``, ``hybrid``, ``fast-rt``, or ``rt`` |
| ``--preview`` | ``False`` | Enter interactive embed mode after loading |

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``sim`` and ``asset`` in scope
- **``s <N>``** — step the simulation *N* times (default 10)
- **``q``** — quit

---

## Run Environment

Launch a Gymnasium environment for data generation, interactive preview, or trajectory replay.

Task environments are **auto-discovered**: any installed package that declares
an ``embodichain.tasks`` entry point (e.g. the official ``embodichain_tasks``
package) is imported at startup, registering its environments via
``@register_env``. Make sure your task package is pip-installed
(``pip install -e .``) so its tasks are visible to the CLI. The task to launch
is selected by the ``"id"`` field of the gym config.

```bash
# Run an environment with a gym config file
python -m embodichain run-env --gym_config path/to/config.yaml

# Run with multiple environments on GPU
python -m embodichain run-env \
    --gym_config config.yaml \
    --num_envs 4 \
    --device cuda \
    --gpu_id 0

# Preview mode for interactive development
python -m embodichain run-env --gym_config config.yaml --preview

# Headless execution
python -m embodichain run-env --gym_config config.yaml --headless

# Generate data AND record trajectories for later replay
python -m embodichain run-env --gym_config config.yaml --record_trajectory
# trajectories auto-save to ~/.cache/embodichain_data/trajectories/<run_id>/

# Replay a recorded trajectory (kinematic - exact reproduction, default)
python -m embodichain run-env --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt

# Replay with physics re-simulation (dynamic)
python -m embodichain run-env --gym_config config.yaml \
    --replay --replay_trajectory path/to/traj.pt --replay_mode dynamic

# Interactive scrubber (kinematic; step forward/back/jump via terminal)
python -m embodichain run-env --gym_config config.yaml \
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
| ``--renderer`` | ``hybrid`` | Renderer backend: ``legacy``, ``hybrid``, ``fast-rt`` or ``rt`` |
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

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``env`` in scope
- **``q``** — quit

### Replay Mode

When ``--replay`` is enabled (with ``--replay_trajectory <path>``), the env loads a recorded ``.pt`` trajectory and drives it via ``ReplayWrapper``. The replay env must use the same gym config (robot/objects/ActionManager) as the recording env.

Trajectories are recorded by passing ``--record_trajectory`` (or setting ``record_trajectory: true`` in the gym config); recorded episodes auto-save to ``~/.cache/embodichain_data/trajectories/<run_id>/`` (or ``--trajectory_save_dir``) at episode end, and the save path is logged at the end of the run. Point ``--replay_trajectory`` at one of these files (or any ``.pt`` produced by ``env.save_trajectory(path)``).

``--replay_mode`` selects how the trajectory is replayed:

- **``kinematic``** (default) - disable physics and write the recorded object states directly each step. Exact reproduction; produces observations only.
- **``dynamic``** - feed the recorded robot actions back through ``env.step`` so physics re-simulates the scene. Produces the full ``obs/reward/terminated/truncated/info``. Faithful even with an ``ActionManager`` (the raw action is re-preprocessed).
- **``control``** - interactive kinematic scrubber. Terminal commands:

  - **``n``** (or Enter) - next step
  - **``p``** - previous step
  - **``<N>``** - jump to step N
  - **``a``** - auto-play to the end
  - **``r``** - reset to step 0
  - **``q``** - quit

  ``control`` mode needs a render window (re-run without ``--headless``).

``--replay`` and ``--preview`` are mutually exclusive.

---

## Train RL

Launch reinforcement learning training from a JSON or YAML config file.

```bash
# Train with a config file (JSON or YAML)
python -m embodichain train-rl --config embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.yaml

# JSON configs remain supported
python -m embodichain train-rl --config embodichain_tasks/configs/agents/rl/push_cube/train_config.json

# Multi-GPU distributed training
torchrun --nproc_per_node=2 -m embodichain train-rl \
    --config embodichain_tasks/configs/agents/rl/push_cube/train_config.yaml \
    --distributed
```

The direct module entry point remains available:

```bash
python -m embodichain.learning.rl.train --config embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.yaml
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--config`` | *(required)* | Path to the RL training config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--distributed`` | ``None`` | Enable multi-GPU distributed training. If omitted, uses ``trainer.distributed`` from the config. Use ``--no-distributed`` to force single-process training. |

Outputs are written to ``./outputs/<exp_name>_<timestamp>/`` (TensorBoard logs and checkpoints). See the :doc:`../tutorial/rl` tutorial for config structure and training workflow.
