# CLI Reference

EmbodiChain provides a unified CLI via ``python -m embodichain <subcommand>``.

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

Launch a Gymnasium environment for data generation or interactive preview.

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

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``env`` in scope
- **``q``** — quit

---

## Train RL

Launch reinforcement learning training from a JSON or YAML config file.

```bash
# Train with a config file (JSON or YAML)
python -m embodichain train-rl --config configs/agents/rl/basic/cart_pole/train_config.yaml

# JSON configs remain supported
python -m embodichain train-rl --config configs/agents/rl/push_cube/train_config.json

# Multi-GPU distributed training
torchrun --nproc_per_node=2 -m embodichain train-rl \
    --config configs/agents/rl/push_cube/train_config.yaml \
    --distributed
```

The direct module entry point remains available:

```bash
python -m embodichain.training.rl.train --config configs/agents/rl/basic/cart_pole/train_config.yaml
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--config`` | *(required)* | Path to the RL training config file (``.json``, ``.yaml``, or ``.yml``) |
| ``--distributed`` | ``None`` | Enable multi-GPU distributed training. If omitted, uses ``trainer.distributed`` from the config. Use ``--no-distributed`` to force single-process training. |

Outputs are written to ``./outputs/<exp_name>_<timestamp>/`` (TensorBoard logs and checkpoints). See the :doc:`../tutorial/rl` tutorial for config structure and training workflow.
