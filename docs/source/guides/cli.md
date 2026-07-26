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

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``sim`` and ``asset`` in scope
- **``s <N>``** — step the simulation *N* times (default 10)
- **``q``** — quit

---

## Run Environment

Launch a Gymnasium environment for data generation or interactive preview.

Task environments are **auto-discovered**: any installed package that declares
an ``embodichain.tasks`` entry point (e.g. the official ``embodichain_tasks``
package) is imported at startup, registering its environments via
``@register_env``. Make sure your task package is pip-installed
(``pip install -e .``) so its tasks are visible to the CLI. The task to launch
is selected by the ``"id"`` field of the gym config.

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

### Preview Mode

When ``--preview`` is enabled, an interactive REPL is available:

- **``p``** — enter an IPython embed session with ``env`` in scope
- **``q``** — quit

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

Outputs are written to ``./outputs/<exp_name>_<timestamp>/`` (TensorBoard logs and checkpoints). See the :doc:`../tutorial/rl` tutorial for config structure and training workflow.

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
