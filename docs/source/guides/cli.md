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
python -m embodichain run-env --gym_config path/to/config.json

# Run with multiple environments on GPU
python -m embodichain run-env \
    --gym_config config.json \
    --num_envs 4 \
    --device cuda \
    --gpu_id 0

# Preview mode for interactive development
python -m embodichain run-env --gym_config config.json --preview

# Headless execution
python -m embodichain run-env --gym_config config.json --headless
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| ``--gym_config`` | *(required)* | Path to gym config file |
| ``--action_config`` | ``None`` | Path to action config file |
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
