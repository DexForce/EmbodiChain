# Asset Preview

The `preview_asset` script loads a USD or mesh asset into the simulation for visual inspection and debugging, without requiring a full gym environment. It supports both rigid objects (meshes) and articulations (robot-like assets), with an optional interactive session for manipulation.

## Quick Start

Preview a rigid object from a USD file:

```bash
embodichain preview-asset \
    --asset_path /path/to/sugar_box.usda \
    --asset_type rigid
```

Preview an articulation:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation
```

## Viser Browser Preview

Use `--viser` to inspect rigid objects and articulations in a browser without
opening the native simulation window:

```bash
embodichain preview-asset \
    --asset_path /path/to/asset.usda \
    --asset_type rigid \
    --viser
```

Viser implies headless simulation. The command keeps stepping the simulation
until `Ctrl+C`, so dynamic assets continue moving and their poses update in the
browser. Assets are published immediately after loading because the Viser
server starts together with `SimulationManager`.

### Articulation Joint Controls

For articulation previews, `--viser` adds an **Articulation joints** panel by
default. Each articulation has its own folder:

- joints with two finite position limits use sliders;
- joints with one or both limits missing use numeric inputs;
- rotational values are displayed in degrees and prismatic values in meters;
- mimic joints are omitted, and reset buttons restore the pose captured when
  the asset was loaded.

Commands are validated and applied on the simulation thread before every
physics step. The preview controller writes both the current and target joint
positions, and clears velocity and effort, so the selected pose remains stable
even when the asset has no configured joint drive. Articulations whose active
joint names do not map one-to-one to scalar DOFs are left read-only.

Disable the panel with `--no-joint-control`. Joint controls are currently a
Viser-only preview feature; the native DexSim window remains unchanged until it
provides a GUI integration point. When `--preview` is also active, queued
browser changes are applied the next time the REPL executes `s <N>`.

Combine `--viser` with `--preview` to retain the interactive REPL:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.urdf \
    --viser \
    --preview
```

Use `--viser-host`, `--viser-port`, `--viser-fps`, and the other standard
`--viser-*` options to configure the browser server and update rates.

## Asset Type Detection

The asset type is determined as follows:

1. **Explicit**: use `--asset_type rigid` or `--asset_type articulation`.
2. **URDF files**: automatically treated as articulations.
3. **Other files**: loaded as rigid objects when `--asset_type` is omitted.

## Interactive Preview Mode

Pass `--preview` to enter an interactive REPL after the asset is loaded:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation \
    --preview
```

Available commands inside the REPL:

| Command     | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `p`         | Enter an IPython embed session. `sim` and `asset` are in scope.   |
| `s <N>`     | Step the simulation *N* times (default 10).                        |
| `q`         | Quit the simulation.                                               |

Inside the IPython embed session you can freely inspect and manipulate the asset:

```python
# Inspect articulation joint positions
asset.get_qpos()

# Step the simulation
sim.update(step=10)

# Change asset position
asset.set_root_pose(pos=[0, 0, 1.0], rot=[0, 0, 0])
```

## Command-Line Arguments

| Argument             | Description                                                        | Default              |
|----------------------|--------------------------------------------------------------------|----------------------|
| `--asset_path`       | Path to the asset file (`.usd`/`.usda`/`.usdc`/`.obj`/`.stl`/`.glb`/`.urdf`) | **required**         |
| `--asset_type`       | Type of non-URDF asset: `rigid` or `articulation`                  | `rigid` |
| `--uid`              | Unique identifier in the scene                                     | Derived from filename |
| `--init_pos`         | Initial position as `x y z`                                        | `0 0 0.5`            |
| `--init_rot`         | Initial rotation in degrees as `rx ry rz`                          | `0 0 0`              |
| `--body_type`        | Body type for rigid objects: `dynamic`, `kinematic`, `static`      | `kinematic`          |
| `--use_usd_properties` | Use physical properties from the USD file instead of defaults    | `False`              |
| `--fix_base` / `--no-fix_base` | Fix or unfix the base of articulations                  | `True`               |
| `--device`           | Simulation device                                                  | `cpu`                |
| `--headless`         | Run without rendering window                                       | `False`              |
| `--renderer`         | Renderer backend: `hybrid`, `fast-rt` or `rt`            | `hybrid`             |
| `--preview`          | Enter interactive embed mode after loading                         | `False`              |
| `--joint-control` / `--no-joint-control` | Enable or disable Viser articulation controls       | `True`               |
| `--viser`            | Enable the headless Viser browser preview                           | `False`              |
| `--viser-host`       | Viser bind host                                                     | `127.0.0.1`          |
| `--viser-port`       | Viser bind port                                                     | `8080`               |
| `--viser-fps`        | Maximum scene update rate                                           | `15.0`               |
| `--viser-image-fps`  | Maximum camera RGB preview rate                                     | `2.0`                |
| `--viser-soft-body-fps` | Maximum deformable mesh update rate                             | `5.0`                |
| `--viser-env-ids`    | Environment IDs published to Viser, or `all`                       | `0`                  |

## Examples

**Headless smoke test** (no render window):

```bash
embodichain preview-asset \
    --asset_path /path/to/asset.usda \
    --headless
```

**Custom position and rotation**:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation \
    --init_pos 0.5 0 0.0 \
    --init_rot 0 0 90 \
    --preview
```

**Dynamic rigid body** (falls under gravity):

```bash
embodichain preview-asset \
    --asset_path /path/to/box.obj \
    --body_type dynamic \
    --preview
```
