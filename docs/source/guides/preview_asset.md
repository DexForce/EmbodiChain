# Previewing Assets

The `preview-asset` command loads one or more USD, mesh, or URDF assets for
visual inspection and debugging without requiring a Gym environment. It can
open the native DexSim window, publish the scene through Viser, run as a
headless smoke test, or enter an interactive terminal session.

## Quick Start

Preview a rigid object:

```bash
embodichain preview-asset \
    --asset_path /path/to/sugar_box.usda \
    --asset_type rigid
```

URDF files are detected as articulations automatically:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.urdf
```

Pass multiple paths to compare assets in one scene. They are placed along the
positive X axis using `--asset_spacing`:

```bash
embodichain preview-asset \
    --asset_path /path/to/first.usda /path/to/second.usda \
    --asset_spacing 1.5
```

## Visualization Modes

| Mode | Option | Behavior |
|---|---|---|
| Native window | Default | Opens the DexSim viewer and keeps stepping until `Ctrl+C`. |
| Viser browser | `--viser` | Runs headlessly, publishes the live scene, and enables trusted preview controls. |
| Headless check | `--headless` | Loads and validates the asset without opening a viewer. |
| Interactive terminal | `--preview` | Adds a REPL for inspecting state and stepping manually. |

Viser and the native window are mutually exclusive. `--viser` implies
headless simulation; `--headless` alone does not start Viser.

### Viser Browser Preview

Use `--viser` to inspect assets without opening a native window:

```bash
embodichain preview-asset \
    --asset_path /path/to/asset.usda \
    --viser
```

The command prints the browser endpoint, normally
`http://127.0.0.1:8080`, and continues stepping the simulation so dynamic
assets and browser poses stay current.

For articulation previews, Viser adds an **Articulation joints** panel by
default:

- joints with two finite position limits use sliders;
- joints with one or both limits missing use numeric inputs;
- revolute values are displayed in degrees and prismatic values in meters;
- mimic joints are omitted;
- reset buttons restore the joint pose captured after loading.

Commands are validated and applied on the simulation thread before each
physics step. The controller writes current and target positions and clears
velocity and effort, so the selected pose remains stable without configured
joint drives. Articulations whose active joint names do not map one-to-one to
scalar DOFs remain read-only.

Disable the panel with `--no-joint-control`. When `--preview` is also active,
queued browser changes are applied the next time the REPL executes `s <N>`.

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.urdf \
    --viser \
    --preview
```

Use `--viser-host`, `--viser-port`, `--viser-fps`, and the other standard
`--viser-*` options to configure the browser server and update rates. See
{doc}`Viser browser visualization </overview/sim/viser_visualization>` for the
complete backend reference and remote-access guidance.

## Asset Type and Placement

Asset types are resolved as follows:

1. URDF files are always loaded as articulations.
2. Other files use `--asset_type rigid` by default.
3. Pass `--asset_type articulation` for non-URDF articulated assets.

When `--uid` is supplied for multiple assets, it becomes their shared base
identifier and each object receives an `_<index>` suffix. Without it, each
filename supplies the base identifier. `--asset_spacing` controls separation
along the positive X axis.

## Interactive Terminal

Pass `--preview` to enter the preview REPL after loading:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation \
    --preview
```

| Command | Description |
|---|---|
| `p` | Enter an IPython session with `sim` and the `assets` list in scope. |
| `s <N>` | Step the simulation `N` times; the default is 10. |
| `q` | Quit the preview. |

Inside IPython, use the regular simulation APIs:

```python
# Select one loaded asset when several paths were supplied.
asset = assets[0]

# Inspect articulation joint positions.
asset.get_qpos()

# Step the simulation.
sim.update(step=10)

# Change an asset pose.
pose = asset.get_local_pose()
pose[:, 2] = 1.0
asset.set_local_pose(pose)
```

## Command-Line Options

| Option | Default | Description |
|---|---:|---|
| `--asset_path PATH ...` | required | One or more `.usd`, `.usda`, `.usdc`, `.obj`, `.stl`, `.glb`, or `.urdf` files. |
| `--asset_type` | `rigid` | Type for non-URDF files: `rigid` or `articulation`. |
| `--uid` | each filename | Optional shared base identifier; multiple assets receive an index suffix. |
| `--asset_spacing` | `1.0` | Spacing in meters between multiple assets along positive X. |
| `--init_pos X Y Z` | `0 0 0.5` | Initial position of the first asset. |
| `--init_rot RX RY RZ` | `0 0 0` | Initial rotation in degrees. |
| `--body_type` | `kinematic` | Rigid body type: `dynamic`, `kinematic`, or `static`. |
| `--use_usd_properties` | disabled | Use physical properties stored in the USD file. |
| `--fix_base` / `--no-fix_base` | fixed | Fix or unfix articulation bases. |
| `--sim_device` | `cpu` | Simulation device. |
| `--renderer` | `hybrid` | Renderer: `hybrid`, `fast-rt`, or `offline-rt`. |
| `--env_map` | none | Built-in IBL resource name or absolute `.hdr`, `.png`, or `.exr` path. |
| `--headless` | disabled | Run without the native window. |
| `--preview` | disabled | Enter the interactive terminal after loading. |
| `--joint-control` / `--no-joint-control` | enabled | Enable or disable the Viser articulation panel. |
| `--viser` | disabled | Enable the headless Viser browser preview. |
| `--viser-host` | `127.0.0.1` | Viser bind host. |
| `--viser-port` | `8080` | Viser TCP port. |
| `--viser-fps` | `15.0` | Maximum scene update rate. |
| `--viser-image-fps` | `2.0` | Maximum camera RGB preview rate. |
| `--viser-soft-body-fps` | `5.0` | Maximum deformable mesh update rate. |
| `--viser-env-ids` | `0` | Published environment IDs, or `all`. |

Run `embodichain preview-asset --help` for the authoritative option list.

## Additional Examples

Run a headless load check:

```bash
embodichain preview-asset \
    --asset_path /path/to/asset.usda \
    --headless
```

Set a custom initial transform:

```bash
embodichain preview-asset \
    --asset_path /path/to/robot.usd \
    --asset_type articulation \
    --init_pos 0.5 0 0 \
    --init_rot 0 0 90 \
    --preview
```

Preview a dynamic rigid body:

```bash
embodichain preview-asset \
    --asset_path /path/to/box.obj \
    --body_type dynamic
```
