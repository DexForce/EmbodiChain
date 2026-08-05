# Browser visualization with Viser

```{currentmodule} embodichain.lab.visualization
```

EmbodiChain can publish a running simulation to a
[Viser](https://viser.studio/main/) browser interface. This is useful for
headless servers, SSH workflows, multi-environment inspection, and lightweight
debugging when opening the native DexSim window is inconvenient.

Programmatic Viser configurations are read-only by default. The common
`--viser` launcher enables registered simulation controls for trusted clients,
including Gizmo targets and the articulation panel used by `preview-asset`, but
does not allow arbitrary physics, action, or asset operations. Simulation
remains owned by
{class}`~embodichain.lab.sim.SimulationManager`; Viser runs on a background
update thread and keeps only the latest unconsumed frame so a slow browser
cannot accumulate simulation lag.

## Quick start

From the repository root, run a supported tutorial with `--viser`:

```bash
python scripts/tutorials/sim/create_scene.py --viser
```

The terminal prints the server endpoint, normally
`http://127.0.0.1:8080`. Open it in a browser while the simulation is running.

Other tutorials that demonstrate specific visualization paths include:

```bash
# Camera frustum and low-frequency RGB preview
python scripts/tutorials/sim/create_sensor.py --viser

# Rigid-object groups
python scripts/tutorials/sim/create_rigid_object_group.py --viser

# CUDA deformables
python scripts/tutorials/sim/create_softbody.py --viser
python scripts/tutorials/sim/create_cloth.py --viser
```

Gym environments use the same launcher options:

```bash
embodichain run-env --gym_config path/to/config.yaml --viser
```

Enabling `--viser` makes the environment headless automatically. The native
DexSim window is not required. The same one-way rule applies to programmatic
configuration: `SimulationManagerCfg` forces `headless=True` whenever
`visualization.backend == "viser"`. Setting `headless=True` alone does not
enable Viser.

## Programmatic configuration

Set {attr}`~embodichain.lab.sim.SimulationManagerCfg.visualization` when
constructing the simulation:

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        visualization=VisualizationCfg(
            backend="viser",
            scene_fps=15.0,
            sensor_image_fps=2.0,
            soft_body_fps=5.0,
            env_ids=[0],
            viser_server=ViserServerCfg(
                host="127.0.0.1",
                port=8080,
            ),
        ),
    )
)
```

{class}`~embodichain.lab.sim.SimulationManager` starts the configured Viser
runtime during construction. Calling
{meth}`~embodichain.lab.sim.SimulationManager.start_visualization` again is
safe and returns the existing runtime while it is active.

Assets may be added or removed after startup. The manager marks visualization
topology dirty and publishes a new scene manifest on the next simulation
update. Use
{meth}`~embodichain.lab.sim.SimulationManager.refresh_visualization` when an
immediate manual refresh is required, and
{meth}`~embodichain.lab.sim.SimulationManager.stop_visualization` to release
the server before destroying the simulation.

## Supported scene content

The browser scene currently includes:

- `RigidObject`, including multi-segment render meshes;
- each constituent object in a `RigidObjectGroup`;
- every visible link of `Robot` and `Articulation`;
- dynamic `SoftObject` and `ClothObject` geometry;
- camera frustums and low-frequency RGB previews;
- read-only Gizmo frames, or interactive transform controls when commands are
  explicitly enabled;
- a default XY ground grid with 1 m cells and 10 m major sections;
- optional coordinate-frame, target, trajectory, and point-cloud overlays when
  using {class}`SceneOverlays`.

Static geometry is content-addressed and batched. Normal scene frames update
only poses, orientations, and visibility.

Lights and rigid constraints do not own ordinary scene meshes, so they are not
exported as Viser mesh nodes. Gizmos are exported separately under
`/interactions/gizmos`; their affected objects remain ordinary scene nodes.
Camera depth, segmentation masks, normals, and position buffers are also not
currently shown in the browser RGB panel.

## Interactive Gizmos

The `--viser` launcher option enables browser Gizmo commands by default. Create
each Gizmo through `SimulationManager.enable_gizmo`; a pure browser process can
omit the DexSim handle:

```python
sim.enable_gizmo("cube", enable_native=False)
```

Viser and DexSim use the same deferred target-control path:

- rigid-object drags set its local arena pose;
- camera drags set its local pose;
- robot drags invoke FK/IK for the selected `control_part`.

Viser callbacks only enqueue immutable pose commands. `update_gizmos()` drains
and applies them on the simulation thread; manual `SimulationManager.update()`
does this automatically. Automatic-update loops must continue calling
`update_gizmos()` and `capture_visualization_safely()`.

Only one client owns a Gizmo from drag start through drag end or disconnect.
Other clients are returned to the latest authoritative simulation pose. Gizmo
control currently requires `num_envs=1`. The Viser transform-control appearance
is browser-native and therefore does not exactly reproduce DexSim's arrow,
corner, tag, and ring styling.

Programmatic configurations remain explicit:
`VisualizationCfg(backend="viser", allow_commands=True)` enables interaction,
while `allow_commands=False` creates read-only Gizmo frames and disabled joint
inputs. Command-line
`--viser` grants connected browser clients permission to mutate the simulation,
and Viser does not add application authentication here. Keep the default
loopback bind for local use; expose the server only behind an authenticated,
trusted network boundary.

## Asset-preview joint controls

`embodichain preview-asset --asset_path <articulation> --viser` registers a
simulation-thread joint-control provider. Its static control descriptions are
included in the scene manifest and its authoritative values in each scene
frame. Browser callbacks enqueue immutable scalar commands; the preview loop
validates their run and scene revision before writing articulation state.

The **Articulation joints** panel uses degree sliders for bounded revolute
joints, meter sliders for bounded prismatic joints, and numeric inputs when one
or both limits are absent. Mimic joints are omitted. The controller writes both
current and target positions and clears velocity and effort before each step,
which makes the preview independent of drive configuration. Use
`--no-joint-control` to disable it.

This controller is currently specific to the Viser asset-preview path. The
protocol and backend command sink are kept separate from the controller so a
native DexSim GUI can reuse the simulation-side behavior later.

## Deformable objects

Cloth and soft bodies require dynamic vertex updates and are intentionally
sampled independently from rigid-body poses:

- **Cloth** uses the physical cloth vertices and a welded mapping of the source
  render triangles. Its browser topology matches the simulated surface.
- **Soft bodies** expose live PhysX collision vertices through DexSim, but
  DexSim does not expose the collision triangle connectivity. EmbodiChain
  therefore visualizes a stable convex-hull surface over those vertices. The
  preview follows deformation but omits concave render-mesh details.

Viser mesh handles do not support in-place vertex replacement. A deformable
mesh is therefore recreated only when a low-frequency vertex sample is due.
The default `soft_body_fps=5.0` is a deliberate performance tradeoff; reduce it
for large deformable meshes or multiple visible environments.

## Cameras and browser controls

The **Cameras** panel lets you select one environment and sensor. It provides:

- a camera-frustum visibility switch;
- an RGB-preview switch;
- independent environment and camera selectors.

Only the selected frustum and RGB preview are shown. RGB images use a separate
latest-frame queue and `sensor_image_fps`, so image rendering cannot build up a
backlog behind simulation frames. Setting `sensor_image_fps=None` captures
after each eligible simulation step; `run-env --viser` uses this mode by
default.

The **Environments** panel independently hides or shows exported environments.
For more than 16 environments, it switches to a scalable **Show all
environments** toggle plus a selected-environment dropdown instead of creating
one GUI checkbox per environment.
The **Overlays** panel controls frames, trajectories, targets, and point clouds.
Hiding an environment affects its static meshes, deformable meshes, and camera
frustum together.

## Configuration reference

### `VisualizationCfg`

| Field | Default | Description |
|---|---:|---|
| `backend` | `"none"` | Use `"viser"` to enable browser visualization. |
| `scene_fps` | `15.0` | Maximum rigid pose and overlay capture rate. |
| `env_ids` | `[0]` | Environment IDs published to the browser; `None` selects every simulation environment. |
| `max_visible_envs` | `None` | Optional safety limit; `None` disables the limit. |
| `point_cloud_max_points` | `100000` | Per-overlay point-cloud limit. |
| `sensor_image_fps` | `2.0` | Maximum RGB preview capture rate; `None` synchronizes capture to simulation steps. |
| `soft_body_fps` | `5.0` | Maximum cloth and soft-body vertex rate. |
| `allow_commands` | `False` | Allow trusted Viser clients to use registered Gizmos and joint controls that mutate simulation targets. |
| `viser_server` | `ViserServerCfg()` | HTTP/WebSocket bind settings. |

### `ViserServerCfg`

| Field | Default | Description |
|---|---:|---|
| `host` | `"127.0.0.1"` | Server bind interface. |
| `port` | `8080` | Server TCP port. |
| `label` | `"EmbodiChain"` | Browser application label. |
| `verbose` | `False` | Print detailed Viser diagnostics. |

### Command-line options

Scripts using the common environment launcher accept:

| Option | Default | Description |
|---|---:|---|
| `--viser` | disabled | Enable headless Viser with trusted browser simulation controls. |
| `--viser-host` | `127.0.0.1` | Bind interface. |
| `--viser-port` | `8080` | Server port. |
| `--viser-fps` | `15.0` | Scene pose update limit. |
| `--viser-image-fps` | `2.0`; `run-env`: every environment step | Camera RGB update limit when explicitly supplied. |
| `--viser-soft-body-fps` | `5.0` | Deformable mesh update limit. |
| `--viser-env-ids ID ...` | `0` | Environment IDs to publish, or `all`. |

Application launchers only need to preserve an explicit `--headless` request:

```python
if not args.headless:
    sim.open_window()
```

`SimulationManager.open_window()` returns `False` without opening a native
window when the Viser backend is configured or running, so launchers do not
need their own Viser condition. Starting Viser while the native window is
already open is rejected.

## Health and telemetry

Use the manager properties to inspect the running service:

```python
print(sim.visualization_health)
print(sim.visualization_stats)
```

`visualization_health` reports runtime state, endpoint, connected client count,
published scene revision, and worker errors. `visualization_stats` reports
captured, published, dropped, and rejected frames together with approximate
payload bytes and capture/upload time.

## Remote access

The default loopback binding is the safest choice on a remote worker. Forward
the port through SSH:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@worker-host
```

Then open `http://127.0.0.1:8080` locally. Avoid binding an unauthenticated
worker directly to a public interface. Production deployments should place the
Viser port behind an authenticated gateway.

## Troubleshooting

| Symptom | Check |
|---|---|
| Browser cannot connect | Verify the printed endpoint, port, firewall, and SSH forwarding. |
| `env_ids` validation error | Every selected ID must be below `SimulationManager.num_envs` and any configured `max_visible_envs` limit. |
| Newly added asset is absent | Step the simulation once or call `refresh_visualization()`. |
| Soft body looks simplified | This is the collision-vertex convex-hull preview described above. |
| Browser motion is expensive | Lower scene, image, or soft-body FPS and publish fewer environments. |
| Server port remains occupied | Call `stop_visualization()` or destroy the simulation cleanly. |

## Related pages

- {doc}`sim_manager`
- {doc}`sim_assets`
- {doc}`sim_sensor`
- {doc}`/tutorial/create_scene`
- {doc}`/tutorial/sensor`
