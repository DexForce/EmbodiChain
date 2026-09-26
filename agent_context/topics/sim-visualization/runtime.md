# Visualization runtime and controls

Read this when the request needs these details. [Topic overview](sim-visualization.md).

## Manager Lifecycle

| API | Behavior |
|---|---|
| `start_visualization()` | No-op for backend `"none"`; otherwise starts or returns the active runtime and forces an initial capture |
| `notify_visualization_topology_changed()` | Increments the manager-local topology revision |
| `refresh_visualization()` | Rebuilds and asynchronously publishes a manifest immediately |
| `capture_visualization(force=False)` | Refreshes dirty topology, then captures a due frame |
| `capture_visualization_safely()` | Protects simulation progress from visualization errors and logs the first failure |
| `stop_visualization()` | Stops the worker/backend and releases the server port |
| `destroy()` | Stops visualization before queuing deferred simulation cleanup |

During explicit `SimulationManager.update()`, each physics step increments the
visualization step/time counters and attempts a rate-limited capture. A topology
revision mismatch publishes a fresh manifest before its first matching frame.
`BaseEnv.reset()` also requests a forced capture after resetting scene state.
Drawing markers and capturing visualization do not advance physics; interactive
loops call `SimulationManager.update(step=1)` to process Gizmos and step the world.

Manager add methods mark topology dirty for rigid objects, rigid-object groups,
volume/surface deformables, robots, articulations, and `Camera` sensors. Supported
`remove_asset()` branches do the same. The next simulation update refreshes the
browser automatically. Use `refresh_visualization()` when the refresh must
happen before another physics step. Code that changes mesh topology outside
these manager APIs must call `notify_visualization_topology_changed()` itself.

## Scene Representation

`SceneManifest` contains static topology and geometry. `SceneFrame` contains
poses, visibility, camera poses, optional deformable vertices, and overlays.
Both carry `run_id` and `scene_revision`; the backend rejects stale or foreign
frames.

Mesh geometry is identified by a SHA-256 hash of local vertices and faces.
Static nodes sharing geometry are sent through one Viser batched-mesh handle.
Normal frames update only positions, Viser-native `wxyz` quaternions, and visibility.
Identifiers are URL-escaped before becoming Viser path components.

EmbodiChain pose vectors use `(x, y, z, qx, qy, qz, qw)`. The visualization
protocol follows Viser and stores normalized `wxyz` quaternions.
`pose_to_position_wxyz()` converts EmbodiChain `xyz + xyzw` pose vectors at
that boundary and also accepts homogeneous `(..., 4, 4)` matrices. Protocol
fields already named `wxyz` remain protocol-native.

Arena offsets are added to rigid, robot, articulation, and camera poses.
Deformable vertices are stored relative to the corresponding arena node.

### Exported content

| Simulation content | Browser representation |
|---|---|
| `RigidObject` | Complete combined render mesh; multi-segment faces receive matching vertex offsets |
| `RigidObjectGroup` | One node and pose per constituent object |
| `Robot` | One mesh node per non-empty link |
| `Articulation` | One mesh node per non-empty link |
| `VolumeDeformableObject` | Live Newton render-surface vertices and triangles |
| `SurfaceDeformableObject` | Live Newton render-surface vertices and triangles |
| `Camera` | Frustum plus optional low-frequency RGB preview |
| Default ground | 1000 m × 1000 m XY grid, 1 m cells, 10 m sections |
| `SceneOverlays` | Frames, targets, trajectories, and point clouds |

Lights, rigid constraints, markers/gizmos, contact sensors, and stereo cameras
are not exported as scene nodes by the current `SceneExporter`. Visual
materials and textures are not mirrored; meshes use category colors from
`SceneExporter._COLORS`.

## Cameras

Only sensors whose `sensor_type` is exactly `"Camera"` are added to the
manifest. Camera intrinsics produce a vertical field of view and aspect ratio.
DexSim camera poses are converted from OpenGL to the OpenCV convention expected
by Viser frustums by flipping the local Y and Z axes.

RGB capture requires `camera.cfg.enable_color=True`. The exporter calls
`camera.update()`, reads `(num_envs, height, width, 4)` color data, drops alpha,
and publishes RGB. Depth, mask, normal, position, and disparity buffers are not
part of the current camera-preview protocol.

The browser **Cameras** panel selects one environment/camera pair and toggles
its frustum and RGB preview. Camera images use a separate latest-frame queue so
slow rendering or clients cannot accumulate an image backlog.

## Deformables

Volume and surface deformables require the Newton backend, CUDA, and a
particle-capable solver. Their live vertices are sampled at `soft_body_fps`,
independently from `scene_fps`. `SceneExporter` reads the manager's single
deformable registry through `get_surface_vertices()` and
`get_surface_triangles()`; it does not branch on legacy buffer APIs. The facade
returns world-frame render vertices, and the exporter subtracts the arena
offset before publishing below the arena node.

- Both kinds publish the live render surface from DexSim 0.5 typed Newton
  particle-set handles. Visualization does no convex-hull reconstruction or
  nearest-neighbor welding.
- A volume deformable separately exposes its tetrahedral collision surface via
  `get_collision_surface_triangles()`; visualization intentionally uses render
  topology.
- Spawn binding requires identical render vertex/triangle counts across
  replicated instances. A clone topology mismatch fails during preparation.
- Viser does not update mesh vertices in place. `ViserBackend` removes and
  recreates a deformable mesh handle only when a dynamic vertex sample arrives.
  Pose-only frames reuse the current handle.
- A dynamic update with an unknown node or a vertex shape different from the
  manifest geometry is rejected.

## Browser Controls and Overlays

Gizmo implementation lives in `embodichain/lab/sim/objects/gizmo.py`. Native
windows delegate object picking/manipulation to DexSim's entity gizmo and robot
targets to its `IKGizmoController`. Entity interaction defaults on at the first
native window open; `SimulationManagerCfg.enable_entity_gizmo=False` or
`sim.disable_entity_gizmo()` opts out and reopening preserves the choice.
Pure headless and Viser runs do not automatically create native controls.
`SimulationManagerCfg.robot_ik_gizmo` automatically registers robot parts with
solver chain/TCP metadata: native IK activates on I by default, or on the first
update with an open window when `GizmoCfg(ik_start_enabled=True)`. The robot
tutorial uses this startup option. Viser builds IK on the first drag. The manager updates both and preserves native controls across window
reopen. Explicit configuration and disable calls take precedence; caller-owned
native factory controllers are not duplicated. Automatic Viser registration
requires `allow_commands`, independently of native interaction preferences.
Viser transports poses to the
simulation thread. Both robot paths use Newton IK by default or, with
`GizmoCfg(ik_solver="embodichain")`, reuse the configured control-part solver
(including Pink). Chain roots follow the live robot link, including upstream
joint motion; TCP overrides adapt targets without changing the shared solver.

Viser click picking runs on the visualization worker via the existing GUI
event queue. The worker registers the global scene pointer callback only while
**Enable click-to-pick Gizmo** is checked, and removes it when unchecked. Viser
captures camera drags while that callback is registered, so inactive picking
must leave the callback unregistered to preserve orbit and pan controls.
A manifest invalidates cached pick poses until its matching frame
arrives; stale clicks are dropped. The manager validates `PickCommand` run and
revision before attaching a gizmo and tracks picker ownership independently
from explicitly created gizmos. Clearing selection releases only the picker
gizmo. Native entity selection remains entirely DexSim-owned.

The browser GUI has:

- **Environments** — visibility per exported environment;
- **Cameras** — selected environment, camera, frustum, and RGB preview;
- **Overlays** — visibility for frames, targets, trajectories, and point clouds;
- **Gizmos** — transform controls for configured robot, rigid-object, and camera targets.

GUI callbacks enqueue events; the visualization worker applies them, preserving
the single-threaded Viser-handle invariant. Environment visibility affects
static batches, deformables, and camera frustums.

Overlays are backend-neutral protocol objects. Point clouds larger than
`point_cloud_max_points` are deterministically subsampled. The ordinary
`SimulationManager.capture_visualization()` helper currently supplies no
overlay argument; callers needing overlays must use the active
`VisualizationRuntime.capture(..., overlays=...)` path or extend the manager
integration.

## Backpressure, Health, and Telemetry

Scene and camera-image queues each retain one unconsumed frame. A new frame
replaces the old one instead of blocking simulation or accumulating latency.
Topology manifests remain ordered and are always published before their first
matching frame.

Use:

```python
health = sim.visualization_health
stats = sim.visualization_stats
```

`RuntimeHealth` reports status, endpoint, connected-client count, published
scene revision, and a worker error. `RuntimeStats` separates captured,
published, dropped, and rejected scene/image frames and tracks approximate
payload bytes plus capture/upload time.
