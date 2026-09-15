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
soft bodies, cloth, robots, articulations, and `Camera` sensors. Supported
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
Normal frames update only positions, `wxyz` quaternions, and visibility.
Identifiers are URL-escaped before becoming Viser path components.

EmbodiChain pose vectors use `(x, y, z, qw, qx, qy, qz)`. The protocol uses
normalized `wxyz` quaternions. `pose_to_position_wxyz()` is the conversion
boundary and also accepts homogeneous `(..., 4, 4)` matrices.

Arena offsets are added to rigid, robot, articulation, and camera poses.
Deformable vertices are stored relative to the corresponding arena node.

### Exported content

| Simulation content | Browser representation |
|---|---|
| `RigidObject` | Complete combined render mesh; multi-segment faces receive matching vertex offsets |
| `RigidObjectGroup` | One node and pose per constituent object |
| `Robot` | One mesh node per non-empty link |
| `Articulation` | One mesh node per non-empty link |
| `SoftObject` | Live collision vertices with a cached convex-hull surface |
| `ClothObject` | Live physical vertices with render triangles mapped onto the welded physical vertex buffer |
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

Soft bodies and cloth require GPU physics. Their live vertices are sampled at
`soft_body_fps`, independently from `scene_fps`.

- DexSim does not expose soft-body collision triangle connectivity.
  `SoftBodyData.collision_surface_triangles` therefore caches a SciPy
  `ConvexHull` over rest collision vertices. The preview follows deformation
  but cannot preserve concave render detail.
- Cloth maps all render-mesh triangles onto DexSim's welded rest-vertex buffer
  with `cKDTree`. Construction raises `RuntimeError` if the mapping distance
  exceeds the scale-relative tolerance.
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
event queue. A manifest invalidates cached pick poses until its matching frame
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

## Preview Nodes

`SceneExporter.set_preview_groups()` registers translucent copies of a robot or
articulation. Each `PreviewGroupCfg` reuses the source link meshes, adds
`preview_link` nodes with a constant `SceneNode.opacity`, and keeps them hidden
until a pose arrives. A tinted `color` uploads its own geometry entry (and its
own batched mesh); `color=None` reuses the source entries and shares the real
links' batch. The group materializes on the next `build_manifest()`, so a
caller registering at runtime must also publish a manifest through
`refresh_scene()`.

Preview poses never come from the simulation: each frame the caller passes one
`PreviewNodeUpdate` per group to `capture(..., preview_updates=...)`, holding
detached poses in the same per-environment frame as `body_link_pose`. The
exporter adds the arena offset and marks the group visible. Groups without an
update stay hidden at the environment origin.

`embodichain.lab.visualization.authoring.SequencePreview` is the first consumer:
it owns a playback cursor over an `AuthoringSession`'s compiled trajectory and
turns `preview_qpos` waypoints into link poses through the robot's analytic
forward kinematics. It anchors each link with that link's current simulated
pose, so the preview overlaps the rendered robot exactly at the current
configuration and absorbs fixed chain-versus-model offsets. It performs two
forward-kinematics evaluations per frame and never writes joint state.

## Custom Panels

`VisualizationRuntime.register_panel(PanelSpec(...))` adds an
application-defined side panel. The spec carries a `build` callback, an
optional `apply_state` callback, and an optional folder `title` the backend
owns. Registration is queued and applied on the worker thread, so a running
runtime builds the panel without a scene refresh. Publishing a manifest resets
the browser GUI, so every registered panel is rebuilt afterwards and its last
published state is replayed.

The backend stays neutral: it hands `build` a `PanelBuildContext` (GUI
namespace, `emit` sink, event-to-client resolver, run identity, and
`allow_commands`) and forwards whatever immutable value the panel emits through
the existing GUI event queue as a `PanelCommand`. The simulation thread drains
them with `drain_panel_commands()` (empty unless `allow_commands=True`) and
pushes new state with `publish_panel_state(panel_id, state)`, which keeps only
the newest state per panel. Registering nothing leaves browser behavior
bit-for-bit unchanged.

`authoring.SkillSequencePanel` is the first consumer. It renders an immutable
`PanelViewState` (sequence snapshot, preview playback state, picked entity,
status line) into markdown and controls, colors the five `SkillCardState`
values with marker glyphs, and emits authoring commands plus the
`TogglePreviewPlayback` / `SeekPreview` / `StepPreview` playback family. It
never calls a session write method. `authoring.AuthoringBridge` closes the loop
on the simulation thread: once per iteration `update()` drains click-picks into
the selected entity, routes panel commands through
`AuthoringSession.handle_command()` or the preview driver, converts failures
into a status string, and republishes the view state when it changed. Viser
sliders have no mutable range, so the panel rebuilds its seek slider whenever
the compiled length changes and ignores server-side value echoes.

`AuthoringSession.execute()` blocks until the whole trajectory is replayed, so
a browser sees nothing until it returns. `execute_stepwise()` is the additive,
non-blocking counterpart: it validates and resets card states eagerly, then
returns an iterator that performs exactly the same work one simulation update
at a time and carries the blocking call's boolean in its `StopIteration` value.
`execution.StepwiseExecution` wraps that iterator with `advance(count)`,
`is_active`, `succeeded`, and `close()`. `AuthoringBridge(stepwise_execution=True)`
starts one per `ExecuteSequence` and advances it inside `update()`, which is how
the panel shows `RUNNING -> SUCCEEDED` while the robot is moving. A host loop
must then skip its own `sim.update()` while `bridge.execution_active` is true,
and sequence edits are refused for the duration so a browser click cannot
invalidate the trajectory being replayed. Both defaults keep the blocking
behavior.

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
