# sim-visualization

> Topic: Simulation visualization — `SimulationManager` lifecycle,
> backend-neutral scene snapshots, the Viser backend, cameras, deformables,
> interactive Gizmos, and launcher integration.

---

## Entry Points

| File | Role |
|---|---|
| `embodichain/lab/sim/sim_manager.py` | Owns visualization configuration, lifecycle, topology revisions, capture hooks, health, and telemetry |
| `embodichain/lab/visualization/cfg.py` | `VisualizationCfg` and nested `ViserServerCfg` |
| `embodichain/lab/visualization/cli.py` | Common `--viser*` arguments and standalone argument-to-config conversion |
| `embodichain/lab/visualization/protocol.py` | Backend-neutral manifest, frame, mesh, camera, and overlay snapshots |
| `embodichain/lab/visualization/scene_exporter.py` | Reads simulation assets and produces detached CPU snapshots |
| `embodichain/lab/visualization/runtime.py` | Background worker, latest-frame queues, rate limiting, health, and telemetry |
| `embodichain/lab/visualization/backends/base.py` | Visualization backend contract |
| `embodichain/lab/visualization/backends/viser.py` | Viser server, scene handles, GUI controls, and browser publishing |
| `embodichain/lab/gym/utils/gym_utils.py` | Gym config parsing, common launcher arguments, and CLI overrides |
| `embodichain/lab/gym/envs/base_env.py` | Environment scene setup, post-setup start, and forced reset capture |
| `embodichain/lab/scripts/preview_asset.py` | Standalone rigid-object and articulation browser preview |

Human-facing behavior and usage are documented in
`docs/source/overview/sim/viser_visualization.md`.

## Ownership and Data Flow

```
SimulationManager / simulation thread
  ├── SceneExporter.build_manifest()  → static SceneManifest
  ├── SceneExporter.capture()         → dynamic SceneFrame
  └── capture_camera_images()         → rate-limited or step-synced CameraImageFrame
             │ detached contiguous CPU NumPy copies
             ▼
VisualizationRuntime
  ├── one-slot latest scene-frame queue
  ├── one-slot latest camera-image queue
  ├── ordered topology-manifest queue
  └── private "embodichain-visualization" thread
             ▼
VisualizationBackend
             └── ViserBackend → HTTP/WebSocket browser UI
```

The simulation owns physics and assets. Scene export is read-only. When
`allow_commands=True`, Viser Gizmo callbacks enqueue pose commands that
`SimulationManager` later applies on the simulation thread.

All DexSim and Torch reads happen in `SceneExporter` on the simulation thread.
Protocol objects make detached CPU copies before crossing into the runtime.
Viser creation and every Viser handle mutation must happen on the single
visualization worker thread; `ViserBackend._assert_update_thread()` enforces
this invariant.

## Configuration

`SimulationManagerCfg.visualization` owns a `VisualizationCfg`. Server settings
are nested under `visualization.viser_server`; there is no top-level
`SimulationManagerCfg.viser_server`.

### `VisualizationCfg`

| Field | Default | Operational meaning |
|---|---:|---|
| `backend` | `"none"` | `"viser"` enables the runtime |
| `scene_fps` | `15.0` | Maximum rigid pose, camera-frustum pose, and overlay capture rate |
| `env_ids` | `[0]` | Environment instances exported to the browser; `None` selects all |
| `max_visible_envs` | `None` | Optional validation limit for `env_ids` |
| `point_cloud_max_points` | `100000` | Deterministic per-cloud downsampling limit |
| `sensor_image_fps` | `2.0` | Maximum RGB image capture rate; `None` captures once per eligible simulation step |
| `soft_body_fps` | `5.0` | Maximum soft-body and cloth vertex capture rate |
| `allow_commands` | `False` | Allow trusted Viser clients to drag configured Gizmos; common `--viser` launchers set this to `True` |
| `viser_server` | `ViserServerCfg()` | Server bind configuration |

### `ViserServerCfg`

| Field | Default | Operational meaning |
|---|---:|---|
| `host` | `"127.0.0.1"` | Bind interface |
| `port` | `8080` | TCP port, validated in `[1, 65535]` |
| `label` | `"EmbodiChain"` | Browser application label |
| `verbose` | `False` | Viser diagnostics |

Validation rejects an unsupported backend, empty or duplicate explicit
`env_ids`, negative IDs, a violated optional environment limit, non-positive
sampling rates, invalid server fields, and commands enabled without the Viser backend.
`SceneExporter` performs the later check that every selected environment
exists in the simulation.

## Activation

### Programmatic

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        visualization=VisualizationCfg(
            backend="viser",
            env_ids=[0],
            viser_server=ViserServerCfg(host="127.0.0.1", port=8080),
        ),
    )
)
```

`SimulationManager.__init__()` calls `start_visualization()` after creating the
default scene and all configured arenas. `BaseEnv` requests a safe forced
capture after task assets and sensors are initialized, which refreshes the
manifest when manager APIs marked the topology dirty.

### Common launcher

`add_env_launcher_args_to_parser()` always installs:

- `--viser`
- `--viser-host`
- `--viser-port`
- `--viser-fps`
- `--viser-image-fps`
- `--viser-soft-body-fps`
- `--viser-env-ids ID ...` or `--viser-env-ids all`

`merge_args_with_gym_config()` makes `--viser` set `headless=True`, changes the
backend to `"viser"`, and applies the CLI sampling/server overrides. A
file-based config may also contain a `visualization` section. The old nested
key `visualization.server` is accepted and normalized to
`visualization.viser_server`.

`run-env` changes the omitted `--viser-image-fps` default to `None`, so camera
images are captured once per environment step after its final physics substep.
An explicit CLI value or file-based `sensor_image_fps` retains wall-clock rate
limiting.

`visualization_cfg_from_args()` only builds visualization config.
`SimulationManagerCfg` makes the simulation headless whenever the resulting
backend is `"viser"`. The common `--viser` option also enables Gizmo commands;
there is no separate `--viser-gizmo` option. Programmatic configurations may
retain `allow_commands=False` for read-only deployments.

### Asset preview

`embodichain preview-asset --asset_path <path> --viser` loads rigid objects or
articulations and keeps the headless simulation stepping until `Ctrl+C`.
Because `SimulationManager` starts Viser before the requested assets are
loaded, the command forces a safe capture immediately after loading; this
refreshes the dirty topology and publishes the initial poses before entering
the update loop or optional `--preview` REPL.

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

## Invariants

- Scene export must not mutate physics, actions, or asset state.
- Gizmo commands mutate targets only after simulation-thread validation.
- Simulation/DexSim reads stay on the simulation thread.
- Viser server and handle operations stay on one private worker thread.
- Cross-thread snapshots own detached contiguous CPU arrays.
- A frame is accepted only when `run_id`, `scene_revision`, node IDs, and
  dynamic vertex shapes match the published manifest.
- Topology changes require a new manifest; pose-only changes require only a
  frame.
- Sampling rates are wall-clock limits, not simulation-time guarantees.
- Bind to loopback by default. Viser has no EmbodiChain authentication layer;
  use SSH forwarding or an authenticated gateway for remote access.

## Common Failure Modes

| Symptom | Cause / action |
|---|---|
| `Visualization env_ids ... outside simulation range` | Selected IDs do not exist in the configured arenas. Validate them against `SimulationManager.num_envs`. |
| Startup timeout or address-in-use error | The Viser worker did not become ready or the configured port is occupied. Select another port and inspect `visualization_health.worker_error`. |
| Asset added after startup is missing | Step once, call `refresh_visualization()`, or mark topology dirty if the change bypassed manager APIs. |
| Browser stops updating after an exporter/backend exception | `capture_visualization_safely()` latches the first error to protect simulation. Inspect health/logs, then stop and restart after fixing the cause. |
| Soft body looks inflated or loses cavities | The surface is a collision-vertex convex hull, not the render topology. |
| Cloth construction raises a mapping error | Render vertices do not match the welded physical rest vertices within tolerance. |
| Camera frustum exists but preview is blank | Color capture is disabled, no image has been captured yet, or the selected camera/environment is hidden. |
| Stereo/contact sensor is absent | Current camera export accepts only `sensor_type == "Camera"`; non-mesh sensors are not exported. |
| Browser lags or upload cost is high | Reduce scene/image/deformable FPS, select fewer environments, or lower point-cloud limits. |
| Frame is counted as rejected | Its revision/run/node topology is stale, or dynamic vertex shape differs from the manifest. |
| Port remains occupied during teardown | Call `stop_visualization()` or ensure `SimulationManager.destroy()` reaches its visualization cleanup. |

## Verification

Relevant tests:

- `tests/visualization/test_cfg.py`
- `tests/visualization/test_cli.py`
- `tests/visualization/test_protocol.py`
- `tests/visualization/test_scene_exporter.py`
- `tests/visualization/test_runtime.py`
- `tests/visualization/test_viser_backend.py`
- `tests/sim/test_sim_manager.py`
- `tests/gym/utils/test_gym_utils.py`
- `tests/lab/scripts/test_preview_asset.py`
