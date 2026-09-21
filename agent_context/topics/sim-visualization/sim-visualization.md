# sim-visualization

> Topic: Simulation visualization — `SimulationManager` lifecycle,
> backend-neutral scene snapshots, the Viser backend, cameras, deformables,
> interactive Gizmos, render-only marker groups, and launcher integration.

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
| `embodichain/lab/visualization/markers/` | Marker prototype/group configuration, geometry generation, validation, and lifecycle |
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

`SimulationManager.add_marker_group()` owns named render-only marker groups.
`MarkerPrototypeCfg` supplies box, sphere, cylinder, capsule, cone, arrow,
frame, or caller-provided mesh geometry; `MarkerGroup.update()` publishes
batched pose, scale, prototype, color, and visibility arrays without stepping
physics. Groups default to all sub-environments; `env_ids` selects mutations,
while `scope="world"` creates one global batch. Attach/detach follows prepared
registered roots or links through public pose reads on the simulation thread.
Native rendering creates ordinary `MeshObject` instances through the builder's
Scene with `Scene.add_mesh_object(MeshObjectDesc(..., physics=None))`.
`RenderDesc` and `MaterialDesc` configure overlay routing, shadow/picking and
unlit RGBA blending before GPU build. Scene owns the stable `SpawnedRigidBody`
registration; Arena owns the native object. Marker updates reuse these handles,
and removal uses `Scene.remove_mesh_object(path)`. Creation failures and failed
group updates preserve existing handles and snapshots. The lightweight Scene
is available after `prepare_arenas()` for either physics backend; creating or
updating markers does not prepare physics. Handles survive Newton physics
rebuild and become invalid on Scene close; renderer cleanup tolerates already
invalidated handles. Unlit/alpha-mode controls require the shared native RT
material type; Filament materials reject them. Actor removal and handle
invalidation are immediate; unreferenced materials are reclaimed by DexSim's
reference-aware periodic manager collection, so resource release can span
later render updates. Shared or externally retained materials remain valid.
The consumer keeps no native material cache and does not force collection;
the dependency must provide this manager lifecycle alongside the Scene and
overlay descriptor APIs. File-backed
overlay geometry is unsupported. Browser rendering carries meshes in
`SceneOverlays`. Native submission remains per-object; true native batching and
GPU instancing are tracked separately in DexSim issue 227. `draw_marker()` remains
the compatibility API for legacy axis markers. Point clouds and polylines
remain separate overlay APIs rather than marker prototypes.

`SimulationManager` may start the Viser server during its own construction,
before a standalone caller has declared any assets. That empty visualization
startup must not call `prepare()` or finalize the Spawn scene. The caller's
first asset-bearing `prepare()` remains the initial topology commit; the next
capture observes the topology revision and republishes the manifest.

All DexSim and Torch reads happen in `SceneExporter` on the simulation thread.
Protocol objects make detached CPU copies before crossing into the runtime.
Viser creation and every Viser handle mutation must happen on the single
visualization worker thread; `ViserBackend._assert_update_thread()` enforces
this invariant.

## Read on demand

- [Configuration and activation](configuration.md): backend, Viser settings, launcher and asset preview.
- [Runtime and controls](runtime.md): export, cameras, deformables, overlays, queues and telemetry.
- [Native gizmos](native-gizmos.md): entity manipulation, IK ownership and native-window behavior.

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
- Starting Viser with empty asset registries must not finalize Spawn. Initial
  scene materialization belongs to the first `prepare()` after declaration.
- Newton articulation export combines every render mesh belonging to a link
  and offsets face indices into the concatenated vertex array. Links without
  render meshes export empty geometry without querying mesh zero.
- Sampling rates are wall-clock limits, not simulation-time guarantees.
- Marker mutation never prepares or synchronizes an unprepared scene. When
  topology is dirty, browser publication waits for the next explicit host
  capture or `SimulationManager.update()`.
- Bind to loopback by default. Viser has no EmbodiChain authentication layer;
  use SSH forwarding or an authenticated gateway for remote access.

## Common Failure Modes

| Symptom | Cause / action |
|---|---|
| `Visualization env_ids ... outside simulation range` | Selected IDs do not exist in the configured arenas. Validate them against `SimulationManager.num_envs`. |
| Startup timeout or address-in-use error | The Viser worker did not become ready or the configured port is occupied. Select another port and inspect `visualization_health.worker_error`. |
| Asset added after startup is missing | Step once, call `refresh_visualization()`, or mark topology dirty if the change bypassed manager APIs. |
| Native marker creation reports missing render-actor support | Use a DexSim build exposing Scene render-only mesh lifecycle and generic RenderBody/MaterialInst overlay properties, or run with `--viser`. Overlay routing supports Hybrid, FastRT and OfflineRT. |
| Newton Viser shows only the grid after declaring a robot | Check that Viser startup did not finalize an empty Spawn scene before asset declaration and that link poses remain finite after the first update. |
| A Newton articulation link is missing geometry or emits `mesh_id 0 out of range` | Export all render-body mesh segments and treat a zero-mesh link as empty geometry. Do not use the single-mesh articulation helper for Newton render bodies. |
| Browser stops updating after an exporter/backend exception | `capture_visualization_safely()` latches the first error to protect simulation. Inspect health/logs, then stop and restart after fixing the cause. |
| Replicated soft body fails with a render-vertex-count mismatch | DexSim produced clone render topology different from the source. Use one environment or a compatible mesh until replication preserves the topology. |
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
- `tests/visualization/test_markers.py`
- `tests/visualization/test_native_markers.py`
- `tests/sim/test_sim_manager.py`
- `tests/sim/test_marker_rendering.py`
- `tests/gym/utils/test_gym_utils.py`
- `tests/lab/scripts/test_preview_asset.py`
