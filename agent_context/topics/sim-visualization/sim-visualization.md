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
- `tests/sim/test_sim_manager.py`
- `tests/gym/utils/test_gym_utils.py`
- `tests/lab/scripts/test_preview_asset.py`
