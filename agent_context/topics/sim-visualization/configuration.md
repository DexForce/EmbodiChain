# Visualization configuration and activation

Read this when the request needs these details. [Topic overview](sim-visualization.md).

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
