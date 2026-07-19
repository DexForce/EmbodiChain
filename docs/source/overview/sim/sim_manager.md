# Simulation Manager

```{currentmodule} embodichain.lab.sim
```

The {class}`SimulationManager` is the central class in EmbodiChain's simulation framework for managing the simulation lifecycle. It handles:
- **Asset Management**: Loading and managing robots, rigid objects, soft objects, articulations, and lights.
- **Simulation Loop**: Controlling the physics stepping and rendering updates.
- **Rendering**: Managing the simulation window, camera rendering, material settings and ray-tracing configuration.
- **Interaction**: Providing gizmo controls for interactive manipulation of objects.

## Configuration

The simulation is configured using the {class}`SimulationManagerCfg` class.

```python
from embodichain.lab.sim import SimulationManagerCfg

sim_config = SimulationManagerCfg(
    width=1920,               # Window width
    height=1080,              # Window height
    num_envs=10,              # Number of parallel environments
    physics_dt=0.01,          # Physics time step
    sim_device="cpu",         # Simulation device ("cpu" or "cuda:0", etc.)
    arena_space=5.0           # Spacing between environments
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` | `int` | `1920` | The width of the simulation window. |
| `height` | `int` | `1080` | The height of the simulation window. |
| `headless` | `bool` | `False` | Whether to run the simulation in headless mode (no Window). |
| `render_cfg` | `RenderCfg` | `RenderCfg()` | The rendering configuration parameters. |
| `gpu_id` | `int` | `0` | The gpu index that the simulation engine will be used. Affects gpu physics device. |
| `thread_mode` | `ThreadMode` | `RENDER_SHARE_ENGINE` | The threading mode for the simulation engine. |
| `cpu_num` | `int` | `1` | The number of CPU threads to use for the simulation engine. |
| `num_envs` | `int` | `1` | The number of parallel environments (arenas) to simulate. |
| `arena_space` | `float` | `5.0` | The distance between each arena when building multiple arenas. |
| `physics_dt` | `float` | `0.01` | The time step for the physics simulation. |
| `sim_device` | `str` \| `torch.device` | `"cpu"` | The device for the physics simulation. |
| `physics_config` | `PhysicsCfg` | `PhysicsCfg()` | The physics configuration parameters. |
| `gpu_memory_config` | `GPUMemoryCfg` | `GPUMemoryCfg()` | The GPU memory configuration parameters. |

### Physics Configuration

The {class}`~cfg.PhysicsCfg` class controls the global physics simulation parameters.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `gravity` | `np.ndarray` | `[0, 0, -9.81]` | Gravity vector for the simulation environment. |
| `bounce_threshold` | `float` | `2.0` | The speed threshold below which collisions will not produce bounce effects. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection (CCD) for fast-moving objects. |
| `length_tolerance` | `float` | `0.05` | The length tolerance for the simulation. Larger values increase speed. |
| `speed_tolerance` | `float` | `0.25` | The speed tolerance for the simulation. Larger values increase speed. |

For more parameters and details, refer to the [PhysicsCfg](https://dexforce.github.io/EmbodiChain/api_reference/embodichain/embodichain.lab.sim.html#embodichain.lab.sim.cfg.PhysicsCfg) documentation.

### Render Configuration

The {class}`~cfg.RenderCfg` class controls the rendering backend and quality settings.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `renderer` | `str` | `"auto"` | Renderer backend to use. Options are `'auto'` (pick a default based on the detected GPU), `'hybrid'` (ray tracing for shadows/reflections + rasterization), `'fast-rt'` (full ray tracing), and `'rt'` (offline ray-traced renderer for maximum visual fidelity). |
| `enable_denoiser` | `bool` | `True` | Whether to enable denoising. Only valid when `renderer` is `'hybrid'`, `'fast-rt'` or `'rt'`. |
| `spp` | `int` | `64` | Samples per pixel for ray tracing rendering. Only valid when `renderer` is `'hybrid'`, `'fast-rt'` or `'rt'` and `enable_denoiser` is `False`. |

#### Automatic Renderer Selection

By default (`renderer="auto"`), EmbodiChain selects the renderer based on the GPU detected at the configured `gpu_id` when the {class}`SimulationManager` is constructed:

| GPU class | Examples | Selected renderer |
| :--- | :--- | :--- |
| RTX-series (consumer/workstation) | RTX 4090, RTX 6000 Ada | `hybrid` |
| Datacenter accelerators | A100, A800, H100, H800, H200, H20 | `fast-rt` |
| No CUDA device / unknown GPU | — | `hybrid` (fallback) |

You can override the global default at runtime — useful for forcing a renderer across all simulations regardless of hardware:

```python
from embodichain.lab.sim import SimulationManager

# Resolve the default from the current GPU, or force a specific backend.
SimulationManager.set_default_renderer("auto")       # auto-detect from GPU
SimulationManager.set_default_renderer("fast-rt")    # force full ray tracing
```

Setting `render_cfg.renderer` explicitly always takes precedence over auto-selection:

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg

sim_config = SimulationManagerCfg(
    render_cfg=RenderCfg(
        renderer="fast-rt",    # Use full ray tracing (overrides auto-selection)
        enable_denoiser=True,  # Enable denoising
        spp=64,                # Samples per pixel (used when denoiser is off)
    )
)
```

### DLSS 3.5 (OfflineRT Only)

When `render_cfg.renderer="rt"` (OfflineRT) and the simulation is running with a visible window, you can enable NVIDIA DLSS 3.5 for AI-powered denoising and upscaling through `render_cfg.dlss`.

DLSS 3.5 combines two features:

- **Ray Reconstruction (RR)** — replaces the OptiX denoiser for the window camera.
- **Super Resolution (SR)** — upscales the RR output from render resolution to the target/display resolution.

Both features are enabled together when `dlss_enabled=True`. The exact upscale ratio can be controlled either with the `dlss_quality` preset or with the `upsample_ratio` convenience parameter.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dlss_enabled` | `bool` | `True` | Master DLSS enable toggle. |
| `dlss_quality` | `int` | `-1` | DLSS quality preset. `-1`=auto, `0`=UltraPerformance (~3.0×), `1`=Performance (~2.25×), `2`=Balanced (~1.75×), `3`=Quality (~1.5×), `4`=UltraQuality (~1.3×), `5`=DLAA (1.0×). |
| `upsample_ratio` | `float` | `1.0` | Convenience ratio `target_width / render_width` (and height). When `render_width`/`render_height` are `0`, the render resolution is computed as `target / upsample_ratio`. Defaults to `1.0` (1:1 / DLAA mode). Must be `>= 1.0`. |
| `render_width` | `int` | `0` | Internal render resolution width. `0` means use the effective target width. |
| `render_height` | `int` | `0` | Internal render resolution height. `0` means use the effective target height. |
| `target_width` | `int` | `0` | Target/display resolution width. `0` means use the window width. |
| `target_height` | `int` | `0` | Target/display resolution height. `0` means use the window height. |
| `exposure_compensation` | `float` | `1.0` | Multiplier on the DLSS-RR pre-exposure scalar. `>1` brightens the HDR input to RR, which helps reject ghosting in dark areas; `<1` darkens. Typical range `0.5–8.0`. |

#### Resolution selection rules

- The **target** resolution defaults to the window size unless `target_width`/`target_height` are set explicitly.
- The **render** resolution defaults to the target resolution (1:1 / DLAA mode) unless overridden.
- If `upsample_ratio > 1.0` and `render_width`/`render_height` are `0`, the render resolution is computed as `target / upsample_ratio`. At `1.0`, render defaults to the target resolution (1:1 / DLAA mode).
- Explicit `render_width`/`render_height` or `target_width`/`target_height` take precedence over computed values.
- The window is always resized to match the effective target resolution when DLSS is active.

#### Examples

Use the `upsample_ratio` shortcut to render at half resolution and upscale to the window size:

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import DLSSCfg, RenderCfg

sim_config = SimulationManagerCfg(
    width=1920,
    height=1080,
    render_cfg=RenderCfg(
        renderer="rt",
        dlss=DLSSCfg(
            dlss_enabled=True,
            upsample_ratio=2.0,  # render at 960x540, upscale to 1920x1080
        ),
    ),
)
```

Set explicit render and target resolutions:

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import DLSSCfg, RenderCfg

sim_config = SimulationManagerCfg(
    render_cfg=RenderCfg(
        renderer="rt",
        dlss=DLSSCfg(
            dlss_enabled=True,
            render_width=1280,
            render_height=720,
            target_width=1920,
            target_height=1080,
        ),
    ),
)
```

> **Note:** Offscreen cameras always use the OptiX denoiser regardless of DLSS settings. DLSS only affects the interactive viewer window when the OfflineRT renderer is used.


## Initialization

Initialize the manager with the configuration object:

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

# User can customize the config as needed.
sim_config = SimulationManagerCfg()
sim = SimulationManager(sim_config)
```

## Assets Management

The manager provides methods to add, retrieve and remove various simulation assets including:
- Rigid Objects
- Soft Objects
- Articulations
- Robots
- Lights
- Materials

For more details on simulation assets, please refer to their respective documentation pages.

### USD Import and Export

#### Importing USD Files

EmbodiChain supports importing USD files (`.usd`, `.usda`, `.usdc`) for both rigid objects and articulations. When importing USD files, you can choose whether to use the physical properties defined in the USD file or override them with configuration values:

```python
# Import rigid object with USD properties
rigid_cfg = RigidObjectCfg(
    shape=MeshCfg(fpath=get_data_path("path/to/object.usd")),
    use_usd_properties=True  # Use properties from USD file
)
obj = sim.add_rigid_object(cfg=rigid_cfg)

# Import articulation with USD properties
robot_cfg = ArticulationCfg(
    fpath=get_data_path("path/to/robot.usd"),
    use_usd_properties=True  # Use joint drive properties from USD
)
robot = sim.add_articulation(cfg=robot_cfg)
```

#### Exporting to USD

You can export the current simulation scene to a USD file using the `export_usd()` method:

```python
# Export the entire scene to USD
sim.export_usd("my_scene.usda")
```

This exports all objects, articulations, robots, and their current states to a USD file, which can be:
- Reimported into EmbodiChain with preserved properties
- Opened in USD-compatible tools (e.g., USD Viewer, Omniverse)
- Used as assets for other simulations

See `scripts/tutorials/sim/export_usd.py` for a complete example.

## Simulation Loop

### Manual Update mode

In this mode, the physics simulation should be explicitly stepped by calling {meth}`~SimulationManager.update()` method, which provides precise control over the simulation timing. 

The use case for manual update mode includes:
- Data generation with openai gym environments, in which the observation and action must be synchronized with the physics simulation.
- Applications that require precise dynamic control over the simulation timing.

```python
while True:
    # Step physics simulation.
    sim.update(step=1)

    # Perform other tasks such as get data from the scene or apply sensor update.
```

> The default mode is manual update mode. To switch to automatic update mode, call `set_manual_update(False)`. 

### Automatic Update mode

In this mode, the physics simulation stepping is automatically handling by the physics thread running in dexsim engine, which makes it easier to use for visualization and interactive applications.

> When in automatic update mode, user are recommanded to use CPU `sim_device` for simulation.


## Mainly used methods

- **`SimulationManager.update(physics_dt=None, step=1)`**: Steps the physics simulation with optional custom time step and number of steps. If `physics_dt` is None, uses the configured physics time step.
- **`SimulationManager.enable_physics(enable: bool)`**: Enable or disable physics simulation.
- **`SimulationManager.set_manual_update(enable: bool)`**: Set manual update mode for physics.


## Multiple instances

`SimulationManager` supports multiple instances to run separate simulations world independently. Each instance maintains its own simulation state, assets, and configurations.

- To get current instance number of `SimulationManager`: `SimulationManager.get_instance_num()`
- To get specific instance: `SimulationManager.get_instance(instance_id)`.

> Currently, multiple instances are not supported for ray tracing rendering backend. Good news is that we are working on adding this feature in future releases.


For more methods and details, refer to the [SimulationManager](https://dexforce.github.io/EmbodiChain/api_reference/embodichain/embodichain.lab.sim.html#embodichain.lab.sim.SimulationManager) documentation.

### Related Tutorials

- [Basic scene creation](https://dexforce.github.io/EmbodiChain/tutorial/create_scene.html)
- [Interactive simulation with Gizmo](https://dexforce.github.io/EmbodiChain/tutorial/gizmo.html)