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
| `profiler` | `ProfilerCfg` \| `None` | `None` | Optional hierarchical wall-time profiler for simulation updates. |
| `sim_device` | `str` \| `torch.device` | `"cpu"` | The device for the physics simulation. |
| `physics_config` | `PhysicsCfg` | `PhysicsCfg()` | The physics configuration parameters. |
| `gpu_memory_config` | `GPUMemoryCfg` | `GPUMemoryCfg()` | The GPU memory configuration parameters. |
| `visualization` | `VisualizationCfg` | `VisualizationCfg()` | Browser visualization, opt-in Gizmo commands, and Viser server settings. |

### Physics Configuration

The {class}`~cfg.PhysicsCfg` class controls the global physics simulation parameters.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `gravity` | `np.ndarray` | `[0, 0, -9.81]` | Gravity vector for the simulation environment. |
| `bounce_threshold` | `float` | `2.0` | The speed threshold below which collisions will not produce bounce effects. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection (CCD) for fast-moving objects. |
| `length_tolerance` | `float` | `0.05` | The length tolerance for the simulation. Larger values increase speed. |
| `speed_tolerance` | `float` | `0.25` | The speed tolerance for the simulation. Larger values increase speed. |

PCM and TGS remain enabled, enhanced determinism remains disabled, and friction
is evaluated on every solver iteration. These solver implementation details use
fixed defaults and are not exposed by `PhysicsCfg`.

### Render Configuration

The {class}`~cfg.RenderCfg` class controls the rendering backend and quality settings.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `renderer` | `str` | `"auto"` | Renderer backend to use. Options are `'auto'` (pick a default based on the detected GPU), `'hybrid'` (ray tracing for shadows/reflections + rasterization), `'fast-rt'` (full ray tracing), and `'rt'` (offline ray-traced renderer for maximum visual fidelity). |
| `spp` | `int` | `1` | Samples per pixel for ray-traced rendering. Must be at least 1. |
| `tone_mapping_enabled` | `bool` | `False` | Whether to map HDR RGB output with the modified Reinhard curve. |
| `tone_mapping_exposure` | `float` | `1.0` | Non-negative fixed linear exposure multiplier applied before tone mapping. |

Ray-traced output always uses DexSim's default OptiX denoiser. Tone mapping
affects RGB output only; depth, segmentation masks, normals, and position
buffers remain unchanged.

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
        renderer="fast-rt",         # Override automatic renderer selection
        spp=4,                      # Render four samples per pixel
        tone_mapping_enabled=True,  # Convert HDR RGB to display-referred RGB
        tone_mapping_exposure=1.0,  # Fixed exposure for reproducible frames
    )
)
```


## Initialization

Initialize the manager with the configuration object:

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

# User can customize the config as needed.
sim_config = SimulationManagerCfg()
sim = SimulationManager(sim_config)
```

## Profiling simulation updates

Configure {class}`ProfilerCfg` directly on the simulation manager when using
the simulation without a Gym environment:

```python
from embodichain.lab.sim import ProfilerCfg, SimulationManager, SimulationManagerCfg

sim = SimulationManager(
    SimulationManagerCfg(
        profiler=ProfilerCfg(enable_time=True, warmup_steps=0),
    )
)
sim.update(step=4)
sim.profiler.report()
```

Each standalone {meth}`SimulationManager.update` call creates a `sim_update`
root. The `manual_update` section contains one `gizmo_update` and one
`world_update` sample per physics substep, plus optional
`window_record_capture` and `visualization_capture` samples when those features
are enabled. Consequently, `world_update.calls` is the total number of physics
substeps, its mean is the mean cost of one substep, and its total is the
aggregate physics-update time.

When a Gym environment owns the manager, it reuses the same profiler instance.
Simulation sections compose below `step.sim_update` without adding another
`sim_update` path component, so existing environment reports remain compatible.

## Browser visualization

{class}`SimulationManager` owns the optional Viser runtime. Configure it through
{attr}`SimulationManagerCfg.visualization`:

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import VisualizationCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        visualization=VisualizationCfg(
            backend="viser",
            env_ids=[0],
        ),
    )
)
print(sim.visualization_health.endpoint)
```

When `backend="viser"`, the manager starts the server during construction.
Assets added or removed later are published automatically on the next
{meth}`SimulationManager.update`. The runtime is stopped by
{meth}`SimulationManager.destroy`, or explicitly with
{meth}`SimulationManager.stop_visualization`.

The browser supports rigid objects and groups, robot and articulation links,
cloth, soft bodies, camera frustums, low-frequency RGB preview, overlays, and a
1 m ground grid. For configuration, performance behavior, deformable-object
limitations, remote access, and troubleshooting, see
{doc}`viser_visualization`.

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
- **`SimulationManager.start_visualization()`**: Start or return the configured visualization runtime.
- **`SimulationManager.refresh_visualization()`**: Immediately republish scene topology.
- **`SimulationManager.capture_visualization(force=False)`**: Capture the current scene state.
- **`SimulationManager.capture_visualization_safely(force=False)`**: Capture without allowing a visualization failure to interrupt simulation progress.
- **`SimulationManager.stop_visualization()`**: Stop Viser and release its server port.
- **`SimulationManager.visualization_health`**: Return endpoint, client count, revision, and worker status.
- **`SimulationManager.visualization_stats`**: Return capture, queue, payload, and upload telemetry.


## Multiple instances

`SimulationManager` supports multiple instances to run separate simulations world independently. Each instance maintains its own simulation state, assets, and configurations.

- To get current instance number of `SimulationManager`: `SimulationManager.get_instance_num()`
- To get specific instance: `SimulationManager.get_instance(instance_id)`.

> Currently, multiple instances are not supported for ray tracing rendering backend. Good news is that we are working on adding this feature in future releases.


For more methods and details, refer to the [SimulationManager](https://dexforce.github.io/EmbodiChain/api_reference/embodichain/embodichain.lab.sim.html#embodichain.lab.sim.SimulationManager) documentation.

### Related Tutorials

- [Basic scene creation](https://dexforce.github.io/EmbodiChain/tutorial/create_scene.html)
- [Interactive simulation with Gizmo](https://dexforce.github.io/EmbodiChain/tutorial/gizmo.html)
- {doc}`Viser browser visualization <viser_visualization>`
