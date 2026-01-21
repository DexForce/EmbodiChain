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
| `enable_rt` | `bool` | `False` | Whether to enable ray tracing rendering. |
| `enable_denoiser` | `bool` | `True` | Whether to enable denoising for ray tracing rendering. |
| `spp` | `int` | `64` | Samples per pixel for ray tracing rendering. Only valid when ray tracing is enabled and denoiser is False. |
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