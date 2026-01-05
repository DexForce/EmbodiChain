# Simulation Manager

The `SimulationManager` is the central class in EmbodiChain's simulation framework for managing the simulation lifecycle. It handles:
- **Asset Management**: Loading and managing robots, rigid objects, soft objects, articulations, sensors, and lights.
- **Simulation Loop**: Controlling the physics stepping and rendering updates.
- **Rendering**: Managing the simulation window, camera rendering, material settings and ray-tracing configuration.
- **Interaction**: Providing gizmo controls for interactive manipulation of objects.

## Configuration

The simulation is configured using the `SimulationManagerCfg` class.

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

## Initialization

Initialize the manager with the configuration object:

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

# User can customize the config as needed.
sim_config = SimulationManagerCfg()
sim = SimulationManager(sim_config)
```

## Asset Management

The manager provides methods to add and retrieve various simulation assets.

### Robots

Add robots using `RobotCfg`.

```python
from embodichain.lab.sim.cfg import RobotCfg

robot_cfg = RobotCfg(uid="my_robot", ...)
robot = sim.add_robot(robot_cfg)

# Retrieve existing robot
robot = sim.get_robot("my_robot")
```

### Rigid Objects

Add rigid bodies (e.g., cubes, meshes) using `RigidObjectCfg`.

```python
from embodichain.lab.sim.cfg import RigidObjectCfg

obj_cfg = RigidObjectCfg(uid="cube", ...)
obj = sim.add_rigid_object(obj_cfg)
```

### Sensors

Add sensors (e.g., Cameras) using `SensorCfg`.

```python
from embodichain.lab.sim.sensors import CameraCfg

camera_cfg = CameraCfg(uid="cam1", ...)
camera = sim.add_sensor(camera_cfg)
```

### Lights

Add lights to the scene using `LightCfg`.

```python
from embodichain.lab.sim.cfg import LightCfg

light_cfg = LightCfg(uid="sun", light_type="point", ...)
light = sim.add_light(light_cfg)
```

## Simulation Loop

The simulation loop typically involves stepping the physics and rendering the scene.

```python
while True:
    # Step physics and render
    sim.update()
    
    # Or step manually
    # sim.step_physics()
    # sim.render()
```

### Methods

- **`update(physics_dt=None, step=1)`**: Steps the physics simulation and updates the rendering.
- **`enable_physics(enable: bool)`**: Enable or disable physics simulation.
- **`set_manual_update(enable: bool)`**: Set manual update mode for physics.

## Rendering

- **`render_camera_group()`**: Renders all cameras in the scene.
- **`open_window()`**: Opens the visualization window.
- **`close_window()`**: Closes the visualization window.

## Gizmos

Gizmos allow interactive control of objects in the simulation window.

```python
# Enable gizmo for a robot
sim.enable_gizmo(uid="my_robot", control_part="arm")

# Toggle visibility
sim.toggle_gizmo_visibility(uid="my_robot", control_part="arm")

# Disable gizmo
sim.disable_gizmo(uid="my_robot", control_part="arm")
```

## Example Usage

Below is a complete example of setting up a simulation with a robot and a sensor.

```python
import argparse
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.cfg import RobotCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg

# 1. Configure Simulation
config = SimulationManagerCfg(
    headless=False,
    sim_device="cuda",
    enable_rt=True,
    physics_dt=0.01
)
sim = SimulationManager(config)

# 2. Add a Robot
# (Assuming robot_cfg is defined)
# robot = sim.add_robot(robot_cfg)

# 3. Add a Rigid Object
cube_cfg = RigidObjectCfg(
    uid="cube",
    shape=CubeCfg(size=[0.05, 0.05, 0.05]),
    init_pos=[1.0, 0.0, 0.5]
)
sim.add_rigid_object(cube_cfg)

# 4. Add a Sensor
camera_cfg = CameraCfg(
    uid="camera",
    width=640,
    height=480,
    # ... other params
)
camera = sim.add_sensor(camera_cfg)

# 5. Run Simulation Loop
while True:
    sim.update()
    
    # Access sensor data
    # data = camera.get_data()
```
```