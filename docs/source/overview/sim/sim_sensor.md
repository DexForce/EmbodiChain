# Sensors

```{currentmodule} embodichain.lab.sim.sensors
```

The Simulation framework provides sensor interfaces for agents to perceive the environment. Currently, the primary supported sensor type is the **Camera**.

## Camera

### Configuration

The {class}`CameraCfg` class defines the configuration for camera sensors. It inherits from {class}`~SensorCfg` and controls resolution, clipping planes, intrinsics, and active data modalities.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `width` | `int` | `640` | Width of the captured image. |
| `height` | `int` | `480` | Height of the captured image. |
| `intrinsics` | `tuple` | `(600, 600, 320.0, 240.0)` | Camera intrinsics `(fx, fy, cx, cy)`. |
| `extrinsics` | `ExtrinsicsCfg` | `ExtrinsicsCfg()` | Pose configuration (see below). |
| `near` | `float` | `0.005` | Near clipping plane distance. |
| `far` | `float` | `100.0` | Far clipping plane distance. |
| `enable_color` | `bool` | `True` | Enable RGBA image capture. |
| `enable_depth` | `bool` | `False` | Enable depth map capture. |
| `enable_mask` | `bool` | `False` | Enable segmentation mask capture. |
| `enable_normal` | `bool` | `False` | Enable surface normal capture. |
| `enable_position` | `bool` | `False` | Enable 3D position map capture. |

### Camera Extrinsics

The `ExtrinsicsCfg` class defines the position and orientation of the camera.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `parent` | `str` | `None` | Name of the link to attach to (e.g., `"ee_link"`). If `None`, camera is fixed in world. |
| `pos` | `list` | `[0.0, 0.0, 0.0]` | Position offset `[x, y, z]`. |
| `quat` | `list` | `[0.0, 0.0, 0.0, 1.0]` | Orientation quaternion `[x, y, z, w]`. |
| `eye` | `tuple` | `None` | (Optional) Camera eye position for look-at mode. |
| `target` | `tuple` | `None` | (Optional) Target position for look-at mode. |
| `up` | `tuple` | `None` | (Optional) Up vector for look-at mode. |

### Usage

You can create a camera sensor using `sim.add_sensor()` with a `CameraCfg` object.

#### Code Example

```python
from embodichain.lab.sim.sensors import Camera, CameraCfg

# 1. Define Configuration
camera_cfg = CameraCfg(
    width=640,
    height=480,
    intrinsics=(600, 600, 320.0, 240.0),
    extrinsics=CameraCfg.ExtrinsicsCfg(
        parent="ee_link",        # Attach to robot end-effector
        pos=[0.09, 0.05, 0.04],  # Relative position
        quat=[1, 0, 0, 0],       # Relative rotation [x, y, z, w]
    ),
    enable_color=True,
    enable_depth=True,
)

# 2. Add Sensor to Simulation
camera: Camera = sim.add_sensor(sensor_cfg=camera_cfg)
```
### Observation Data
Retrieve sensor data using camera.get_data(). The data is returned as a dictionary of tensors on the specified device.

| Key | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `color` | `torch.uint8` | `(B, H, W, 4)` | RGBA image data. |
| `depth` | `torch.float32` | `(B, H, W)` | Depth map in meters. |
| `mask` | `torch.int32` | `(B, H, W)` | Segmentation mask / Instance IDs. |
| `normal` | `torch.float32` | `(B, H, W, 3)` | Surface normal vectors. |
| `position` | `torch.float32` | `(B, H, W, 3)` | 3D Position map (OpenGL coords). |

*Note: `B` represents the number of environments (batch size).*

## Stereo Camera

### Configuration

The {class}`StereoCameraCfg` class defines the configuration for stereo camera sensors. It inherits from {class}`CameraCfg` and includes additional settings for the right camera and stereo-specific features like disparity computation.

In addition to the standard {class}`CameraCfg` parameters, it supports the following:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `intrinsics_right` | `tuple` | `(600, 600, 320.0, 240.0)` | The intrinsics for the right camera `(fx, fy, cx, cy)`. |
| `left_to_right_pos` | `tuple` | `(0.05, 0.0, 0.0)` | Position offset `[x, y, z]` from the left camera to the right camera. |
| `left_to_right_rot` | `tuple` | `(0.0, 0.0, 0.0)` | Rotation offset `[x, y, z]` (Euler angles in degrees) from the left camera to the right camera. |
| `enable_disparity` | `bool` | `False` | Enable disparity map computation. *Note: Requires `enable_depth` to be `True`.* |

### Usage

You can create a stereo camera sensor using `sim.add_sensor()` with a `StereoCameraCfg` object.

#### Code Example

```python
from embodichain.lab.sim.sensors import StereoCamera, StereoCameraCfg

# 1. Define Configuration
stereo_cfg = StereoCameraCfg(
    width=640,
    height=480,
    # Intrinsics for Left (inherited) and Right cameras
    intrinsics=(600, 600, 320.0, 240.0),
    intrinsics_right=(600, 600, 320.0, 240.0),
    # Baseline configuration (e.g., 5cm baseline)
    left_to_right_pos=(0.05, 0.0, 0.0),
    extrinsics=StereoCameraCfg.ExtrinsicsCfg(
        parent="head_link",
        pos=[0.1, 0.0, 0.0],
    ),
    # Data modalities
    enable_color=True,
    enable_depth=True,
    enable_disparity=True,
)

# 2. Add Sensor to Simulation
stereo_camera: StereoCamera = sim.add_sensor(sensor_cfg=stereo_cfg)
```

## Contact Sensor

### Configuration

The {class}`ContactSensorCfg` class defines the configuration for contact sensors. It inherits from {class}`~SensorCfg` and enables filtering and monitoring of contact events between specific rigid bodies and articulation links in the simulation. The same API works with the Default (DexSim adapter) and Newton physics backends.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rigid_uid_list` | `List[str]` | `[]` | List of rigid body UIDs to monitor for contacts. |
| `articulation_cfg_list` | `List[ArticulationContactFilterCfg]` | `[]` | List of articulation link contact filter configurations. |
| `filter_need_both_actor` | `bool` | `True` | Whether to filter contact only when both actors are in the filter list. If `False`, contact is reported if either actor is in the filter. |
| `max_contacts_per_env` | `int` | `64` | Maximum number of contacts per environment that the sensor can handle. |

The sensor forwards `max_contacts_per_env` to DexSim as a per-Arena query
quota. If the global contact buffer is also full, DexSim distributes retained
rows across Arenas before filling later rows from the same Arena.

### Articulation Contact Filter Configuration

The `ArticulationContactFilterCfg` class specifies which articulation links to monitor for contacts.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `articulation_uid` | `str` | `""` | Unique identifier of the articulation (robot or articulated object). |
| `link_name_list` | `List[str]` | `[]` | List of link names in the articulation to monitor. If empty, all links are monitored. |

### Usage

You can create a contact sensor using `sim.add_sensor()` with a `ContactSensorCfg` object.

#### Code Example

```python
from embodichain.lab.sim.sensors import ContactSensor, ContactSensorCfg, ArticulationContactFilterCfg
import torch

# 1. Define Contact Filter Configuration
contact_filter_cfg = ContactSensorCfg()

# Monitor contacts for specific rigid bodies
contact_filter_cfg.rigid_uid_list = ["cube0", "cube1", "cube2"]

# Monitor contacts for specific articulation links
contact_filter_art_cfg = ArticulationContactFilterCfg()
contact_filter_art_cfg.articulation_uid = "UR10_PGI"
contact_filter_art_cfg.link_name_list = ["finger1_link", "finger2_link"]
contact_filter_cfg.articulation_cfg_list = [contact_filter_art_cfg]

# Only report contacts when both actors are in the filter list
contact_filter_cfg.filter_need_both_actor = True

# Set maximum contacts per environment
contact_filter_cfg.max_contacts_per_env = 128

# 2. Add Sensor to Simulation
contact_sensor: ContactSensor = sim.add_sensor(sensor_cfg=contact_filter_cfg)

# 3. Update and Retrieve Contact Data
sim.update(step=1)
contact_sensor.update()
contact_report = contact_sensor.get_data()

# Access contacts for a specific environment using is_valid mask
env_id = 0
env_valid_mask = contact_report["is_valid"][env_id]
env_contact_positions = contact_report["position"][env_id][env_valid_mask]

# Or get all valid contacts across all environments
valid_mask = contact_report["is_valid"]
all_valid_positions = contact_report["position"][valid_mask]  # Shape: (total_valid_contacts, 3)

# 4. Filter contacts by backend-neutral contact actor IDs
filter_user_ids = torch.as_tensor(
    [
        actor_id
        for actor_id in contact_sensor.item_user_ids.tolist()
        if contact_sensor.get_actor_info(actor_id).path.endswith("/cube2")
        or contact_sensor.get_actor_info(actor_id).link_name == "finger1_link"
    ],
    dtype=torch.int32,
    device=sim.device,
)
# Filter for specific environments
filter_contact_report = contact_sensor.filter_by_user_ids(filter_user_ids, env_ids=[env_id])

# 5. Visualize Contact Points
contact_sensor.set_contact_point_visibility(
    visible=True,
    rgba=(0.0, 0.0, 1.0, 1.0),  # Blue color
    point_size=6.0,
    env_ids=[env_id],  # Optional: visualize only specific environments
)
```

### Observation Data

Retrieve contact data using `contact_sensor.get_data()`. The data is returned as a dictionary of tensors on the specified device.

| Key | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `position` | `torch.float32` | `(num_envs, max_contacts_per_env, 3)` | Contact positions in arena frame (world coordinates minus arena offset). |
| `normal` | `torch.float32` | `(num_envs, max_contacts_per_env, 3)` | Unit normal vectors pointing from actor 0 toward actor 1. |
| `friction` | `torch.float32` | `(num_envs, max_contacts_per_env, 3)` | Tangential contact impulse applied to actor 0. Availability is reported by `contact_capabilities.friction`. |
| `impulse` | `torch.float32` | `(num_envs, max_contacts_per_env)` | Normal contact impulse magnitudes. |
| `distance` | `torch.float32` | `(num_envs, max_contacts_per_env)` | Signed contact separation (negative means penetration). |
| `user_ids` | `torch.int32` | `(num_envs, max_contacts_per_env, 2)` | Pair of query-local, backend-neutral contact actor IDs. The legacy field name is retained; resolve IDs with `get_actor_info()`. |
| `is_valid` | `torch.bool` | `(num_envs, max_contacts_per_env)` | Boolean mask indicating which contact slots contain valid data. Use this mask to filter out unused slots. |

**Note**: Use the `is_valid` mask to access only valid contacts:
```python
# Get all valid contacts across all environments
valid_mask = contact_report["is_valid"]
valid_positions = contact_report["position"][valid_mask]  # Shape: (total_valid_contacts, 3)

# Or access per-environment
env_id = 0
num_valid = contact_report["is_valid"][env_id].sum().item()
env_positions = contact_report["position"][env_id, :num_valid]
```

Each valid row is the strongest-normal-impulse representative for one ordered
native collision-shape pair. Multiple shape pairs can map to the same actor
pair, and geometry-only contacts with zero impulse remain valid. The fixed-size
numeric buffers are not cleared every update; values where `is_valid=False`
are unspecified and may be left over from an earlier update.

### Additional Methods

- **`get_actor_info(actor_id)`**: Resolve a contact actor ID to its Spawn path, articulation link, Arena, and environment ID.
- **`contact_capabilities`**: Report whether geometry, normal impulse, and friction impulse are available for the active backend/solver.
- **`filter_by_user_ids(item_user_ids, env_ids=None)`**: Filter contact report by contact actor IDs. The method name is retained for compatibility. Optionally filter by specific environment IDs.
- **`set_contact_point_visibility(visible, rgba, point_size, env_ids=None)`**: Enable/disable visualization of contact points with customizable color and size. Optionally visualize only specific environments.

Newton MuJoCo-Warp exposes contact forces, so both impulse fields are available. Other supported Newton rigid solvers currently expose contact geometry with zero-valued impulse fields. MuJoCo CPU mode does not expose device contact buffers, and MJVBD does not currently publish rigid contacts through `ContactQuery`; those modes are therefore unsupported by this sensor.

PhysX Direct GPU reports static counterparts with actor ID `-1` because its raw contact buffer does not expose their object identity. To monitor a dynamic body or articulation link against arbitrary static geometry, select the dynamic/link object and set `filter_need_both_actor=False`. Default CPU and Newton can identify registered static shapes.
