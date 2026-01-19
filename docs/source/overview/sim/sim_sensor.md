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
| `quat` | `list` | `[1.0, 0.0, 0.0, 0.0]` | Orientation quaternion `[w, x, y, z]`. |
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
        quat=[0, 1, 0, 0],       # Relative rotation [w, x, y, z]
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

The {class}`ContactSensorCfg` class defines the configuration for contact sensors. It inherits from {class}`~SensorCfg` and enables filtering and monitoring of contact events between specific rigid bodies and articulation links in the simulation.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `rigid_uid_list` | `List[str]` | `[]` | List of rigid body UIDs to monitor for contacts. |
| `articulation_cfg_list` | `List[ArticulationContactFilterCfg]` | `[]` | List of articulation link contact filter configurations. |
| `filter_need_both_actor` | `bool` | `True` | Whether to filter contact only when both actors are in the filter list. If `False`, contact is reported if either actor is in the filter. |

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

# 2. Add Sensor to Simulation
contact_sensor: ContactSensor = sim.add_sensor(sensor_cfg=contact_filter_cfg)

# 3. Update and Retrieve Contact Data
sim.update(step=1)
contact_sensor.update()
contact_report = contact_sensor.get_data()

# 4. Filter contacts by specific user IDs
cube2_user_ids = sim.get_rigid_object("cube2").get_user_ids()
finger1_user_ids = sim.get_robot("UR10_PGI").get_user_ids("finger1_link").reshape(-1)
filter_user_ids = torch.cat([cube2_user_ids, finger1_user_ids])
filter_contact_report = contact_sensor.filter_by_user_ids(filter_user_ids)

# 5. Visualize Contact Points
contact_sensor.set_contact_point_visibility(
    visible=True, 
    rgba=(0.0, 0.0, 1.0, 1.0),  # Blue color
    point_size=6.0
)
```

### Observation Data

Retrieve contact data using `contact_sensor.get_data()`. The data is returned as a dictionary of tensors on the specified device.

| Key | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `position` | `torch.float32` | `(n_contact, 3)` | Contact positions in arena frame (world coordinates minus arena offset). |
| `normal` | `torch.float32` | `(n_contact, 3)` | Contact normal vectors. |
| `friction` | `torch.float32` | `(n_contact, 3)` | Contact friction forces. *Note: Currently this value may not be accurate.* |
| `impulse` | `torch.float32` | `(n_contact,)` | Contact impulse magnitudes. |
| `distance` | `torch.float32` | `(n_contact,)` | Contact penetration distances. |
| `user_ids` | `torch.int32` | `(n_contact, 2)` | Pair of user IDs for the two actors in contact. Use with `rigid_object.get_user_ids()` to identify objects. |
| `env_ids` | `torch.int32` | `(n_contact,)` | Environment IDs indicating which parallel environment each contact belongs to. |

*Note: `N` represents the number of contacts detected.*

### Additional Methods

- **`filter_by_user_ids(item_user_ids)`**: Filter contact report to include only contacts involving specific user IDs.
- **`set_contact_point_visibility(visible, rgba, point_size)`**: Enable/disable visualization of contact points with customizable color and size.