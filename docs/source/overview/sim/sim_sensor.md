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