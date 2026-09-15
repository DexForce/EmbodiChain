# Stereo Camera

```{currentmodule} embodichain.lab.sim.sensors
```

## Configuration

The {class}`StereoCameraCfg` class defines the configuration for stereo camera sensors. It inherits from {class}`CameraCfg` and includes additional settings for the right camera and stereo-specific features like disparity computation.

In addition to the standard {class}`CameraCfg` parameters, it supports the following:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `intrinsics_right` | `tuple` | `(600, 600, 320.0, 240.0)` | The intrinsics for the right camera `(fx, fy, cx, cy)`. |
| `left_to_right_pos` | `tuple` | `(0.05, 0.0, 0.0)` | Position offset `[x, y, z]` from the left camera to the right camera. |
| `left_to_right_rot` | `tuple` | `(0.0, 0.0, 0.0)` | Rotation offset `[x, y, z]` (Euler angles in degrees) from the left camera to the right camera. |
| `enable_disparity` | `bool` | `False` | Enable disparity map computation. *Note: Requires `enable_depth` to be `True`.* |

## Usage

You can create a stereo camera sensor using `sim.add_sensor()` with a `StereoCameraCfg` object.

### Code Example

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
