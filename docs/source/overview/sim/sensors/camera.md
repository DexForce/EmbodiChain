# Camera

```{currentmodule} embodichain.lab.sim.sensors
```

## Configuration

The {class}`CameraCfg` class defines the configuration for camera sensors. It inherits from {class}`~SensorCfg` and controls resolution, clipping planes, intrinsics, and active data modalities.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `backend` | `str` | `"native"` | Image producer: `native` or `newton-tiled`. |
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

## Camera Extrinsics

### CUDA images without a native renderer

`CameraCfg(backend="newton-tiled")` uses the finalized Newton Scene's CUDA tiled
camera. It supports `RenderCfg(renderer="no-render")`, CUDA simulation, RGBA
color and optical-axis depth. The physics model, collision geometry, existing
sensor definitions, and physical stepping stay unchanged. Image geometry comes
from retained visual assets and follows the simulated body poses.

Select `enable_color=True` and/or `enable_depth=True`. Mask, normal, position,
stereo, and native renderer material effects are not supported by this backend;
unsupported configurations raise an error when the sensor is created. Intrinsics,
clipping, and fixed or rigid-link extrinsics use the same `CameraCfg` fields.

In a Gym environment, sensor updates happen at the existing observation update
boundary. In a standalone simulation, call `camera.update()` after physics has
advanced, then read `camera.get_data()`. Outputs stay on CUDA. The next update
reuses the output buffers, so copy observations into rollout storage when they
must survive another frame. Reset clears selected rows before the next sample;
destroying the camera releases its scene resources, while retained tensors keep
their data storage alive. Depth is measured in metres along the optical axis,
with zero for pixels outside the clipping range.

The first frame captures the camera CUDA Graph. Subsequent motion updates poses
and refits visual geometry without recapture; a supported scene rebuild refreshes
the mapping and graph. Camera creation or sampling does not load native Vulkan
or OptiX rendering modules.

### Pose fields

The `ExtrinsicsCfg` class defines the position and orientation of the camera.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `parent` | `str` | `None` | Name of the link to attach to (e.g., `"ee_link"`). If `None`, camera is fixed in world. |
| `pos` | `list` | `[0.0, 0.0, 0.0]` | Position offset `[x, y, z]`. |
| `quat` | `list` | `[0.0, 0.0, 0.0, 1.0]` | Orientation quaternion `[x, y, z, w]`. |
| `eye` | `tuple` | `None` | (Optional) Camera eye position for look-at mode. |
| `target` | `tuple` | `None` | (Optional) Target position for look-at mode. |
| `up` | `tuple` | `None` | (Optional) Up vector for look-at mode. |

## Usage

You can create a camera sensor using `sim.add_sensor()` with a `CameraCfg` object.

### Code Example

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
        quat=[0, 0, 0, 1],       # Relative rotation [x, y, z, w]
    ),
    enable_color=True,
    enable_depth=True,
)

# 2. Add Sensor to Simulation
camera: Camera = sim.add_sensor(sensor_cfg=camera_cfg)
```
## Observation Data
Retrieve sensor data using camera.get_data(). The data is returned as a dictionary of tensors on the specified device.

| Key | Data Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `color` | `torch.uint8` | `(B, H, W, 4)` | RGBA image data. |
| `depth` | `torch.float32` | `(B, H, W)` | Depth map in meters. |
| `mask` | `torch.int32` | `(B, H, W)` | Segmentation mask / Instance IDs. |
| `normal` | `torch.float32` | `(B, H, W, 3)` | Surface normal vectors. |
| `position` | `torch.float32` | `(B, H, W, 3)` | 3D Position map (OpenGL coords). |

*Note: `B` represents the number of environments (batch size).*
