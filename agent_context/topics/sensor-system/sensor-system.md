# Sensor System

## Entry Points

| What | Path |
|---|---|
| Sensor registry | `embodichain/lab/sim/sensors/__init__.py` |
| Base sensor class & config | `embodichain/lab/sim/sensors/base_sensor.py` → `BaseSensor`, `SensorCfg` |
| Camera | `embodichain/lab/sim/sensors/camera.py` → `Camera`, `CameraCfg` |
| Stereo camera | `embodichain/lab/sim/sensors/stereo.py` → `StereoCamera`, `StereoCameraCfg` |
| Contact sensor | `embodichain/lab/sim/sensors/contact_sensor.py` → `ContactSensor`, `ContactSensorCfg` |

## Overview

All sensors inherit from `BaseSensor`, which extends `BatchEntity`. Each sensor:
- Is configured via a `SensorCfg` subclass (uses `@configclass`).
- Maintains a `TensorDict` data buffer (`_data_buffer`) sized `[num_envs]`.
- Must implement `update()` and `get_data()`.
- Supports dynamic instantiation via `SensorCfg.from_dict()`, which resolves the config class from `sensor_type` string.

## Sensor Hierarchy

```
ObjectBaseCfg
  └─ SensorCfg            sensor_type, OffsetCfg, from_dict(), get_data_types()
      ├─ CameraCfg         width, height, intrinsics, extrinsics, enable_* flags
      │   └─ StereoCameraCfg   intrinsics_right, left_to_right_pos/rot, enable_disparity
      └─ ContactSensorCfg  rigid_uid_list, articulation_cfg_list, max_contacts_per_env

BatchEntity
  └─ BaseSensor            _data_buffer (TensorDict), SUPPORTED_DATA_TYPES
      ├─ Camera
      │   └─ StereoCamera
      └─ ContactSensor
```

## Available Sensors

| Sensor | Config | `sensor_type` string | Data Types | Notes |
|---|---|---|---|---|
| Camera | `CameraCfg` | `"Camera"` | color, depth, mask, normal, position | Single RGB-D camera; configurable intrinsics/extrinsics |
| StereoCamera | `StereoCameraCfg` | `"StereoCamera"` | color/depth/mask/normal/position (left + right), disparity | Extends Camera; adds right camera with baseline transform |
| ContactSensor | `ContactSensorCfg` | `"ContactSensor"` | contact data tensors | Collision detection between rigid bodies and articulation links; uses Warp kernels |

## Sensor Configuration

### SensorCfg.OffsetCfg

Defines the sensor pose relative to its parent frame:

| Field | Type | Default | Notes |
|---|---|---|---|
| `pos` | `Tuple[float, float, float]` | `(0, 0, 0)` | Position in parent frame |
| `quat` | `Tuple[float, float, float, float]` | `(1, 0, 0, 0)` | Orientation as `(w, x, y, z)` quaternion |
| `parent` | `str \| None` | `None` | Parent frame name (e.g. robot link); `None` = arena frame |

The `transformation` property returns a `4×4 torch.Tensor` homogeneous matrix.

### Dynamic Sensor Creation

`SensorCfg.from_dict(init_dict)` creates the correct config class by looking up `init_dict["sensor_type"] + "Cfg"` in the sensors module. Nested configclass fields are recursively initialized via their own `from_dict()`.

## Camera System

### CameraCfg

| Field | Type | Default | Notes |
|---|---|---|---|
| `width` | `int` | `640` | Image width in pixels |
| `height` | `int` | `480` | Image height in pixels |
| `near` | `float` | `0.005` | Near clipping plane (meters) |
| `far` | `float` | `100.0` | Far clipping plane (meters) |
| `intrinsics` | `Tuple[float, float, float, float]` | `(600, 600, 320, 240)` | `(fx, fy, cx, cy)` |
| `enable_color` | `bool` | `True` | Enable RGBA output |
| `enable_depth` | `bool` | `False` | Enable depth output |
| `enable_mask` | `bool` | `False` | Enable instance segmentation mask |
| `enable_normal` | `bool` | `False` | Enable surface normal output |
| `enable_position` | `bool` | `False` | Enable 3D position output |

### CameraCfg.ExtrinsicsCfg

Extends `SensorCfg.OffsetCfg` with look-at support:

| Field | Type | Default | Notes |
|---|---|---|---|
| `eye` | `Tuple[float,float,float] \| None` | `None` | Camera position |
| `target` | `Tuple[float,float,float] \| None` | `None` | Look-at target |
| `up` | `Tuple[float,float,float] \| None` | `None` | Up vector; defaults to `(0, 0, 1)` if `eye` is set |

When `eye` is provided, the transformation is computed via `look_at_to_pose()`. Otherwise falls back to `pos`/`quat`.

### StereoCameraCfg

Extends `CameraCfg` with stereo-specific fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `intrinsics_right` | `Tuple[float,float,float,float]` | `(600, 600, 320, 240)` | Right camera intrinsics |
| `left_to_right_pos` | `Tuple[float,float,float]` | `(0.05, 0, 0)` | Baseline translation (5cm default) |
| `left_to_right_rot` | `Tuple[float,float,float]` | `(0, 0, 0)` | Rotation in degrees |
| `enable_disparity` | `bool` | `False` | Enable disparity map output |

Properties `left_to_right` and `right_to_left` return `4×4` transform tensors. All enabled data types are duplicated for left and right (e.g. `color`, `color_right`).

### ContactSensorCfg

| Field | Type | Default | Notes |
|---|---|---|---|
| `rigid_uid_list` | `List[str]` | `[]` | UIDs of rigid bodies to monitor |
| `articulation_cfg_list` | `List[ArticulationContactFilterCfg]` | `[]` | Articulation link filters |
| `filter_need_both_actor` | `bool` | `True` | Require both actors in filter list |
| `max_contacts_per_env` | `int` | `64` | Max contacts per environment |

`ArticulationContactFilterCfg` specifies `articulation_uid` and `link_name_list` to filter which links report contacts.

## Common Failure Modes

- **`sensor_type` string mismatch** — `SensorCfg.from_dict()` looks up `sensor_type + "Cfg"` in the sensors module. A typo (e.g. `"camera"` instead of `"Camera"`) causes `AttributeError`.
- **Depth not enabled** — `enable_depth` defaults to `False`. Accessing depth data without enabling it returns empty tensors.
- **Parent frame not found** — `OffsetCfg.parent` must exactly match a link name in the scene. A wrong name silently places the sensor at the arena origin.
- **Stereo baseline sign** — `left_to_right_pos` defines translation from left to right camera. Flipping the sign inverts the disparity.
- **Contact sensor buffer overflow** — `max_contacts_per_env` caps the contact count. Exceeding it silently drops contacts; increase if the scene has dense collisions.
- **View attribute flags** — `Camera.get_view_attrib()` computes `dr.ViewFlags` from enabled booleans. Adding a new data type requires both the `enable_*` flag and the corresponding `ViewFlags` bit.
