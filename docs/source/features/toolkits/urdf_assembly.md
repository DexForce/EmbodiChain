# URDF Assembly Tool

The URDF Assembly Tool is a modular system for building and assembling Unified Robot Description Format (URDF) files for robotic systems. It enables users to combine individual robot components (chassis, arms, legs, sensors, etc.) into a complete robot description.

## Overview

The tool provides a programmatic way to:

- **Add robot components**: Import URDF files for different robot parts (chassis, legs, torso, head, arms, hands)
- **Attach sensors**: Add sensors (camera, lidar, IMU, GPS, force/torque) to specific components
- **Merge URDFs**: Combine multiple component URDFs into a single unified robot description
- **Apply transformations**: Position and orient components using 4x4 transformation matrices
- **Validate assemblies**: Ensure URDF integrity and format compliance
- **Cache assemblies**: Use signature checking to skip redundant processing

## Quick Start

```python
from pathlib import Path
import numpy as np
from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

# Initialize the assembly manager
manager = URDFAssemblyManager()

# Add robot components
manager.add_component(
    component_type="chassis",
    urdf_path="path/to/chassis.urdf",
)

manager.add_component(
    component_type="torso",
    urdf_path="path/to/torso.urdf",
    transform=np.eye(4)  # 4x4 transformation matrix
)

manager.add_component(
    component_type="left_arm",
    urdf_path="path/to/arm.urdf"
)

# Attach sensors
manager.attach_sensor(
    sensor_name="front_camera",
    sensor_source="path/to/camera.urdf",
    parent_component="chassis",
    parent_link="base_link",
    transform=np.eye(4)
)

# Merge all components into a single URDF
manager.merge_urdfs(output_path="assembly_robot.urdf")
```

## Supported Components

The tool supports the following robot component types:

| Component | Description |
|:----------|:------------|
| `chassis` | Base platform of the robot |
| `legs` | Leg system for legged robots |
| `torso` | Main body/torso section |
| `head` | Head/upper section |
| `left_arm` | Left arm manipulator |
| `right_arm` | Right arm manipulator |
| `left_hand` | Left end-effector/gripper |
| `right_hand` | Right end-effector/gripper |
| `arm` | Single arm (bimanual robots) |
| `hand` | Single end-effector/gripper |

### Wheel Types

For chassis components, the following wheel types are supported:

- `omni`: Omnidirectional wheels
- `differential`: Differential drive
- `tracked`: Tracked locomotion

## Supported Sensors

The following sensor types can be attached to robot components:

| Sensor | Description |
|:-------|:------------|
| `camera` | RGB/depth cameras |
| `lidar` | 2D/3D LIDAR sensors |
| `imu` | Inertial measurement units |
| `gps` | GPS receivers |
| `force` | Force/torque sensors |

## Connection Rules

The tool automatically generates connection rules based on available components. The default rules include:

- Chassis → Legs → Torso (for legged robots)
- Chassis → Torso (for wheeled robots)
- Torso → Head
- Torso → Arms → Hands
- Chassis → Arms (when no torso exists)

## Mesh Formats

Supported mesh file formats for visual and collision geometries:

- STL
- OBJ
- PLY
- DAE
- GLB

## API Reference

### URDFAssemblyManager

*Located in:* `embodichain/toolkits/urdf_assembly/urdf_assembly_manager.py`

The main class for managing URDF assembly operations.

#### Methods

##### add_component()

Add a URDF component to the component registry.

```python
manager.add_component(
    component_type: str,
    urdf_path: Union[str, Path],
    transform: np.ndarray = None,
    **params
) -> bool
```

**Parameters:**

- `component_type` (str): Type of component (e.g., 'chassis', 'head')
- `urdf_path` (str or Path): Path to the component's URDF file
- `transform` (np.ndarray, optional): 4x4 transformation matrix for positioning
- `**params`: Additional component-specific parameters (e.g., `wheel_type` for chassis)

**Returns:** `bool` - True if component added successfully

##### attach_sensor()

Attach a sensor to a specific component and link.

```python
manager.attach_sensor(
    sensor_name: str,
    sensor_source: Union[str, ET.Element],
    parent_component: str,
    parent_link: str,
    transform: np.ndarray = None,
    **kwargs
) -> bool
```

**Parameters:**

- `sensor_name` (str): Unique name for the sensor
- `sensor_source` (str or ET.Element): Path to sensor URDF or XML element
- `parent_component` (str): Component to attach sensor to
- `parent_link` (str): Link within the parent component
- `transform` (np.ndarray, optional): Sensor transformation matrix

**Returns:** `bool` - True if sensor attached successfully

##### merge_urdfs()

Merge all registered components into a single URDF file.

```python
manager.merge_urdfs(
    output_path: str = "./assembly_robot.urdf",
    use_signature_check: bool = True
) -> ET.Element
```

**Parameters:**

- `output_path` (str): Path where the merged URDF will be saved
- `use_signature_check` (bool): Whether to check signatures to avoid redundant processing

**Returns:** `ET.Element` - Root element of the merged URDF

##### get_component()

Retrieve a registered component by type.

```python
manager.get_component(component_type: str) -> URDFComponent | None
```

##### get_attached_sensors()

Get all attached sensors.

```python
manager.get_attached_sensors() -> dict
```

##### Component name prefixes (`component_prefix`)

`URDFAssemblyManager` uses `component_prefix` to configure name prefixes for
each supported component type. This attribute is a list of 2-tuples:

- Form: `[(component_name, prefix), ...]`
- The default value is:

    ```python
    [
        ("chassis", None),
        ("legs", None),
        ("torso", None),
        ("head", None),
        ("left_arm", "left_"),
        ("right_arm", "right_"),
        ("left_hand", "left_"),
        ("right_hand", "right_"),
        ("arm", None),
        ("hand", None),
    ]
    ```

You can configure it in a *patch-style* manner via the property:

```python
# Only override prefixes for existing components; do not introduce
# new component names.
manager.component_prefix = [
    ("left_arm", "L_"),
    ("right_arm", "R_"),
    ("left_hand", "L_"),
    ("right_hand", "R_"),
]
```

Semantics:

- Only components that already exist in the default configuration (e.g. `chassis/torso/left_arm/...`) may be overridden; new component names are not allowed.
- Components not listed in `new_prefixes` keep their original prefix.
- If `new_prefixes` contains an unknown component name, a `ValueError` is raised indicating that new component types cannot be introduced.

##### Name casing policy (`name_case`)

`URDFAssemblyManager` supports a global name casing policy that controls how
link and joint names are normalized during assembly. This is configured via
the optional `name_case` argument in the constructor:

```python
manager = URDFAssemblyManager(
        name_case={
                "joint": "upper",  # or "lower" / "none"
                "link": "lower",  # or "upper" / "none"
        }
)
```

Semantics:

- Valid keys: `"joint"`, `"link"`.
- Valid values: `"upper"`, `"lower"`, `"none"`.
- Default behavior matches the legacy implementation:
  - joints are normalized to **UPPERCASE**,
  - links are normalized to **lowercase**.
- This policy is propagated to the internal component and connection managers,
    and is also included in the assembly signature. Changing `name_case` will
    therefore force a rebuild of the assembled URDF.

## Using with URDFCfg for Robot Creation

The URDF Assembly Tool can be used directly with `URDFCfg` to create robots with multiple components in the simulation. This is the recommended approach when building robots from assembled URDF files.

### URDFCfg Overview

The `URDFCfg` class provides a convenient way to define multi-component robots:

```python
from embodichain.lab.sim.cfg import RobotCfg, URDFCfg

cfg = RobotCfg(
    uid="my_robot",
    urdf_cfg=URDFCfg(
        components=[
            {
                "component_type": "arm",
                "urdf_path": "path/to/arm.urdf",
            },
            {
                "component_type": "hand",
                "urdf_path": "path/to/hand.urdf",
                "transform": hand_transform,  # 4x4 transformation matrix
            },
        ]
    ),
    control_parts={...},
    drive_pros={...},
)
```

When using `URDFCfg` to build multi-component robots, you can pass custom
component prefixes to the internal `URDFAssemblyManager` via
`URDFCfg.component_prefix`. Its semantics are identical to
`URDFAssemblyManager.component_prefix`:

- Each element is a `(component_name, prefix)` tuple.
- Only prefixes for components that exist in the default configuration may be overridden; no new component names can be added.
- Components not explicitly listed keep their original prefix.

Example:

```python
urdf_cfg = URDFCfg(
    components=[...],
)
urdf_cfg.component_prefix = [
    ("left_arm", "L_"),
    ("right_arm", "R_"),
]
```

### Complete Example

Here's a complete example from `scripts/tutorials/sim/create_robot.py`:

```python
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path


def create_robot(sim):
    """Create and configure a robot with arm and hand components."""

    # Get URDF paths for robot components
    arm_urdf_path = get_data_path("Rokae/SR5/SR5.urdf")
    hand_urdf_path = get_data_path(
        "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
    )

    # Define control parts - joint names can be regex patterns
    CONTROL_PARTS = {
        "arm": ["JOINT[1-6]"],      # Matches JOINT1, JOINT2, ..., JOINT6
        "hand": ["LEFT_.*"],         # Matches all joints starting with LEFT_
    }

    # Define transformation for hand attachment
    hand_transform = np.eye(4)
    hand_transform[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()

    # Create robot configuration
    cfg = RobotCfg(
        uid="sr5_with_hand",
        urdf_cfg=URDFCfg(
            components=[
                {
                    "component_type": "arm",
                    "urdf_path": arm_urdf_path,
                },
                {
                    "component_type": "hand",
                    "urdf_path": hand_urdf_path,
                    "transform": hand_transform,
                },
            ]
        ),
        control_parts=CONTROL_PARTS,
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[1-6]": 1e4, "LEFT_.*": 1e3},
            damping={"JOINT[1-6]": 1e3, "LEFT_.*": 1e2},
        ),
    )

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=cfg)

    return robot


# Initialize simulation and create robot
sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=4))
robot = create_robot(sim)
print(f"Robot created with {robot.dof} joints")
```

```python
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.data import get_data_path


def create_robot(sim):
    """Create and configure a robot with arm and hand components."""

    # Get URDF paths for robot components
    arm_urdf_path = get_data_path("Rokae/SR5/SR5.urdf")
    hand_urdf_path = get_data_path(
        "BrainCoHandRevo1/BrainCoLeftHand/BrainCoLeftHand.urdf"
    )

    # Define transformation for hand attachment
    hand_transform = np.eye(4)
    hand_transform[:3, :3] = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()

    left_arm_base_xpos = np.eye(4)
    left_arm_base_xpos[1, 3] = 0.3

    right_arm_base_xpos = np.eye(4)
    right_arm_base_xpos[1, 3] = -0.3

    # Create robot configuration
    cfg = RobotCfg(
        uid="dual_sr5",
        urdf_cfg=URDFCfg(
            components=[
                {
                    "component_type": "left_arm",
                    "urdf_path": arm_urdf_path,
                    "transform": left_arm_base_xpos,
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": arm_urdf_path,
                    "transform": right_arm_base_xpos,
                },
                {
                    "component_type": "left_hand",
                    "urdf_path": hand_urdf_path,
                    "transform": hand_transform,
                },
                {
                    "component_type": "right_hand",
                    "urdf_path": hand_urdf_path,
                    "transform": hand_transform,
                },
            ],
            component_prefix=[("left_arm", "L_"), ("right_arm", "R_"), ("right_hand", "left_"), ("right_hand", "right_")],
            name_case={
                "joint": "lower",
                "link": "lower",
            }
        ),
    )

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=cfg)

    return robot


# Initialize simulation and create robot
sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=4))
robot = create_robot(sim)
print(f"Robot created with {robot.dof} joints")
```

### Component Configuration

Each component in the `components` list supports the following keys:

| Key | Type | Description |
|:-----|:-----|:------------|
| `component_type` | str | Type of component (arm, hand, chassis, etc.) |
| `urdf_path` | str | Path to the component's URDF file |
| `transform` | np.ndarray | Optional 4x4 transformation matrix for positioning |

### Control Parts

Control parts define which joints belong to different subsystems of the robot. Joint names can be specified as:

- **Exact match**: `"JOINT1"` matches only `JOINT1`
- **Regex patterns**: `"JOINT[1-6]"` matches `JOINT1` through `JOINT6`
- **Wildcards**: `"LEFT_.*"` matches all joints starting with `LEFT_`

### Drive Properties

Drive properties control the joint behavior:

- `stiffness`: Position control gain (P gain) for each joint/group
- `damping`: Velocity control gain (D gain) for each joint/group

Both support regex pattern matching for convenient configuration.

## File Structure

```
embodichain/toolkits/urdf_assembly/
├── __init__.py                  # Package exports
├── urdf_assembly_manager.py     # Main assembly manager
├── component.py                  # Component classes and registry
├── sensor.py                     # Sensor attachment and registry
├── connection.py                 # Connection/joint management
├── mesh.py                       # Mesh file handling
├── file_writer.py                # URDF file output
├── signature.py                  # Assembly signature checking
├── logging_utils.py              # Logging configuration
└── utils.py                      # Utility functions
```
