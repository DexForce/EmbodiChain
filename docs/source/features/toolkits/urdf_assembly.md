# URDF Assembly

The URDF assembly toolkit builds one robot description from multiple component
URDFs. It is intended for modular robots whose chassis, arms, end effectors, or
sensors are maintained as separate assets.

During assembly, the toolkit loads each component, copies its links, joints, and
meshes into a unified layout, creates fixed joints between compatible
components, attaches sensors, normalizes names, and writes the merged URDF. A
content signature avoids rebuilding an unchanged assembly.

## Capabilities

- **Component assembly** — combines common robot parts such as a chassis,
  torso, arms, and hands.
- **Automatic connections** — creates fixed joints from built-in parent-child
  rules and applies optional 4×4 transforms.
- **Sensor attachment** — inserts sensor URDFs or XML elements at a selected
  component link.
- **Name management** — adds per-component prefixes and applies a consistent
  casing policy to links and joints.
- **Asset collection** — copies referenced meshes and related material assets
  into the output layout.
- **Incremental rebuilds** — hashes component files and assembly settings so an
  up-to-date output can be reused.

## Internal Modules

The public entry point is `URDFAssemblyManager`. Its work is divided among the
following submodules:

| Module | Responsibility |
|---|---|
| `urdf_assembly_manager.py` | Coordinates registration, connection generation, merging, and output. |
| `component.py` | Loads component URDFs, applies prefixes, and manages component registries. |
| `connection.py` | Creates fixed joints according to connection rules and component transforms. |
| `sensor.py` | Registers sensor attachments and merges their links and joints. |
| `mesh.py` | Resolves and copies mesh, material, and texture assets. |
| `name_normalizer.py` | Applies link and joint casing policies. |
| `file_writer.py` | Formats and writes the assembled URDF. |
| `signature.py` | Calculates and checks assembly signatures. |
| `logging_utils.py` | Provides assembly-specific logging. |

Most applications should use `URDFAssemblyManager` or
`embodichain.lab.sim.cfg.URDFCfg` instead of calling these internal helpers
directly.

## Quick Start

```python
import numpy as np

from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

manager = URDFAssemblyManager()

manager.add_component(
    component_type="arm",
    urdf_path="assets/arm.urdf",
)
manager.add_component(
    component_type="hand",
    urdf_path="assets/hand.urdf",
    transform=np.eye(4),
)

manager.merge_urdfs(output_path="build/arm_with_hand.urdf")
```

`add_component()` returns `False` and logs an error if registration fails.
Check its return value in asset-processing pipelines before calling
`merge_urdfs()`.

## Component Assembly

### Supported Component Types

The manager recognizes these component roles:

| Component | Purpose |
|---|---|
| `chassis` | Mobile base or central platform. |
| `legs` | Legged locomotion system. |
| `torso` | Main body between the base and upper-body components. |
| `head` | Head or upper sensor structure. |
| `left_arm`, `right_arm` | Side-specific manipulators. |
| `left_hand`, `right_hand` | Side-specific end effectors. |
| `arm` | Single manipulator without a side designation. |
| `hand` | Single end effector without a side designation. |

For a chassis, the optional `wheel_type` parameter accepts `omni`,
`differential`, or `tracked`. Referenced meshes can use STL, OBJ, PLY, DAE, or
GLB formats.

### Connection Rules

The manager connects registered components using built-in parent-child rules:

- `chassis` → `legs` → `torso`
- `chassis` → `torso`
- `torso` → `head`
- `torso` or `chassis` → side-specific arms
- `left_arm` → `left_hand`
- `right_arm` → `right_hand`
- `arm` → `hand`

Components without a matching parent are attached to the assembly base link.
The `transform` passed to `add_component()` controls the fixed joint that
attaches that component. It must be a 4×4 homogeneous transformation matrix.

```python
hand_transform = np.eye(4)
hand_transform[2, 3] = 0.05

manager.add_component(
    component_type="hand",
    urdf_path="assets/hand.urdf",
    transform=hand_transform,
)
```

## Sensor Attachment

`attach_sensor()` accepts either a path to a sensor URDF or an
`xml.etree.ElementTree.Element`. The attachment identifies both the component
and the link within that component.

```python
manager.attach_sensor(
    sensor_name="front_camera",
    sensor_source="assets/camera.urdf",
    parent_component="chassis",
    parent_link="base_link",
    transform=np.eye(4),
)
```

The predefined sensor categories are `camera`, `lidar`, `imu`, `gps`, and
`force`. Use a unique `sensor_name` for each attachment.

## Naming Configuration

### Component Prefixes

Prefixes prevent duplicate link and joint names when two components originate
from the same URDF. The default side-specific prefixes are `left_` and
`right_`; the other component types have no prefix.

Set `component_prefix` with a list of `(component_type, prefix)` tuples:

```python
manager.component_prefix = [
    ("left_arm", "L_"),
    ("right_arm", "R_"),
    ("left_hand", "L_"),
    ("right_hand", "R_"),
]
```

This property uses patch semantics: omitted component types keep their current
prefix. It does not accept new component types, and an unknown type raises
`ValueError`.

### Link and Joint Casing

The `name_case` policy controls how names are normalized:

```python
manager.name_case = {
    "joint": "upper",
    "link": "lower",
}
```

Both `joint` and `link` keys are required. Supported modes are `upper`, `lower`,
and `original`; `none` is retained as a legacy alias for `original`.
`URDFAssemblyManager` defaults to uppercase joints and lowercase links.

```{note}
`URDFCfg` defaults to preserving the source casing for both links and joints.
Set `name_case` explicitly when direct-manager and simulation-config workflows
must produce identical names.
```

Prefix and casing changes are included in the assembly signature, so they
trigger a rebuild even when the component files are unchanged.

## Public API

### `add_component()`

Registers a component URDF:

```python
manager.add_component(
    component_type: str,
    urdf_path: str | Path,
    transform: np.ndarray | None = None,
    **params,
) -> bool
```

### `attach_sensor()`

Registers a sensor attachment:

```python
manager.attach_sensor(
    sensor_name: str,
    sensor_source: str | Element,
    parent_component: str,
    parent_link: str,
    transform: np.ndarray | None = None,
    **kwargs,
) -> bool
```

### `merge_urdfs()`

Builds and writes the unified description:

```python
manager.merge_urdfs(
    output_path: str = "./assembly_robot.urdf",
    use_signature_check: bool = True,
) -> Element
```

When signature checking is enabled, the manager reuses an existing output if
the component contents, transforms, parameters, prefix configuration, casing
policy, and output name have not changed.

### Registry Access

Use `get_component(component_type)` to retrieve one registered component and
`get_attached_sensors()` to retrieve all sensor attachments.

## Using `URDFCfg` in Simulation

`URDFCfg` is the convenient integration point for robots created through
`SimulationManager`. It invokes the assembly toolkit automatically when more
than one component is configured.

```python
import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.cfg import RobotCfg, URDFCfg

hand_transform = np.eye(4)
hand_transform[:3, :3] = Rotation.from_euler(
    "x", 90, degrees=True
).as_matrix()

cfg = RobotCfg(
    uid="arm_with_hand",
    urdf_cfg=URDFCfg(
        components=[
            {
                "component_type": "arm",
                "urdf_path": "assets/arm.urdf",
            },
            {
                "component_type": "hand",
                "urdf_path": "assets/hand.urdf",
                "transform": hand_transform,
            },
        ],
        component_prefix=[("hand", "tool_")],
        name_case={"joint": "original", "link": "original"},
    ),
)
```

Each component dictionary requires `component_type` and `urdf_path`; `transform`
and component-specific parameters are optional. `URDFCfg` also supports:

| Setting | Purpose |
|---|---|
| `sensors` | Sensor attachment configurations. |
| `base_link_name` | Name of the assembly root link. |
| `component_prefix` | Per-component prefix overrides. |
| `name_case` | Link and joint casing policy. |
| `use_signature_check` | Enables incremental assembly reuse. |
| `fpath` | Explicit output URDF path. |
| `fname`, `fpath_prefix` | Generated output name and parent directory. |

For a runnable robot example, see
`scripts/tutorials/sim/create_robot.py` in the repository.
