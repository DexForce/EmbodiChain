# Parallel-jaw grippers

Concrete `ParallelJawGripperCfg` subclasses own native simulation and grasp data.
They are registered in the existing `embodichain.robots` catalog and can be
previewed independently with `embodichain preview-robot --robot DHPGI14080`.

| Catalog ID | Native source |
|---|---|
| `FrankaPandaHand` | Pinned FR3/Panda Hand subtree |
| `DHPGI14080` | DH PGI-140-80 |
| `CobotMagicV100Gripper` | V100 hand subtree, distinct from V70 |
| `PikaGripper` | Standalone Pika hand including its root adapter |
| `TianjiMarvinGripper` | Explicit `left` or `right` variant |
| `RobotiqArg2F85` | ARG2F-85 in the existing Robotiq archive |
| `RobotiqArg2F140` | ARG2F-140 in the existing Robotiq archive |

`RobotiqArg2F85` does not select the separate `Robotiq2F85` asset. Independent
simulation drive defaults are presets, not manufacturer actuator calibration.
Embedded hands are materialized as content-addressed source subtrees only when
used independently; original assets are retained and derivatives record hashes.

## Embodiment binding

```yaml
grippers:
  tool:
    class_type: DHPGI14080
    mode: mounted
    component_type: hand
    motion_part: arm
    control_part: eef
```

A mounted binding owns the hand component and control group. Its optional
`transform` is the existing URDF-component installation transform. `existing`
binds a hand already present in the selected robot; it appends no component.
Marvin requires `variant: left` or `variant: right`. Native names are mapped
through the same name-normalization policy as ordinary URDF assembly.

A declared grasp endpoint opts into reuse with `gripper: tool`, its matching
`control_part`, and explicit `interaction.grasp` capability. Omitted presets
receive `gripper.tool`; a default grasp generator is created only for those
explicit endpoints. Existing preset IDs may be retained, but their physical
commands must not also be repeated inline. Algorithm policy remains under the
existing `runtime_services.grasp_pose_generators` target and task override route.
Handwritten/RL embodiments may omit the semantic profile completely.

Commands are complete mappings of native joint names, including mimic followers.
`control_joint_names` can explicitly project an active-only control group. The
binding's optional `commands` replaces complete commands by command name; task
integrations cannot replace hardware, native commands or calibration.

## TCP and asset validation

The hand supplies `default_root_from_tcp` and a separate
`root_from_contact_frame`. A binding may explicitly replace `root_from_tcp` or
`root_from_contact_frame`. Authors do not need to duplicate solver TCP values:

```text
T_solver_end_tcp = T_solver_end_hand_root @ T_hand_root_tcp
T_contact_tcp = inverse(T_hand_root_contact) @ T_hand_root_tcp
```

Static composition reads no asset and creates no Robot. Complete configuration
construction projects known fixed references; runtime assembly verifies actual
joint ownership, mimic commands, collision meshes and fixed end/root linkage.
An explicit solver TCP must agree with the binding. GraspKit converts canonical
candidates to the selected TCP once; consumers must not add another copy of the
hand calibration. Fixed TCP/object-to-EEF inputs retain their existing bypass.

## Python and migration

```python
from embodichain.robot_registry import resolve_gripper_defaults, resolve_robot_cfg
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator, build_gripper_collision_geometry,
)

data = resolve_gripper_defaults("DHPGI14080")  # Pure declarations, no asset I/O.
hand = resolve_robot_cfg("DHPGI14080")        # Full standalone RobotCfg.
generator = AntipodalGraspPoseGenerator(
    ParallelJawGripperModelCfg(**data["grasp_cfg"]),
    collision_geometry=build_gripper_collision_geometry(data),
)
```

The former `get_parallel_jaw_gripper_model` and
`get_parallel_jaw_collision_geometry` APIs are removed. The former generator
`model` string/mapping is removed; select a bound gripper or supply an explicit
physical description. No similarly named hardware is selected as a fallback.
