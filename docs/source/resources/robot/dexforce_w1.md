
# Dexforce W1

Dexforce W1 is a dual-arm humanoid robot developed by DexForce Technology Co., Ltd.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; flex-wrap: wrap;">
  <figure style="text-align: center; margin: 10px;">
    <img src="../../_static/robots/dexforcew1.jpg" alt="Dexforce W1" style="height: 400px; width: auto;"/>
    <figcaption><b>Dexforce W1</b></figcaption>
  </figure>
</div>

## Key Features

- Supports dual 7-DOF arms
- Supports version-owned asset layouts and calibration parameters
- Configurable left/right hand brand and version
- Flexible URDF assembly and simulation configuration
- Compatible with SimulationManager simulation environment

## Configuration with `DexforceW1Cfg.from_dict`

`DexforceW1Cfg.from_dict` is the single public construction entry point. A W1
configuration always describes the complete chassis, torso, head, eyes, wrist
cameras, and both arms for one robot version.

**Parameters:**

- `hand_types`: Dict specifying hand brand for each arm side (`LEFT`/`RIGHT`).
- `hand_versions`: Dict specifying hand version for each arm side.
- `with_default_eef`: Whether to install the registered default end effectors.
- `hand_attach_xposes`: Optional hand-specific mounting transforms.

```python
from embodichain.lab.sim.robots import DexforceW1Cfg

cfg = DexforceW1Cfg.from_dict(
    {
        "uid": "dexforce_w1",
        "version": "v025",
        "hand_types": {
            "left": "BRAINCO_HAND",
            "right": "BRAINCO_HAND",
        },
        "hand_versions": {"left": "v021", "right": "v021"},
    }
)
robot = sim.add_robot(cfg=cfg)
print("DexforceW1 robot added to the simulation.")
```

## Arm Joint Design: Mirrored Configuration

DexforceW1's left and right arms are designed with mirrored joint configurations. This means the joint angles for the left and right arms are symmetric but opposite in sign for certain axes, making it easier to coordinate bimanual tasks and maintain natural robot postures.

### Example: Setting Arm Joint Positions

```python
import numpy as np
# Set left arm joint positions (mirrored)
robot.set_qpos(qpos=[0, -np.pi/4, 0.0, -np.pi/2, -np.pi/4, 0.0, 0.0], joint_ids=robot.get_joint_ids("left_arm"))
# Set right arm joint positions (mirrored)
robot.set_qpos(qpos=[0, np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0], joint_ids=robot.get_joint_ids("right_arm"))
```

This mirrored design simplifies motion planning and ensures that both arms can perform coordinated or symmetric actions efficiently.

## Type Descriptions

| Type                    | Options / Values                                      | Description                        |
|-------------------------|-------------------------------------------------------|------------------------------------|
| `DexforceW1HandBrand`   | `BRAINCO_HAND`, `DH_PGC_GRIPPER`, `DH_PGC_GRIPPER_M`  | Hand brand                         |
| `DexforceW1Version`     | `V021`, `V022`, `V025`                                | Release version                    |
| `DexforceW1HandVersion` | `V021`                                                  | External hand/gripper asset version |
| `DexforceW1ArmSide`     | `LEFT`, `RIGHT`                                       | Left/right hand identifier         |

## Unified asset layout and version extension

V022 and V025 use one unified Hugging Face archive per release:

```text
dexforce_w1/<version>/w1.zip
└── w1/
    ├── robot.urdf
    ├── chassis.urdf
    ├── torso.urdf
    ├── head.urdf
    ├── left_arm.urdf
    ├── right_arm.urdf
    ├── visual/
    └── collision/
```

The runtime downloads each release archive once. Direct FK/IK uses `robot.urdf`
or the arm URDFs, while configurable robot assembly reads all components from
the same extracted directory. V022 and V025 assets are resolved through the
registered Hugging Face dataset archives and the shared asset cache.

### Version-owned end-effector offset

Different arm revisions may place the physical mounting surface at different
positions relative to the arm `ee` frame. This difference belongs to the W1
revision, not to a BrainCo hand, DH gripper, PIKA gripper, or any other
end-effector.

`W1VersionSpec.default_eef_attach_xpos` is the single source of truth for this
revision offset:

| Version | Left arm | Right arm |
|---------|----------|-----------|
| V021 | Identity | Identity |
| V022 | Identity (provisional; calibration required) | Identity (provisional; calibration required) |
| V025 | `+0.012 m` along the `ee` frame Z axis | `+0.012 m` along the `ee` frame Z axis |

The final assembly transform and solver TCP are derived as follows:

```python
final_attach_xpos = version_attach_xpos @ eef_attach_xpos
final_tcp = version_attach_xpos @ solver_tcp
```

Therefore, the V025 offset affects both the assembled end-effector position and
FK/IK results. The offset is also applied when callers provide a custom
`hand_attach_xposes` value, a custom `left_hand`/`right_hand` component
transform, or an explicit arm TCP through `DexforceW1Cfg.from_dict`.
Serialization removes the derived offset and restores it on loading, so a
configuration round trip does not apply the offset twice.

Do not manually add the 12 mm correction to an end-effector transform or TCP.
Those values must describe the end-effector relative to the standard mounting
surface; the W1 version layer adds the robot revision correction.

One `DexforceW1Version` selects the complete W1 robot release. Mixed body/arm
versions are intentionally unsupported because their assets, kinematics, TCP,
and camera calibration must remain consistent.

Robot and hand versions are independent. `DexforceW1Version` selects the W1
body, arm assets, kinematics, and flange calibration. `DexforceW1HandVersion`
selects an external hand or gripper asset from
`embodichain/lab/sim/robots/dexforce_w1/hand_specs.py`.

The currently released BrainCo hand and DH gripper registrations all use
`DexforceW1HandVersion.V021`. Therefore W1 V021, V022, and V025 select hand V021
by default. This is an explicit default, not a compatibility alias keyed by the
robot version. A future hand release is added by registering a new hand version
and hand spec; it does not require adding aliases for every W1 robot version.

To add another W1 revision:

1. Register the new `DexforceW1Version` value and dataset archive.
2. Add one `W1VersionSpec` entry in
   `embodichain/lab/sim/robots/dexforce_w1/specs.py`.
3. Register component URDF paths and the full-robot URDF path.
4. Set the arm kinematic parameters and
   `default_eef_attach_xpos` for both sides. Use identity only after confirming
   that the arm `ee` frame is already at the physical mounting surface.
5. Verify that the same version offset is present in both the assembled
   end-effector pose and the final FK/IK TCP.
6. Validate hand/gripper mounting, wrist cameras, FK/IK, VR teleoperation, and
   real2sim task regression.

To add another hand release:

1. Add a `DexforceW1HandVersion` value.
2. Register each supported brand and side in `hand_specs.py`, including its
   URDF, joint names, root/end links, and mounting transform.
3. Select it explicitly through `hand_versions`; an unregistered version fails
   during config construction.

Control parts, assembly, TCP selection, analytical parameters and FK/IK then use
the version specification without revision-specific branches.
