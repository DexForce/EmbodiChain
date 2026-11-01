# Franka robot family

`FrankaCfg` selects a physical arm and tool explicitly. It uses the existing
`Robot` runtime and supports two qualified configuration combinations:

| Variant | Arm and hand | Solver | Independent DOFs |
| --- | --- | --- | ---: |
| `fr3/panda_hand` | Historical FR3-named asset with integrated Panda Hand | FEP | 8 |
| `panda/robotiq_2f85` | Original Panda arm with Robotiq 2F-85 | Pink | 8 |

Other cross-combinations are rejected. The default is `fr3/panda_hand`.
Model and vendor package imports stay lazy; listing the catalog downloads no
assets and creates no simulator.

```python
from embodichain_robots import FrankaCfg
from embodichain.robot_registry import resolve_robot_cfg

default = FrankaCfg.from_dict({})
tool_robot = FrankaCfg.from_dict({
    "arm_model": "panda",
    "gripper": "robotiq_2f85",
    "gripper_mount_z_m": 0.018,
    "gripper_mount_yaw_rad": 0.0,
    "tcp_offset_m": 0.15,
})
same_model = resolve_robot_cfg("Franka", variant="panda/robotiq_2f85")
```

The 18 mm mounting value is an example calibration, not a universal hardware
default. The nominal external mount defaults to zero and the tool offset to
150 mm. An embodiment supplies its calibrated values. The assembly and solver
use the same transform once; changing calibration isolates the assembly cache.
The integrated FR3 hand uses its authored `fr3_hand_tcp`; the three external
calibration fields must remain `None` for that combination. Standard
`solver_cfg` overrides remain available.

The reusable `franka_panda_robotiq85.yaml` embodiment selects `class_type: Franka`
and owns its tool calibration and semantic open/grasp commands. It exposes all
six Robotiq linkage joints, including five mimic followers with signed motion.
The tool keeps the shared ``Robotiq/robotiq_arg2f_85`` asset reference and its
calibration. The separately registered ``Robotiq2F85`` archive is a different
model identity and is not substituted. Grasp planning uses full open-hand URDF collision surfaces, including authored
box-shaped rubber pads. Contact width alone cannot establish grasp feasibility.

## Compatibility and physical identity

`FrankaPandaCfg` and the `FrankaPanda` catalog ID remain compatibility entries.
Their legacy `robot_type: panda` selects exactly the historical asset, joints,
UID, initial pose, FEP settings and drive parameters. That asset is internally
named `fr3`; existing applications do not silently switch to original-Panda
limits. New configurations select `arm_model` and `gripper` through `FrankaCfg`.
The legacy entry retains its existing dual-arm composition; Robotiq dual-arm
qualification is not advertised.

The original Panda arm is pinned through `FrankaPandaArm` to `FrankaV2.zip`.
Its original URDF is retained alongside a separately reported exact-quarter-turn
RPY variant shared by simulation and FK. Model-owned inertial overlays preserve
identified Panda parameters, upstream Robotiq nominal CAD inertials and the
explicit 5 g pad-mass assumption. These are simulation parameters and do not
certify hardware dynamics. The model README records hashes, licenses and sources.

`embodichain_robots.robotiq.two_finger.robotiq85_joint_positions()` maps an
observed normalized closed fraction to the complete linkage. It does not replace
measured aperture with a full-close command. No dedicated Robotiq robot class
is required.
