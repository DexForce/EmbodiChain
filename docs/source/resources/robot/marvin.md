# Marvin

Marvin is a fixed-base dual-arm robot with two symmetric 7-DOF arms and two
parallel grippers. The configuration uses the complete
`robot_with_ee.urdf` and exposes both arms and grippers as independent control
parts.

## Asset setup

Place the complete Marvin asset directory (including `collision/` and
`visual/`) at `$EMBODICHAIN_DATA_ROOT/Marvin`, or pass an absolute path:

```python
from embodichain.lab.sim.robots import MarvinCfg

cfg = MarvinCfg.from_dict({
    "urdf_path": "/path/to/Marvin/robot_with_ee.urdf",
})
robot = sim.add_robot(cfg=cfg)
```

Without an override, `MarvinCfg` resolves `Marvin/robot_with_ee.urdf` through
the EmbodiChain data root. The arm-only `robot.urdf` is not the default because
it omits the gripper links and joints.

## Kinematic configuration

| Control part | Root link | End link | Degrees of freedom |
|---|---|---|---:|
| `left_arm` | `left_arm_base` | `left_ee` | 7 |
| `right_arm` | `right_arm_base` | `right_ee` | 7 |

The gripper control parts are `left_eef` and `right_eef`. Each contains two
prismatic finger joints with a URDF range of 0–0.05 m. The second finger is a
mimic of the first, so each gripper has one independently actuated DOF.

Joint names, axes, position limits, actuator effort limits, and velocity limits
come directly from the supplied URDF. The URDF also defines a 95 mm fixed
transform from each wrist-roll link to its corresponding EE link.

## Provisional parameters

The following values require physical validation:

- The solver TCP is provisionally 140 mm along the positive Z axis of each EE
  frame. Although the complete URDF contains gripper geometry, its
  `left_hand_tool_link` and `right_hand_tool_link` contain no translation to a
  defined grasp center, so this offset still cannot be verified from the URDF.
- Position-drive stiffness is 600 N·m/rad for shoulder joints, 400 N·m/rad for
  elbow joints, and 120 N·m/rad for wrist joints.
- Drive damping is respectively 50, 35, and 8 N·m·s/rad. These are conservative
  simulation starting points inferred from link masses and inertias, not
  measured controller gains.
- Gripper stiffness is provisionally 1000 N/m. The gripper damping of 8 N·s/m,
  joint friction of 0.2, and active-finger effort limit of 30 N come directly
  from the URDF. Because the source URDF declares a zero velocity limit, the
  usable simulation velocity limit of 0.1 m/s is provisional and needs vendor
  or real-robot validation.
- Rigid-body friction uses the same conservative contact defaults as the
  CobotMagic configuration because the URDF provides no contact material data.

`max_effort` is not estimated: it is set to the URDF limits of 108 N·m for J1
and J2, 66 N·m for J3 and J4, and 18 N·m for J5 through J7. All arm joints use
the URDF velocity limit of 3.1416 rad/s.

## Interactive TCP calibration

Use the calibration script to display the URDF EE frames and configured TCP
frames, adjust either TCP in the EE-local frame, and save matrices ready for
`solver_cfg.tcp`:

```bash
python examples/sim/robot/calibrate_marvin_tcp.py \
    --urdf-path /path/to/Marvin/robot_with_ee.urdf
```

Enter `help` in the terminal for commands. Common adjustments are `select
left`, `t z 0.001` (translate 1 mm), `r x 1` (rotate 1 degree), `show all`, and
`save outputs/marvin_tcp.json`. The saved JSON is a robot-config fragment and
contains end-link-relative TCP matrices rather than scene/world poses.
