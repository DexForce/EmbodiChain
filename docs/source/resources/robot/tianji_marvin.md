# Tianji Marvin

`TianjiMarvinCfg` loads the complete Tianji Marvin ACD URDF with its original
joint and link names. Select the two variants through `with_gripper`:

| Setting | Asset | Control parts |
| --- | --- | --- |
| `True` (default) | `TianjiMarvin/robot_with_ee_acd.urdf` | `left_arm`, `right_arm`, `left_hand`, `right_hand` |
| `False` | `TianjiMarvin/robot_acd.urdf` | `left_arm`, `right_arm` |

```python
from embodichain.lab.sim.robots import TianjiMarvinCfg

cfg = TianjiMarvinCfg.from_dict({"with_gripper": True})
robot = sim.add_robot(cfg=cfg)
```

Each arm has seven joints, ordered from shoulder to wrist. Each hand contains
`LEFT_HAND_FINGER_1/2` or `RIGHT_HAND_FINGER_1/2`; finger 2 mimics finger 1.
The hands use joint-space control. Their joint range is -0.05 to 0 metres.

The default arm solvers are `PytorchSolverCfg`. Their root frames are
`left_arm_base` and `right_arm_base`. With grippers, the end frames are
`left_hand_tool_link` and `right_hand_tool_link`; without grippers they are
`left_ee` and `right_ee`. No extra TCP offset is applied.
`build_pk_serial_chain()` returns the two seven-joint arm chains; the parallel
fingers are not a single serial chain. Both FK chains use the same URDF as the
simulation, including an explicit `fpath` override.

The root is fixed. Drive gains are simulation defaults based on Aloha Mini and
can be overridden through `joint_drive_props`. They are not factory motor
specifications. Construct presets through `from_dict()` so variant defaults
are assembled before overrides; `to_dict()` preserves the variant selection.

Run the preview and FK/IK smoke check in the project environment:

```bash
conda run -n embodichain python -m embodichain.lab.sim.robots.tianji_marvin --headless
conda run -n embodichain python -m embodichain.lab.sim.robots.tianji_marvin --headless --no-with-gripper
```

Omit `--headless` to open the simulation window and an interactive session.
