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
`left_arm_base` and `right_arm_base`. The source end links depend on the variant:

| `with_gripper` | Left source end link | Right source end link |
| --- | --- | --- |
| `True` | `left_hand_tool_link` | `right_hand_tool_link` |
| `False` | `left_ee` | `right_ee` |

**Both variants apply an additional end-link-to-TCP transform.** The TCP origin
is 0.13 m along the source end link's +X axis, and its orientation is rotated
+90 degrees about that link's Y axis. The default `solver_cfg[part].tcp` is:

```text
T_end_link_tcp = [[ 0, 0, 1, 0.13],
                 [ 0, 1, 0, 0   ],
                 [-1, 0, 0, 0   ],
                 [ 0, 0, 0, 1   ]]
```

Solver FK returns `T_root_tcp = T_root_end_link @ T_end_link_tcp`, and solver IK
expects this TCP pose relative to the arm root. Named `Robot.compute_fk()`
returns the TCP pose in the local arena frame; `Robot.compute_ik()` expects its
target in that same frame. If an application starts with a source end-link pose
in the arena, convert it with `T_arena_tcp = T_arena_end_link @ T_end_link_tcp`
before passing it to IK.

`build_pk_serial_chain()` returns the two seven-joint raw URDF chains, ending at
the source links in the table. Their FK matrices omit the configured TCP
transform; right-multiply by `solver_cfg[part].tcp` to obtain solver TCP poses.
The parallel fingers are not a single serial chain. Both chains use the same
URDF as the simulation, including an explicit `fpath` override.

To make named robot FK/IK operate directly on the source end-link poses, set
identity TCP transforms for both arms. This override works with either variant:

```python
import numpy as np
from embodichain.lab.sim.robots import TianjiMarvinCfg

cfg = TianjiMarvinCfg.from_dict({
    "with_gripper": False,
    "solver_cfg": {
        "left_arm": {"tcp": np.eye(4).tolist()},
        "right_arm": {"tcp": np.eye(4).tolist()},
    },
})
```

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
