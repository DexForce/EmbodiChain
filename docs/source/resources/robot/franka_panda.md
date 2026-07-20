# Franka Panda

`FrankaPandaCfg` configures the Franka Panda manipulator together with its
parallel-jaw hand. The bundled model contains the seven-joint arm and two
finger joints in one URDF.

<div style="text-align: center;">
  <img src="../../_static/robots/franka_panda.png" alt="Franka Panda robot" style="height: 400px; width: auto;"/>
  <p><b>Franka Panda with parallel-jaw hand</b></p>
</div>

## Key Features

- **Seven-axis manipulator** with the two-joint Panda hand.
- **Ready-to-use robot configuration** for the URDF, control parts, joint
  drives, and initial joint positions.
- **Numerical forward and inverse kinematics** through `PytorchSolverCfg` and
  `pytorch-kinematics`.
- **Multi-environment simulation support** through `SimulationManager`.
- **Dual-arm composition support** through the `"franka"` base-robot key.

## Usage

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import FrankaPandaCfg

sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=4))
cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
robot = sim.add_robot(cfg=cfg)
```

`from_dict` also accepts standard `RobotCfg` overrides. For example, the base
pose and initial joint positions can be changed when the configuration is
created:

```python
cfg = FrankaPandaCfg.from_dict(
    {
        "robot_type": "panda",
        "init_pos": [0.5, 0.0, 0.0],
        "init_qpos": [0.0, -0.5, 0.0, -2.5, 0.0, 2.8, 0.7, 0.04, 0.04],
    }
)
```

## Robot Configuration

| Item | Default |
|------|---------|
| Configuration class | `FrankaPandaCfg` |
| `robot_type` | `"panda"` (the only currently supported variant) |
| UID | `FrankaPanda` |
| URDF | `Franka/Panda/PandaWithHand.urdf` |
| Arm joints | `fr3_joint1` through `fr3_joint7` |
| Hand joints | `fr3_finger_joint1`, `fr3_finger_joint2` |
| Kinematics root / end link | `base` / `fr3_hand_tcp` |
| Solver | `PytorchSolverCfg` with 30 samples |

The configuration exposes two independently addressable control parts:

```python
arm_joint_ids = robot.get_joint_ids("arm")
hand_joint_ids = robot.get_joint_ids("hand")
```

The hand's second finger joint is a mimic of the first in the bundled URDF.

## Default Joint State

The default state places the arm in a neutral ready pose and opens both fingers:

| Joint | Position (rad) |
|-------|----------------|
| `fr3_joint1` | 0.000 |
| `fr3_joint2` | -0.569 |
| `fr3_joint3` | 0.000 |
| `fr3_joint4` | -2.810 |
| `fr3_joint5` | 0.000 |
| `fr3_joint6` | 3.037 |
| `fr3_joint7` | 0.741 |
| `fr3_finger_joint1` | 0.040 |
| `fr3_finger_joint2` | 0.040 |

## Dual-Arm Composition

Use `DualArmRobotCfg` with `base_robot="franka"` to create a pair of Panda
arms. The dual-arm builder prefixes the generated joints, control parts, and
solver links with `left_` and `right_`.

```python
from embodichain.lab.sim.robots import DualArmRobotCfg

cfg = DualArmRobotCfg.from_dict(
    {
        "base_robot": "franka",
        "mount": {"preset": "side_by_side", "separation": 0.6},
    }
)
robot = sim.add_robot(cfg=cfg)
```

See :doc:`dual_arm` for the available mount presets and configuration options.

## See Also

- :doc:`/overview/sim/solvers/index` - IK solver reference
- :doc:`/guides/add_robot` - Adding a new robot (quick reference)
