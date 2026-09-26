# OPWSolver

`OPWSolver` provides analytical IK for 6-DOF arms that match the OPW
parameterization. It supports batched solution branches, joint-limit checks,
and continuous solution selection along a path.

## Configuration

The solver is configured using the `OPWSolverCfg` class, which defines OPW parameters and solver options.

```python
import torch
import numpy as np
from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers.opw_solver import OPWSolver, OPWSolverCfg

cfg = OPWSolverCfg(
    urdf_path=get_data_path("CobotMagicArm/CobotMagicNoGripper.urdf"),
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    end_link_name="link6",
    root_link_name="arm_base",
    a1 = 0.0,
    a2 = -21.984,
    b = 0.0,
    c1 = 123.0,
    c2 = 285.03,
    c3 = 250.75,
    c4 = 91.0,
    offsets = (
        0.0,
        82.21350356417211 * np.pi / 180.0,
        -167.21710113148163 * np.pi / 180.0,
        0.0,
        0.0,
        0.0,
    ),
    flip_axes = (False, False, False, False, False, False),
    has_parallelogram = False,
)

solver = OPWSolver(cfg, device="cuda")
```

## Solver-specific behavior

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern and return order.

`return_all_solutions=True` returns candidate validity flags and joint
solutions. The default selects a valid solution near `qpos_seed`.

Continuous selection is requested at the robot boundary with
`Robot.compute_batch_ik(..., continuous=True)`. The robot performs arena/root
frame conversion and obtains the candidates in one flattened batch. OPW's
sequential selector returns validity `(B, N)` and continuous joint positions
`(B, N, 6)`.

## References

* [OPW Kinematics Paper](https://doi.org/10.1109/TRO.2017.2776312)
* [warp Documentation](https://github.com/NVIDIA/warp)
