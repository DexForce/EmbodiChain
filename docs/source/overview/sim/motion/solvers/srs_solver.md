# SRSSolver

`SRSSolver` provides analytical IK for 7-DOF arms with a
spherical-rotational-spherical (S-R-S) joint structure, including joint limits
and multiple-solution sampling.

S-R-S describes three intersecting shoulder axes, one elbow axis, and three
intersecting wrist axes. The extra degree of freedom permits multiple joint
configurations for the same end-effector pose.

## Configuration

SRSSolver is configured via the `SRSSolverCfg` class, allowing detailed control over kinematic parameters and solver behavior.

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.params import (
    W1ArmKineParams,
)
from embodichain.lab.sim.motion.solvers.srs_solver import SRSSolver, SRSSolverCfg

arm_params = W1ArmKineParams(
    arm_side=DexforceW1ArmSide.RIGHT,
    version=DexforceW1Version.V021,
)

cfg = SRSSolverCfg(
    urdf_path=get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf"),
    joint_names=[f"RIGHT_J{i+1}" for i in range(7)],
    end_link_name="right_ee",
    root_link_name="right_arm_base",
    dh_params=arm_params.dh_params,
    user_qpos_limits=arm_params.qpos_limits,
    T_e_oe=arm_params.T_e_oe,
    T_b_ob=arm_params.T_b_ob,
    link_lengths=arm_params.link_lengths,
    rotation_directions=arm_params.rotation_directions,
)

solver = SRSSolver(cfg, num_envs=1, device="cuda")
```

## Solver-specific behavior

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern and return order.

The default `get_ik()` result contains success flags `(B,)` and a selected
solution `(B, 1, 7)`. `return_all_solutions=True` retains multiple solutions.
Configure arm-angle sampling and joint limits for the desired redundancy
resolution; the robot must match the S-R-S geometry.

## References

* **Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution**  
  Masayuki Shimizu, Hiromu Kakuya, Woo-Keun Yoon, Kosei Kitagaki, Kazuhiro Kosuge  
  *IEEE Transactions on Robotics*, 2008  
  [DOI: 10.1109/TRO.2008.2003266](https://doi.org/10.1109/TRO.2008.2003266)  
  This paper presents an analytical approach for solving the inverse kinematics of 7-DOF redundant manipulators, including joint limit handling and redundancy resolution strategies.

* **Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance**  
  Carlos Faria, Flora Ferreira, Wolfram Erlhagen, Sérgio Monteiro, Estela Bicho  
  *Mechanism and Machine Theory*, 2018  
  [DOI: 10.1016/j.mechmachtheory.2017.10.025](https://doi.org/10.1016/j.mechmachtheory.2017.10.025)  
  This work introduces position-based kinematic algorithms for 7-DOF manipulators, focusing on global configuration control, joint limit enforcement, and singularity avoidance.
