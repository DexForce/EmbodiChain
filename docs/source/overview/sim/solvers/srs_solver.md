# SRSSolver

`SRSSolver` is a high-performance inverse kinematics (IK) solver specifically designed for 7-DOF manipulators with a Spherical-Rotational-Spherical (S-R-S) joint structure. This architecture is common in anthropomorphic and redundant industrial arms, providing high dexterity and redundancy for advanced manipulation tasks. SRSSolver supports batch computation, joint limits, GPU acceleration, and seamless integration with PyTorch workflows.

## What is S-R-S Kinematics?

The S-R-S (Spherical-Rotational-Spherical) structure refers to a 7-joint manipulator arrangement:

* **Spherical (S)**: A 3-DOF spherical joint (often realized by three intersecting revolute joints), enabling arbitrary orientation in 3D space.
* **Rotational (R)**: A single revolute joint, typically located at the "elbow, " providing an extra degree of freedom for redundancy.
* **Spherical (S)**: Another 3-DOF spherical joint at the wrist, allowing full orientation control of the end-effector.

This structure enables:
* **Redundancy**: The arm can reach the same pose with multiple joint configurations, useful for obstacle avoidance and singularity avoidance.
* **High Dexterity**: Suitable for tasks requiring complex manipulation and orientation.
* **Wide Application**: Common in humanoid robots and collaborative arms.

## Key Features

* Optimized for S-R-S 7-DOF kinematic chains
* Supports position and orientation constraints, joint limits
* Batch sampling and multi-solution output for robust IK
* GPU acceleration for large-scale or real-time applications
* Flexible configuration for DH parameters, joint limits, link lengths, and more

## Configuration

SRSSolver is configured via the `SRSSolverCfg` class, allowing detailed control over kinematic parameters and solver behavior.

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.params import (
    W1ArmKineParams,
)
from embodichain.lab.sim.solvers.srs_solver import SRSSolver, SRSSolverCfg

arm_params = W1ArmKineParams(
    arm_side=DexforceW1ArmSide.RIGHT,
    arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
    version=DexforceW1Version.V020,
)

cfg = SRSSolverCfg(
    urdf_path=get_data_path("DexforceW1V020/DexforceW1_v02_1.urdf"),
    joint_names=[f"{'RIGHT'}_J{i+1}" for i in range(7)],
    end_link_name="left_ee",
    root_link_name="left_arm_base",
    dh_params=arm_params.dh_params,
    qpos_limits=arm_params.qpos_limits,
    T_e_oe=arm_params.T_e_oe,
    T_b_ob=arm_params.T_b_ob,
    link_lengths=arm_params.link_lengths,
    rotation_directions=arm_params.rotation_directions,
)

solver = SRSSolver(cfg, num_envs=1, device="cuda")
```

## Main Methods

* `get_fk(self, qpos: torch.Tensor) -> torch.Tensor`  
  Computes the end-effector pose (homogeneous transformation matrix) for the given joint positions.

  **Parameters:**
  + `qpos` (`torch.Tensor` or `list[float]`): Joint positions, shape `(num_envs, num_joints)` or `(num_joints,)`.

  **Returns:**
  + `torch.Tensor`: End-effector pose(s), shape `(num_envs, 4, 4)`.

  **Example:**

```python
  fk = solver.get_fk(qpos=[0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0])
  print(fk)
  # Output:
  # tensor([[[ 0.0,     -1.0,      0.0,      0.0],
  #          [ 0.0,      0.0,     -1.0,     -0.33],
  #          [ 1.0,      0.0,      0.0,      0.3625],
  #          [ 0.0,      0.0,      0.0,      1.0]]])
```

* `get_ik(self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor = None, return_all_solutions: bool = False, jacobian: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]`  
  Computes joint positions (inverse kinematics) for the given target end-effector pose.

  **Parameters:**
  + `target_xpos` (`torch.Tensor`): Target end-effector pose(s), shape `(num_envs, 4, 4)`.
  + `qpos_seed` (`torch.Tensor`, optional): Initial guess for joint positions, shape `(num_envs, num_joints)`. If `None`, a default is used.
  + `return_all_solutions` (`bool`, optional): If `True`, returns all possible solutions. Default is `False`.
  + `jacobian` (`torch.Tensor`, optional): Custom Jacobian. Usually not required.

  **Returns:**
  + `Tuple[torch.Tensor, torch.Tensor]`:
    - First element: Joint positions, shape `(num_envs, num_joints)`.
    - Second element: Convergence info or error for each environment.

  **Example:**

```python
  import torch
  xpos = torch.tensor([[[ 0.0,     -1.0,      0.0,     -0.0],
                        [ 0.0,      0.0,     -1.0,     -0.33],
                        [ 1.0,      0.0,      0.0,      0.3625],
                        [ 0.0,      0.0,      0.0,      1.0]]])
  qpos_seed = torch.zeros((1, 7))
  qpos_sol, info = solver.get_ik(target_xpos=xpos)
  print("IK solution:", qpos_sol)
  print("Convergence info:", info)
  # IK solution: tensor([True], device='cuda:0')
  # Convergence info: tensor([[[-0.022269, 0.045214, -0.022273, -1.570796, 0.045204, -0.001007, 0.044519]]], device='cuda:0')
```

## References

The following key references provide the theoretical foundation and algorithmic background for this implementation:

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

These publications provide the mathematical models and solution strategies that underpin the SRSSolver's design and functionality.
