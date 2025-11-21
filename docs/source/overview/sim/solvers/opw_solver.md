# OPWSolver

`OPWSolver` is a specialized inverse kinematics (IK) solver for 6-DOF industrial robots using the OPW kinematic parameterization. It provides fast, analytical solutions for robots with parallel and offset axes, supporting both CPU and GPU acceleration. The solver is suitable for large-scale batch IK tasks and real-time control.

## Key Features

* Analytical IK for OPW-parameterized 6-DOF manipulators
* Supports both parallel and offset axes, with custom axis flipping
* Fast batch computation for multiple target poses
* Configurable for CPU (py_opw_kinematics) and GPU (warp) backends
* Flexible configuration via `OPWSolverCfg`
* Strict enforcement of joint limits
* Forward kinematics (FK) and multiple IK solution branches

## Configuration

The solver is configured using the `OPWSolverCfg` class, which defines OPW parameters and solver options.

```python
import torch
import numpy as np
from embodichain.data import get_data_path
from embodichain.lab.sim.solvers.opw_solver import OPWSolver, OPWSolverCfg

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

## Main Methods

* `get_fk(self, qpos: torch.Tensor) -> torch.Tensor`  
  Computes the end-effector pose (homogeneous transformation matrix) for the given joint positions.

  **Parameters:**
  + `qpos` (`torch.Tensor` or `list[float]`): Joint positions, shape `(num_envs, num_joints)` or `(num_joints,)`.

  **Returns:**
  + `torch.Tensor`: End-effector pose(s), shape `(num_envs, 4, 4)`.

  **Example:**

```python
  fk = solver.get_fk(qpos=[0.0, 0.0, 0.0, 1.5708, 0.0, 0.0])
  print(fk)
  # Output:
  # tensor([[[ 0.0,      0.087093,      0.996200,      0.056135],
  #          [-1.0,      0.0     ,     -0.0     ,     -0.0     ],
  #          [ 0.0,     -0.996200,      0.087093,      0.213281],
  #          [ 0.0,      0.0     ,      0.0     ,      1.0     ]]], device=solver.device)
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
  xpos = torch.tensor([[[ 0.0,      0.087093,      0.996200,      0.056135],
                        [-1.0,      0.0     ,     -0.0     ,     -0.0     ],
                        [ 0.0,     -0.996200,      0.087093,      0.213281],
                        [ 0.0,      0.0     ,      0.0     ,      1.0     ]]], device=solver.device)

  qpos_seed = torch.zeros((1, 6))
  qpos_sol, info = solver.get_ik(target_xpos=xpos)
  print("IK solution:", qpos_sol)
  print("Convergence info:", info)
  # IK solution: tensor([1], device='cuda:0', dtype=torch.int32)
  # Convergence info: tensor([[-3.141593, 0.793811, 0.0, 0.0, 2.522188, 1.570792]], device='cuda:0')

```

## References

* [OPW Kinematics Paper](https://doi.org/10.1109/TRO.2017.2776312)
* [py_opw_kinematics Documentation](https://github.com/UM-ARM-Lab/py_opw_kinematics)
* [warp Documentation](https://github.com/NVIDIA/warp)
