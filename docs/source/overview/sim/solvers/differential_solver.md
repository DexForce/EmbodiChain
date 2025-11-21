# DifferentialSolver

The `DifferentialSolver` is a differential inverse kinematics (IK) controller designed for robot manipulators. It computes joint-space commands to achieve desired end-effector positions or poses using various Jacobian-based methods.

## Key Features

* Supports multiple IK methods: pseudo-inverse (`pinv`), singular value decomposition (`svd`), transpose (`trans`), and damped least squares (`dls`)
* Configurable for position or pose control, with absolute or relative modes
* Efficient batch computation for multiple environments
* Flexible configuration via `DifferentialSolverCfg`

## Configuration Example

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.solvers.differential_solver import DifferentialSolver
from embodichain.lab.sim.solvers.differential_solver import DifferentialSolverCfg

cfg = DifferentialSolverCfg(
    urdf_path=get_data_path("UniversalRobots/UR5/UR5.urdf"),
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    end_link_name="ee_link",
    root_link_name="base_link",
    command_type="pose",
    use_relative_mode=False,
    ik_method="pinv",
    ik_params={"k_val": 1.0}
)

solver = DifferentialSolver(cfg)
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
  # tensor([[[ 0.0,     -1.0,      0.0,     -0.722600],
  #          [ 0.0,      0.0,     -1.0,     -0.191450],
  #          [ 1.0,      0.0,      0.0,      0.079159],
  #          [ 0.0,      0.0,      0.0,      1.0     ]]])
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
  xpos = torch.tensor([[[ 0.0,     -1.0,      0.0,     -0.722600],
                        [ 0.0,      0.0,     -1.0,     -0.191450],
                        [ 1.0,      0.0,      0.0,      0.079159],
                        [ 0.0,      0.0,      0.0,      1.0     ]]])
  qpos_seed = torch.zeros((1, 6))
  qpos_sol, info = solver.get_ik(target_xpos=xpos)
  print("IK solution:", qpos_sol)
  print("Convergence info:", info)
  # IK solution: tensor([True])
  # Convergence info: tensor([[0.0, -0.231429, 0.353367, 0.893100, 0.0, 0.555758]])
```

> **Tip:**
> - `get_fk` is for forward kinematics (joint to end-effector), `get_ik` is for inverse kinematics (end-effector to joint).
> - For batch computation, the first dimension of `qpos` and `target_xpos` is the batch size.

## IK Methods Supported

* **pinv**: Jacobian pseudo-inverse
* **svd**: Singular value decomposition
* **trans**: Jacobian transpose
* **dls**: Damped least squares

## References

* [Isaac Sim Library](https://github.com/isaac-sim/IsaacLab)
