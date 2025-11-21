# PytorchSolver

`PytorchSolver` is a high-performance inverse kinematics (IK) solver for robot manipulators, leveraging [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) for efficient computation and seamless integration with PyTorch workflows. It supports both position and orientation constraints, joint limits, batch sampling, and GPU acceleration, making it suitable for real-time and large-scale applications.

## Key Features

* Full support for position-only or full pose (position + orientation) constraints
* Configurable convergence tolerance, damping, and iteration limits
* Enforces joint limits during optimization
* Batch sampling for robust IK seed initialization and solution diversity
* Efficient batched computation for multiple target poses
* PyTorch integration for GPU acceleration and tensor-based workflows
* Flexible configuration via `PytorchSolverCfg` class

## Configuration

The solver is configured using the `PytorchSolverCfg` class, which allows detailed control over solver parameters and robot model setup.

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.solvers.pytorch_solver import PytorchSolver
from embodichain.lab.sim.solvers.pytorch_solver import PytorchSolverCfg
import torch

cfg = PytorchSolverCfg(
    urdf_path=get_data_path("UniversalRobots/UR5/UR5.urdf"),
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    end_link_name="ee_link",
    root_link_name="base_link",
    max_iterations=1000,
    pos_eps=1e-4,
    rot_eps=1e-4,
    dt=0.05,
    damp=1e-6,
    num_samples=30,
    is_only_position_constraint=False,
)

solver = PytorchSolver(cfg)
```

### Dynamic Parameter Adjustment

Solver parameters can be updated at runtime using `set_iteration_params` :

```python
solver.set_iteration_params(
    pos_eps=1e-5,
    rot_eps=1e-5,
    max_iterations=500,
    num_samples=50,
    damp=1e-7,
)
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
  # IK solution: tensor([True], device='cuda:0')
  # Convergence info: tensor([[0.0, -0.244575, 0.373442, 0.853886, 0.0, 0.588007]], device='cuda:0')

```

## References

* [pytorch_kinematics Documentation](https://github.com/UM-ARM-Lab/pytorch_kinematics)
