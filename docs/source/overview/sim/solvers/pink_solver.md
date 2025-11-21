# PinkSolver

`PinkSolver` is an advanced inverse kinematics (IK) solver for robot manipulators, built on [Pinocchio](https://github.com/stack-of-tasks/pinocchio) and [Pink](https://github.com/stephane-caron/pink). It supports flexible task definitions, robust optimization, and null space posture control.

## Key Features

- Supports both position-only and full pose (position + orientation) constraints
- Configurable convergence tolerance (`pos_eps`, `rot_eps`), damping, and iteration limits
- Handles joint limits and safety checks during optimization
- Allows variable and fixed task definitions for flexible control (see `FrameTask`, `NullSpacePostureTask`)
- Integrates with Pinocchio robot models and Pink task framework
- Supports multiple solver backends: `osqp`, `clarabel`, `ecos`, `proxqp`, `scs`, `daqp`
- Provides joint mapping between simulation and solver for flexible robot integration
- Null space posture task for redundancy resolution and secondary objectives
- Torch and numpy compatible for seamless integration in simulation pipelines

## Configuration Example

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.solvers.pink_solver import PinkSolver
from embodichain.lab.sim.solvers.pink_solver import PinkSolverCfg

cfg = PinkSolverCfg(
    urdf_path=get_data_path("UniversalRobots/UR5/UR5.urdf"),
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    end_link_name="ee_link",
    root_link_name="base_link",
    max_iterations=500,
    pos_eps=1e-4,
    rot_eps=1e-4,
    dt=0.05,
    damp=1e-10,
    is_only_position_constraint=False,
    fail_on_joint_limit_violation=True,
    solver_type="osqp",
    variable_input_tasks=None,
    fixed_input_tasks=None,
)

solver = PinkSolver(cfg)
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


## References

- [Pinocchio Library](https://github.com/stack-of-tasks/pinocchio)
- [Pink Library](https://github.com/stephane-caron/pink)
- [Null Space Posture Task](https://github.com/stephane-caron/pink#null-space-posture-task)
