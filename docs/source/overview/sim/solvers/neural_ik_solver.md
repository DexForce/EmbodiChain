# NeuralIKSolver

````{admonition} Experimental
:class: warning

`NeuralIKSolver` is an **experimental** feature. The API, checkpoint format,
and default parameters may change without a deprecation cycle. It is currently
only validated on the **Franka Panda** robot.
````

`NeuralIKSolver` is a learning-based inverse kinematics (IK) solver that uses a
trained neural network policy to iteratively solve IK queries. It requires a
pre-trained checkpoint and supports batch processing.

## Key Features

* Iterative neural policy inference for IK solving
* Batch processing for multiple target poses simultaneously
* Multi-seed sampling: generate several random initial guesses and return the best solution
* Joint limit enforcement at every iteration
* PyTorch-based — supports both CPU and CUDA devices

## Configuration

The solver is configured using the `NeuralIKSolverCfg` class. Pre-trained
checkpoints are hosted on HuggingFace and can be downloaded with
`download_neural_ik_checkpoint()` (requires `HF_TOKEN` environment variable).

```python
from embodichain.data.assets.solver_assets import download_neural_ik_checkpoint
from embodichain.lab.sim.solvers.neural_ik_solver import NeuralIKSolverCfg

checkpoint_path = download_neural_ik_checkpoint()

cfg = NeuralIKSolverCfg(
    checkpoint_path=checkpoint_path,
    num_arm_joints=7,
    max_steps=30,
    action_scale=0.2,
    hidden_dims=[256, 256],
    pos_eps=0.01,
    rot_eps=0.1,
    num_samples=1,
)
```

## Main Methods

* `get_ik(self, target_xpos, qpos_seed=None, num_samples=None, **kwargs)`
  Solve IK for the given target end-effector pose(s).

  **Parameters:**
  + `target_xpos` (`torch.Tensor`): Target pose(s) as 4x4 matrix, shape `(4, 4)` or `(B, 4, 4)`.
  + `qpos_seed` (`torch.Tensor`, optional): Initial joint positions, shape `(dof,)` or `(B, dof)`.
  + `num_samples` (`int`, optional): Override `cfg.num_samples` for this call.
  + `return_all_solutions` (`bool`): If `True`, return all sampled solutions with shape `(B, num_samples, dof)`.

  **Returns:**
  + `Tuple[torch.Tensor, torch.Tensor]`:
    - Success flags, shape `(B,)`.
    - Joint positions, shape `(B, 1, dof)` or `(B, num_samples, dof)`.

  **Example:**

```python
  import torch
  success, ik_qpos = solver.get_ik(target_xpos=target_pose, qpos_seed=qpos_seed)
  print("Success:", success)
  print("IK solution:", ik_qpos)
```

