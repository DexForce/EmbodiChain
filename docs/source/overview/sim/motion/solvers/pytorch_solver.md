# PytorchSolver

`PytorchSolver` solves batched numerical IK using
[pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics). It supports
position-only or full-pose targets, multiple seeds, joint limits, and GPU computation.

## Configuration

The solver is configured using the `PytorchSolverCfg` class, which allows detailed control over solver parameters and robot model setup.

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers.pytorch_solver import PytorchSolver
from embodichain.lab.sim.motion.solvers.pytorch_solver import PytorchSolverCfg
import torch

cfg = PytorchSolverCfg(
    urdf_path=get_data_path("UniversalRobots/UR5/UR5.urdf"),
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

## Solver-specific behavior

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern and return order.

`get_ik()` returns success flags `(B,)` and, on success, the selected solution
`(B, 1, DOF)`. An entirely failed batch returns joint values with shape
`(B, DOF)`. With `return_all_solutions=True`, the second dimension contains
the sampled solutions. Tune `num_samples` and convergence settings when the
initial seed is insufficient.

## References

* [pytorch_kinematics Documentation](https://github.com/UM-ARM-Lab/pytorch_kinematics)
