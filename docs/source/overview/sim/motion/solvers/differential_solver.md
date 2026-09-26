# DifferentialSolver

The `DifferentialSolver` is a differential inverse kinematics (IK) controller designed for robot manipulators. It computes joint-space commands to achieve desired end-effector positions or poses using various Jacobian-based methods.

## Configuration Example

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers.differential_solver import DifferentialSolver
from embodichain.lab.sim.motion.solvers.differential_solver import DifferentialSolverCfg

cfg = DifferentialSolverCfg(
    urdf_path=get_data_path("UniversalRobots/UR5/UR5.urdf"),
    end_link_name="ee_link",
    root_link_name="base_link",
    command_type="pose",
    use_relative_mode=False,
    ik_method="pinv",
    ik_params={"k_val": 1.0}
)

solver = DifferentialSolver(cfg)
```

## Solver-specific behavior

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern and return order.

Select `ik_method` from `pinv` (pseudo-inverse), `svd` (singular value
decomposition), `trans` (Jacobian transpose), or `dls` (damped least squares).
`get_ik()` returns success flags `(B,)` and joint positions `(B, DOF)`;
use `max_iterations=1` for a single differential step.
`return_all_solutions=True` still returns one solution per target.

## References

* [Isaac Sim Library](https://github.com/isaac-sim/IsaacLab)
