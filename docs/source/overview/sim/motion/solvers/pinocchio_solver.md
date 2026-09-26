# PinocchioSolver

`PinocchioSolver` solves single-target numerical IK with
[Pinocchio](https://github.com/stack-of-tasks/pinocchio). Its active solve path
iterates on pose error with damped Jacobian updates; the CasADi optimization
path is currently disabled.

## Configuration Example

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers.pinocchio_solver import PinocchioSolver
from embodichain.lab.sim.motion.solvers.pinocchio_solver import PinocchioSolverCfg

cfg = PinocchioSolverCfg(
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

solver = PinocchioSolver(cfg)
```

## Solver-specific behavior

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern and return order.

`get_ik()` currently solves one target per call: if given `(B, 4, 4)`, it
uses the first pose. Call it separately for each target. It returns success
first and joint positions second. A failed solve may return its last iterate;
check the success flag before using the result.

## References

* [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)
* [CasADi Documentation](https://web.casadi.org/)
