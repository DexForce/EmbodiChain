# PinkSolver

`PinkSolver` uses [Pink](https://github.com/stephane-caron/pink) and
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) for task-based IK,
including frame targets and null-space posture objectives.

## Configuration Example

```python
from embodichain.data import get_data_path
from embodichain.lab.sim.motion.solvers.pink_solver import PinkSolverCfg
from embodichain.lab.sim.motion.solvers.null_space_posture_task import NullSpacePostureTask

posture_task = NullSpacePostureTask(
    cost=1e-3,
    controlled_frames=["ee_link"],
)

cfg = PinkSolverCfg(
    urdf_path=get_data_path("UniversalRobots/UR5/UR5.urdf"),
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
    fixed_input_tasks=[posture_task],
)

solver = cfg.init_solver(device="cpu")
solver.update_null_space_joint_targets([0.0] * 6)
```


## Solver-specific behavior

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern and return order.

`get_ik()` returns success flags `(B,)` and joint solutions `(B, 1, DOF)`.
Failed targets retain their seeds. `return_all_solutions=True` still returns
one locally optimal solution per target. Seeds may be unbatched, broadcast
from `(1, DOF)`, or supplied per target.

Use `fixed_input_tasks` for posture objectives and
`update_null_space_joint_targets()` to change their target posture.

## References

- [Pinocchio Library](https://github.com/stack-of-tasks/pinocchio)
- [Pink Library](https://github.com/stephane-caron/pink)
- [Pink documentation](https://stephane-caron.github.io/pink/)
