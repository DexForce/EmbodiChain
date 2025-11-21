# ToppraPlanner

`ToppraPlanner` is a trajectory planner based on the [TOPPRA](https://toppra.readthedocs.io/) (Time-Optimal Path Parameterization via Reachability Analysis) library. It generates time-optimal joint trajectories under velocity and acceleration constraints.

## Features

- **Time-optimal trajectory generation**: Computes the fastest possible trajectory between waypoints, given joint velocity and acceleration limits.
- **Flexible sampling**: Supports sampling by time interval or by number of points.
- **Constraint handling**: Automatically formats velocity and acceleration constraints for the TOPPRA solver.
- **Dense and sparse waypoints**: Supports both dense and sparse waypoint interpolation.

## Usage

### Initialization

```python
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanner
planner = ToppraPlanner(
    dofs=6,
    max_constraints={
        "velocity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "acceleration": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    }
)
```

### Planning

```python
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanner
success, positions, velocities, accelerations, times, duration = planner.plan(
    current_state={
        "position": [0, 0, 0, 0, 0, 0],
        "velocity": [0, 0, 0, 0, 0, 0],
        "acceleration": [0, 0, 0, 0, 0, 0]
    },
    target_states=[
        {"position": [1, 1, 1, 1, 1, 1]}
    ],
    sample_method=TrajectorySampleMethod.TIME,
    sample_interval=0.01
)
```

- `positions`, `velocities`, `accelerations` are arrays of sampled trajectory points.
- `times` is the array of time stamps.
- `duration` is the total trajectory time.

## Notes

- The planner requires the `toppra` library (`pip install toppra==0.6.3`).
- For dense waypoints, the default spline interpolation is used. For sparse waypoints, you may need to adjust the interpolation method.
- The number of grid points (`gridpt_min_nb_points`) is important for accurate acceleration constraint handling.

## References

- [TOPPRA Documentation](https://hungpham2511.github.io/toppra/index.html)
