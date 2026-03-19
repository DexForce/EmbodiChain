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
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanner, ToppraPlannerCfg

# Configuration - constraints are now specified in ToppraPlanOptions, not here
cfg = ToppraPlannerCfg(
    robot_uid="UR5"
)
planner = ToppraPlanner(cfg=cfg)
```

### Planning

```python
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod, PlanState
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions

# Create options with constraints and sampling parameters
options = ToppraPlanOptions(
    constraints={
        "velocity": 0.2,          # Joint velocity limit (rad/s) - can also be a list per joint
        "acceleration": 0.5,      # Joint acceleration limit (rad/s²) - can also be a list per joint
    },
    sample_method=TrajectorySampleMethod.TIME,  # Or TrajectorySampleMethod.QUANTITY
    sample_interval=0.01  # Time interval in seconds (if TIME) or number of samples (if QUANTITY)
)

# Plan trajectory - only target_states needed now (current_state is handled internally)
result = planner.plan(
    target_states=[
        PlanState(qpos=[1, 1, 1, 1, 1, 1])
    ],
    options=options
)
```

- `result.positions`, `result.velocities`, `result.accelerations` are arrays of sampled trajectory points.
- `result.dt` is the array of time intervals between each point.
- `result.duration` is the total trajectory time.
- `result.success` indicates whether planning succeeded.

## Notes

- The planner requires the `toppra` library (`pip install toppra==0.6.3`).
- For dense waypoints, the default spline interpolation is used. For sparse waypoints, you may need to adjust the interpolation method.
- The number of grid points (`gridpt_min_nb_points`) is important for accurate acceleration constraint handling.
- Constraints can be specified as a single float (applied to all joints) or as a list of values for each joint.
- The `sample_method` can be `TrajectorySampleMethod.TIME` for uniform time intervals or `TrajectorySampleMethod.QUANTITY` for a fixed number of samples.

## References

- [TOPPRA Documentation](https://hungpham2511.github.io/toppra/index.html)
