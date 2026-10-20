# ToppraPlanner

`ToppraPlanner` time-parameterizes joint paths under velocity and acceleration
limits with the in-tree Warp implementation of
[TOPPRA](https://github.com/hungpham2511/toppra). CPU and CUDA use the same
calculation chain; the external `toppra` package is not required. For Cartesian goals, use
[MotionGenerator](../motion_generator.md) to generate joint waypoints through IK.

## Direct planning

The example assumes a registered `robot` in `sim`. Supply at least a start
and goal as batched tensors; the direct planner does not insert the current
robot state.

```python
from embodichain.lab.sim.motion.planners import (
    PlanState,
    ToppraPlanner,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    TrajectorySampleMethod,
)

planner = ToppraPlanner(
    ToppraPlannerCfg(robot_uid=robot.uid, sim_instance_id=sim.instance_id)
)
start_qpos = robot.get_qpos(name="arm")  # (B, DOF)
goal_qpos = start_qpos.clone()
goal_qpos[:, 0] += 0.1  # Choose a goal within this robot's limits.

result = planner.plan(
    [PlanState.from_qpos(start_qpos), PlanState.from_qpos(goal_qpos)],
    ToppraPlanOptions(
        control_part="arm",
        constraints={"velocity": 0.2, "acceleration": 0.5},
        sample_method=TrajectorySampleMethod.TIME,
        sample_interval=0.01,
    ),
)
assert result.is_all_success()
planner.close()
```

## Constraints and sampling

- Set `constraints` on `ToppraPlanOptions`; positive scalars apply to all
  controlled joints, vectors specify per-joint magnitudes, and `(DOF, 2)`
  arrays specify signed bounds containing zero. Explicit velocity requests
  are capped by the robot's runtime speed limits. `None` uses robot speed
  limits and the default planning acceleration budget.
- `control_part` selects the trajectory's joint order. A partial-joint path
  needs an explicit group when the mapping is ambiguous. Runtime zero-speed
  joints must remain constant at every waypoint, including points that
  would otherwise be removed by duplicate filtering.
- `TIME` uses `sample_interval` as the requested time spacing in seconds;
  samples include both endpoints, so actual spacing can be smaller.
- `QUANTITY` uses `sample_interval` as the output count, with at least two
  samples.

Results follow the shared
[PlanResult contract](../motion_generator.md#inputs-and-results). Planner
results use float32 on the robot's device. Internal spline fitting, interval
bounds, timing and sampling use float64. `grid_size` sets the minimum timing
grid size; segment alignment and path length can increase the actual size.
The interval bounds are conservative, so durations can differ from the
former external solver. Timing alone does not certify collision freedom.

## CPU execution and configuration migration

CPU inference automatically splits sufficiently large batches across a shared
pool of at most eight threads. Small requests and calls recording derivatives
run inline. CUDA numerical work remains on the input device. Planning does
not create a CPU process pool, and individual planners own no worker processes.
`close()` is an idempotent lifecycle hook; it does not close the shared CPU
thread executor.

Remove `backend`, `max_workers`, and `mp_context` from existing
`ToppraPlannerCfg` constructors and configuration dictionaries. These options
are no longer accepted. The robot selects the device, and Warp is the only
runtime implementation:

```python
cfg = ToppraPlannerCfg(
    robot_uid=robot.uid,
    sim_instance_id=sim.instance_id,
    grid_size=100,
)
```

The NumPy implementation is an offline numerical reference used by tests;
it cannot be selected as a planner backend.

## First-order gradients

Waypoint tensors and tensor velocity/acceleration requests retain their graphs
through `ToppraPlanOptions` copying and the timing solve on CPU or CUDA. For
example, with a valid joint goal in the direct-planning setup above:

```python
goal = goal_qpos.detach().requires_grad_()
result = planner.plan(
    [PlanState.from_qpos(start_qpos), PlanState.from_qpos(goal)],
    ToppraPlanOptions(
        control_part="arm",
        constraints={"velocity": 0.2, "acceleration": 0.5},
        sample_method=TrajectorySampleMethod.QUANTITY,
        sample_interval=101,
    ),
)
assert result.is_all_success()
result.duration.sum().backward()
assert goal.grad is not None
```

Gradients are local to fixed waypoint filtering, timing-grid topology, output
counts and active constraints. Higher-order derivatives are unsupported.
Failed rows contribute zero gradients; stationary rows differentiate only
their retained position. Robot speed-cap saturation stops gradients through
the capped request. Supply trainable boundary states through `PlanState.qpos`.

## Retiming an existing path

`retime_joint_paths()` exposes the same Warp timing solve for a `(B, N, DOF)`
tensor without constructing a planner. Provide explicit velocity and
acceleration limits; optional `path_lengths` exclude per-row tail padding
from spline fitting. Outputs preserve float32/float64 input precision, while
other input dtypes are promoted to float32. Recompute poses from the returned
positions because the helper resamples along the fitted spline.

## References

- [TOPPRA documentation](https://hungpham2511.github.io/toppra/index.html)
- {doc}`Planner API </api_reference/embodichain/embodichain.lab.sim.motion.planners>`
