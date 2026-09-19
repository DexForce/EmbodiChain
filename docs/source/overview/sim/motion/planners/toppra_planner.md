# ToppraPlanner

`ToppraPlanner` time-parameterizes a joint path with
[TOPPRA](https://github.com/hungpham2511/toppra) under velocity and acceleration
limits. Install `toppra==0.6.3` before use. For Cartesian goals, use
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
        constraints={"velocity": 0.2, "acceleration": 0.5},
        sample_method=TrajectorySampleMethod.TIME,
        sample_interval=0.01,
    ),
)
assert result.is_all_success()
planner.close()
```

## Constraints and sampling

- Set `constraints` on `ToppraPlanOptions`; a scalar applies to all controlled
  joints and a list specifies per-joint limits.
- `TIME` uses `sample_interval` as the requested time spacing in seconds;
  samples include both endpoints, so actual spacing can be smaller.
- `QUANTITY` uses `sample_interval` as the output count, with at least two
  samples.

Results follow the shared
[PlanResult contract](../motion_generator.md#inputs-and-results). Planning
runs on CPU; batched workloads can use the configured worker pool. Call
`close()` when a directly owned planner is no longer needed.

## References

- [TOPPRA documentation](https://hungpham2511.github.io/toppra/index.html)
- {doc}`Planner API </api_reference/embodichain/embodichain.lab.sim.motion.planners>`
