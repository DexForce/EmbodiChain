# TrajectorySampleMethod

`TrajectorySampleMethod` names sampling modes. A backend supports only the
modes documented on its own page.

| Member | Meaning of `sample_interval` | Example |
|---|---|---|
| `TIME` | Requested time spacing in seconds; endpoint handling is backend-specific. | `0.01` seconds. |
| `QUANTITY` | Requested number of samples. | `100` points. |
| `DISTANCE` | Path-distance spacing, where supported. | A distance step in the path's units. |

[TOPPRA](toppra_planner.md) and [TrapezoidalPlanner](trapezoidal_planner.md)
support `TIME` and `QUANTITY`. The presence of `DISTANCE` in the enum does not
imply support in those planners. Sampling controls the output grid; physical
execution uses the simulator or environment's control clock, as described in
[playback](../motion_generator.md#execute-in-simulation).
