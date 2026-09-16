# MotionGenerator

`MotionGenerator` turns batched joint or end-effector goals into timed
trajectories. Choose a [planner backend](planners/index.rst), then select one
of two strategies for each request:

| `MotionGenOptions.strategy` | Behavior | Required inputs |
|---|---|---|
| `"motion_gen"` (default) | Run the configured planner. | Targets and backend options; provide the start state explicitly in the examples below. |
| `"ik_interp"` | Solve EEF waypoints sequentially with IK, or interpolate joint waypoints directly. | `start_qpos`, `sample_count`, and `interpolation_dt`. |

The strategy selects who generates the timed trajectory. `motion_gen` can
also interpolate a path and solve IK before planning; it still invokes the
backend. Unsupported target types raise an error instead of switching strategy.

## Plan a trajectory

These examples assume `sim` already contains a `robot` with an `"arm"` control
part and a configured IK solver. See the
{doc}`simulation tutorial </tutorial/motion_gen>` for scene setup.
`B` is the number of robot instances and `DOF` is the controlled joint count.

### Initialize a backend

```python
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import (
    PlanState,
    ToppraPlannerCfg,
    ToppraPlanOptions,
    TrajectorySampleMethod,
)

motion_gen = MotionGenerator(
    MotionGenCfg(
        planner_cfg=ToppraPlannerCfg(
            robot_uid=robot.uid,
            sim_instance_id=sim.instance_id,
        ),
    )
)
plan_opts = ToppraPlanOptions(
    constraints={"velocity": 0.2, "acceleration": 0.5},
    sample_method=TrajectorySampleMethod.TIME,
    sample_interval=0.01,
)
start_qpos = robot.get_qpos(name="arm")  # (B, DOF)
```

### Joint goals

Supply both the start and goal when calling TOPPRA without pre-interpolation.
Choose a goal within the robot's joint limits and reachable workspace.

```python
goal_qpos = start_qpos.clone()
goal_qpos[:, 0] += 0.1

result = motion_gen.generate(
    [PlanState.from_qpos(start_qpos), PlanState.from_qpos(goal_qpos)],
    MotionGenOptions(
        strategy="motion_gen",
        start_qpos=start_qpos,
        control_part="arm",
        plan_opts=plan_opts,
    ),
)
assert result.is_all_success()
```

### Cartesian goals

For a joint-only backend such as TOPPRA, enable pre-interpolation to convert
Cartesian waypoints through the robot's IK solver. Native Cartesian backends
receive the original targets directly.

Pre-interpolation solves the sampled poses in order, using each solution as
the next IK seed. A failed required pose fails that environment's trajectory;
the generator never drops it to report a shorter path as successful. The
backend then applies its own path and timing behavior, so these input samples
do not guarantee an exact Cartesian line in the final output.

```python
goal_pose = robot.compute_fk(qpos=start_qpos, name="arm", to_matrix=True)
goal_pose = goal_pose.clone()  # (B, 4, 4), arena-local TCP pose
goal_pose[:, 2, 3] += 0.02

result = motion_gen.generate(
    [PlanState.from_xpos(goal_pose)],
    MotionGenOptions(
        strategy="motion_gen",
        start_qpos=start_qpos,
        control_part="arm",
        plan_opts=plan_opts,
        is_interpolate=True,
        is_linear=True,
        interpolate_position_step=0.002,
    ),
)
```

### Deterministic interpolation

Use explicit sampling and timing to generate a path without invoking the
configured planner. EEF targets use the previous IK solution as the next seed;
joint targets are interpolated directly.

```python
result = motion_gen.generate(
    [PlanState.from_qpos(goal_qpos)],
    MotionGenOptions(
        strategy="ik_interp",
        start_qpos=start_qpos,
        control_part="arm",
        sample_count=50,
        interpolation_dt=0.02,
    ),
)
```

Interpolation alone does not enforce velocity or acceleration limits or check
collisions. Passing `plan_opts`, `velocity_limit`, or `acceleration_limit` with
`ik_interp` raises an error. Use a suitable planner when those constraints
are required.

To preserve a supplied Cartesian path, use `ik_interp` with
`preserve_cartesian_samples=True` and provide exactly `sample_count - 1` EEF
poses after the start. Each pose is solved with IK and retained at the explicit
`interpolation_dt`; no joint-space resampling follows. `is_linear=True` in
this strategy requires those explicit samples. The caller defines their
geometry, including whether they form a line.

`motion_gen` with `preserve_cartesian_samples=True` is unsupported and raises
an error: the current backend contract cannot guarantee exact Cartesian
samples while also applying backend timing and path constraints.

## Inputs and results

`PlanState.from_qpos()` takes `(B, DOF)` tensors and
`PlanState.from_xpos()` takes `(B, 4, 4)` tensors. For a single environment,
`PlanState.single()` also accepts unbatched inputs. All waypoints must use the
same batch size as the robot.

`generate()` returns a `PlanResult`:

| Field | Shape | Meaning |
|---|---|---|
| `success` | `(B,)` | Per-environment planning status; `is_all_success()` checks every row. |
| `positions` | `(B, N, DOF)` | Joint position samples; `generate()` requires a trajectory from the backend. |
| `velocities`, `accelerations` | `(B, N, DOF)` when present | Joint derivatives. Missing velocities are derived from the final timed positions. |
| `dt` | `(B, N)` | Arrival intervals, required whenever positions are present. |
| `duration` | `(B,)` | Derived as `dt.sum(dim=1)`. |
| `constraint_report` | mapping or `None` | Backend diagnostics, retained only while the reported trajectory is unchanged. |

When `sample_count` changes a backend's output grid, the generator resamples
in time, preserves each row's duration, recomputes velocities, and invalidates
accelerations and constraint diagnostics. If the backend supports joint
trajectory validation, the final samples are checked again against the resolved
collision scene, and a failed check fails that environment. This sampled check
does not certify continuous motion between samples or enforce derivative limits.
Backends may preserve native samples;
see [TrapezoidalPlanner](planners/trapezoidal_planner.md) and
[cuRobo](planners/curobo_planner.md) for their controls.

Failed rows hold `start_qpos` with zero velocity when it is supplied, except
for backends that explicitly preserve failed rollout positions, such as
NeuralPlanner. Always check `success` before execution.

## Execute in simulation

Planning returns a trajectory; playback applies it on the simulator's fixed
control clock. With `sim` in explicit-update mode:

```python
from embodichain.lab.sim.motion.execution import (
    JointTrajectoryPlaybackCfg,
    play_joint_trajectory,
)

assert result.is_all_success()
play_joint_trajectory(
    sim,
    robot,
    positions=result.positions,
    dt=result.dt,
    joint_ids=robot.get_joint_ids("arm"),
    cfg=JointTrajectoryPlaybackCfg(joint_command_mode="position_velocity"),
)
```

Playback defaults to one command per physics step. An explicit `control_dt`
must be an integer multiple of `physics_dt`; playback retimes the trajectory
and recomputes velocity targets on that grid without changing the physics
period. Position-only playback is the default; the example opts into velocity
targets, with zero velocity at the first, terminal, and padded hold samples.

Atomic Skills and Gym experts own their execution cadence and holds. See
{doc}`Atomic actions </overview/sim/atomic_actions/index>` and the
{doc}`expert data tutorial </tutorial/data_generation>` for those
workflows. Teleporting through samples, as the cuRobo visualization demo does,
shows a path but does not measure physical tracking.

For a physical position-only versus velocity-target comparison, run
`examples/sim/motion/trajectory_velocity_tracking.py`. For complete options and
helper methods such as `estimate_trajectory_sample_count()`, see the
{doc}`MotionGenerator API </api_reference/embodichain/embodichain.lab.sim.motion.motion_generator>`.
