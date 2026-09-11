# TrapezoidalPlanner

`TrapezoidalPlanner` is a lightweight, natively batched joint-trajectory
planner. It time-parameterizes piecewise-linear joint paths with either a
trapezoidal velocity profile or a jerk-limited seven-phase Double-S profile.
The Torch implementation is the reference and CPU fallback; the NVIDIA Warp
backend parallelizes profile construction, phase evaluation, and batched joint
trajectory composition for CUDA workloads.

Use this planner when you need deterministic joint-space timing, explicit
velocity and acceleration outputs, and no collision-world planning dependency.
For collision-aware Cartesian planning, use cuRobo instead.

## Capabilities

* Batched input and output with a leading environment dimension.
* Scalar or per-joint velocity, acceleration, and jerk limits.
* Synchronized multi-joint motion for every path segment.
* Triangular fallback when a short move cannot reach the requested trapezoidal
  cruise velocity.
* Jerk-limited seven-phase Double-S trajectories without display filtering.
* Fixed-count and approximately fixed-time sampling.
* Optional minimum-duration scaling without changing the path.
* Redundant collinear-waypoint compression and optional quintic corner blends.
* Explicit `torch`, `warp`, and `auto` backend selection.
* Native `positions`, `velocities`, `accelerations`, `dt`, and constraint
  diagnostics in `PlanResult`.

The planner accepts `JOINT_MOVE` waypoints. Cartesian straight-line planning in
the accompanying tutorial is a separate pipeline: it applies a scalar time law
in Cartesian space first, solves the complete pose path through continuous
batched IK, and derives joint velocity and acceleration using differential
kinematics. This ordering avoids distorting the requested end-effector path by
applying joint-space timing after independently sampled IK.

## MotionGenerator integration

Configure the planner through the common `MotionGenerator` facade:

```python
import torch

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import (
    PlanState,
    TrapezoidalPlannerCfg,
    TrapezoidalPlanOptions,
)

generator = MotionGenerator(
    MotionGenCfg(
        planner_cfg=TrapezoidalPlannerCfg(robot_uid=robot.uid),
    )
)

# PlanState tensors are batched: (B, DOF).
start_qpos = robot.get_qpos(name="arm")
goal_qpos = torch.zeros_like(start_qpos)

result = generator.generate(
    [PlanState.from_qpos(goal_qpos)],
    MotionGenOptions(
        start_qpos=start_qpos,
        control_part="arm",
        plan_opts=TrapezoidalPlanOptions(
            profile="double_s",
            constraints={
                "velocity": 0.5,
                "acceleration": 1.0,
                "jerk": 3.0,
            },
            sample_interval=200,
            stop_at_waypoints=False,
            backend="auto",
        ),
    ),
)
```

`TrapezoidalPlanner` owns sparse joint-waypoint timing. `MotionGenerator`
prepends `start_qpos` when the first requested waypoint is not already the
start, and does not run generic joint pre-interpolation. It also preserves the
planner's native sample grid and its analytical velocity and acceleration
outputs. `MotionGenOptions.sample_count` therefore does not replace an explicit
`TrapezoidalPlanOptions.sample_interval`.

Every successful result with positions contains:

| Field | Shape | Meaning |
|---|---|---|
| `positions` | `(B, N, DOF)` | Joint position samples. |
| `velocities` | `(B, N, DOF)` | Native joint velocity samples. |
| `accelerations` | `(B, N, DOF)` | Native joint acceleration samples. |
| `dt` | `(B, N)` | Arrival intervals; the first interval is zero. |
| `duration` | `(B,)` | Derived as `dt.sum(dim=1)`. |
| `constraint_report` | mapping | Peaks, per-joint utilization, limits, and status. |

## Profiles and constraints

Set `profile="trapezoidal"` for acceleration-limited trapezoidal timing. The
planner automatically uses a triangular profile for moves too short to reach a
constant-velocity phase. Acceleration changes discontinuously at phase
boundaries, so this profile does not enforce the configured jerk limit.

Set `profile="double_s"` for a seven-phase profile constrained by velocity,
acceleration, and jerk. The `constraints` mapping accepts a positive scalar or
one value per joint for each derivative. All joints within a segment are
synchronized to the same segment duration.

`minimum_duration` stretches trajectories that would otherwise finish sooner.
It does not shorten slower trajectories or relax derivative limits.

## Waypoint behavior

By default, `stop_at_waypoints=True`, so every supplied waypoint is a rest
point. Set it to `False` when intermediate samples describe a continuous path:

* duplicate and same-direction collinear points are compressed;
* real corners remain in the path;
* `collinearity_tolerance` controls the angular comparison;
* positive `blend_tolerance` inserts quintic corner blends and requires
  `stop_at_waypoints=False`.

If the caller passes `start_qpos` and also includes exactly the same first
waypoint, `MotionGenerator` avoids inserting a duplicate zero-length segment.

## Sampling and backends

`TrajectorySampleMethod.QUANTITY` interprets `sample_interval` as the requested
integer output count. It must provide at least one sample for every retained
waypoint. `TrajectorySampleMethod.TIME` interprets it as a positive time step in
seconds. In a batch with different durations, shorter rows hold their exact
final position with zero velocity, zero acceleration, and zero arrival
intervals after completion.

Backend selection follows these rules:

* `backend="torch"` uses the reference implementation.
* `backend="warp"` requests the Warp implementation and requires float32
  tensors plus an available Warp runtime; it can be exercised on Warp CPU or
  CUDA devices.
* `backend="auto"` selects Warp for CUDA float32 inputs and Torch for CPU or
  float64 inputs.

No new runtime dependency is needed: Warp is already part of EmbodiChain and is
only required when its backend is selected.

## Atomic Skill integration

`MoveJoints` reaches the same planner through `AtomicActionEngine`:

```python
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    JointPositionGoal,
    MotionPolicy,
)

engine = AtomicActionEngine(generator)
binding = engine.bind_control_parts(
    "move_joints",
    {"primary": {"motion": "arm"}},
)
invocation = ActionInvocation(
    skill_id="move_joints",
    goal=JointPositionGoal(target=goal_qpos),
    binding=binding,
    motion_policy=MotionPolicy(
        strategy="motion_gen",
        plan_opts=TrapezoidalPlanOptions(
            profile="double_s",
            constraints={
                "velocity": 0.5,
                "acceleration": 1.0,
                "jerk": 3.0,
            },
            sample_interval=100,
            backend="auto",
        ),
    ),
)
plan = engine.plan(invocation, latest_context)
```

The planner's timed trajectory is retained at the `MotionGenerator` boundary.
Atomic execution may retime it to the environment control grid and derives the
velocity targets for that executed grid; terminal and stationary holds command
zero velocity.

## Tutorials and benchmark

Run the full batched simulation tutorial for joint planning, Cartesian path IK,
timed replay, and position/velocity/acceleration diagnostics:

```bash
python scripts/tutorials/sim/planner/trapezoidal_planner.py \
    --path joint --profile both --backend auto
```

The minimal scalar-profile example is
`scripts/tutorials/sim/planner/trapezoidal_profile.py`. Reproducible Torch/Warp
measurements are provided by
`scripts/benchmark/motion_generation/trapezoidal_planner.py`.

See the [MotionGenerator guide](../motion_generator.md) for the shared planning
contract and the
{doc}`API reference </api_reference/embodichain/embodichain.lab.sim.motion.planners>`
for complete class and option documentation.
