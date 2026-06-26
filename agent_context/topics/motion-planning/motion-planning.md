# Motion Planning

## Entry Points

| What | Path |
|---|---|
| Planner registry | `embodichain/lab/sim/planners/__init__.py` |
| Base planner class & config | `embodichain/lab/sim/planners/base_planner.py` → `BasePlanner`, `BasePlannerCfg`, `PlanOptions` |
| TOPPRA planner | `embodichain/lab/sim/planners/toppra_planner.py` → `ToppraPlanner`, `ToppraPlannerCfg`, `ToppraPlanOptions` |
| Neural planner | `embodichain/lab/sim/planners/neural_planner.py` → `NeuralPlanner`, `NeuralPlannerCfg`, `NeuralPlanOptions` |
| Planner assets | `embodichain/data/assets/planner_assets.py` → `download_neural_planner_checkpoint()` |
| Motion generator | `embodichain/lab/sim/planners/motion_generator.py` → `MotionGenerator`, `MotionGenCfg`, `MotionGenOptions` |
| Planner utilities & data types | `embodichain/lab/sim/planners/utils.py` → `PlanState`, `PlanResult`, `MoveType`, `MovePart`, `TrajectorySampleMethod` |

## Overview

The planning stack has two layers:
1. **BasePlanner** — low-level trajectory planner that takes a list of `PlanState` waypoints and produces a `PlanResult` with joint trajectories.
2. **MotionGenerator** — high-level wrapper that composes a planner with optional interpolation, IK resolution, and multi-part coordination.

All planners resolve their robot at init via `SimulationManager.get_instance().get_robot(cfg.robot_uid)`.

## Planner Hierarchy

```
BasePlanner (ABC)
  ├─ ToppraPlanner        Time-optimal path parameterization
  └─ NeuralPlanner (experimental)   APG waypoint rollout

MotionGenerator           Wraps any BasePlanner; adds interpolation and multi-part support
```

Config hierarchy:
```
BasePlannerCfg            robot_uid (MISSING), planner_type
  ├─ ToppraPlannerCfg     planner_type = "toppra"
  └─ NeuralPlannerCfg     planner_type = "neural", checkpoint_path (MISSING)

MotionGenCfg              planner_cfg (MISSING — must be a BasePlannerCfg subclass)

PlanOptions               (empty base)
  ├─ ToppraPlanOptions    constraints, sample_method, sample_interval
  └─ NeuralPlanOptions    control_part, start_qpos, max_steps

MotionGenOptions          start_qpos, control_part, plan_opts, is_interpolate,
                          interpolate_nums, is_linear, interpolate_position_step,
                          interpolate_angle_step
```

## Available Planners

### ToppraPlanner

Time-optimal path parameterization using the [toppra](https://github.com/hungpham2511/toppra) library.

- **Dependency**: `pip install toppra==0.6.3` (import-time error if missing).
- **Single-instance only**: raises `NotImplementedError` if `robot.num_instances > 1`.
- **Method**: `plan(target_states, options=ToppraPlanOptions()) → PlanResult`

`ToppraPlanOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `constraints` | `dict` | `{"velocity": 0.2, "acceleration": 0.5}` | Per-joint or scalar limits |
| `sample_method` | `TrajectorySampleMethod` | `QUANTITY` | `TIME`, `QUANTITY`, or `DISTANCE` |
| `sample_interval` | `float \| int` | `0.01` | Time interval (seconds) or sample count depending on method |

### NeuralPlanner (experimental)

Learning-based EEF waypoint planner. Franka Panda only.

- Checkpoint: `download_neural_planner_checkpoint()` from HuggingFace (gated, needs `HF_TOKEN`)
- Use via `MotionGenerator` with `planner_type="neural"` and `plan_opts=NeuralPlanOptions(...)`
- Input: `EEF_MOVE` `PlanState` list with 4×4 `xpos`
- Key cfg: `checkpoint_path` (from download), `control_part`

### MotionGenerator

Unified interface for trajectory planning with optional pre-interpolation.

- Wraps a `BasePlanner` instance (resolved from `planner_cfg.planner_type`).
- Supported planner types: `{"toppra": (ToppraPlanner, ToppraPlannerCfg), "neural": (NeuralPlanner, NeuralPlannerCfg)}`.
- `MotionGenCfg.planner_cfg` is **MISSING** — must be provided.

`MotionGenOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `start_qpos` | `torch.Tensor \| None` | `None` | Override starting joint config; `None` = use current robot state |
| `control_part` | `str \| None` | `None` | Robot control part name (must match `RobotCfg.control_parts` key) |
| `plan_opts` | `PlanOptions \| None` | `None` | Passed to the underlying planner |
| `is_interpolate` | `bool` | `False` | Pre-interpolate waypoints before planning |
| `interpolate_nums` | `int \| list[int]` | `10` | Points per segment (scalar or per-segment list) |
| `is_linear` | `bool` | `False` | `True` = Cartesian linear interpolation; `False` = joint-space |
| `interpolate_position_step` | `float` | `0.002` | Cartesian step size (meters) or joint step size (radians) |
| `interpolate_angle_step` | `float` | `π/90` | Angular step in joint space (radians); only if `is_linear=False` |

## Planner Interface

### PlanState (input)

Describes one waypoint or action:

| Field | Type | Notes |
|---|---|---|
| `move_type` | `MoveType` | `TOOL`, `EEF_MOVE`, `JOINT_MOVE`, `SYNC`, `PAUSE` |
| `move_part` | `MovePart` | `LEFT`, `RIGHT`, `BOTH`, `TORSO`, `ALL` |
| `xpos` | `torch.Tensor \| None` | 4×4 target TCP pose (for `EEF_MOVE`) |
| `qpos` | `torch.Tensor \| None` | Target joint angles `(DOF,)` (for `JOINT_MOVE`) |
| `is_open` | `bool` | Tool open/close (for `TOOL`) |
| `is_world_coordinate` | `bool` | `True` = world frame; `False` = relative |
| `pause_seconds` | `float` | Duration for `PAUSE` move type |

### PlanResult (output)

| Field | Type | Notes |
|---|---|---|
| `success` | `bool \| torch.Tensor` | Whether planning succeeded |
| `xpos_list` | `torch.Tensor \| None` | EEF poses `(N, 4, 4)` |
| `positions` | `torch.Tensor \| None` | Joint positions `(N, DOF)` |
| `velocities` | `torch.Tensor \| None` | Joint velocities `(N, DOF)` |
| `accelerations` | `torch.Tensor \| None` | Joint accelerations `(N, DOF)` |
| `dt` | `torch.Tensor \| None` | Per-step time durations `(N,)` |
| `duration` | `float \| torch.Tensor` | Total trajectory time (seconds) |

### MoveType enum

| Value | Meaning |
|---|---|
| `TOOL` | Tool open/close command |
| `EEF_MOVE` | End-effector Cartesian move (IK + trajectory) |
| `JOINT_MOVE` | Joint-space move (trajectory planning only) |
| `SYNC` | Synchronized dual-arm movement |
| `PAUSE` | Pause for `pause_seconds` |

### MovePart enum

| Value | Meaning |
|---|---|
| `LEFT` | Left arm/EEF |
| `RIGHT` | Right arm/EEF |
| `BOTH` | Both arms/EEFs |
| `TORSO` | Torso (humanoid) |
| `ALL` | All joints |

## Configuration

### Registering a new planner

1. Create a `BasePlanner` subclass with a `plan()` method decorated with `@validate_plan_options`.
2. Create a `BasePlannerCfg` subclass with a unique `planner_type` string.
3. Optionally create a `PlanOptions` subclass for planner-specific options.
4. Register in `MotionGenerator._support_planner_dict`:
   ```python
   _support_planner_dict = {
       "toppra": (ToppraPlanner, ToppraPlannerCfg),
       "neural": (NeuralPlanner, NeuralPlannerCfg),
   }
   ```
5. Export from `embodichain/lab/sim/planners/__init__.py`.

### validate_plan_options decorator

Applied to `plan()` methods to type-check the `options` argument at runtime. Supports three styles:
- `@validate_plan_options` — bare; validates against base `PlanOptions`.
- `@validate_plan_options()` — called with no args; same as above.
- `@validate_plan_options(options_cls=MyPlanOptions)` — custom options class.

### Constraint checking

`BasePlanner.is_satisfied_constraint(vels, accs, constraints)` verifies trajectory outputs stay within limits. Tolerance: 10% for velocity, 25% for acceleration. Supports batch dimensions `(B, N, DOF)`.

## Common Failure Modes

- **`robot_uid` is MISSING** — `BasePlannerCfg.robot_uid` defaults to `MISSING`. Forgetting to set it raises `ValueError` at planner init.
- **Robot not found** — planner init calls `SimulationManager.get_instance().get_robot(uid)`. If the robot hasn't been added to the sim yet, this returns `None` and raises `ValueError`.
- **toppra not installed** — `ToppraPlanner` import fails with `ImportError` at module load time if `toppra==0.6.3` is not installed.
- **Multi-instance robot with ToppraPlanner** — `ToppraPlanner.__init__` raises `NotImplementedError` if `robot.num_instances > 1`. Use a batch-capable planner or single-instance setup.
- **Wrong PlanOptions subclass** — `@validate_plan_options(options_cls=ToppraPlanOptions)` rejects non-matching options types. Passing a base `PlanOptions()` to `ToppraPlanner.plan()` will error.
- **MotionGenerator planner_type not registered** — if `planner_cfg.planner_type` is not in `_support_planner_dict`, `MotionGenerator.__init__` fails. Register new planners there first.
- **Interpolation with unsupported MoveType** — pre-interpolation in `MotionGenOptions` only works for `EEF_MOVE` and `JOINT_MOVE`. Using it with `TOOL`, `SYNC`, or `PAUSE` is ignored or produces unexpected results.
- **Constraint tolerance** — `is_satisfied_constraint` allows 10% velocity / 25% acceleration overshoot. Dense waypoint trajectories may appear to violate constraints but pass validation.
