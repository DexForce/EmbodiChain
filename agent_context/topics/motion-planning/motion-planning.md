# Motion Planning

## Entry Points

| What | Path |
|---|---|
| Planner registry | `embodichain/lab/sim/planners/__init__.py` |
| Base planner class & config | `embodichain/lab/sim/planners/base_planner.py` → `BasePlanner`, `BasePlannerCfg`, `PlanOptions`, `validate_plan_options` |
| TOPPRA planner | `embodichain/lab/sim/planners/toppra_planner.py` → `ToppraPlanner`, `ToppraPlannerCfg`, `ToppraPlanOptions` |
| Neural planner | `embodichain/lab/sim/planners/neural_planner.py` → `NeuralPlanner`, `NeuralPlannerCfg`, `NeuralPlanOptions` |
| Planner assets | `embodichain/data/assets/planner_assets.py` → `download_neural_planner_checkpoint()` |
| Motion generator | `embodichain/lab/sim/planners/motion_generator.py` → `MotionGenerator`, `MotionGenCfg`, `MotionGenOptions` |
| Planner utilities & data types | `embodichain/lab/sim/planners/utils.py` → `PlanState`, `PlanResult`, `MoveType`, `MovePart`, `TrajectorySampleMethod`, `interpolate_xpos_batched` |

## Overview

The planning stack has two layers:
1. **BasePlanner** — low-level trajectory planner that takes a list of `PlanState` waypoints and produces a `PlanResult` with joint trajectories.
2. **MotionGenerator** — high-level wrapper that composes a planner with optional interpolation, IK resolution, and multi-part coordination.

All planners resolve their robot at init via `SimulationManager.get_instance().get_robot(cfg.robot_uid)`.

The entire stack is **env-batched** (`B = num_envs`). `PlanState` / `PlanResult` tensors carry a leading `B` dimension; `BasePlanner.plan()` and `MotionGenerator.generate()` operate on `B` environments in one call.

## Planner Hierarchy

```
BasePlanner (ABC)
  ├─ ToppraPlanner        Time-optimal path parameterization (fork-pool fan-out)
  └─ NeuralPlanner (experimental)   APG waypoint rollout (native batching)

MotionGenerator           Wraps any BasePlanner; adds interpolation and multi-part support
```

Config hierarchy:
```
BasePlannerCfg            robot_uid (MISSING), planner_type
  ├─ ToppraPlannerCfg     planner_type = "toppra", max_workers, mp_context
  └─ NeuralPlannerCfg     planner_type = "neural", checkpoint_path (MISSING)

MotionGenCfg              planner_cfg (MISSING — must be a BasePlannerCfg subclass)

PlanOptions               (empty base)
  ├─ ToppraPlanOptions    constraints, sample_method, sample_interval
  └─ NeuralPlanOptions    control_part, start_qpos, max_steps

MotionGenOptions          start_qpos (B, DOF), control_part, plan_opts, is_interpolate,
                          interpolate_nums, is_linear, interpolate_position_step,
                          interpolate_angle_step
```

## Available Planners

### ToppraPlanner

Time-optimal path parameterization using the [toppra](https://github.com/hungpham2511/toppra) library.

- **Dependency**: `pip install toppra==0.6.3` (import-time error if missing).
- **Batched**: accepts `target_states` whose tensor fields carry leading batch dim `B`. Internally fans out `B` independent single-env TOPPRA solves across a `ProcessPoolExecutor`.
- **Method**: `plan(target_states, options=ToppraPlanOptions()) -> PlanResult`

`ToppraPlannerCfg` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_workers` | `int \| None` | `None` | Worker process count. `None` → `min(cpu_count() // 2, B)`. |
| `mp_context` | `str \| None` | `None` | Multiprocessing start method. `None` auto-selects `fork` on CPU and `spawn` on GPU; can be set to `"fork"` or `"spawn"`. |

`ToppraPlanOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `constraints` | `dict` | `{"velocity": 0.2, "acceleration": 0.5}` | Per-joint or scalar limits |
| `sample_method` | `TrajectorySampleMethod` | `QUANTITY` | `TIME`, `QUANTITY`, or `DISTANCE` |
| `sample_interval` | `float \| int` | `0.01` | Time interval (seconds) or sample count depending on method |

Worker details:
- The pure-numpy module-level worker `_toppra_solve_one_env` is picklable and never touches CUDA/Warp/sim state.
- `B == 1` or `max_workers == 1` uses an inline fallback (no IPC).
- `TIME` sampling can produce per-env waypoint counts; shorter trajectories are tail-padded by repeating the final waypoint and `duration` records the real endpoint per env.
- Per-env failures set `success[b] = False` and fill the env's trajectory with its start qpos; other envs continue. `BrokenProcessPool` tears the pool down and rebuilds it on the next call.

### NeuralPlanner (experimental)

Learning-based EEF waypoint planner. Franka Panda only.

- Checkpoint: `download_neural_planner_checkpoint()` from HuggingFace (gated, needs `HF_TOKEN`)
- Use via `MotionGenerator` with `planner_type="neural"` and `plan_opts=NeuralPlanOptions(...)`
- Input: `EEF_MOVE` `PlanState` list with batched `xpos:(B, 4, 4)`
- Key cfg: `checkpoint_path` (from download), `control_part`
- Natively batched: transformer forward, reach checks, and convergence holds all operate on `(B, ...)`.

### MotionGenerator

Unified interface for trajectory planning with optional pre-interpolation.

- Wraps a `BasePlanner` instance (resolved from `planner_cfg.planner_type`).
- Supported planner types: `{"toppra": (ToppraPlanner, ToppraPlannerCfg), "neural": (NeuralPlanner, NeuralPlannerCfg)}`.
- `MotionGenCfg.planner_cfg` is **MISSING** — must be provided.
- `generate()` and `interpolate_trajectory()` are env-batched (`B, N, DOF`).

`MotionGenOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `start_qpos` | `torch.Tensor \| None` | `None` | Override starting joint config, shape `(B, DOF)`; `None` = use current robot state |
| `control_part` | `str \| None` | `None` | Robot control part name (must match `RobotCfg.control_parts` key) |
| `plan_opts` | `PlanOptions \| None` | `None` | Passed to the underlying planner |
| `is_interpolate` | `bool` | `False` | Pre-interpolate waypoints before planning |
| `interpolate_nums` | `int \| list[int]` | `10` | Points per segment (scalar or per-segment list) |
| `is_linear` | `bool` | `False` | `True` = Cartesian linear interpolation; `False` = joint-space |
| `interpolate_position_step` | `float` | `0.002` | Cartesian step size (meters) or joint step size (radians) |
| `interpolate_angle_step` | `float` | `π/90` | Angular step in joint space (radians); only if `is_linear=False` |

## Planner Interface

### PlanState (input)

Describes one waypoint or action. Tensor fields carry a leading batch dim `B`; enum/scalar fields are shared across `B`.

| Field | Type | Notes |
|---|---|---|
| `move_type` | `MoveType` | `TOOL`, `EEF_MOVE`, `JOINT_MOVE`, `SYNC`, `PAUSE` |
| `move_part` | `MovePart` | `LEFT`, `RIGHT`, `BOTH`, `TORSO`, `ALL` |
| `xpos` | `torch.Tensor \| None` | Target TCP pose `(B, 4, 4)` for `EEF_MOVE` |
| `qpos` | `torch.Tensor \| None` | Target joint angles `(B, DOF)` for `JOINT_MOVE` |
| `qvel` / `qacc` | `torch.Tensor \| None` | Target joint velocities / accelerations `(B, DOF)` |
| `is_open` | `bool` | Tool open/close (for `TOOL`) |
| `is_world_coordinate` | `bool` | `True` = world frame; `False` = relative |
| `pause_seconds` | `float` | Duration for `PAUSE` move type |

Convenience constructors:
- `PlanState.from_qpos(qpos:(B,DOF), move_type=JOINT_MOVE, ...) -> PlanState`
- `PlanState.from_xpos(xpos:(B,4,4), move_type=EEF_MOVE, ...) -> PlanState`
- `PlanState.single(qpos=(DOF,)\|None, xpos=(4,4)\|None, ...) -> PlanState` — unsqueezes single-env tensors to `B=1` (idempotent on already-batched tensors).

### PlanResult (output)

| Field | Type | Notes |
|---|---|---|
| `success` | `bool \| torch.Tensor` | Per-env success `(B,)` bool tensor (or scalar bool) |
| `xpos_list` | `torch.Tensor \| None` | EEF poses `(B, N, 4, 4)` |
| `positions` | `torch.Tensor \| None` | Joint positions `(B, N, DOF)` |
| `velocities` | `torch.Tensor \| None` | Joint velocities `(B, N, DOF)` |
| `accelerations` | `torch.Tensor \| None` | Joint accelerations `(B, N, DOF)` |
| `dt` | `torch.Tensor \| None` | Per-step time durations `(B, N)` |
| `duration` | `float \| torch.Tensor` | Total trajectory time per env `(B,)` |

Helper: `PlanResult.is_all_success() -> bool` returns `True` only when every env succeeded.

### MoveType enum

| Value | Meaning |
|---|---|
| `TOOL` | Tool open or close command |
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

Applied to `plan()` methods to type-check the `options` argument at runtime and enforce batch consistency. Supports three styles:
- `@validate_plan_options` — bare; validates against base `PlanOptions`.
- `@validate_plan_options()` — called with no args; same as above.
- `@validate_plan_options(options_cls=MyPlanOptions)` — custom options class.

The decorator checks that every `PlanState` in `target_states` shares the same leading batch dim `B` and that `B` matches `robot.num_instances` (or is `1`).

### Constraint checking

`BasePlanner.is_satisfied_constraint(vels, accs, constraints)` verifies trajectory outputs stay within limits. Tolerance: 10% for velocity, 25% for acceleration. Supports batch dimensions `(B, N, DOF)`.

## Common Failure Modes

- **`robot_uid` is MISSING** — `BasePlannerCfg.robot_uid` defaults to `MISSING`. Forgetting to set it raises `ValueError` at planner init.
- **Robot not found** — planner init calls `SimulationManager.get_instance().get_robot(uid)`. If the robot hasn't been added to the sim yet, this returns `None` and raises `ValueError`.
- **toppra not installed** — `ToppraPlanner` import fails with `ImportError` at module load time if `toppra==0.6.3` is not installed.
- **Batch dim mismatch** — `@validate_plan_options` raises `ValueError` if `PlanState` entries have inconsistent `B` or if `B` does not equal `robot.num_instances`.
- **Single-env caller shape mismatch** — legacy callers passing `(DOF,)` qpos or `(4,4)` xpos must wrap with `PlanState.single(...)` or call `from_qpos`/`from_xpos` with a leading `B=1` dim.
- **MotionGenerator planner_type not registered** — if `planner_cfg.planner_type` is not in `_support_planner_dict`, `MotionGenerator.__init__` fails. Register new planners there first.
- **Interpolation with unsupported MoveType** — pre-interpolation in `MotionGenOptions` only works for `EEF_MOVE` and `JOINT_MOVE`. Using it with `TOOL`, `SYNC`, or `PAUSE` is ignored or produces unexpected results.
- **Constraint tolerance** — `is_satisfied_constraint` allows 10% velocity / 25% acceleration overshoot. Dense waypoint trajectories may appear to violate constraints but pass validation.
- **Fork safety with GPU sim** — `ToppraPlannerCfg.mp_context=None` defaults to `spawn` on GPU to avoid fork-after-CUDA-init hazards. Force `fork` only when the sim device is CPU or you have verified it is safe.
