# Planner implementation details

Read this when the request needs these details. [Topic overview](motion-planning.md).

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

### TrapezoidalPlanner

`TrapezoidalPlanOptions` selects trapezoidal or Double-S timing for joint
waypoints, with optional quintic corner blending. `TIME` sampling pads shorter
batch rows at their exact final position with zero velocity and acceleration,
starting at each row's actual endpoint. Padding has zero arrival intervals.

With `stop_at_waypoints=False`, straight runs are compressed using normalized
edge directions and a cosine tolerance relative to the first edge of each
retained run. Small waypoint spacing does not change the angular test, and
successive small turns cannot accumulate into an unchecked shortcut.
Positive `blend_tolerance` requires `stop_at_waypoints=False`; conflicting
settings raise `ValueError` both during option construction and at planning
entry, including when options have been mutated after construction.

Constraint reports use joint units for both derivative peaks and `*_limit`
summaries. Each summary limit is the maximum configured limit over joints;
per-joint utilization and `within_limits` retain heterogeneous limits. The
projected scalar path limits are internal timing inputs, not report limits.

Torch and Warp scalar sampling clamp times outside the trajectory to the
endpoint state. Scalar trapezoidal acceleration uses the one-sided phase value
at an endpoint; the joint planner explicitly turns completed rows into holds.

The tutorial `scripts/tutorials/sim/planner/trapezoidal_planner.py` replays
position samples by interpolating targets at physics ticks. A partial final
tick preserves total duration divided by replay speed. Zero-time samples do
not advance physics; multiple environments require identical timing rows.

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
- Supported planner types: TOPPRA, TrapezoidalPlanner, NeuralPlanner, and cuRobo.
- `MotionGenCfg.planner_cfg` is **MISSING** — must be provided.
- `generate()` and `interpolate_trajectory()` are env-batched (`B, N, DOF`).
- `generate()` always returns a normalized `PlanResult`; failed rows hold the
  supplied `start_qpos`, and every returned trajectory has explicit `dt` and
  a `duration` derived from it.

Grasp-pose generation is a sibling planning service, not a
`MotionGenerator` feature. `embodichain.toolkits.graspkit` owns its standalone
contract and the `pg_grasp` implementation without importing `embodichain.lab`.
Atomic Skills and Task Program install a generator instance in
`AtomicActionEngine`. Reference deployments select execution policies under
`configs/components/execution_policies/`, while reusable embodiment components
own the simulation robot and sensors; their optional `skill_profile` owns
grasp-generator geometry/tuning. Task-local integrations retain only semantic
action options and typed task-specific overrides.
`task_program/integrations/_configured_composition.py` checks the component
contracts and lowers the result through the strict decoder in `configured.py`;
the config loader registers the existing `EmbodiedEnv` under the selected ID.
There are no task-specific environment subclasses or direct-planning paths.

`MotionGenOptions` fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `strategy` | `"motion_gen" \| "ik_interp"` | `"motion_gen"` | Use the configured backend or deterministic waypoint IK/joint interpolation |
| `sample_count` | `int \| None` | `None` | Requested normalized output count; backend default when omitted |
| `velocity_limit` | `float \| None` | `None` | Optional backend-neutral velocity limit |
| `acceleration_limit` | `float \| None` | `None` | Optional backend-neutral acceleration limit |
| `start_qpos` | `torch.Tensor \| None` | `None` | Optional backend context, shape `(B, DOF)`; required by `strategy="ik_interp"` |
| `control_part` | `str \| None` | `None` | Robot control part name (must match `RobotCfg.control_parts` key) |
| `plan_opts` | `PlanOptions \| None` | `None` | Passed to the underlying planner |
| `is_interpolate` | `bool` | `False` | Pre-interpolate waypoints before planning |
| `interpolation_dt` | `float \| None` | `None` | Required explicit waypoint interval for `strategy="ik_interp"` and automatic joint interpolation fallback |
| `interpolate_nums` | `int \| list[int]` | `10` | Points per segment (scalar or per-segment list) |
| `is_linear` | `bool` | `False` | `True` = Cartesian linear interpolation; `False` = joint-space |
| `interpolate_position_step` | `float` | `0.002` | Cartesian step size (meters) or joint step size (radians) |
| `interpolate_angle_step` | `float` | `π/90` | Angular step in joint space (radians); only if `is_linear=False` |

## Configuration

### Registering a new planner

1. Create a `BasePlanner` subclass with a `plan()` method decorated with `@validate_plan_options`; every result containing positions must include `dt`, from which `duration` is derived.
2. Create a `BasePlannerCfg` subclass with a unique `planner_type` string.
3. Optionally create a `PlanOptions` subclass for planner-specific options.
4. For a planner that accepts live obstacles, override `collision_world_info`
   with a `CollisionWorldInfo` whose `supports_updates=True`, and implement
   `with_collision_world()` without mutating caller-owned reusable options.
5. Register in `MotionGenerator._support_planner_dict`:
   ```python
   _support_planner_dict = {
       "toppra": (ToppraPlanner, ToppraPlannerCfg),
       "neural": (NeuralPlanner, NeuralPlannerCfg),
   }
   ```
6. Export from `embodichain/lab/sim/motion/planners/__init__.py`.

### validate_plan_options decorator

Applied to `plan()` methods to type-check the `options` argument at runtime and enforce batch consistency. Supports three styles:
- `@validate_plan_options` — bare; validates against base `PlanOptions`.
- `@validate_plan_options()` — called with no args; same as above.
- `@validate_plan_options(options_cls=MyPlanOptions)` — custom options class.

The decorator checks that every `PlanState` in `target_states` shares the same leading batch dim `B` and that `B` matches `robot.num_instances` (or is `1`).

### Constraint checking

`BasePlanner.is_satisfied_constraint(vels, accs, constraints)` verifies trajectory outputs stay within limits. Tolerance: 10% for velocity, 25% for acceleration. Supports batch dimensions `(B, N, DOF)`.
