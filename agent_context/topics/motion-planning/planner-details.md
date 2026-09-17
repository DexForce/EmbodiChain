# Planner integration contracts

Read when changing a backend or the facade. Entry points and validation are in
[the overview](motion-planning.md); obstacle identity and updates belong to
[collision worlds](collision-worlds.md). Config/API defaults remain in source.

## Normalization and capabilities

`MotionGenerator` prepares Cartesian targets with sequential seeded IK. Required
IK/FK-consistency failures fail the environment. `interpolate_trajectory()` raises
on any failed row; `generate()` carries per-row success in a normalized
`PlanResult`. Fixed Cartesian sample preservation requires explicit `ik_interp`;
it is not an implicit fallback from backend planning.

Preserve these coupled result rules in `motion_generator.py`:

- Missing velocities are derived from final positions and arrival intervals.
- Resampling is time-based, preserves each row's duration, recomputes velocities
  and invalidates accelerations. Unchanged samples retain native derivatives.
- `uses_sparse_joint_waypoints` lets a backend receive `start_qpos` plus goals
  without generic pre-interpolation. `preserve_plan_samples` keeps its native
  grid and derivatives through normalization. TrapezoidalPlanner declares both.
- Failed rows hold the supplied start pose unless the backend declares
  `preserve_failed_plan_positions`; NeuralPlanner uses that capability to keep
  partial rollout evidence while reporting failure.
- Preserve `constraint_report` only while the corresponding samples remain
  unchanged. Resampling or replacing failed rows invalidates the entire report;
  generic normalization cannot reconstruct planner-specific diagnostics.
- A backend offering joint-trajectory validation rechecks resampled output
  using the resolved control part and current obstacle poses. Invalid rows fail;
  this sampled check is not continuous collision/derivative certification.

`resolve_plan_options()` copies typed caller options or obtains backend defaults
and applies supported generic timing constraints. Atomic Skills call this facade
rather than branching on concrete planner option classes.

## Backend-specific boundaries

### TOPPRA

`toppra_planner.py` fans batched solves into independent CPU work. The module-level
NumPy worker must remain picklable and must not touch CUDA/Warp/simulation state.
Single-worker/single-row execution can run inline. The automatic multiprocessing
context uses spawn with GPU simulation to avoid forking initialized CUDA state.
Inspect the config for overrides; do not force fork merely to reduce overhead.

Variable-length outputs pad with the terminal position without adding duration.
Per-row failures preserve other rows; a broken worker pool is discarded and
rebuilt on a later call. Test these lifecycle boundaries in
`tests/sim/motion/planners/test_toppra_batched.py` and keep dependency versions
in the package declarations rather than this context.

### Trapezoidal and Double-S

`trapezoidal_planner.py` owns sparse joint-waypoint timing and optional corner
blending. Completed rows become exact endpoint holds with zero derivatives;
padding has zero arrival intervals. Scalar Torch/Warp samplers clamp outside
trajectory time, but the joint planner owns completed-row hold semantics.

Straight-run compression must not accumulate small turns into an unchecked
shortcut. Blending option compatibility is checked at construction and planning
entry so mutated options cannot bypass it. Continuous derivative bounds belong
to `_blend_constraints.py`; inspect algorithm tests instead of duplicating
thresholds or formulas here.

Constraint reports use joint units, not projected scalar timing limits.
Aggregate limit summaries do not replace per-joint utilization/validity for
heterogeneous limits. Validate reports, geometry and padding together in
`tests/sim/motion/planners/test_trapezoidal_planner.py`.

### Neural policy adapter

`neural_planner.py` consumes a standalone NMG ONNX export with normalization in
the graph. Validate complete observation-layout metadata/fingerprint before
rollout; model dimensions and waypoint capacity belong to the export contract.
Inspect the `policy-deploy` package extra for runtime dependencies.

Runtime/training frame differences must be expressed through explicit policy
frame and TCP transforms. Target/FK quaternions follow the shared
[simulation convention](../simulation-system/simulation-system.md); do not
convert a value twice. Failed waypoint convergence retains native rollout
samples through the capability described above. Validate export metadata and
batched behavior in `test_neural_planner.py` and `test_neural_batched.py` under
`tests/sim/motion/planners/`.

### cuRobo import isolation

`curobo_planner.py::_require_curobo()` lazily imports the optional backend and
restores the caller's Torch matmul precision and CUDA/cuDNN TF32 flags on success
or failure. Backend import side effects must not change numerical policy for
other planners/solvers. Native derivatives are mapped into simulator joint order;
only missing velocity segments are derived. Scene conversion and batch-world
semantics are owned by [collision worlds](collision-worlds.md).

## Retiming and playback

`compute/trajectory/timing.py` owns time-domain differentiation/resampling and
`retime_to_control_grid()`. Zero-time position changes are invalid; repeated-time
unchanged samples may represent padding/junctions. Time resampling preserves
first-arrival offset and total duration but does not certify motion limits.

Control-grid retiming requires zero first-arrival intervals. It rounds each
row's duration up to the destination clock, samples uniform source-path phase,
and recomputes qvel with zero first/last-valid and padded-hold velocities.
Source `PlanResult.dt` remains planner timing.

`motion/execution.py::play_joint_trajectory()` requires `control_dt` to be an
integer multiple of `physics_dt` and advances that many unchanged substeps.
Position-plus-velocity command mode is opt-in. Gym execution instead owns its
`step_dt`; dataset encoding must not apply another retiming pass. The legacy
`ExpertJointTrajectory` wrapper is a compatibility boundary, not a second
canonical trajectory format. Terminal target policy belongs to
[Atomic Skills execution](../atomic-actions/execution.md).

## Adding or changing a planner

A planner extends `BasePlanner`, provides a unique config `planner_type`, applies
`validate_plan_options` to its `plan()` entry, and returns explicit timing with
positions. The decorator validates option type and waypoint batch consistency;
inspect it for supported single-row behavior rather than hand-validating a
second contract in the backend.

Update `MotionGenerator._support_planner_dict` and the public planner exports.
Choose sparse-waypoint, sample-preservation and failure-preservation capabilities
explicitly and test normalization through the facade. A dynamic-world backend
also implements `collision_world_info` and `with_collision_world()` without
mutating caller-owned options. Keep lazy import behavior and optional dependency
failures covered when changing registrations.
