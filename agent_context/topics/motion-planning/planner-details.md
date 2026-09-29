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

`toppra_planner.py` selects Warp on CUDA or when input gradients are required,
and otherwise selects NumPy on CPU. The config can explicitly select either
backend. Numerical ownership is in
`compute/trajectory/_toppra.py`, `_toppra_warp.py`, and `_warp/toppra.py`.
Both preserve uniform not-a-knot cubic geometry, the existing waypoint
deduplication tolerance, and signed velocity/acceleration bounds containing
rest. A run of repeated samples at the end of a path moves the last kept knot
onto the final sample rather than appending a zero-length end segment, so a
batched environment that converges early and holds its pose is retimed exactly
as if it had stopped. Planner outputs retain float32 and quantity sampling retains
the existing integer coercion, independently of backend or input precision.
Interval bounds are conservative; timing need not match the former external
TOPPRA package. Tiny real moves must retain positive duration.

Warp constructs splines, bounds, controllable sets and samples on the input
device. It honors the current Torch stream and reads a scalar to size output;
it does not use a CPU worker pool. NumPy fans batched solves into independent
CPU workers, which must remain picklable and avoid CUDA/Warp/simulation state.
Single-worker/single-row NumPy execution can run inline. The automatic process
context uses spawn with GPU simulation; do not force fork to reduce overhead.

Warp records first-order adjoints when Torch gradient tracking is enabled and
waypoints or tensor velocity/acceleration bounds require gradients. Both time
and quantity sampling propagate position, velocity, acceleration and duration
losses through the timing solve. Scalar, per-joint and signed tensor bounds keep
their graphs on either device; TOPPRA option copies preserve these tensor
references while copying mutable containers. Explicit NumPy selection rejects
gradient requests; `torch.no_grad()` retains ordinary forward planning.
Filtering, grid topology, sample counts and active-constraint choices are
discrete: derivatives apply locally away from their switches. Time sampling
keeps `ceil(duration / sample_dt) + 1` points uniformly spread across the full
duration, with the integer count fixed during backward. Failed rows contribute
zero gradients; stationary rows differentiate only their retained position.
Higher-order gradients are unsupported. Test finite-difference agreement,
generator option copying/resampling and CUDA backward stream behavior in the
TOPPRA tests.

Variable-length outputs pad with the terminal position without adding duration.
Per-row failures preserve other rows; a broken worker pool is discarded and
rebuilt on a later call. Test these lifecycle boundaries in
`tests/sim/motion/planners/test_toppra_batched.py`; numerical limits, geometry,
CPU/Warp agreement and CUDA stream behavior are covered in
`tests/compute/test_toppra.py`.

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
