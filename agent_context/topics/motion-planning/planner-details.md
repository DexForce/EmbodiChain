# Planner integration contracts

Read when changing a backend or the facade. Entry points and validation are in
[the overview](motion-planning.md); obstacle identity and updates belong to
[collision worlds](collision-worlds.md). Config/API defaults remain in source.

## Normalization and capabilities

`MotionGenerator` prepares Cartesian targets through `Robot.compute_batch_ik`,
batching environments and waypoints in bounded chunks. Solvers advertising
continuous batch selection receive a path seed carried between chunks. Other
solvers first solve each chunk's endpoint and interpolate joint seeds for its
interior batch; this encourages consistent branches without guaranteeing
continuous selection. Failed samples retain the last valid seed across chunks.
Required IK/FK-consistency failures fail the environment. For each row, an initial pose
whose full transform agrees with the seed FK retains the known starting joints;
the comparison aligns dtype/device with the interpolated poses, since FK may
use float32 buffers for float64 Cartesian inputs. It needs no fresh IK solution
at a singularity; an entirely confirmed starting batch is omitted from IK.
Unconfirmed starts and later poses still require successful IK and FK consistency.
`interpolate_trajectory()` raises
on any failed row; `generate()` carries per-row success in a normalized
`PlanResult`. Fixed Cartesian sample preservation requires explicit `ik_interp`;
it is not an implicit fallback from backend planning. When velocity or
acceleration limits expand such a path, `MotionGenOptions.cartesian_path_sampler`
may regenerate poses from normalized minimum-jerk progress. The callback must
end at the requested target, begin within the active IK solver's accepted pose
tolerance, and preserve the batch, dtype, and device. Without it, the facade
resamples the caller-supplied Cartesian polyline.

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

`planners/_joint_limits.py` resolves effective speeds through
`Robot.get_qvel_limits()` and existing articulation mimic metadata. It selects
trajectory joints, reduces participating environments to their strictest caps,
and intersects speed requests with those caps. Acceleration is a planner-owned
budget, supplied through plan options or backend defaults; it is not
an inferred asset or physics limit. Partial-joint trajectories require an
explicit control part when their column mapping is ambiguous. Explicit zero
runtime speeds hold those joints; moving them must fail planning. Imported
nonpositive speeds follow DexSim's unspecified-limit semantics, and unavailable
runtime caps first use positive authored speeds from the assembled `cfg.fpath`
URDF, mapped by joint name, before planner fallbacks. Valid runtime values,
including zero, remain authoritative. The asset lookup is cached by path and
file revision; mimic projection applies after fallback.

A finite backend value does not prove that a speed was authored in the asset:
DexSim's legacy URDF reader substitutes a positive default when the field is
missing. Preserve backend import semantics instead of guessing provenance from
the number; calibrated presets can supply explicit limits. Asset zero placeholders
must not be interpreted as explicit runtime zero-speed commands.

cuRobo writes absolute speed caps into a cache-owned URDF copy and acceleration
caps into YAML in active cspace order, excluding mimic followers. Keep velocity
scales at one so loading and cloning preserve these caps. Mesh paths retain the
source asset root. cuRobo rejects controlled zero-speed joints before loading;
it cannot represent zero-width velocity intervals. `CuroboPlannerCfg.max_velocity`
optionally tightens effective Robot speeds; a request cannot loosen a valid cap.
Both it and `max_acceleration` accept a scalar or a vector in `active_joint_ids`
order. The default acceleration budget remains 15.0; CPU timing planners retain
their 0.5 default. Its YAML and backend caches include resolved speed and
acceleration vectors; changing either a request or robot limits closes and
rebuilds the affected backend. Neural policy rollout retains its configured timing. The
`ik_interp` path resolves robot speed caps and generic motion limits in the
facade independently of the selected planner backend.

## Backend-specific boundaries

### TOPPRA

`toppra_planner.py` uses Warp on CPU and CUDA. Numerical ownership is in
`compute/trajectory/_toppra_warp.py` and `_warp/toppra.py`, with cooperative
CUDA inference in `_warp/toppra_cuda.py`; shared validation
lives in `_toppra_common.py`. `_toppra.py` is an offline NumPy oracle and must
not be imported by the runtime planner or Warp adapter.
Robot-backed planning retains shared speed-cap and control-part resolution,
including gradients of uncapped requested limits.
The runtime and reference preserve uniform not-a-knot cubic geometry, the existing waypoint
deduplication tolerance and final knot, and signed velocity/acceleration bounds
containing rest. Planner outputs retain float32 and quantity sampling retains
the existing integer coercion, independently of backend or input precision.
Interval bounds are conservative; timing need not match the former external
TOPPRA package. Tiny real moves must retain positive duration.

Warp constructs splines, bounds, controllable sets and samples on the input
device. It honors the current Torch stream and reads a scalar to size output.
CUDA inference reduces constraint planes across one block per path; all
early exits must remain uniform across the block. Inference may use fast
polygon projection, while recorded calls retain the selected-expression
projection and serial solver needed by the adjoint. Dense CPU inference sampling
shares scalar timing and interval searches across the joints of each sample;
recorded sampling retains its per-joint adjoint path.
CPU inference amortizes sufficiently large batches over a shared bounded thread
pool; small requests and adjoint-recording calls stay inline. Row blocks own
their numerical buffers, inherit Torch grad/inference modes and merge using
the global endpoint/zero-derivative/zero-dt padding contract. Ordinary inference
launches must not record into Warp's process-global tape. CPU pool state is
discarded after fork; it is not owned by an individual planner's `close()`.
Planning has no process backend, per-planner worker state or process cleanup.
The removed `backend`, `max_workers` and `mp_context` configuration fields must
be removed from callers; do not restore them as ignored compatibility options.

Warp records first-order adjoints when Torch gradient tracking is enabled and
waypoints or tensor velocity/acceleration bounds require gradients. Both time
and quantity sampling propagate position, velocity, acceleration and duration
losses through the timing solve. Scalar, per-joint and signed tensor bounds keep
their graphs on either device; TOPPRA option copies preserve these tensor
references while copying mutable containers. `torch.no_grad()` retains
ordinary forward planning.
Filtering, grid topology, sample counts and active-constraint choices are
discrete: derivatives apply locally away from their switches. Time sampling
keeps `ceil(duration / sample_dt) + 1` points uniformly spread across the full
duration, with the integer count fixed during backward. Failed rows contribute
zero gradients; stationary rows differentiate only their retained position.
Higher-order gradients are unsupported. Test finite-difference agreement,
generator option copying/resampling and CUDA backward stream behavior in the
TOPPRA tests.

Variable-length outputs pad with the terminal position without adding duration.
Per-row failures preserve other rows. Test planner contracts and the absence
of process workers in
`tests/sim/motion/planners/test_toppra_batched.py`; numerical limits, geometry,
CPU/Warp agreement and CUDA stream behavior are covered in
`tests/compute/test_toppra.py`; CPU projection, thread isolation, variable-length
assembly and pool inheritance are covered in `tests/compute/test_toppra_cpu.py`.

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

`NeuralPlannerCfg.dt` is bookkeeping, not solved timing: the rollout integrates
a joint delta per step, so a saturated step implies `action_scale / dt` rad/s
and overshoots a typical arm's velocity limit several times over. Setting
`constraints` swaps that nominal timing for a TOPPRA parameterization through
the shared entry point below; it resamples the path, so positions and poses
change with the timing, and it cannot reduce the path's own jerk. Default
remains `None`, which preserves the nominal-timing behavior.

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

`toppra_planner.py::retime_joint_paths()` is the shared entry point for
re-parameterizing an already-planned joint path under velocity and acceleration
limits, used by planners that emit geometry without executable timing. It is
the planner's own time parameterization. All CPU/CUDA rows run on Warp,
including single-row calls, so batch size does not change waypoint cleanup
precision or the helper's float32/float64 output dtype. It resamples
along the fitted spline, so recompute derived poses from its output rather
than reusing the input samples. Optional per-row path lengths exclude
rectangular tail holds from spline fitting; without them, an explicit repeated
final waypoint retains the historical final-knot geometry.

`compute/trajectory/timing.py` owns time-domain differentiation/resampling and
`retime_to_control_grid()`. Zero-time position changes are invalid; repeated-time
unchanged samples may represent padding/junctions. Time resampling preserves
first-arrival offset and total duration but does not certify motion limits.

Control-grid retiming requires zero first-arrival intervals. It rounds each
row's duration up to the destination clock, samples uniform source-path phase,
and recomputes qvel with zero first/last-valid and padded-hold velocities.
Source `PlanResult.dt` remains planner timing.

`compute/trajectory/interpolation.py::interpolate_minimum_jerk()` expands joint
keyframes onto a fixed control interval while preserving every keyframe. The
requested sample count is a minimum; per-segment counts grow to satisfy scalar
or ordered per-joint velocity and acceleration limits, and spare samples are
distributed by joint displacement rather than equally across unequal moves.
The emitted dtype is checked on the destination grid so float32 cancellation
cannot silently violate those derivative limits.

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
