# Motion Planning

## Find the owner

| Change or question | Start here |
|---|---|
| Planning interface, options and batch validation | `embodichain/lab/sim/motion/planners/base_planner.py` |
| Timed inputs/results | `embodichain/lab/sim/motion/planners/utils.py` → `PlanState`, `PlanResult` |
| Strategy, IK preparation, result normalization | `embodichain/lab/sim/motion/motion_generator.py` → `MotionGenerator` |
| Joint-path timing | `embodichain/lab/sim/motion/planners/toppra_planner.py`, `trapezoidal_planner.py` in that directory |
| Cartesian geometry and scalar timing | `embodichain/lab/sim/motion/planners/se3.py`, `bezier.py`, `_scalar_time_law.py`, `_blend_constraints.py` in that directory |
| Collision planning and scene conversion | `embodichain/lab/sim/motion/planners/curobo/` |
| NMG policy rollout and export metadata | `embodichain/lab/sim/motion/planners/neural_planner.py` |
| Pure interpolation, resampling and retiming | `embodichain/compute/trajectory/` |
| Standalone physical playback | `embodichain/lab/sim/motion/execution.py` |
| Candidate generation and coverage bookkeeping | `embodichain/lab/sim/motion/expansion/` |
| Distinct trajectory variants for fixed waypoints | `embodichain/lab/sim/motion/expansion/variants.py` |

## Choose the layer

`BasePlanner` consumes waypoints and returns timed trajectories.
`MotionGenerator` is the stateful facade for strategy selection, sequential IK,
backend composition, normalization and multi-part coordination. Import it from
`motion.motion_generator`, not `motion.planners`. Motion siblings and the parent
namespace remain lazy to protect Robot initialization.

Planners resolve the robot through the manager identified by
`BasePlannerCfg.sim_instance_id`; there is no fallback to another manager with
the same robot UID. Environment factories copy configs before supplying their
own manager ID. Keep manager selection at this construction boundary.

Read [planner details](planner-details.md) when changing normalization, backend
capabilities or planner registration. Read [collision worlds](collision-worlds.md)
when changing obstacle identity, cache generation or dynamic scene binding.
[IK solvers](../ik-solvers/ik-solvers.md) own candidate/frame contracts;
[Atomic Skills](../atomic-actions/atomic-actions.md) own motion execution policy.

## Timing and batch boundary

Planning is environment-batched: waypoint qpos is `(B, dof)`, TCP transforms
are `(B, 4, 4)`, and output positions are `(B, N, dof)`. Use `PlanState.single()`
for single-environment inputs instead of losing the batch axis. Concrete enum
values, fields and option defaults remain in their source definitions.

`PlanResult` is the canonical timed trajectory across planning and execution.
Whenever positions exist, `dt` contains per-sample arrival intervals `(B, N)`;
`duration` is derived from their sum. A failed result may omit positions.
Timing does not alter simulator `physics_dt`.

`MotionGenOptions.strategy` selects the timing owner. `motion_gen` invokes the
backend; `ik_interp` requires explicit start/count/dt, rejects backend options
and derivative limits, and supports joint/EEF moves. Required Cartesian IK
failures fail the row; waypoints are never discarded to salvage success.
Unsupported routes raise instead of silently selecting another strategy.

Normalization must preserve the meaning of timing, derivatives, failure rows
and backend diagnostics together. Read the
[result contract](planner-details.md#normalization-and-capabilities) before
changing interpolation or output sample counts. Final sampled collision checks
do not establish continuous collision freedom or derivative compliance.

## Compute and execution ownership

`compute.trajectory` owns pure interpolation, differentiation, resampling and
warping. New consumers import it directly; `lab.sim.utility.action_utils` keeps
solver-dependent pose/IK adaptation and compatibility re-exports. Trapezoidal/
Double-S Warp profiles live in `compute/kinematics/_warp/trapezoidal.py`; compute
must not import simulation objects.

`retime_to_control_grid()` maps source trajectories onto a fixed command clock
without shortening their durations and recomputes velocity references.
[Planner details](planner-details.md#retiming-and-playback) describe the execution
boundary. Standalone `play_joint_trajectory()` advances unchanged physics
substeps; Gym experts use the environment-owned command period and
[expert execution contract](../env-framework/env-framework.md). Dataset writers
encode prepared actions rather than retiming a second time.

## Trajectory augmentation boundary

`motion/expansion/` owns immutable candidate contracts, strict config decoding,
phase-authorized residuals/retiming, sampled motion-limit checks and coverage.
Candidate identity and local random seeds are independent of physical env slots.
`GenerationSession` owns budgets, pending writes, coverage reservations and
idempotent commit receipts. It never steps/resets an environment or writes a
dataset: host integrations own restoration, rollout, validation and persistence.
Keep algorithm modules free of direct Gym imports.

Motion-limit checks are not collision/task-success certification.
`rotate_grasp_about_object_axis` and `perturb_approach_direction` change TCP
candidates around a fixed object axis or approach cone; both emit Cartesian
poses that callers must replan, and neither certifies the contact. Grasp
generation itself belongs to `embodichain.toolkits.graspkit`, composed by
Atomic Skills/Task Program rather than embedded in `MotionGenerator`.

`variants.py` varies execution when waypoints are already fixed. Its qpos
operators (`joint_residual`, `via_points`, `nullspace_residual`, `retime`
profiles) vanish with
zero derivative at every phase endpoint and never touch `contact`/`hold`
phases, so annotated waypoints stay exact. Joint-limit rejection covers only
the joints an operator moved; an observed reference can hold an untouched joint
microradians outside its range, and checking it would reject every proposal.
`nullspace_residual` holds the caller's declared task rows to first order only;
the caller owns the Jacobian frame and column order, and a fully constrained
task raises. At most one joint-path operator runs per variant, though `spatial.method`
may name several so each becomes its own variant; a single name is still
accepted.
`expand_trajectory_variants` deduplicates one fixed scene on measured
geometry/timing and counts every rejection under a key naming its reason.
Configuration is optional: omitting it resolves `default_variant_factors`,
which enables every implemented factor the inputs support and drops the ik
factor when no Jacobians are supplied. An explicit config is never overridden, and an
explicitly enabled factor that cannot produce a qpos variant raises rather than
disappearing: `ik` without Jacobians, and `approach`, which owns Cartesian
standoff poses the caller must replan. The resolved config is reported on the result.

`scripts/tutorials/atomic_action/place.py` is the demonstration host, hooked
the way the Affordance sampling tutorials are: shared helpers live in
`tutorial_utils.py`, the tutorial declares only which named segments may move,
and one variant replays per simulation row. Phases come from the action's own
`TrajectorySegment` ranges; only motion at or after the lift is retimed so the
shared `clear_dynamics()` step stays aligned. `--variant_plot_dir` writes a
joint-trajectory figure and a tool-path figure with waypoint markers. Task
Program, Gym lifecycle, dataset persistence and `GenerationSession` collection
bookkeeping are deliberately out of scope; this is a direct-simulation
contract. Keep `joint_dedup_normalized_tol` below `joint_offset_scale`, or
genuinely different variants are discarded as duplicates.

`expansion/manipulability.py` profiles posture conditioning through
`embodichain.compute.kinematics.yoshikawa_manipulability`. Callers supply the
Jacobians, so the module stays host-independent. `ManipulabilityBands`
normalizes scores against a reference bottleneck held per initial state, and
`manipulability_guided_residual` keeps whichever of several `joint_residual`
draws from one local generator lands nearest a requested band. The factor is
opt-in through `augmentation.factors.manipulability` and disabled by default.
Once enabled, `register_case` requires a positive `manipulability_reference`
for each initial state, which may differ within one case; `CoverageIndex`
enforces a per-band quota so one well-conditioned posture cannot absorb the
collection budget; and `GenerationSession` classifies bands from the measured
`manipulability` observation rather than any planned score. Band guidance ranks
postures only: path, dynamic and task validation stay separate.

## Focused validation

| Changed boundary | Existing coverage |
|---|---|
| Imports and generator normalization | `tests/sim/motion/test_motion_imports.py`, `test_motion_generator.py`, `test_motion_generator_batched.py` in that directory |
| Planner options, timing and backend behavior | Matching files under `tests/sim/motion/planners/` |
| Pure path/timing functions | `tests/compute/test_trajectory.py`, `test_trajectory_timing.py` in that directory |
| Fixed-cadence playback | `tests/sim/motion/test_execution.py` |
| Augmentation contracts, operators and session accounting | `tests/sim/motion/expansion/` |

For stale obstacles, inspect collision capability/identity before planner tuning.
For altered trajectory duration or acceleration, inspect normalization and the
destination clock before changing backend limits. Tutorials remain examples;
validate these production boundaries directly.
