# Motion Planning

## Find the owner

| Change or question | Start here |
|---|---|
| Planning interface, options and batch validation | `embodichain/lab/sim/motion/planners/base_planner.py` |
| Timed inputs/results | `embodichain/lab/sim/motion/planners/utils.py` → `PlanState`, `PlanResult` |
| Strategy, IK preparation, result normalization | `embodichain/lab/sim/motion/motion_generator.py` → `MotionGenerator` |
| Joint-path timing | `embodichain/lab/sim/motion/planners/toppra_planner.py`, `trapezoidal_planner.py` in that directory |
| Cartesian geometry and scalar timing | `embodichain/lab/sim/motion/planners/se3.py`, `bezier.py`, `_scalar_time_law.py`, `_blend_constraints.py` in that directory |
| Sampled Cartesian geometry → timing → robot IK | `embodichain/lab/sim/motion/planners/_cartesian_profiles.py` (internal adapter) |
| Collision planning and scene conversion | `embodichain/lab/sim/motion/planners/curobo/` |
| NMG policy rollout and export metadata | `embodichain/lab/sim/motion/planners/neural_planner.py` |
| Installable evaluation suites and external planner adapters | `extensions/embodichain_benchmarks/motion_generation/api.py`, `planners/base.py` in that package |
| Pure interpolation, resampling and retiming | `embodichain/compute/trajectory/` |
| Standalone playback and B=1 expansion host lifecycle | `embodichain/lab/sim/motion/execution.py` |
| Expansion Profile loading, source adapters, candidate coordination and coverage | `embodichain/lab/sim/motion/expansion/` |
| Distinct trajectory variants for fixed waypoints | `embodichain/lab/sim/motion/expansion/variants.py` |

## Choose the layer

`BasePlanner` consumes waypoints and returns timed trajectories.
`MotionGenerator` is the stateful facade for strategy selection, batched IK,
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
backend; `ik_interp` requires explicit start/count/dt, rejects backend-specific
options, consumes optional scalar velocity/acceleration limits, and supports
joint/EEF moves. Limited interpolation uses minimum-jerk rest-to-rest segments
and treats the requested count as a lower bound. Required Cartesian IK
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

The internal Cartesian profile adapter accepts caller-owned endpoint poses and
optional Bézier geometry, times distance before batched IK, and derives joint
references using the root-frame Jacobian. Its pure derivative and joint-wrap
calculations live in `compute/kinematics/_path_derivatives.py`; tests target those
owners rather than tutorial functions. Tutorial geometry, CLI and plots remain
sample-owned. This adapter is not a new planner backend or a collision/joint-limit
compliance certificate.

`retime_to_control_grid()` maps source trajectories onto a fixed command clock
without shortening their durations and recomputes velocity references.
[Planner details](planner-details.md#retiming-and-playback) describe the execution
boundary. Standalone `play_joint_trajectory()` advances unchanged physics
substeps; Gym experts use the environment-owned command period and
[expert execution contract](../env-framework/env-framework.md). Dataset writers
encode prepared actions rather than retiming a second time.

## Trajectory augmentation boundary

Scene layout proposals and their Task Program execution adapter belong to
[scene expansion](../scene-expansion/scene-expansion.md). The trajectory
contracts below operate within a specified scene and initial state.

`motion/expansion/` owns callable-free Expansion Profiles, immutable templates
and candidates, sampled limit checks and coverage. `load_expansion_profile()`
strictly decodes the task-local resource, separate from environment
and Task Program deployments. `SourceAdapter` converts handwritten, planner or
grounded Atomic sources; `CandidateCoordinator` expands, deduplicates and queues
them. `ExpansionSession` owns identities, randomness, budgets,
reservations and receipts. None owns physical slots, environment lifecycle or
persistence; host integrations own those boundaries. Keep expansion algorithms
free of direct Gym imports.

The `expansion.combined` contracts extend this boundary to episode-level
expansion: they decode the combined profile, enumerate reference-family,
affordance and trajectory recipes, assign one visual profile per episode, and
reserve generic compatible slots. `schedule_digest` provides deterministic
rerun identity; restoration, measured acceptance, and artifact writing remain
host responsibilities. A profile that declares a conservative
`trajectory.spatial.joint_offset_scale` also names a smaller
`trajectory.spatial.dedup_tolerance`, which the Task Program integration
forwards as the augmentation coverage tolerance; omitting it keeps the contract
default.

The generic B=1 host orchestration and its restore/executor/sink ports live in
`motion/execution.py`, alongside standalone playback. The runner drives injected
host ports and applies `ExpansionSession` transitions; it does not move those
state or coverage contracts out of `motion/expansion/`.

Motion-limit checks are not collision/task-success certification.
`rotate_grasp_about_object_axis` and `perturb_approach_direction` change TCP
candidates around a fixed object axis or approach cone; both emit Cartesian
poses that callers must replan, and neither certifies the contact. Grasp
grasp generation itself belongs to `embodichain.toolkits.graspkit`, composed by
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
geometry/timing and counts every rejection under a key naming its reason. Only
`ProposalRejected` is a rejection; malformed arguments stay plain `ValueError`
and propagate rather than hiding in a rejection count.
Configuration is optional: omitting it resolves `default_variant_factors`,
which enables every implemented factor the inputs support and drops the ik
factor when no Jacobians are supplied. An explicit config is never overridden, and an
explicitly enabled factor that cannot produce a qpos variant raises rather than
disappearing: `ik` without Jacobians, and `approach`, which owns Cartesian
standoff poses the caller must replan. The resolved config is reported on the result.

`scripts/tutorials/atomic_action/place.py` is the demonstration host, hooked
the way the Affordance sampling tutorials are: shared helpers live in
`examples/sim/atomic_action/support.py` through the `tutorial_utils.py` adapter;
the tutorial declares only which named segments may move,
and one variant replays per simulation row. Phases come from the action's own
`TrajectorySegment` ranges; only motion at or after the lift is retimed so the
shared `clear_dynamics()` step stays aligned. `--variant_plot_dir` writes a
joint-trajectory figure and a tool-path figure with waypoint markers. Task
Program, Gym lifecycle, dataset persistence and `ExpansionSession` collection
bookkeeping are deliberately out of scope; this is a direct-simulation
contract. Keep `joint_dedup_normalized_tol` below `joint_offset_scale`, or
genuinely different variants are discarded as duplicates; an enabled spatial
factor at or below the tolerance now fails configuration decoding instead of
silently keeping the nominal variant alone.

The twelve-scene Atomic CLI uses the same variant operators through the
shared Atomic plan adapter and owns its replay provenance; see the
[Atomic host boundary](../atomic-actions/atomic-actions.md). It does not extend
the Gym Task Program collection lifecycle.

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
collection budget; and `ExpansionSession` classifies bands from the measured
`manipulability` observation rather than any planned score. Band guidance ranks
postures only: path, dynamic and task validation stay separate.

## Focused validation

The installed benchmark host lives in `embodichain_benchmarks.motion_generation`.
`run_benchmark()` accepts a packaged suite, external YAML or assembled `SuiteCfg`,
optional explicit planner specs and a run-local adapter mapping. It copies caller
configuration; explicit planners replace the suite list and are enabled.
`PlannerAdapter` owns build/prepare/plan/close and returns `PlanResult` for
free-space evaluation; Atomic Skills also require an exposed `MotionGenerator`.
The host owns simulation lifetime, seeded cases, evaluation, and artifacts.
Canonical case tensors and nested parameters stay independent of adapter
mutations through untimed copies at capacity, preparation and plan boundaries.
Free-space Cartesian populations, bounded FK-only numerical reachability screening,
measured difficulty and batch semantics live in
`scenarios/_free_space_sampling.py`. Optional mask projections reuse screened
Cartesian bases, retain `base_case_id`, and share independent maximum-joint/constrained
matching in `metrics/_constraints.py`. Homogeneous Joint/FullPose cases keep
classical input contracts; partial/mixed cases require `masked_constraints`.
Validation preserves native samples with deterministic joint subdivisions.
`NeuralPlanOptions.waypoint_modalities` supplies explicit meanings, and
position-only goals require coincident runtime/policy TCP origins. The public
API accepts a hashed `case_manifest` to replay fixed inputs without regeneration;
scenario `bind_cases()` restores runtime bindings for both generated and loaded
inputs, including Atomic skill providers and grasp resources. Experiment owners
keep repeated timing subsets out of quality denominators. The
configuration retains explicit legacy joint templates. Reports preserve workload strata and provide seed-cluster intervals at both
condition and workload levels; balanced geometry allocations must not hide the
independent seed count. Measured difficulty partitions batch rows by conditioning
and joint margin separately. The screener's solver and selection condition stay
explicit; decoupling runtime IK does not remove solver-conditioned sampling.
Distances can use an explicit robot-reach reference. Waypoint layouts do not
promise path following; sampled polyline deviations and native start-error
quantiles remain diagnostics.
Geometry validation includes the requested initial joint state. Atomic providers
declare gripper requirements; typed pose settings flow through generation rather
than being expanded into private configuration keys. Built-in Atomic suites
compile and replay on their declared command clock; compiled replay consumes
arrival intervals on physics ticks and preserves observation segment indices.
Legacy fixed-sample replay remains explicit in trial metadata. The benchmark
README and design contract own population and protocol details.
Run calls serially within a process. Suite resources ship in the wheel; the
installed CLI dispatches through `embodichain/cli/benchmark.py`. Source launchers
under `scripts/benchmark/` only forward motion-generation evaluation.
`tests/benchmark/motion_generation/` covers contracts and lifecycle, while
`tests/test_benchmark_package.py` builds and installs the wheel outside the checkout.

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
