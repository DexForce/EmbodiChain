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
| Bounded interaction pose refinement before expansion | `embodichain/lab/sim/motion/refinement.py`, with numeric fitting in `compute/trajectory/refinement.py`; [Real2Sim boundary](../data-pipeline/real2sim.md) |
| Standalone playback and B=1 expansion host lifecycle | `embodichain/lab/sim/motion/execution.py` |
| Expansion Profile loading, source adapters, candidate coordination and coverage | `embodichain/lab/sim/motion/expansion/` |
| Distinct trajectory variants for fixed waypoints | `embodichain/lab/sim/motion/expansion/variants.py` |
| Endpoint-preserving task-space deformation and trajectory IK | `embodichain/lab/sim/motion/expansion/task_space.py` |
| Phase-aligned demonstration banks, stitching and bridges | `embodichain/lab/sim/motion/expansion/stitching.py` |

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

For Atomic case preparation, preserve the original batched grasp-pool RNG scope.
An explicitly configured finite position search reuses that pool; an approach
search constructs one original batched pool per direction. Both freeze successful
rows independently, while the public grasp screen still raises for unresolved
rows. Per-row sampler resets and hard approach filtering before NMS/ranking can
change the selected full-pose population. Freeze the chosen grasp, direction and
position before evaluating planners, and review population fingerprints when the
selection law changes. See the benchmark package README for the case options;
preparation diagnostics and incomplete corpora are not formal quality results.
MoveHeldObject/Place may explicitly condition goal-local streams on complete
reference endpoint evidence with `target_candidate_count`; freeze each accepted
row and retain rejected goals. This changes the population, requires reviewed
fingerprints, and supplies no path or physical-success certificate.

## Trajectory augmentation boundary

Real2Sim source templates and protected mapped retiming integrate with this
boundary; their control-grid, source-permission and host limits are documented
in [Real2Sim motion](../data-pipeline/real2sim.md).

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
poses that callers must replan, and neither certifies the contact.
Grasp generation belongs to `embodichain.toolkits.graspkit`, composed by Atomic
Skills/Task Program rather than embedded in `MotionGenerator`. Its antipodal
service requires explicit URDF collision geometry at the caller's open joint
state. Builtin asset/TCP mappings live in `pg_grasp/model_geometry.py`; custom
hands provide their own geometry. Sampling/collision stays in the canonical
contact frame, and public poses map once to the declared runtime TCP. Do not
add a second hand-specific conversion. See the [GraspKit contract](../../../embodichain/toolkits/graspkit/README.md)
for geometry ownership and the finite collision-check scope.

`variants.py` is the common candidate-generation path for joint-space,
task-space and manipulability-guided expansion; `CandidateCoordinator` supplies
the same host-owned kinematics callable to these factors. Its qpos operators
(`joint_residual`, `via_points`, `nullspace_residual`, `retime` profiles) vanish with
zero derivative at every phase endpoint and never touch `contact`/`hold`
phases, so annotated waypoints stay exact. Joint-limit rejection covers only
the joints an operator moved; an observed reference can hold an untouched joint
microradians outside its range, and checking it would reject every proposal.
`nullspace_residual` holds the caller's declared task rows to first order only;
the caller owns the Jacobian frame and column order, and a fully constrained
task raises. At most one joint-path operator runs per proposal, though `spatial.method`
may name several so each becomes its own variant; a single name is still
accepted. A task-space proposal projects before joint edits and re-projects
after them; null-space edits use Jacobians of the projected path. Retiming runs
last. Reference FK and guidance normalization are cached per expansion call.
`expand_trajectory_variants` deduplicates one fixed scene on measured
geometry/timing and counts every rejection under a key naming its reason. Only
`ProposalRejected` is a rejection; malformed arguments stay plain `ValueError`
and propagate rather than hiding in a rejection count. `max_attempts` bounds all
numerical draws, including guidance's inner proposals; valid unselected draws
are counted as `guidance_unselected`. The nominal branch remains unchanged.
Configuration is optional: omitting it resolves `default_variant_factors`,
which enables the joint-path factors the inputs support and drops the ik
factor when neither Jacobians nor kinematics are supplied. Task-space and
manipulability guidance remain explicit opt-ins. An explicit config is never overridden, and an
explicitly enabled factor that cannot produce a qpos variant raises rather than
disappearing: `ik` without Jacobians or kinematics, task-space/guidance without
kinematics, and `approach`, which owns Cartesian
standoff poses the caller must replan. The resolved config is reported on the result.

`task_space.py` is the host-independent route when fixed joint waypoints are too
restrictive. It evaluates one controlled frame through an injected batched
kinematics callable, applies bounded root-frame translation/rotation residuals
only to explicitly authorized free phases, and preserves each phase endpoint.
`project_task_space_trajectory` solves all editable interior samples with the
shared fixed-step DLS compute API and returns a normal full-joint
`TrajectoryTemplate` plus per-sample pose residuals and success evidence.
Locked samples keep the reference qpos. Task-space permissions must be present
globally and on each editable phase. Conversion only grants new permissions
when the caller explicitly names `editable_phase_ids`; segment extraction
retains the intersection of source template and phase permissions.
The caller still owns frame binding,
collision/dynamics checks, task validation and rejection of unsuccessful rows.

`stitching.py` extracts annotated phases as source-owned, local-time
`TrajectorySegment` values. `TrajectorySegmentBank` admits replacements only
when semantic phase, motion kind, joint layout, controlled joints and tensor
contracts agree; selection is deterministic. Stitching de-duplicates shared
boundaries while adjacent output phases share that one endpoint index, so every
phase remains independently extractable and augmentable. It gives every output
phase a unique identity and retains multi-source lineage. A discontinuous
boundary must go through an injected
`TrajectoryBridgePlanner`; `MinimumJerkBridgePlanner` is only a sampled
velocity/acceleration-limited geometric fallback and makes no collision claim.
The resulting template enters the existing `TemplateSourceAdapter` path, so
execution, reset, validation and persistence remain host responsibilities.
`examples/expansion/trajectory/pure_trajectory_augmentation_showcase.py`
renders both paths analytically on CPU and records IK residuals, source lineage
and bridge phases without constructing a simulator.
`trajectory_augmentation_visualizer.py` is the CPU-only browser overview for
the implemented trajectory operators: Python calls the production APIs and
embeds their candidates in a self-contained HTML page. The page uses the
official Atomic Action example's seven-DoF Franka Panda serial-chain geometry
for orbitable 3D robot and EEF-path playback. A fixed task object and placement
target clarify pick-and-place phases without participating in augmentation;
JavaScript owns playback, drawing, and light/dark/system theme selection using
the documentation's stored preference. The checked-in generated page lives
under
`docs/source/labs/trajectory-augmentation/`; Sphinx copies that directory into
each versioned documentation build through `html_extra_path`, so the same page
is published by the existing GitHub Pages workflow. It deliberately excludes
affordance, contact-point, observation and scene augmentation.

`examples/sim/atomic_action/place.py` is the demonstration host. Shared helpers
in `examples/sim/atomic_action/support.py` convert the compiled action's named
segments to the expansion contracts and replay one variant per simulation row.
The default `joint_space` mode retains the earlier residual/via/null-space and
retiming behavior. The optional `task_space` mode deforms only selected free
lift/transport/retract segments, projects them through the production
trajectory-IK API, and requires every requested row to be distinct and IK-valid.
`examples/expansion/trajectory/task_environment_augmentation_showcase.py`
selects that mode and defaults to one nominal plus eight variants in nine
parallel arenas; recording frames the complete arena grid. `--variant_plot_dir`
writes joint and tool-path figures with waypoint markers. Task Program, Gym
lifecycle, dataset persistence and `ExpansionSession` collection bookkeeping
remain out of scope; this is a direct-simulation contract. In joint-space mode,
keep `joint_dedup_normalized_tol` below `joint_offset_scale`, or genuinely
different variants are discarded as duplicates.

For joint-space replay, only motion at or after the lift is retimed so the
shared `clear_dynamics()` callback stays aligned. Enabled joint residual
scales at or below the deduplication tolerance fail configuration validation.

The twelve-scene Atomic CLI uses the same variant operators through the
shared Atomic plan adapter and owns its replay provenance; see the
[Atomic host boundary](../atomic-actions/atomic-actions.md). It does not extend
the Gym Task Program collection lifecycle.

`expansion/manipulability.py` profiles posture conditioning through
`embodichain.compute.kinematics.yoshikawa_manipulability`. Callers supply the
Jacobians, so the module stays host-independent. `ManipulabilityBands`
normalizes scores against a reference bottleneck held per initial state, and
`manipulability_guided_residual` keeps whichever of several `joint_residual`
draws from one local generator lands nearest a requested band. The shared
selector also accepts task-space and combined proposals from `variants.py`.
The factor is opt-in through `augmentation.factors.manipulability` and disabled
by default. `guided_proposals > 1` enables bounded guided sampling;
`guided_proposals == 1` retains coverage-only behavior.
Once enabled, `register_case` requires a positive `manipulability_reference`
for each initial state, which may differ within one case; `CoverageIndex`
enforces a per-band quota so one well-conditioned posture cannot absorb the
collection budget; and `ExpansionSession` classifies bands from the measured
`manipulability` observation rather than any planned score. Band guidance ranks
postures only: path, dynamic and task validation stay separate. The coordinator
uses `ExpansionSession.manipulability_reference(case)` for proposal guidance so
its normalization agrees with the registered initial state. Selection scores,
actual bands and scored-proposal counts remain candidate metadata.

## Focused validation

The installed benchmark host lives in `embodichain_benchmarks.motion_generation`.
`run_benchmark()` accepts a packaged suite, external YAML or assembled `SuiteCfg`,
optional explicit planner specs and a run-local adapter mapping. It copies caller
configuration; explicit planners replace the suite list and are enabled.
`PlannerAdapter` owns build/prepare/plan/close and returns `PlanResult` for
free-space evaluation; Atomic Skills also require an exposed `MotionGenerator`.
`registry.planner_configuration_features()` loads built-in registrations without
constructing backends and reports supported configuration semantics for preflight.
Runtime request support remains the adapter instance's `capabilities` contract.
The host owns simulation lifetime, seeded cases, evaluation, and artifacts.
Canonical case tensors and nested parameters stay independent of adapter
mutations through untimed copies at capacity, preparation and plan boundaries.
Free-space Cartesian populations, bounded FK-only numerical reachability screening,
measured difficulty and batch semantics live in
`scenarios/_free_space_sampling.py`. Optional mask projections reuse screened
Cartesian bases, retain `base_case_id`, and share independent maximum-joint/constrained
matching in `metrics/_constraints.py`. Homogeneous Joint/FullPose cases keep
classical input contracts; partial/mixed cases require `masked_constraints`.
The TOPPRA and cuRobo adapters expose that capability only when their explicit
`config.masked_constraints` opt-in is enabled, preserving historical populations.
TOPPRA uses sequential component IK; cuRobo changes native terminal pose criteria.
Neither consumes inactive pose payloads or Cartesian reference joint witnesses.
See [planner details](planner-details.md) for component and arm-frame contracts.
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
than being expanded into private configuration keys. Object antipodal screening
samples the pose batch once, then freezes each row's first aligned, independently
IK-feasible candidate from its own pool; AxisAlign ranks against each row's
rotation axis. Failed chains do not truncate other rows or change their choices.
Built-in Atomic suites
compile and replay on their declared command clock; compiled replay consumes
arrival intervals on physics ticks and preserves observation segment indices.
Legacy fixed-sample replay remains explicit in trial metadata. The benchmark
README and design contract own population and protocol details.
Run host calls serially within a process. `run_benchmark(workers>1)` isolates
quality shards with spawned processes after freezing the full input corpus;
`prepare_case_manifest()` can prepare that corpus without running planners.
Frozen B=1 Free-space NMG-only replay optionally uses `quality_batch_size` to
pack compatible requests into a larger simulator. `_quality_batching.py` owns
packing, valid-prefix recovery and explicit environment-row FK mapping; the
runner validates original cases separately and preserves their manifest and
trial identities. Raw endpoint derivatives are recomputed after trimming, and
retimed zero-dt holds are excluded. A single batch measurement is metadata,
not a per-case latency. The option composes with spawn workers; Atomic,
traditional backends and performance replay retain their declared batches.
The parent validates complete case/method/repeat evidence and recomputes
aggregates from raw trials. Parallel raw timing is marked contended and excluded
from timing reports; performance replay stays serial. Planner RNG is
backend-owned, so frozen-input pairing does not guarantee identical stochastic
solver outcomes across schedules. Worker thread limits also bound ONNX Runtime
through the host Torch intra-op setting. Suite resources ship in the wheel;
the installed CLI dispatches through `embodichain/cli/benchmark.py`. Source
launchers under `scripts/benchmark/` forward motion-generation evaluation.
`BenchmarkRunner.run_paired_arm_performance()` delegates to
`_paired_arm_performance.py` for serial replay of independent two-arm FullPose
requests captured by a declared TOPPRA reference before candidate measurement.
Its immutable cache binds source cases, suite protocol and embodiment;
candidate inputs are copied per call. Classical facade fallback stays sequential,
while capable NMG configurations additionally report packed dispatch and parity.
Scoring and a separate warmed memory call sit outside synchronized latency.
Request replicas do not enter task-quality counts, and native pair success does
not certify compiled timing or physical execution. The benchmark README owns
usage, measurement scope and artifact interpretation.
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
