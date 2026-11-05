# Motion Planning

`MotionGenerator.generate()` owns sampled Cartesian segment recovery through its
private `_cartesian.py` helper. Explicit linear keyframes use fixed-grid sequential
IK; `preserve_cartesian_samples=True` retains the separate pre-sampled contract.
Segment alternatives are opt-in and B1-only. Their local counts and continuity
receipt live in `PlanResult.recovery`, separate from complete-action retries.
Dense attempts advance the original RNG stream; continuous IK uses the original
grid and its existing Torch fork. Exhaustion retains the first failed result and
candidate evidence. `defer_recovery=True` lets a caller inspect the nominal result.
Each actual generation attempt charges the existing host budget once; the public
entry checks cancellation before FK and adds no planning charge of its own.
Atomic skills own which segment enables recovery; Task Engine declares policy.

## Find the owner

| Change or question | Start here |
|---|---|
| Planning interface, options and batch validation | `embodichain/lab/sim/motion/planners/base_planner.py` |
| Timed inputs/results | `embodichain/lab/sim/motion/planners/utils.py` → `PlanState`, `PlanResult` |
| Strategy, IK preparation, result normalization | `embodichain/lab/sim/motion/motion_generator.py` → `MotionGenerator` |
| Fixed-time continuous IK and Cartesian target sampling | `embodichain/lab/sim/motion/ik.py`; numeric branch search in `compute/kinematics/ik_path.py` |
| Explicit search measurements and cooperative deadlines | `embodichain/lab/sim/motion/search.py` |
| Joint-path timing | `embodichain/lab/sim/motion/planners/toppra_planner.py`, `trapezoidal_planner.py` in that directory |
| Cartesian geometry and scalar timing | `embodichain/lab/sim/motion/planners/se3.py`, `bezier.py`, `_scalar_time_law.py`, `_blend_constraints.py` in that directory |
| Sampled Cartesian geometry → timing → robot IK | `embodichain/lab/sim/motion/planners/_cartesian_profiles.py` (internal adapter) |
| Discrete full-robot collision queries and owned query scopes | `embodichain/lab/sim/motion/mesh_collision.py` |
| Bounded empty-arm joint clearance | `embodichain/lab/sim/motion/joint_clearance.py` |
| Collision planning and scene conversion | `embodichain/lab/sim/motion/planners/curobo/` |
| NMG policy rollout and export metadata | `embodichain/lab/sim/motion/planners/neural_planner.py` |
| Installable evaluation suites and external planner adapters | `extensions/embodichain_benchmarks/motion_generation/api.py`, `planners/base.py` in that package |
| Pure interpolation, resampling and retiming | `embodichain/compute/trajectory/` |
| Contact-qualified paired grasp sampling and explicit URDF surfaces | `embodichain/toolkits/graspkit/pg_grasp/coordinated.py`, `urdf_surfaces.py` in that directory |
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

Native convex queries reuse the shared isolated Newton collision pipeline.
`lab/sim/scene/_collision_backends.py::_require_newton` checks availability before
capture; no FCL extra or backend fallback is selected. Distance diagnostics are
capped intervals under the query precision policy. A threshold inside an interval
remains unavailable, never a fabricated contact or free path. CPU comparisons
remain separate from GPU and physical-task qualification.

`ik_branches.CpuIKBranchSearch` owns the explicit B1 CPU endpoint/continuation
search, with fixed seed order, separated branch selection and original residual
rules. `capture` reads the configured root once; a supplied-root constructor
keeps geometry reads caller-owned. Its legacy CPU-only Torch fork plus
`manual_seed(0)` also reseeds visible CUDA streams. Do not claim CUDA RNG isolation
or change that behavior as part of ownership migration. Numeric branch selection
and pose residuals live in `compute.kinematics.seed_selection` and
`compute.geometry.pose_checks`.

`goal_seeded` supplies bounded endpoints and distinguished-goal/common-seed pose
paths. Callers supply mandatory poses, source time samples and same-observation
state. Motion owns solver/FK masks and limit qualification; Atomic Skills own
phase composition, retry choices and physical effects. `solvers.tolerances` reads
explicit full-pose solver budgets without replacing them with task thresholds.

`cpu_threads.cpu_planning_threads` is an explicit isolated-worker scope, never a
shared default. It captures capacity before overrides and restores process-wide
Torch settings on exit; `candidate_ik_threads` uses that capacity only for
eligible CPU PytorchSolver pose products. Hosts serialize those process-wide
changes and retain lifecycle ownership.

`RevolutePathQueryFactory` prepares original URDF collision inputs for one exact
Default/B1 robot, articulation and support. It borrows native owners and accepts
an explicit complete motion-root subtree; it never creates a simulator, task,
skill or planning context. Fresh `NativeQueryScope` queries bind supplied root,
parent/child and measured-joint observations. Source/model/frame/owner drift
rejects without recapturing a replacement baseline. Closing releases query
resources, never borrowed Robot/native objects.

`RevolutePathReport` keeps unsupported acquisition distinct from an accepted or
rejected supported screen. `RevoluteDependencyReport` describes the preserved
source-input command-frame proxy and conservative sweep qualification, not
native cooking, continuous collision or physical effects. The original FK batch,
sampling order and collision domains remain with this Motion query. Scene's
`revolute_setting_attributes` only exposes authored fixed-base setting identities;
it chooses no Semantic Call or recovery policy.

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

`interpolation_mode="fixed_grid"` explicitly preserves the requested count and
arrival intervals. It requires absent scalar velocity/acceleration overrides,
intersects runtime speeds with every controlled joint's authored URDF cap, and
rejects overspeed rows without retiming. `ik_seed_mode="sequential"` preserves
previous-valid-posture seeds, callback order and RNG consumption; batched IK and
adaptive timing remain defaults. Non-default modes require `ik_interp`.
Failed execution rows hold their start. `PlanResult.rejected_candidate` retains
owned original positions, intervals and rejected-row identities for retries and
audit; checking the zero-speed hold is not evidence that the candidate passed.

`compute.trajectory.interpolation.interpolate_velocity_grid` allocates fixed
B=1 joint-edge intervals from explicit speed caps. It preserves ordered mandatory
endpoints and requested sample count, rejecting insufficient duration instead of
retiming. The `limits.joint_velocity_diagnostics` companion reports the worst
frame/joint operands with nonfinite values represented as None; it does not
replace or weaken the existing validity mask. Ordinary interpolation remains
unchanged unless callers explicitly choose the fixed-count helper.

The bounded `resolve_continuous_ik_path()` adapter preserves a single path's
time grid and restores the caller's random stream. Its numerical seed search
accepts only B=1 and performs no simulation or state mutation.
Treat perturbed IK seeds as numerical hints, not executed waypoints: velocity
checks compare each candidate to the actual preceding posture. Search ledgers
share an outer caller-owned deadline across nested previews and isolate separate
contexts. They do not replace `_budget.py`'s existing automatic planning hooks,
preempt native work, or certify physical task success.

Atomic policy propagation keeps whole-request sample alternatives separate from
MotionGenerator's segment alternatives. Required final speed
qualification checks emitted full-robot samples after transforms/recovery; it
never treats zero-speed rejected holds as original candidate evidence. The
ordinary automatic planning/cancellation hooks remain on AtomicAction.plan.

Normalization must preserve the meaning of timing, derivatives, failure rows
and backend diagnostics together. Read the
[result contract](planner-details.md#normalization-and-capabilities) before
changing interpolation or output sample counts. Final sampled collision checks
do not establish continuous collision freedom or derivative compliance.

Joint scene/deployment searches install a context-local cooperative call budget
through `motion/_budget.py`. Robot IK, motion generation and Atomic planning
boundaries share that host budget, including retries. See
[scene feasibility](../scene-expansion/feasibility.md) for quota and evidence scope.

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
add a second hand-specific conversion. GraspKit owns collision-qualified end-region
selection through `pg_grasp.regions`; callers select a policy without supplying
a Task Engine implementation. Explicit `pg_grasp.constraints` and `opening`
views filter complete object geometry after ordinary ranking and preserve source
order and infinite-cost failure padding. `preshape` consumes captured numeric
hand sweeps; `proposals` defers final-TCP qualification without bypassing required
ground filtering. These APIs import without a Robot, simulation or target
decomposition backend; `pg_grasp` resolves concrete implementations lazily. Nominal dimensions
in `hand_envelope` remain separate from physical collision geometry.
The original mesh-region path remains the default. `pg_grasp.contact_fit` owns
sampled closing geometry and contact fitting with explicit thresholds; robot/FK
sampling and physical contact evidence remain Lab responsibilities.
`pg_grasp.native_contact` owns discrete convex closing/opening and rigid-clearance
qualification through an injected `ConvexMeshQuery` factory; it imports no Lab
runtime. `ClosingContactPolicy` declares geometric rules without experiment
revision dispatch. This exact-shape route retains native pair offsets and contact
ordering; it does not replace the surface-based checker. Scene captures physical
geometry, and `ActionPlanningServices` bind an explicitly supplied geometric
policy to the existing Newton backend. Queries own only isolated collision resources and preserve unavailable
threshold results; hosts retain simulation ownership and physical-effect evidence.
`SampledHandGeometry.at_width()` interpolates the supplied hand samples.
The public `get_dual_arm_grasp_proposals()` path retains candidate generation
and ranking while leaving geometric qualification to the caller; normal
validated queries retain their existing checker. `pg_grasp.coordinated` composes
these proposals with `HandContactChecker.refine()` and shares one target-surface
query across both hands for the same mesh identity/version. It owns bounded
paired sampling with explicit contact settings
and optional observed world bounds; clearance preference changes costs after
population selection. A proposal `role` skips unused pose/NMS/ranking work while
preserving both sides' original direction draws and the selected population.
Raw/validated backends share read-only annotations, with separate collision state.
Mesh mutation retires superseded versions; hosts call `clear_cache()` between
unrelated asset cohorts to release distinct-mesh queries/backends while keeping
sampled hand geometry. `pg_grasp.urdf_surfaces` samples explicitly selected
collision links without robot state, sharing shape/origin decoding with fixed
geometry while retaining each sampling policy's original point population. See
[hand geometry ownership](../robot-system/robot-system.md#sampled-hand-geometry)
for deferred Robot capture and cache lifetime, and the
[GraspKit contract](../../../embodichain/toolkits/graspkit/README.md)
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
Mounted-hand selection must agree with command joint order and contact links.
PGI and Robotiq collision samplers bind the actual flange-relative solver TCP;
the benchmark README owns the hand-specific fixture and suite selections.
Legacy fixed-sample replay remains explicit in trial metadata. The benchmark
README and design contract own population and protocol details.
Physical grasp/retention scoring consumes synchronized, target-filtered finger
contacts rather than gripper joint-position proxies. Explicit `finger_links`
belong to benchmark gripper settings; missing bindings or truncated queries are
unavailable evidence. The collector and target/role binding live in
`scenarios/_grasp_contacts.py`; contact geometry, distance conventions and current
scoring version are specified in the benchmark README and `evaluation.py`.
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
