# Motion Planning

Motion owns trajectory proposals and timing. Execution, physical acceptance and
collection belong to their hosts.

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

`BasePlanner` returns timed trajectories; `MotionGenerator` composes strategy,
IK, backends, normalization and named-arm requests. Import the facade from
`motion.motion_generator`, not `motion.planners`. Motion parent/sibling imports
stay lazy to protect Robot initialization.

Planner construction selects the manager through `sim_instance_id`; it cannot
fall back to another manager with the same robot UID. Environment factories
copy configurations before binding their own manager.

Read [planner details](planner-details.md) for backend capabilities, limits,
normalization and bounded Cartesian recovery; [collision worlds](collision-worlds.md)
for obstacle identity and query lifetime; [IK contracts](../ik-solvers/ik-solvers.md)
for candidate frames and shapes. Atomic Skills select execution/recovery policy.
Native convex queries use the isolated Newton pipeline and preserve unavailable
threshold intervals; they do not switch silently to another backend.

## Timing and batch boundary

Planning preserves an environment axis: qpos `(B, dof)`, TCP `(B, 4, 4)`, output
positions `(B, N, dof)`. Use `PlanState.single()` for one environment.
`PlanResult.dt` is `(B, N)` per-sample arrival intervals whenever positions exist;
duration is their sum. Failed results may omit positions. Planning timing never
changes simulator `physics_dt`.

`MotionGenOptions.waypoint_modes` explicitly selects joint/Cartesian paths to
individual keyframes under fixed-grid sequential IK. Adjacent equal modes share
ordinary native requests; composition retains the total control grid and row
failure masks. No robot UID or waypoint count enables the behavior. Atomic
`MotionPolicy.phase_paths` names the PickUp/Place phases that opt into this path;
ordinary calls and other primitives retain their own planning semantics.

`motion_gen` delegates timing to its backend. `ik_interp` needs explicit
start/count/dt, rejects backend-specific options and uses minimum-jerk segments
when limits apply. Adaptive timing and batched IK remain defaults. Explicit
`fixed_grid` preserves count/intervals and rejects overspeed rows against runtime
and authored joint caps; `sequential` retains previous-valid-posture seeds and
callback/RNG order. Neither mode is a generic option for other strategies.

Rejected rows hold their start; `rejected_candidate` retains original samples
and rejected-row identity for audit/recovery. A zero-speed hold does not prove
that its candidate passed. Required IK failures fail the row instead of dropping
waypoints. Normalization must preserve timing, derivatives, row validity and
backend diagnostics together. Sampled checks do not certify continuous collision
freedom or physical success.

`search.py` and `_budget.py` retain caller-owned cooperative deadlines and
planning charges across nested work. They do not preempt native calls. Complete
Atomic-request retries remain distinct from opt-in B=1 segment alternatives;
see [Cartesian recovery](planner-details.md#cartesian-recovery).

## Compute and execution ownership

`compute.trajectory` owns pure interpolation, differentiation, retiming and
warping; compute cannot import simulation objects. New consumers use that
package. `action_utils` retains solver-dependent adaptation and compatibility
exports. Cartesian profile geometry and joint derivatives stay in their
existing planner/compute owners, not tutorials.

`retime_to_control_grid()` rounds duration up to the command grid without
shortening it and recomputes velocity references. Standalone playback advances
unchanged physics substeps; Gym experts use their environment's command period. Writers encode
prepared actions without retiming again. See
[retiming and playback](planner-details.md#retiming-and-playback).

GraspKit owns sampled/native geometric qualification from explicit hand inputs;
Scene captures geometry and Robot adapters provide FK/calibration. GraspKit
performs no hardware lookup. Sampling uses the contact frame and maps to runtime
TCP once. Read the [GraspKit contract](../../../embodichain/toolkits/graspkit/README.md)
and [hand-capture ownership](../robot-system/robot-system.md#sampled-hand-geometry)
for policies, caches, offsets and finite-check limits.

## Trajectory augmentation boundary

[Trajectory expansion](trajectory-expansion.md) owns templates, operators,
permissions, deduplication and coverage within an explicit scene/initial state.
Its algorithms remain host-independent. [Expansion Engine](../expansion-engine/expansion-engine.md)
owns cross-domain collection; [scene expansion](../scene-expansion/scene-expansion.md)
owns layout proposals and host integration. [Real2Sim](../data-pipeline/real2sim.md)
owns protected source mapping and retiming.

The installed benchmark host is `embodichain_benchmarks.motion_generation`.
It owns scene lifetime, frozen cases, evaluation and artifacts; adapters return
`PlanResult` and expose `MotionGenerator` for Atomic evaluation. Configuration
preflight and runtime capability declarations remain separate. Frozen inputs do
not guarantee identical stochastic outcomes across schedules. Parallel quality
work uses spawned processes; performance replay stays serial and excludes
contended timing. Population, masked-constraint, contact-scoring and uncertainty
rules live in the [benchmark README](../../../extensions/embodichain_benchmarks/motion_generation/README.md)
and [design contract](../../../extensions/embodichain_benchmarks/motion_generation/BENCHMARK_DESIGN.md).

## Focused validation

| Boundary | Coverage |
|---|---|
| Imports, facade and result normalization | `tests/sim/motion/test_motion_imports.py`, `test_motion_generator.py`, `test_motion_generator_batched.py` |
| Planner behavior and pure timing | `tests/sim/motion/planners/`, `tests/compute/test_trajectory.py`, `test_trajectory_timing.py` |
| Playback and expansion | `tests/sim/motion/test_execution.py`, `tests/sim/motion/expansion/` |
| Benchmark protocols and installed resources | `tests/benchmark/motion_generation/`, `tests/test_benchmark_package.py` |

For stale obstacles inspect identity/capabilities; for duration or acceleration
changes inspect normalization and the destination clock before tuning limits.
Tests target production owners rather than importing tutorial implementations.
