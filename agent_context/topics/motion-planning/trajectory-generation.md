# Fixed-scene trajectory generation

[Motion planning overview](motion-planning.md).

`rotate_grasp_about_object_axis` rotates a reference TCP pose about a fixed
object-local axis through the object origin. The caller chooses geometry-valid
angles and replans the resulting pose candidates; the operator neither moves
the object nor certifies the grasp. A cube's quarter-turn symmetry is used by
`examples/sim/motion/trajectory_generation/cube_grasp_parallel.py` to compare
0/90-degree grasp orientations crossed with reference/residual transit paths.
Four physics rows execute together using existing MoveEndEffector/PickUp plans;
only the pre-grasp transit allows `joint_residual`. The example saves a tiled
camera MP4 with measured TCP trails, `rollout.npz`, and `report.json`. Physical
acceptance checks sustained lift and TCP-relative position stability. It uses
IK interpolation and does not claim contact-aware collision validation or
LeRobot expert qualification; the separate `cube_pickup_collection.py` performs
that bounded collection workflow.

`lab/trajectory_generation/runner.py` owns the synchronous qpos job for
handwritten free motion and offline atomic PickUp exports. It resolves supplied host/planner/executor/sink policy identities,
reserves bounded rollout capacity through `GenerationSession`, and restores the
full batch only when ready candidates need another round. The concrete
`QposRolloutExecutor` uses normal Gym demo/controller ports or one pure-sim
step owner, captures actual submitted targets and T+1 observations, and checks
the manager's actual control-time increments. Measured qpos is rechecked for
collision, dynamics, reference-relative length/duration, and task success.
Only `LeRobotEpisodeSink` receipts confirmed after sealed readback update
committed counts and coverage; the runner closes its host/sink and writes
`generation_report.json` on completion or failure.

`integrations/planning.py` exposes `EEFPath` and `EnvRowMotionPlanner`: logical
candidate count C is scheduled over real physical B rows, without changing the
backend's batch size. It supports fully annotated free paths, exact qpos checks,
EEF IK/FK conversion and explicitly supplied solved branches. Contact, held
objects, changing locked joints, and unavailable backend checks cannot pass.
Joint-path collision checks use bounded sample densification, not a continuous
collision guarantee. The cuRobo lock model cache includes actual locked-joint
initial values and rejects runtime configuration drift until explicitly closed.

Generation currently requires every physical rigid UID in both the collision
world and its pose-update IDs, so each initial-state snapshot supplies the
correct per-row geometry pose. Different relative layouts require a per-env
world. The executor certifies fixed root/rigid poses at observation boundaries;
implicit floor geometry and other fixed conditions remain the trusted profile's
responsibility. This free-motion integration does not qualify atomic PickUp,
attachment geometry, or the planned four source/host combinations.

Focused integration tests are in `tests/lab/trajectory_generation/`; use the
CPU tests for orchestration and resource limits, and opt into the serialized
real cuRobo/physics smoke tests when changing those adapters.

`examples/sim/motion/trajectory_generation/free_motion.py` runs the complete
normal-gravity pure-arm UR5 pipeline and emits a LeRobot shard plus report.
`--record-video --duration 5 --joint-displacement 0.4` also streams a 640x480,
20 fps offscreen camera preview to `preview.mp4`. Rendering observes the actual
rollout without stepping physics; video includes both initial and terminal
observations and stays separate from the numeric LeRobot schema.
`--robot panda` is a real rejection case: motion-induced hand/mimic displacement
exceeds the locked-joint model tolerance, so actual collision verification is
unavailable and no expert episode is committed. These two real cases are in
`tests/lab/trajectory_generation/test_runner_real.py`; they do not replace the
planned PickUp source/host qualification matrix.

## PickUp Collection Integration

`integrations/atomic.py::export_pickup_templates` accepts a successful offline
MoveEndEffector → PickUp compilation, preserves its explicit phase boundaries,
expands mimic coordinates, and adds real terminal hold commands. Only transit
allows residuals. It exports qpos references, without committing projected
symbolic effects or executing AtomicActionRuntime recovery/tracking.

`integrations/contact.py::PickUpMotionValidator` is the shared Runner planner
and executor contact validator. It uses conservative URDF collision hulls,
full-joint FK, cuboid world geometry, an implicit floor, phase/contact-entry
permissions, and planned held-object geometry. Actual collision checks use
measured fingers/object poses. Native contacts are checked after every 5 ms
CPU physics step; forbidden/unknown/cross-row/deep contacts, buffer overflow,
missing finger contact, slipping and dropping cannot qualify. Mounting links
must be fixed to the robot root. Two-hop self-pair exclusions match cuRobo.
The executor exempts only this target and declared mimic joints from its
free-motion immobility rules; all remaining root/world checks remain active.

`examples/sim/motion/trajectory_generation/cube_pickup_collection.py` runs four
rows, repeatedly restores the full initial state, and collects confirmed
LeRobot shards with optional synchronized video. Its 0/90-degree targets use
independent residual transit paths. Controller stiffness/opening margin are
set before capture; collision and tracking thresholds are not relaxed per row.
The sink flattens numeric matrices in C order, records `observation_shapes`,
and preserves terminal matrices. `commanded_joint_indices` distinguishes active
controller labels from full-joint passive target columns.

This is CPU-physics / unscaled fixed-base URDF / cuboid-rigid support. The
optional `trajectory-generation` dependencies supply FCL/trimesh/yourdfpy.
It is sampled validation, not continuous collision detection. Contact-aware
Gym, atomic runtime replay, general via points and a registry/CLI remain work.
Focused tests: `test_contact.py`, `test_atomic_source.py`, `test_episode_sinks.py`,
`test_pickup_collection.py` under `tests/lab/trajectory_generation/`.

## Multi-grasp atomic candidate source

`integrations/atomic_candidates.py::AtomicTrajectoryGenerator` is a separate
offline source, re-exported from `integrations/atomic.py`. It accepts a frozen
`graspkit.GraspCandidateBatch`, preserves grasp/roll identities, and compiles
optional same-arm MoveJoints/MoveEndEffector prefixes followed by one PickUp.
Only `ik_interp` is supported. Prefixes cannot change the solver root or start
with a held object. The engine evaluates selected grasps at the prefix's
projected context, not at a stale pre-prefix qpos.

`replicas.py::SceneReplicaPool` binds canonical source cases to distinct real
physical rows. It verifies identical local-arena robot/object states and fixed
conditions; host-backed pools also verify the preparation epoch before every
wave. Logical candidates are scheduled in waves over those rows; env IDs are
never duplicated and the backend batch size is never changed. A one-row host
therefore materializes one full trajectory per wave.

The source returns a compact `CandidateTrajectoryBatch` with
`positions[B_out,N_max,D_full]`, arrival intervals, valid lengths, last-qpos
padding, full joint names, phases and identities. Invalid grasp poses and
failed IK/FK/limit/continuity/path checks remove only their candidate; audit
records retain the first failing invocation/stage. An empty `(0,0,D_full)`
result is normal. Proposal, output-count/byte/waypoint, wall-time and rejection
record budgets are bounded. Retry/resampling and held-object suffixes are
explicitly unsupported in this first source.

Without an external validator, results are planning-only: no world-collision,
physical grasp success or expert-data claim. An injected `GenerationSession`
requires an explicit validator including `path_collision`; successful results
enter its bounded ready queue, not committed coverage. The source does not
add `GenerationRunner.run_source`, reset the simulator, or write a dataset.

`examples/sim/motion/trajectory_generation/affordance_parallel.py` uses real
antipodal mesh sampling and four same-case UR5 instances, writes
`trajectories.npz` and `report.json`, and reports partial/empty batches.
The sampler uses tutorial target-local gripper geometry, not support-plane or
full-world collision validation. Focused CPU contracts are in
`test_atomic_candidates.py`, `test_replicas.py`, and
`test_affordance_parallel.py`; the latter also provides an opt-in GPU smoke test.

## Live-row affordance tutorials

`integrations/atomic_affordance.py::plan_affordance_batch`, re-exported from
`integrations/atomic.py`, selects distinct raw grasp IDs over caller-owned real
rows. It accepts a single PickUp or Slide invocation using `ik_interp`, with
bounded replacement from remaining candidates after row-local failure.
`AtomicAffordanceBatch.trajectory` holds all physical rows on one control grid;
`compact_positions` exports only successful rows as `(B_success,T,D_full)`.
Failed rows hold their observed start and never count as generated trajectories.
The zero-time sample preserves measured passive residuals; subsequent samples
expand mimic geometry. This adapter does not own resets, replica certification,
GenerationSession accounting or physics acceptance.

`scripts/tutorials/atomic_action/pickup.py` and `slide.py` accept
`--n_affordance_multi_gen N`, overriding `--num_envs` to create N actual rows.
Without this flag their legacy winner path is unchanged. Shared
`affordance_utils.py` samples one raw set, reprojects object-local poses per env,
and optionally writes compact NPZ/JSON via `--affordance_output`. N counts raw
grasps, not PickUp roll variants; shortage returns partial/empty without copies.
Slide push preserves each pull row's selected raw grasp and rebases it against
the observed post-pull handle pose, using a fresh observed robot context.
Qpos replay is a demonstration, not collision/contact or expert certification.

Focused tests: `tests/lab/trajectory_generation/test_atomic_affordance.py` and
`tests/sim/atomic_actions/test_{slide_candidates,tutorial_affordance,affordance_tutorials}.py`.
