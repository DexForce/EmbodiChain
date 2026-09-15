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
