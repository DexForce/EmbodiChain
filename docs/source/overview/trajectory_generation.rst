Fixed-Scene Trajectory Generation
==================================

Repeated expert rollouts need the same declared scene and initial state before
each candidate executes. The :mod:`embodichain.lab.trajectory_generation` layer
provides that preparation lifecycle for an entire simulator batch, either
directly or through an ``EmbodiedEnv``. Its synchronous qpos runner connects
candidate planning, physical execution, measured validation, and
persistence of accepted episodes. It consumes a scene that the caller already
created; it does not sample layouts.

The :mod:`embodichain.lab.sim.motion.trajectory_augmentation` package provides
qpos candidates, constrained variation, coverage, and generation accounting.
Two collection paths are available: free motion through the existing locked-joint
planner, and a bounded CPU-physics PickUp integration with full-joint geometry
and physical contact validation. PickUp can replay an offline compilation or
consume its augmented candidates through the observed atomic runtime described
below. Contact-aware Gym collection and the unified generation CLI remain
unimplemented. Initial-state preparation
alone does not certify a candidate trajectory or confirm that an episode was saved.

Supported Initial States
------------------------

The physical adapter supports one configuration-owned fixed-base robot and all
ordinary rigid objects in the simulator. Every entity must cover every simulator
row. Cases are supplied in physical row order, so one batch can retain different
fixed cases in its rows. Partial-row restoration and replica rebinding are not
implemented.

Snapshots own local poses, rigid velocities, complete joint positions and
velocities, drive targets, and joint effort state. They include mimic and gripper
joints. Additional articulations, object groups, deformables, rigid constraints,
and robots using USD articulation properties are rejected. The initial state
must not depend on pending external forces or an active contact constraint;
this is episode initialization, not a mid-contact physics checkpoint.

The physical adapter has been exercised against a real headless CPU backend
with physics both enabled and disabled, using one robot and one cube. Those
checks cover backend state reads, writes, and restoration. The PickUp collection
example below additionally exercises repeated restoration after real grasps;
it does not restore a contact solver checkpoint.

A Trusted Preparation Profile
-----------------------------

The caller supplies an
:class:`~embodichain.lab.trajectory_generation.initial_state.InitialStateProfile`
implemented for the task, controller, and scene configuration:

* ``prepare`` deterministically initializes task state and controller history,
  including state outside the standard Gym episode managers. It must preserve
  the fixed scene conditions.
* ``signature`` identifies fixed geometry, physics, materials, visuals, sensors,
  and control conditions. It excludes evolving poses and joint positions and
  must change when a condition relevant to the case changes.
* ``verify`` returns nonempty, passing ``ValidationResult`` evidence for every
  declared case after settling. It must check the task-specific initial
  conditions; an unconditional passing result is not a preparation policy.
* ``physics_dt`` declares the settling clock and must match ``env.physics_dt``
  for Gym. ``settling_steps`` controls preparation-only simulation steps.
* ``allowed_interval_events`` names Gym interval events explicitly checked to
  preserve the fixed conditions. Any active interval event outside this list
  prevents host acquisition or reuse.

Callbacks come from trusted integration code. There is no generic safe default
profile and no loading of profile callbacks from job YAML strings. The physical
adapter checks scene structure and numeric state; it cannot infer the fixed
condition identity or reset controller-specific memory for the profile.

Preparation and Rollout Order
-----------------------------

:class:`~embodichain.lab.trajectory_generation.initial_state.FixedSceneHost`
owns the complete simulator batch. With Gym it also acquires an environment
generation lease, which suppresses automatic resets and rejects ordinary
``env.reset()`` until ownership is released. The owner must serialize all host
access and remain the sole physical stepper. Direct simulation hosts and Gym
leases share ownership of the same simulator batch; a second owner through
either interface is rejected.

The first ``acquire_case(cases)`` performs deterministic preparation, settling,
and verification, then captures the prepared physical states. Later
``restore_initial()`` calls reconstruct those captured states. In Gym, each
preparation follows this order:

1. Invalidate the previous epoch and active Task Program bridge; disable
   stepping while preparation is incomplete.
2. Discard pending camera and episode recording state. The caller must already
   own copies of the previous episode and its validation evidence.
3. Run deterministic profile preparation, restore the captured physical state
   when one exists, then perform the explicit settling steps.
4. Reset observation history, reward, and dataset managers, then verify task
   initial conditions and, for restoration, the captured physical fields. The
   validator observes the new episode's manager state. A failed attempt publishes
   no execution binding.
5. Obtain the initial observation and seed enabled recorders from the verified
   state.

Preparation does not run reset/interval events, reseed the environment RNG, or
call normal Gym ``step``. Profile callbacks must not call ``env.step`` or
``env.reset``. Settling and the articulation root setter may advance physics;
those preparation updates are outside the recorded rollout.

``PreparedBatch`` is an execution binding for one host and epoch. Check it with
``host.assert_current(binding)`` before issuing candidate commands. Restoration,
failed preparation, and ownership release invalidate previous bindings. A new
binding requires new runtime requests and Task Program bridges where applicable.

``host.snapshots(binding)`` returns independent planning snapshots of the
captured initial batch. For Gym, ``host.initial_observation(binding)`` copies the
first observation already produced by preparation, preserving observation-history
semantics without another ``get_obs`` call. It returns ``None`` for direct sim.

Integrating an Existing Gym Scene
---------------------------------

The following helper assumes an already-created unwrapped ``EmbodiedEnv``, one
``SceneCase`` per row, a task-specific trusted profile, and explicit execution
and evidence-copy callbacks. ``execute_candidate`` uses the normal environment
step path; ``freeze_episode`` copies all data needed by later validation and
persistence before the next restoration discards the live buffers.

.. code-block:: python

   from embodichain.lab.trajectory_generation.initial_state import FixedSceneHost
   from embodichain.lab.trajectory_generation.integrations.sim import (
       SimInitialStateAdapter,
   )

   def run_two_candidates(env, cases, profile, execute_candidate, freeze_episode):
       adapter = SimInitialStateAdapter(env.sim, env.robot)
       episodes = []
       with FixedSceneHost(adapter, profile, env=env) as host:
           binding = host.acquire_case(cases)
           for candidate_index in range(2):
               if candidate_index:
                   binding = host.restore_initial()
               host.assert_current(binding)
               execute_candidate(env, binding, candidate_index)
               episodes.append(freeze_episode(env, binding))
       return episodes

For direct simulation, construct the same host without ``env=``. The caller
then owns observation refresh, rollout timing, recording, and task execution.
Closing the host releases ownership and invalidates bindings without implicitly
saving or resetting the final episode.

Checking Free-Motion Candidates
-------------------------------

The
:class:`~embodichain.lab.trajectory_generation.integrations.planning.EnvRowMotionPlanner`
adapter connects explicit EEF or qpos candidates to a configured
``MotionGenerator`` and the actual robot batch. Logical candidate count ``C`` can
exceed physical row count ``B``: candidates are checked in rounds,
with at most one candidate assigned to its source row in each round. Snapshots
must cover the complete batch, and backend calls retain size ``B``.

For EEF input,
:class:`~embodichain.lab.trajectory_generation.integrations.planning.EEFPath`
declares local-arena TCP samples, a source row, identity, explicit arrival
intervals, and phases. ``plan_eef`` propagates successful IK seeds between
samples and checks FK position/orientation residuals. Supplying solved joint
samples preserves that branch without another IK solve. The output is a
full-joint ``CandidateTrajectoryBatch`` and one validation result per candidate.

For qpos input, ``validate_qpos`` checks the original candidate paths. It inserts
bounded joint-space samples between every adjacent pair, including phase
boundaries, before asking the backend to check self/environment collision.
Sampling changes only the check inputs. Original candidate positions and timing
remain intact.

This initial adapter supports fully annotated free motion, explicitly declared
empty ``held_object_ids``, and unchanged uncontrolled joints at the configured
initial values. Contact/hold phases, unannotated paths, held-object sweeps, and
changing locked joints
such as gripper closure are outside this capability. Missing collision support
or a path requiring more than ``max_validation_samples`` also returns
``unavailable``. Such candidates cannot enter accepted rollout generation.

An unchanged gripper command does not guarantee unchanged measured finger
positions. The measured path must also match the backend's locked-joint values
within ``1e-6``. A real Panda run under ordinary gravity exceeded this bound
through finger drift and was rejected with no committed episode, even though its
task, motion-quality, and speed/acceleration checks passed. This is an unsupported
locked-model state for measured collision validation.

The host must keep robot roots and canonical-ID collision obstacles synchronized
with the snapshots. A passing sampled collision result does not provide a
continuous collision guarantee, controller speed/acceleration validation, or
task success. Those checks remain explicit later gates; this adapter is not yet
a contact-aware PickUp planner.

The adapter's real cuRobo smoke uses one Panda with CPU physics and CUDA collision
checking. A short free path passes with a distant dynamic cube and fails after
the cube is moved to the TCP, reusing the same backend. This checks dynamic-world
collision updates for that sampled path; held-object and contact semantics remain
outside the supported scope.

Executing Qpos Candidates
-------------------------

:class:`~embodichain.lab.trajectory_generation.execution.QposRolloutExecutor`
accepts a prepared host binding, one single-row candidate or ``None`` per
physical row, and stable episode/commit IDs. Each candidate contains an initial
sample followed by commands on the host's fixed control clock. Inactive and
mimic joints must remain unchanged; the executor checks initial state, joint
limits, layout, and payload size before submitting a command.

The executor's byte limit includes tensors plus a 64 KiB metadata reserve.
Executor metadata is limited to 32 KiB so the runner and sink can add their
evidence within that reserve. Initial observation size and later schema changes
are checked before the executor copies observation payloads.

Gym execution uses the existing demo loop and ``ControllerAction``. It reads the
actual controller position targets before physics and consumes the existing
``env.step`` observation afterward. Direct sim uses explicit integer physics
substeps and a full-batch observation callback. A Gym encoder can transform the
prepared/step observations; its default flattens all tensor fields without
dropping channels. The executor always includes measured full-joint
``joint_positions``.
The default encoder requires nonempty string keys without embedded dots and
tensor leaves; other observation structures require an explicit encoder.

Timestamps are differences of ``SimulationManager.simulation_time`` and each
transition must advance exactly one control period within tolerance. That clock
counts manager-owned updates; external direct backend updates are outside it and
must not occur during a rollout owned by this executor.

Completed rows freeze their causal frames while other rows finish. Idle,
finished, or terminated rows receive holds, and those holds do not enter the
returned training sequence. Installing holds clears velocity/effort targets
without advancing physics. A row with no complete transition returns ``None``;
an interrupted row with some complete transitions returns rejected evidence.
``executor.last_failures`` keeps bounded candidate-specific runtime reasons even
when no complete transition exists. It is read-only and resets at the next call;
the runner records these reasons when releasing failed candidates.
The final task validator consumes owned row observations, actions, and times.
It must implement the task's real success conditions and include its configured
check ID.

The executor checks root and rigid-object pose stability at every observation
boundary and adds ``fixed_collision_world`` evidence. The runner uses that
evidence when rechecking the measured qpos path against the initial collision
snapshot. Boundary checks and sampled collision checks do not establish
continuous collision freedom or support contact/held-object tasks.

Assembling the Handwritten Runner
---------------------------------

:class:`~embodichain.lab.trajectory_generation.runner.GenerationRunner` combines
the explicit host, planning, execution, and persistence interfaces documented
here. The caller supplies qualified references, initial-state and validation
profiles, and full-joint limits through
:class:`~embodichain.lab.trajectory_generation.runner.MotionLimitsProfile`.

Every host rigid-object physical UID must appear in the backend's collision-world
and dynamic-pose entity IDs. This lets the runner supply the captured initial
poses for all rows after preparation. Dynamic here means updateable collision
poses, including physically static bodies. A baked static world is insufficient;
different relative layouts across rows require per-environment backend worlds.
Semantic aliases are not yet connected. The trusted profile must also establish
coverage of relevant implicit planes and other geometry outside the rigid
registry. The executor's per-row byte limit must not exceed the sink or job
pending-byte limit.

.. code-block:: python

   from embodichain.lab.trajectory_generation.runner import GenerationRunner

   # All ports are explicitly constructed for the same scene, robot, and clock.
   # host owns the batch, but has not yet called acquire_case().
   runner = GenerationRunner(
       cfg, host, planner, executor, sink, motion_limits=motion_limits
   )
   report = runner.run(cases, templates)

``cases`` and ``templates`` contain one item per physical row in matching order.
The runner verifies profile/source/template IDs and clock compatibility, proposes
allowed residual and retiming variants, validates their paths and motion limits,
then reserves capacity before each rollout. Measured paths, task evidence,
speed/acceleration, and path-length/duration quality gates determine which
episodes reach the sink. Write retries reuse frozen evidence.

Planning evidence is retained as ``planned_*`` checks. The measured-path collision
check, ``actual_motion_limits``, and ``motion_quality`` remain separate gates;
planned feasibility alone cannot qualify the saved episode.

``run`` is single-use and closes the supplied host and sink on exit. It returns
counts, coverage, audit, resolved configuration, and ``target_reached`` and writes
``generation_report.json`` under the sink root. Reaching a proposal, rollout, or
wall-time budget can finish without reaching the collection target. The PickUp
adapter below adds contact-aware offline and observed atomic execution; a unified configuration
registry and generation CLI remain subsequent work.

Running the Real Free-Motion Example
-------------------------------------

From the repository root, with the simulation dependencies, CUDA, cuRobo, and
LeRobot installed, use a new or empty output directory:

.. code-block:: bash

   python examples/sim/motion/trajectory_generation/free_motion.py --output /tmp/ur5-free-motion

The example constructs its own fixed scene and trusted preparation/validation
profiles. It uses a pure-arm UR5 with ordinary gravity, one CPU physics row,
CUDA collision checking, and no desktop window. The default run has no camera.
A registered ground proxy
matches the simulator's implicit floor so the collision world covers that
geometry. ``--cuda-device`` selects the GPU, defaulting to ``0``.

One second of motion produces 21 measured observations and 20 actual commands.
The verified run committed one episode after planned and measured collision,
execution, task, quality, and motion-limit checks passed. It measured about
``0.078021`` rad of movement, ``0.003212`` rad endpoint error, and ``0.010872`` rad
maximum tracking error. These are results of this short example, not a PickUp
or throughput acceptance result.

The script prints counts, ``target_reached``, and the report path. The output
contains ``generation_report.json``, a committed-shard ``manifest.json``, and
the LeRobot dataset plus required evidence files described below. A run that
does not reach its target exits with status ``1``. Only manifest-listed shards
are committed training data.

To save an offscreen video of a larger, five-second motion, install
``imageio-ffmpeg`` in the same environment and run:

.. code-block:: bash

   python examples/sim/motion/trajectory_generation/free_motion.py --output /tmp/ur5-video --record-video --duration 5 --joint-displacement 0.4

``--record-video`` adds a 640 x 480 RGB camera and streams its actual execution
frames to ``preview.mp4`` using H.264 at 20 fps. It includes the initial frame
and each subsequent observation, including the terminal frame: five seconds of
physical motion produces 101 frames, so the video lasts 5.05 seconds. Numeric
LeRobot observations retain their existing schema; this preview is a separate
visual record. A video may also show a rejected attempt, so use
``generation_report.json`` and the committed manifest to assess whether the
episode qualified as expert data.

``--duration`` accepts 0.05 through 30 seconds in multiples of 0.05; its default
is 1 second. ``--joint-displacement`` controls the first arm joint's positive
displacement, up to 0.5 rad, and defaults to 0.08 rad. Changing these values
preserves the validation gates and does not guarantee acceptance.

Using ``--robot panda`` with a separate empty output directory runs the rejection
example: ordinary-gravity finger drift exceeds the locked-model tolerance, so
measured collision validation is unavailable and no episode is committed.
The real integration test covers both outcomes:

.. code-block:: bash

   pytest tests/lab/trajectory_generation/test_runner_real.py --run-gpu -m gpu -q

Both cases passed. The UR5 case also decodes the saved H.264 preview and checks
that frame timestamps match all measured observations, including the terminal
state, without extending the physical rollout. Real acceptance currently covers
this one-row direct-sim free-motion example; Gym lifecycle/execution contracts
have separate CPU tests. PickUp has a separate collection example below. The
full source/runtime and sim/Gym qualification matrix remains pending.

Parallel Cube-Grasp Augmentation Preview
----------------------------------------

The standalone ``cube_grasp_parallel.py`` example compares grasp-pose and
transit-path augmentation with four synchronized physical rows:

.. code-block:: bash

   python examples/sim/motion/trajectory_generation/cube_grasp_parallel.py --output /tmp/cube-grasp-parallel

Each row has the same initial UR5/PGI configuration and the same 5 cm cube on a
bench, with normal gravity. The rows cross two object-relative grasp
orientations (0 and 90 degrees) with reference and augmented transit paths.
``rotate_grasp_about_object_axis`` uses the cube's quarter-turn symmetry to
derive TCP goals. Existing ``MoveEndEffector`` and ``PickUp`` Atomic Skills
replan each goal using the UR IK solver and ``ik_interp`` strategy.
``joint_residual`` changes only the transit before the pre-grasp corridor,
preserving its endpoints and the later approach, close, and lift commands.

Four offscreen cameras capture the same physics tick and stream a 1280 x 1056,
20 fps four-panel ``preview.mp4``. Colored trails show measured TCP motion;
labels show grasp orientation, phase, time, and measured cube lift. The
``--seed`` option selects the local residual random stream (default ``13``),
and ``--cuda-device`` selects the renderer GPU. Simulation dependencies, the
UR5/PGI assets, TOPPRA, and ``imageio[ffmpeg]`` are required; this example does
not initialize a cuRobo collision backend. Use a new or empty output directory.

``rollout.npz`` stores reference and commanded qpos, measured full-joint qpos,
TCP/cube poses, grasp goals, and measured timestamps. ``report.json`` records
augmentation factors, phase boundaries, path separation, and per-row outcomes.
The final hold requires at least 12 cm of sustained cube lift, at most 1 cm of
TCP-relative position drift, and a cube-to-TCP distance below 6 cm. A run exits
with status 1 if any row fails. Cubes are moved through physical gripper contact;
no attachment constraint or pose update supplies the lift.

The default run passed all four grasp checks, with approximately 17 cm of
maximum separation between paired transit paths. Its 13.55 seconds of physical
motion produces 272 frames including the initial and terminal observations.
This validates the demonstrated grasp behavior. Contact-aware collision
certification and expert LeRobot qualification remain outside this preview;
use the separate PickUp collection example below for expert-data validation.

Saving Accepted Episodes
------------------------

:class:`~embodichain.lab.trajectory_generation.sinks.LeRobotEpisodeSink` consumes
an owned ``ExpertEpisode`` whose validation checks have passed. Each submission
creates a local LeRobot dataset shard containing one episode. The sink has no
live environment dependency and can receive frozen data from either a sim or Gym
host. The caller still owns task/trajectory validation and episode construction.

An episode with ``T`` commands is stored as:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Artifact
     - Content
   * - ``dataset/``
     - ``T`` LeRobot frames pairing observations ``0..T-1`` with actual commands;
       RGB observations are stored as images.
   * - ``terminal.npz``
     - Observation ``T`` and all ``T+1`` measured timestamps, including their
       original time origin.
   * - ``episode.json``
     - Candidate/episode/commit identity, action representation, validation,
       phases, feature mapping, and user metadata.
   * - Collection ``manifest.json``
     - The committed episode-to-shard mapping. Unlisted shards are not committed.

The complete causal sequence consists of the LeRobot training frames plus the
required terminal sidecar. Measured timestamps must match the configured integer
``fps`` within ``timestamp_tolerance``. LeRobot frame timestamps are relative to
the episode origin; the exact measured clock remains in the sidecar.

Numeric observations have shape ``(T+1,D)`` or ``(T+1,M,N)`` and dtype float32,
float64, int32, int64, or uint8. Matrices flatten to vectors in C order for
LeRobot, with their source layout in ``observation_shapes`` and their terminal
matrix preserved in the sidecar. RGB images use uint8 ``(T+1,H,W,3)``. Actions use float32 or
float64 ``(T,A)``. Unsupported layouts, conflicting feature names, invalid
clocks, and payloads exceeding ``max_episode_bytes`` are rejected before episode
writes. The byte limit covers raw tensors and metadata per submission.

After writing, the sink finalizes the LeRobot writer, opens every training frame
and image again, verifies both sidecars, and replaces and reads back the manifest.
Only then does ``submit`` return ``receipt.confirmed=True``. Persistence errors
return an unconfirmed receipt with an error message; they do not count toward
confirmed generation coverage.

.. code-block:: python

   from embodichain.lab.trajectory_generation.sinks import LeRobotEpisodeSink

   # episode already owns the accepted rollout observations and actual commands.
   with LeRobotEpisodeSink("outputs/accepted-episodes", fps=50) as sink:
       receipt = sink.submit(episode)
       if not receipt.confirmed:
           # Reuse the same episode and commit ID; distinguish this submission.
           receipt = sink.submit(episode, submission_id=1)
       if not receipt.confirmed:
           raise RuntimeError(receipt.error)

The output directory must be new or empty. An identical payload under a confirmed
commit ID only reads and verifies the existing artifacts; a different payload
under that ID raises. A confirmed duplicate does not rewrite files or create
another logical episode.
Readable sealed data can be reused after a metadata failure, and incomplete
uncommitted data can be rebuilt at the same shard path. This retry behavior
applies within the owning sink process; reopening an existing collection after
a process restart is not supported.

When using ``GenerationSession``, first reserve the measured episode with
``accept_episode``; submit only when it returns ``True``. Apply every returned
receipt with ``apply_receipt``.
After a failed write, use ``retry_write(commit_id)`` to obtain the retained
episode and its next submission ID before resubmitting. Successful write retries
reuse the rollout; they do not generate another physical attempt.

Submission is synchronous and serialized. ``drain()`` has no deferred receipts,
and ``close()`` releases ownership without another save operation. Confirmations
cover closed, readable artifacts; they do not promise restart recovery or
power-loss durability.

Public interfaces and supported snapshot fields are documented in the
:doc:`trajectory-generation API </api_reference/embodichain/embodichain.lab.trajectory_generation>`.

Contact-Validated Parallel PickUp Collection
--------------------------------------------

``examples/sim/motion/trajectory_generation/cube_pickup_collection.py`` reuses
the cube preview's four-row scene and MoveEndEffector/PickUp compilation, then
connects them to ``FixedSceneHost``, ``GenerationRunner``, ``GenerationSession``
and ``LeRobotEpisodeSink``. Each row receives its own local residual proposal
in transit; approach, closing, lift and hold remain protected. Two grasp goals
use the cube's 0/90-degree symmetry. All four rows share the physical clock.

.. code-block:: bash

   python -m pip install -e '.[trajectory-generation]' 'imageio[ffmpeg]'
   python examples/sim/motion/trajectory_generation/cube_pickup_collection.py \
     --output /tmp/cube-experts --episodes 8 --record-video

The output directory must be new or empty. ``--episodes`` defaults to 8 and is
bounded to 1–64; ``--seed`` defaults to 13. ``--cuda-device`` selects the renderer.
Physics uses CPU at 200 Hz with normal gravity; control/video use 20 Hz.
The UR5 arm stiffness is 200000 with the existing damping, and the PGI open
command sits 1 mm inside its joint limit to avoid limit overshoot during transit.
These values are fixed before initial-state capture; the same endpoint/tracking
and collision tolerances apply to both grasp orientations.

The PickUp integration adds the following mandatory evidence:

* Planned full-joint geometry includes both fingers and their mimic coupling.
  Conservative convex hulls come from URDF collision shapes, not render meshes.
  Joint-segment sampling is bounded; the held cuboid follows the TCP after lift
  begins. The implicit ground is represented explicitly in the checker.
* Actual geometry uses measured joints, including passive fingers, and the
  measured cube pose. Only the declared target may move; robot root and other
  scene objects must retain their captured poses at observation boundaries.
* Native CPU contacts are read after every 5 ms physics step. The profile allows
  only declared fixed mounting contacts, cube/support contact through initial
  lift-off, and finger/cube contact during closing/lift/hold. During approach,
  finger contact is confined to the declared entry region; this PGI example uses
  6 cm to include fingertip overhang beyond its TCP. Other contacts, unknown
  bodies, cross-row pairs, more than 2 mm penetration, and buffer-budget overflow
  reject the evidence. URDF self pairs within two kinematic hops are structural
  exclusions shared with the existing cuRobo policy.
* During the 1.5-second terminal hold, each finger must report positive normal
  impulse in at least 95% of physics samples and the cube must remain at least
  12 cm above its initial height. Translation and rotation drift relative to TCP
  are checked from lift through hold, with 1 cm / 0.15 rad limits. The cube must
  remain within 6 cm of TCP. Arm endpoint/tracking tolerances are 0.05 / 0.08 rad.
  Motion limits and path/duration quality checks remain mandatory.

These are conservative sampled geometric checks plus actual discrete-physics
contact evidence, not continuous collision detection. The integration currently
supports one unscaled fixed-base URDF robot and cuboid rigid scene objects.
It requires the optional ``python-fcl``, ``trimesh`` and ``yourdfpy`` dependencies.
Contact-aware Gym execution requires equivalent physics-substep evidence and
is rejected at construction today.

Between rounds, the entire robot/cube state is restored and verified, including
current joints, passive coordinates, velocities, drive targets and efforts.
Every returned episode has T actual controller targets and T+1 measured
observations/timestamps. ``commanded_joint_indices`` identifies actuated columns
in the full ordered joint vector; passive target columns are retained as read
from the controller, while collision checks use actual passive positions.

The collection saves ``generation_report.json`` (attempts, rejections, validation
and commit audit), ``pickup_report.json`` (contact profile, prepared rounds and
video frames), and ``manifest.json`` (only read-back-confirmed shards). Each shard
contains LeRobot training frames plus ``episode.json`` and ``terminal.npz``.
Object/TCP 4x4 matrices are flattened in C order into 16-element LeRobot numeric
features; ``observation_shapes`` records their original layout. Terminal matrices
retain their 4x4 shape. ``preview.mp4`` includes accepted and rejected attempts;
it is a four-panel record of actual observations, and each round restarts its
trail and time display. Video is separate from the numeric training dataset.

``source.kind=atomic`` identifies the template's compilation source. The default
mode replays those templates with the physical checks above. To execute them
through the observed atomic runtime, select ``--runtime``.

Observed Atomic PickUp Execution
--------------------------------

.. code-block:: bash

   python examples/sim/motion/trajectory_generation/cube_pickup_collection.py \
     --output /tmp/cube-runtime-experts --episodes 8 --runtime --record-video

``PickUpRuntimeSource`` supplies the selected candidate through
``initial_plan_provider``. Every prepared batch receives a fresh invocation,
measured planning context, pending effect and ``ExecutionSession``. A single
PickUp plan preserves transit/approach/close/lift/hold, including the qualified
joint branch and protected contact coordinates. The runtime goal uses the
reference's URDF TCP FK at grasp closure, which can differ from the nominal
analytic IK target when the asset contains a wrist offset. The initial
observation is omitted from the command sequence, retaining 271 commands and
272 observations for this example.

``ExecutionRunner`` sends typed commands through ``SimulationExecutionAdapter``
and monitors arm position feedback at the existing 0.08 rad in-flight and
0.05 rad terminal thresholds. The grasp endpoint has no position-tracking
channels: the fingers press against the cube before full commanded closure.
Native bilateral contact and hold stability remain mandatory. At completion,
the verifier's measured joint context supplies FK in the same motion endpoint
TCP frame as the plan. Together with the current object pose, it must agree with
the pending held-object transform within the contact profile's tolerances. The
native contact TCP has its own declared frame; both frame sources and effect
errors are recorded in episode metadata. Only then does the session commit its
held-object state. The usual task, geometry and persistence gates still apply.

The collection executor owns every physics step and records the actual submitted
targets. Runtime commands must match the selected candidate and fixed control
clock. Intervals retain float64 precision; zero velocity targets are explicit
for both full and tail batches. Phase or terminal feedback requiring extra waiting fails collection;
replans and retries use the registered skill but their commands are blocked at
the transport boundary. Any runtime failure cancels and holds the entire active
batch, which is retained only as rejected audit evidence. Idle tail-batch rows
are supported. Recovery demonstrations and independent continuation of other
rows after a runtime failure are outside this adapter's contract.
Interrupted phase prefixes produce unavailable path evidence so rejection does
not prevent the runner from preparing its next batch.

The ``atomic_runtime`` episode metadata contains bounded per-row event counts,
command count, plan-attempt count, status and physical-effect result. Its
mandatory validation check must pass before persistence. ``pickup_report.json``
distinguishes observed runtime execution from offline replay; both modes use the
same video and LeRobot formats. Contact-aware Gym, the complete handwritten/atomic
× sim/Gym qualification matrix, general via-point planning, and YAML-based
deployment remain subsequent milestones.
