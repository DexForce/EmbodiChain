embodichain.lab.trajectory_generation
=====================================

.. automodule:: embodichain.lab.trajectory_generation

The trajectory-generation integration layer prepares caller-provided fixed
scenes for repeated expert rollouts. It currently provides full-batch initial
state capture, restoration, verification, exclusive host ownership, qpos rollout
execution, a synchronous runner, and a LeRobot episode sink. Candidate operators
and generation accounting live in
:mod:`embodichain.lab.sim.motion.expansion`.
The planning adapter converts explicit EEF samples and validates supported free
qpos paths against the real environment batch.

The synchronous runner supports free-motion references and an explicitly
bounded CPU-physics PickUp integration. The latter exports an offline atomic
compilation, checks full gripper and held-object geometry, records native
contacts at each physics substep, and persists accepted measured episodes.
AtomicActionRuntime recovery/feedback execution, contact-aware Gym collection,
a configuration registry, and the unified generation CLI remain separate work.
See :doc:`/overview/trajectory_generation` for the host lifecycle and integration
requirements.

Initial-State Host
------------------

.. currentmodule:: embodichain.lab.trajectory_generation.initial_state

.. autosummary::
   :nosignatures:

   InitialStateProfile
   PreparedBatch
   FixedSceneHost

``InitialStateProfile`` supplies trusted Python callbacks for deterministic
task/controller preparation, fixed-condition identity, and task-specific initial
verification. A profile must cover controller history and task state beyond the
standard episode managers, and fixed physics, geometry, visual, sensor, and
control conditions beyond the physical snapshot. Profile IDs are descriptive;
they do not load callbacks from YAML or certify the implementation.

``FixedSceneHost`` owns an entire simulator batch, with one ordered ``SceneCase``
per physical row. ``acquire_case`` prepares and captures the initial states;
``restore_initial`` reconstructs them before a later rollout. ``PreparedBatch``
identifies the current host epoch. Call ``assert_current`` before executing a
candidate; earlier or foreign bindings are rejected. Bindings do not transfer
old action plans, Task Program bridges, or other runtime state into a new epoch.
``snapshots(binding)`` copies planning inputs from the captured initial batch in
physical row order. ``initial_observation(binding)`` returns a copy of the Gym
first frame produced during preparation without querying observations again;
it returns ``None`` for a direct simulation host.

When an ``EmbodiedEnv`` is supplied, the host uses its generation lease and
preparation API. Direct simulation and Gym ownership are mutually exclusive for
the same simulator batch. Gym resets the standard episode managers after
settling and before verification; initial observations and recording seeds are
published only after verification succeeds. The caller must freeze old episode
data and validation evidence before preparation discards its buffers. Closing the
host releases ownership;
it does not save or restore an episode.

.. autoclass:: InitialStateProfile
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: PreparedBatch
   :members:

.. autoclass:: FixedSceneHost
   :members:

Simulation State Adapter
------------------------

.. currentmodule:: embodichain.lab.trajectory_generation.integrations.sim

.. autosummary::
   :nosignatures:

   SimInitialState
   SimInitialStateAdapter

``SimInitialStateAdapter`` supports one configuration-owned fixed-base robot
and ordinary rigid objects spanning the full batch. The owned snapshot includes
local root/object poses, velocities, every movable joint in robot joint order,
position/velocity drive targets, and joint effort state. Tensor inputs are
detached and cloned; consumers must treat those owned tensors as read-only.

The adapter checks topology, field names, shapes, finite values, SE(3) poses,
and joint limits before physical writes. Verification compares every supported
field with absolute tolerance ``atol`` and zero relative tolerance. It does not
snapshot fixed physical/visual properties, task/controller memory, contact
solver state, or pending external forces. Those boundaries require the trusted
profile and a supported episode initial state.
``tolerances_profile_id`` names this explicit tolerance policy for runner/job
matching; supplying the ID does not configure a tolerance or load a profile.

Additional articulations, rigid-object groups, deformables, rigid constraints,
and robots whose articulation properties come from USD are rejected. Restoring
the root uses the existing articulation setter, which can advance the world by
1 ms; remaining joint and object writes follow that operation. A backend failure
can leave partial physical changes, so the owning host invalidates the binding
before restoration and only publishes a replacement after settling and
verification succeed.

.. autoclass:: SimInitialState
   :members:

.. autoclass:: SimInitialStateAdapter
   :members:

Environment-Row Planning
------------------------

.. currentmodule:: embodichain.lab.trajectory_generation.integrations.planning

.. autosummary::
   :nosignatures:

   EEFPath
   EnvRowMotionPlanner

``EEFPath`` owns explicit TCP poses ``(N,4,4)`` in the source row's local arena
frame, arrival intervals ``dt`` with an initial zero, candidate identity, source
row, and phase annotations. Optional ``solved_joint_targets`` have shape
``(N,D_control)`` in the selected control-part joint order. These samples preserve
a chosen IK branch; they are checked by FK without solving them again.
Float64 arrival intervals and supplied joint samples retain their precision.

``EnvRowMotionPlanner`` requires one complete ``MotionSnapshot`` per real robot
row. It schedules candidates in rounds with at most one candidate per source row
and keeps backend calls at the real robot batch size. Input and output candidate
order, source-row identity, phase annotations, and arrival intervals stay aligned.
``plan_eef`` solves explicit EEF samples, propagates valid seeds, checks TCP FK
residuals, and returns a full-joint candidate batch plus per-candidate validation.
Failed rows hold their initial state and remain rejected by that paired result.

``validate_qpos`` leaves the original path unchanged and densifies its segments
for collision checks, including phase boundaries. ``max_joint_step`` bounds the
sample spacing in joint coordinates and ``max_validation_samples`` caps samples
per candidate. Passing evidence is a sampled check, not continuous collision
detection, dynamic feasibility, or task success.
Collision-check scratch is allocated per real-batch round, bounded by ``B`` times
``max_validation_samples``, rather than retaining every candidate's dense path.

Only fully annotated ``free`` motion with explicitly empty ``held_object_ids``
is supported. Uncontrolled joints must stay at both the snapshot values and
robot configuration's initial values, matching the backend locked-joint model.
This requirement also applies when checking measured trajectories: a gripper
can drift under physics even if the candidate commands no gripper motion.
Measured locked-joint drift beyond the current ``1e-6`` comparison makes that
model unavailable for acceptance.
Contact/hold phases, unannotated samples, held-object sweeps, changing locked
joints, absent backend validation,
or excess sampling requirements return ``unavailable``. Invalid trajectories or
detected collisions return ``failed``. The caller maintains the same live robot
roots and collision world as the snapshots and supplies canonical dynamic
obstacle IDs; this adapter does not step or restore the scene.

Measured qpos can be checked after execution while live root frames remain
fixed. Its first sample must match the snapshot within ``1e-6``. The obstacle
input represents one constant world for the whole path; the caller must
establish that it applies to the rollout. A single initial or final obstacle
pose does not reconstruct obstacle motion during execution.

.. autoclass:: EEFPath
   :members:

.. autoclass:: EnvRowMotionPlanner
   :members:

Qpos Rollout Execution
----------------------

.. currentmodule:: embodichain.lab.trajectory_generation.execution

.. autosummary::
   :nosignatures:

   QposRolloutExecutor

``QposRolloutExecutor`` executes at most one candidate per physical row, with
``None`` for idle rows. A candidate's first sample is the initial state;
``valid_length - 1`` later samples are controller targets. The executor checks
case/epoch identity, full-joint order, initial state, active-joint permissions,
limits, the fixed clock, and estimated payload capacity before commands.
The payload limit includes tensors and metadata: execution reserves 64 KiB of
metadata capacity before commands and limits its own lineage, observation-key,
and validation metadata to 32 KiB, leaving room for later runner/storage evidence.
Oversized initial observations and changed schemas are rejected before the
executor creates its owned observation copy.

Gym execution uses ``ControllerAction`` through the common demonstration loop.
The command observer copies actual full-joint position targets after controller
submission and before physics. The step observer consumes the returned Gym
observation without querying it again. Direct simulation uses the supplied
full-batch observation callback and an integer number of physics substeps per
command. Gym uses the authoritative ``env.step_dt``; both paths record differences
of ``SimulationManager.simulation_time`` and reject inconsistent clock advances.

Returned ``ExpertEpisode`` values contain commands actually submitted and their
complete observation transitions. ``joint_positions`` always contains actual
full-joint measurements. Finished rows stop collecting frames while safe holds
allow other rows to finish; holds and padding are excluded from episode data.
Installing a hold clears velocity targets and joint effort without adding a
physics step. Execution remains serialized through task validation and evidence
freezing, including calls made from user-supplied validators.
The trusted task validator receives owned observations, actions, and timestamps
at the final batch boundary and must return its declared validation check.

The executor records ``execution_complete`` and ``fixed_collision_world`` checks.
The latter compares robot roots and all rigid-object poses against the initial
snapshots at observation boundaries; it does not reconstruct motion between
those boundaries. Interrupted rollouts, target disagreement, changed observation
schemas, or runtime errors cannot produce accepted evidence. Rows without a
complete transition return ``None``. ``on_started`` fires after the first command
submission even if physics or observation then fails, so the rollout still
counts. Failure to install the final safe hold raises and stops collection.
``last_failures`` exposes an owned read-only mapping from candidate ID to the
most recent batch's runtime failure reason, including rows returning ``None``.
It resets on the next execution and retains at most one 512-character reason per
physical row, allowing the runner to preserve physics, observation, or cancellation
diagnostics without retaining rollout payloads.

.. autoclass:: QposRolloutExecutor
   :members:

Synchronous Handwritten Runner
------------------------------

.. currentmodule:: embodichain.lab.trajectory_generation.runner

.. autosummary::
   :nosignatures:

   MotionLimitsProfile
   GenerationRunner

``MotionLimitsProfile`` owns positive full-joint speed and acceleration vectors,
copied to CPU float64 in the robot's complete joint order. Its profile ID must
match the requested job policy.

``GenerationRunner`` combines an explicit job configuration, ``FixedSceneHost``,
``EnvRowMotionPlanner``, qpos executor, synchronous LeRobot sink, and motion
limits. It checks integration IDs, robot ownership, joint layout, enabled
operators, and the host/sink clock before collection. ``run`` accepts one fixed
case and qpos template per physical row; the free-motion implementation
requires distinct case/initial-state pairs and the ``env_rows`` / ``full_batch``
execution modes.

With ``EnvRowMotionPlanner``, every host rigid-object physical UID must appear in both
``collision_world_entity_ids`` and ``dynamic_collision_entity_ids`` so the backend
receives each row's captured initial pose. Here dynamic means that backend poses
can be updated; a simulation body may still be static. A baked static world alone
cannot establish the prepared poses or different cases across rows. Different
relative layouts in a multi-row batch require per-environment backend worlds.
Semantic aliases are not connected in this runner. The trusted profile must
also certify collision representation for implicit
planes and other relevant geometry outside that rigid-object registry.
``executor.max_episode_bytes`` must fit both the sink limit and the job's pending
byte limit before rollout allocation.

The runner proposes configured joint residuals and retiming, validates planned
paths and sampled motion limits, reserves episode capacity before rollout, and
restores the fixed initial batch between executed rounds. It rechecks measured
joint paths against the fixed collision snapshot and evaluates actual motion,
duration, task evidence, and motion limits before submitting an
accepted episode. Only confirmed sink receipts increase committed counts and
coverage. Failed writes can retry the same frozen episode within the configured
submission limit.

Final evidence retains planning checks under ``planned_*`` IDs. Measured-path
collision results, ``actual_motion_limits``, and ``motion_quality`` are separate
acceptance checks; a passing planned path cannot replace their results.

The runner is single-use and closes its supplied host and sink when ``run``
exits. It returns a report with counters, coverage, audit, resolved settings,
package version, and ``target_reached``, and writes ``generation_report.json``
under the sink root. Budget exhaustion can return without reaching the requested
target. Required integration, restoration, or persistence failures raise after
cleanup and report generation.

``EnvRowMotionPlanner`` supports explicitly free, unloaded motion. The separate
``PickUpMotionValidator`` below adds changing gripper geometry, held-object
paths and physical contact gates for offline atomic qpos replay. The public
runner interface does not constitute full M1 runtime/source/host acceptance.

.. autoclass:: MotionLimitsProfile
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: GenerationRunner
   :members:

Synchronous Episode Persistence
-------------------------------

.. currentmodule:: embodichain.lab.trajectory_generation.sinks

.. autosummary::
   :nosignatures:

   LeRobotEpisodeSink

``LeRobotEpisodeSink`` writes each accepted ``ExpertEpisode`` into its own local
LeRobot dataset shard. Each shard contains exactly ``T`` causal training frames;
required sidecars retain the terminal observation, all ``T+1`` measured
timestamps, identity, validation, phases, and metadata. A collection manifest
maps committed episode IDs to their shards.

``submit`` owns a CPU copy, writes and finalizes the dataset, reopens every frame
and required image, verifies sidecars, then replaces and reads back the manifest
before returning a confirmed ``CommitReceipt``. Invalid inputs raise before
episode writes; persistence failures return an unconfirmed receipt. Reusing a
confirmed commit ID with an identical payload only verifies existing artifacts;
it does not rewrite them or create another logical episode.
Retries can reuse readable sealed data, while incomplete uncommitted data is
rebuilt at the same shard path. Changed payloads under a known commit ID are
rejected.

The sink accepts fixed-rate float32/float64 action vectors, supported numeric
observation vectors/matrices, and uint8 RGB images. Matrices flatten to C-order
vectors in LeRobot, retaining their source shape in ``observation_shapes`` and
original terminal matrix in the sidecar. Training timestamps use LeRobot's
relative fixed clock; the sidecar retains exact measured times and their origin.
Numeric readback checks the stored Parquet precision, while image readback uses
the LeRobot image decoder.
``max_episode_bytes`` limits raw tensor and metadata payload bytes for each
episode, not total process memory or collection disk size. Submission is
synchronous, so ``drain`` returns no deferred receipts and ``close`` releases the
writer lease after already completed writes.

The sink requires a new or empty directory and one serialized caller. A confirmed
receipt means required artifacts were closed and successfully read back; process
restart and power-loss recovery are not implemented. It does not evaluate the
physical truth of supplied validation evidence or upload anything to the Hub.

.. autoclass:: LeRobotEpisodeSink
   :members:

PickUp Source and Contact Validation
------------------------------------

.. currentmodule:: embodichain.lab.trajectory_generation.integrations.atomic

.. autosummary::
   :nosignatures:

   export_pickup_templates
   AtomicCandidateGenerationCfg
   AtomicCandidateRejection
   AtomicGenerationResult
   AtomicTrajectoryGenerator

The export accepts exactly a successful ``MoveEndEffector`` → ``PickUp``
compilation. It preserves the atomic phase boundaries, expands passive mimic
geometry, makes the shared action boundary an explicit control-period hold,
and appends terminal hold commands. Only transit permits joint residuals.
It produces qpos replay templates; it does not execute AtomicActionRuntime or
commit the compilation's hypothetical symbolic effects.

.. autofunction:: export_pickup_templates

``AtomicTrajectoryGenerator`` provides a separate multi-grasp source. One
canonical scene may occupy several verified real replica rows; independent
grasp/roll branches are compiled in bounded waves. The supported sequence is
optional MoveEndEffector/MoveJoints prefixes followed by one PickUp, using
``ik_interp``. Failed grasp, IK and path branches are filtered independently.
Output is a compact ``CandidateTrajectoryBatch`` with explicit timing, lengths,
IDs and first-failure audit. Short rows repeat the final position with zero
padding time; an all-failed result has shape ``(0, 0, D_full)``.

This source does not step physics, change observed attachment state, or
certify expert episodes. A shared ``GenerationSession`` requires an explicit
passing collision validator for ready admission. Planning-only exports use
private identity accounting without reserving confirmed coverage. Automatic
IK retries/resampling, held-object suffixes, native candidate-capacity buckets
and a general ``GenerationRunner.run_source`` are not implemented.

.. autoclass:: AtomicCandidateGenerationCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: AtomicCandidateRejection
   :members:

.. autoclass:: AtomicGenerationResult
   :members:

.. autoclass:: AtomicTrajectoryGenerator
   :members:

.. currentmodule:: embodichain.lab.trajectory_generation.integrations.atomic_candidates

.. autosummary::
   :nosignatures:

   AtomicCandidateGenerationCfg
   AtomicCandidateRejection
   AtomicGenerationResult
   AtomicTrajectoryGenerator

.. currentmodule:: embodichain.lab.trajectory_generation.replicas

.. autosummary::
   :nosignatures:

   CandidateSlotAssignment
   SceneReplicaPool

``SceneReplicaPool`` verifies equal source initial states, local-arena poses,
joint order and trusted fixed-condition signatures before sharing slots.
``from_host`` additionally binds the preparation epoch and checks the host on
reuse. Pose equality alone does not prove equal collision geometry or physical
parameters; the trusted preparation profile owns that evidence. Physical
environment IDs remain unique and source identities never derive from slots.

.. autoclass:: CandidateSlotAssignment
   :members:

.. autoclass:: SceneReplicaPool
   :members:

.. currentmodule:: embodichain.lab.trajectory_generation.integrations.contact

.. autosummary::
   :nosignatures:

   PickUpContactProfile
   PickUpMotionValidator

Use the same ``PickUpMotionValidator`` as the Runner planner and the executor's
``contact_validator``. Supported geometry is an unscaled fixed-base URDF robot,
cuboid rigid objects, and the simulator's implicit ground. Planning uses
conservative hulls of URDF collision shapes and full-joint FK; actual validation
uses measured joints and measured object poses. The two-hop URDF self-collision
exclusions match the existing cuRobo structural policy. This remains sampled
validation, with additional native contact evidence at each CPU physics substep.

The profile declares the target, support, finger links, fixed mounting links,
TCP, limited finger-contact entry region, penetration tolerance, and physical
hold thresholds. Unknown bodies, cross-row contact, buffer overflow, forbidden
contacts, missing finger contact, slipping and falling fail acceptance. The
profile does not enable contact or timing augmentation. Gym and arbitrary
mesh-shaped rigid objects are not supported by this integration.

.. autoclass:: PickUpContactProfile
   :members:
   :show-inheritance:

.. autoclass:: PickUpMotionValidator
   :members:
   :show-inheritance:
