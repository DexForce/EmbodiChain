embodichain.lab.sim.motion.expansion
==================================================

The :mod:`embodichain.lab.sim.motion.expansion` package provides
qpos contracts, constrained trajectory operators, measured coverage, and a
generation session for simulation expert trajectories. The core algorithms do
not directly import Gym or own simulation stepping. Public imports pass through
``embodichain.lab`` and ``embodichain.lab.sim`` initialization and therefore
require the normal simulation dependencies.

This is the motion core for fixed-scene expert generation. Full-batch physical
initial-state restoration, supported free-motion EEF/qpos checks, actual qpos
execution, the synchronous runner, and episode persistence are provided by the
separate :doc:`generation host API <embodichain.lab.trajectory_generation>`.
That layer supplies actual observations and commands, validation evidence, and
persistence confirmations. The low-level candidate execution entry points in the
simulation and Gym layers are separate from this value-only API.

.. currentmodule:: embodichain.lab.sim.motion.expansion

.. autosummary::
   :nosignatures:

   CandidateIdentity
   CandidateTrajectoryBatch
   CommitReceipt
   ExpertEpisode
   MotionSnapshot
   SceneCase
   TrajectoryAugmentationCfg
   TrajectoryGenerationJobCfg
   TrajectoryPhase
   TrajectoryTemplate
   ValidationCheck
   ValidationResult
   TrajectoryDescriptor
   describe_trajectory
   CoverageIndex
   rotate_grasp_about_object_axis
   joint_residual
   retime
   validate_motion_limits
   GenerationSession

Values and Evidence
~~~~~~~~~~~~~~~~~~~

``SceneCase`` separates scene and initial-state identity from physical execution
slots. ``MotionSnapshot`` copies robot positions and velocities, the robot root
pose, entity poses, and dependency revisions. Pose validation checks finite
proper SE(3) transforms, including rotation orthogonality and handedness.
Tensor inputs are detached and cloned at construction; consumers must still
treat the resulting owned tensors as read-only.

Templates declare the complete joint order and use explicit qpos values.
``dt[0]`` is zero and each subsequent ``dt`` is the positive arrival interval
from the preceding sample. Phase ranges are half-open sample intervals
``[start_index, stop_index)``. Unannotated paths permit replay only; operators
also require explicit template, phase, and controlled-joint permissions.

Candidate batches have shape ``(C, N, D_full)``, where the logical candidate
count ``C`` does not prescribe the host's physical batch size. ``valid_length``
selects actual samples. Padding holds the final valid position and uses zero
time intervals; it is excluded from execution and recording. Extracting a row
copies its tensor payload and retains aligned identity, phase, factor, and
source-row metadata.

An ``ExpertEpisode`` contains ``T`` commands actually sent and ``T+1``
observations and timestamps, including the terminal observation. Episode
metadata accepts finite JSON values rather than live host objects. A validation
result accepts only a nonempty collection of passing checks; ``not_run``,
``failed``, and ``unavailable`` remain distinct rejection states. This structural
check does not generate physical evidence or certify the supplied checker.

Persistence uses stable episode, candidate, scene, and commit IDs. The commit ID
remains fixed across write retries, while ``submission_id`` distinguishes each
submission and its delayed receipt. A persistence receipt is an input contract;
the package itself does not write or verify storage.

.. autoclass:: SceneCase
   :members:

.. autoclass:: MotionSnapshot
   :members:

.. autoclass:: TrajectoryPhase
   :members:

.. autoclass:: TrajectoryTemplate
   :members:

.. autoclass:: CandidateIdentity
   :members:

.. autoclass:: CandidateTrajectoryBatch
   :members:

.. autoclass:: ValidationCheck
   :members:

.. autoclass:: ValidationResult
   :members:

.. autoclass:: ExpertEpisode
   :members:

.. autoclass:: CommitReceipt
   :members:

Configuration and Preflight
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TrajectoryAugmentationCfg`` owns the local seed, provided-start declaration,
spatial and timing factors, and joint-geometry coverage limits.
``TrajectoryGenerationJobCfg`` adds the nested ``source``, ``planning``,
``execution``, ``reset``, ``validation``, ``collection``, and ``persistence``
sections used by a standalone generation job. Both ``from_mapping`` methods
reject unknown fields and invalid types or ranges. ``validate_semantics``
rechecks mutable configuration objects.

The initial schema requires ``planning.batch_mode: env_rows``,
``execution.pool_mode: per_env_case``, ``execution.scheduler: full_batch``,
provided initial states, and synchronous persistence. It limits each candidate
to one rollout attempt. Write retries reuse the same episode and commit ID.
Other scheduling modes, overlapping planning and physics, and unsupported
enabled factors are rejected. Control periods remain owned by the host.

Configuration IDs, including the default ``lerobot`` sink name, do not create
services. ``validate_capabilities`` requires explicit trusted source, validator,
profile, sink, and operator registries; configuration values are not imported
or evaluated. In particular, accepting a restoration profile ID or a supplied
``via_points`` capability does not implement physical restoration or an EEF
planner in this package. Deployment preflight must still verify those services.

.. autoclass:: TrajectoryAugmentationCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: TrajectoryGenerationJobCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

Operators, Coverage, and Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rotate_grasp_about_object_axis`` derives TCP pose candidates from a reference
grasp, rotating its position and orientation about an object-local axis through
the fixed object origin. The caller selects rotations appropriate to the
object's geometry, such as quarter turns for a cube, and replans each candidate.
The operator does not move the object or establish contact/IK validity.

``joint_residual`` samples one smooth residual per explicitly permitted free
phase and preserves its endpoints and uncontrolled joints. ``retime`` rescales
permitted free phases and resamples on the supplied host control period,
remapping phase indices. Contact and hold durations are preserved. Generated
paths still need dynamic, collision, and task checks.
``validate_motion_limits`` supplies a sampled finite-difference speed and
acceleration check; it does not assert collision freedom or task success.

``describe_trajectory`` computes phase-aligned normalized joint geometry and
elapsed-time descriptors from actual observations. ``CoverageIndex`` reserves
capacity before persistence and counts geometry only after confirmation.
Timing variants share geometry-family membership. This first descriptor uses
joint geometry; it does not provide EEF workspace coverage.

``GenerationSession`` owns scene-case registration, stable candidate identity,
local random streams, bounded candidate and pending-episode payloads, collection
budgets, coverage reservations, and commit accounting. Its lifecycle accepts
planning results and measured episode evidence without stepping or resetting a
host. Duplicate receipts do not increment committed counts, and failed writes
release collection and coverage reservations before an explicit write retry.

.. autofunction:: rotate_grasp_about_object_axis

.. autofunction:: joint_residual

.. autofunction:: retime

.. autofunction:: validate_motion_limits

.. autoclass:: TrajectoryDescriptor
   :members:

.. autofunction:: describe_trajectory

.. autoclass:: CoverageIndex
   :members:

.. autoclass:: GenerationSession
   :members:

Implementation Modules
~~~~~~~~~~~~~~~~~~~~~~

The following module paths expose the same contracts and implementations.
The package import path above is convenient for callers combining them.

.. currentmodule:: embodichain.lab.sim.motion.expansion.contracts

.. autosummary::
   :nosignatures:

   SceneCase
   MotionSnapshot
   TrajectoryPhase
   TrajectoryTemplate
   CandidateIdentity
   CandidateTrajectoryBatch
   ValidationCheck
   ValidationResult
   ExpertEpisode
   CommitReceipt

.. currentmodule:: embodichain.lab.sim.motion.expansion.cfg

.. autosummary::
   :nosignatures:

   TrajectoryAugmentationCfg
   TrajectoryGenerationJobCfg

.. currentmodule:: embodichain.lab.sim.motion.expansion.coverage

.. autosummary::
   :nosignatures:

   TrajectoryDescriptor
   describe_trajectory
   CoverageIndex

.. currentmodule:: embodichain.lab.sim.motion.expansion.operators

.. autosummary::
   :nosignatures:

   rotate_grasp_about_object_axis
   joint_residual
   retime
   validate_motion_limits

.. currentmodule:: embodichain.lab.sim.motion.expansion.session

.. autosummary::
   :nosignatures:

   GenerationSession
