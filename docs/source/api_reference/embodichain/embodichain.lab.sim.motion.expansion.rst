embodichain.lab.sim.motion.expansion
==================================================

The :mod:`embodichain.lab.sim.motion.expansion` package provides
qpos contracts, constrained trajectory operators, measured coverage, and a
generation session for simulation expert trajectories. The core algorithms do
not directly import Gym or own simulation stepping. Public imports pass through
``embodichain.lab`` and ``embodichain.lab.sim`` initialization and therefore
require the normal simulation dependencies.

This is the motion core for fixed-scene expert generation. Physical
initial-state restoration, planning, rollout execution, task validation, and
episode persistence must be supplied by separate host integrations. Those
integrations provide actual observations and commands, validation evidence,
and persistence confirmations; this package does not instantiate them.

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
   perturb_approach_direction
   joint_residual
   via_points
   nullspace_residual
   retime
   TIMING_PROFILES
   validate_motion_limits
   NOMINAL_OPERATOR
   TrajectoryVariant
   TrajectoryVariantSet
   plan_trajectory_variants
   apply_trajectory_variant
   expand_trajectory_variants
   expand_row_variants
   sample_approach_cone
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
the spatial, redundancy (``ik``), approach and timing factors, and the
joint-geometry coverage limits.
``TrajectoryGenerationJobCfg`` adds the nested ``source``, ``planning``,
``execution``, ``reset``, ``validation``, ``collection``, and ``persistence``
sections used by a standalone generation job. Both ``from_mapping`` methods
reject unknown fields and invalid types or ranges. ``validate_semantics``
rechecks mutable configuration objects.

The initial schema requires ``planning.batch_mode: env_rows``,
``execution.pool_mode: per_env_case``, ``execution.scheduler: full_batch``,
provided initial states, and synchronous persistence. It limits each candidate
to one rollout attempt. Write retries reuse the same episode and commit ID.
Other scheduling modes, overlapping planning and physics, and the still
unimplemented ``contact``, ``contact_timing`` and ``recovery`` factors are
rejected. Control periods remain owned by the host.

Configuration IDs, including the default ``lerobot`` sink name, do not create
services. ``validate_capabilities`` requires explicit trusted source, validator,
profile, sink, and operator registries; configuration values are not imported
or evaluated. In particular, accepting a restoration profile ID does not implement physical
restoration in this package, and accepting the ``perturb_approach_direction``
capability does not implement the EEF replanning its poses require. Deployment preflight must still verify those services.

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

``perturb_approach_direction`` places standoff poses on a cone around a
nominal approach direction while leaving the contact transform itself exact.
Because it produces Cartesian poses, it must be consumed before IK and path
planning by whichever component builds the waypoints; it cannot be applied to
a qpos template.

``joint_residual`` samples one smooth residual per explicitly permitted free
phase and preserves its endpoints and uncontrolled joints. ``via_points``
routes a phase through several sampled interior knots instead, using clamped
cubic Hermite segments, so two or more knots produce paths that differ in
shape rather than only in amplitude. ``nullspace_residual`` projects a residual
onto the null space of caller-supplied task Jacobians, changing arm posture
while holding the declared task rows to **first order**; its phase endpoints
remain exact because the envelope vanishes there, and interior samples require
forward-kinematics verification by the host.

``retime`` rescales permitted free phases and resamples on the supplied host
control period, remapping phase indices. Contact and hold durations are
preserved. Its ``profile`` argument additionally redistributes time within a
phase without changing that phase's total duration or its geometric path, which
turns one path into several velocity profiles; ``TIMING_PROFILES`` lists the
available warps. Generated paths still need dynamic, collision, and task checks.
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

.. autofunction:: perturb_approach_direction

.. autofunction:: joint_residual

.. autofunction:: via_points

.. autofunction:: nullspace_residual

.. autofunction:: retime

.. autodata:: TIMING_PROFILES

.. autofunction:: validate_motion_limits

.. autoclass:: TrajectoryDescriptor
   :members:

.. autofunction:: describe_trajectory

.. autoclass:: CoverageIndex
   :members:

.. autoclass:: GenerationSession
   :members:

Trajectory Variants
~~~~~~~~~~~~~~~~~~~

These helpers answer a narrower question than affordance expansion: given a
reference trajectory whose annotated waypoints are already settled, how many
genuinely different ways are there to execute it? Only the free motion between
annotated phase endpoints changes, so contacts, grasps, and placements stay
exactly where planning put them.

``plan_trajectory_variants`` enumerates deterministic factor combinations without
touching a trajectory. Ordinal zero is always the unmodified reference, and
later ordinals cycle through the enabled spatial operators, then the duration
scales, then the time warps. ``apply_trajectory_variant`` applies one such
combination, running at most one joint-path operator so a null-space
projection is never stacked on an already displaced joint path.

``expand_trajectory_variants`` collects several modes for one fixed scene. It
rejects proposals that an operator refuses, that fail sampled motion limits, or
whose measured geometry and timing duplicate an accepted row, and it reports
every rejection rather than dropping it silently. ``expand_row_variants`` instead
gives one mode to each of several independent execution rows, such as one per
parallel environment with its own randomized scene. Rows from different scenes
are not comparable, so nothing is deduplicated across them and a rejected row
falls back to its own reference instead of leaving an environment without a
command stream.

Deduplication compares measured joint geometry and elapsed phase time. It does
not certify collision freedom, task success, or dynamic feasibility; each
accepted row must still be executed and validated by the host.
``sample_approach_cone`` samples angles for the Cartesian approach operator and
is likewise a geometric proposal only.

.. autoclass:: TrajectoryVariant
   :members:

.. autoclass:: TrajectoryVariantSet
   :members:

.. autodata:: NOMINAL_OPERATOR

.. autofunction:: plan_trajectory_variants

.. autofunction:: apply_trajectory_variant

.. autofunction:: expand_trajectory_variants

.. autofunction:: expand_row_variants

.. autofunction:: sample_approach_cone

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
   perturb_approach_direction
   joint_residual
   via_points
   nullspace_residual
   retime
   TIMING_PROFILES
   validate_motion_limits

.. currentmodule:: embodichain.lab.sim.motion.expansion.variants

.. autosummary::
   :nosignatures:

   NOMINAL_OPERATOR
   TrajectoryVariant
   TrajectoryVariantSet
   plan_trajectory_variants
   apply_trajectory_variant
   expand_trajectory_variants
   expand_row_variants
   sample_approach_cone

.. currentmodule:: embodichain.lab.sim.motion.expansion.session

.. autosummary::
   :nosignatures:

   GenerationSession
