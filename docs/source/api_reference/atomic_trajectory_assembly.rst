:orphan:

Trajectory-backed Atomic Skill plans
===================================

Ordinary planning, admitted real recordings and trajectory candidates can share
a Skill-owned semantic assembler. The source supplies motion and annotations;
the Skill supplies expected effects and scene-monitor boundaries. The existing
plan builder lowers commands and tracking, and the existing execution runtime
retains effect assurance, recovery and state commits.

Contracts and owned inputs
-------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions

.. autosummary::

   SkillTrajectoryGeometry
   SkillTrajectoryContract
   SkillTrajectoryInput
   SkillPlanSemantics

.. automodule:: embodichain.lab.sim.atomic_actions.trajectory_assembly
   :members:

An installed ``SkillDescriptor.trajectory_contract`` explicitly declares the
accepted schema and revision, semantic stages, event IDs and optional geometric
facts type. Unsupported Skills reject trajectory assembly before ordinary
planning. The declaration is included in the Task Program registration
fingerprint. Geometry snapshots belong to the declared concrete type; they do
not establish measured physical evidence.

Stages cover the execution path with half-open ranges. Event landmarks index
actual samples, so a stage-end event refers to ``stop - 1``. These semantic
annotations do not authorize expansion: existing ``TrajectoryPhase`` windows,
operators and joint permissions remain authoritative.

Assembly and rebuilding
-----------------------

.. automethod:: embodichain.lab.sim.atomic_actions.AtomicAction.assemble_trajectory_plan

.. automethod:: embodichain.lab.sim.atomic_actions.AtomicActionEngine.build_plan_from_trajectory

``build_plan_from_trajectory`` checks the restored start state, full-joint limits,
sampled and interval speeds, and endpoint authorization before Skill assembly.
This is discrete trajectory admission, not collision or physical-success proof.
Sources retain model/joint mapping, calibration, input review and host-specific
validation responsibilities.

``ActionPlan.trajectory_input`` retains an owned assembly snapshot. The existing
``rebuild_plan_from_trajectory`` delegates supported plans to the same assembler,
rebuilding commands, tracking, semantic effects and dependency bounds together.
It still requires identical frame counts and arrival intervals. A candidate
that changes a geometric event while retaining old geometric facts is rejected;
the caller must create a new geometry family and assembly input.

The original ``build_plan_from_recorded_trajectory`` engine method remains a
compatibility adapter using declared stage-end landmarks. It contains no
Pick-specific stage or effect logic. A replay-only source still requires a
fresh reset for another attempt and cannot enable recovery by changing an input
flag.

Implemented Skills
------------------

PickUp and MoveJoints use the shared assembler for both ordinary planning and
external joint motion. PickUp derives an omitted expected grasp pose by FK at
the explicit grasp landmark, then declares an uncommitted held-object effect.
MoveJoints checks its terminal goal and ordered waypoint visits. Existing
non-joint command plans and Skills without an assembly contract retain their
original planning path.

.. autoclass:: embodichain.lab.sim.atomic_actions.primitives.pick_up.PickTrajectoryGeometry
   :members:

The generic assembler supports batches sharing one stage layout and preserves
ordered ``env_ids``. The registered real2sim source remains B=1 fixed-qpos Pick;
the new Skill interface does not expand its admitted capabilities. Ragged
execution, changed-grid rebuilding, additional Skill assemblers and registered
source combined expansion require separate integration.
