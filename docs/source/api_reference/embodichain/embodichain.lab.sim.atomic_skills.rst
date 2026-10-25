embodichain.lab.sim.atomic_skills
==================================

.. automodule:: embodichain.lab.sim.atomic_skills
   :members:
   :no-index:

   .. rubric:: Planning contracts

   .. autosummary::

      SkillBinding
      EndpointBinding
      RuntimeEndpointTarget
      JointPositionTarget
      ControlCommand
      JointPositionCommand
      ControlPartCommandProfile
      SkillControlOverrides
      SkillInvocation
      PhaseEffectGateRequirement
      ResolvedSkillRequest
      SkillOptions
      MotionPolicy
      RecoveryPolicy
      RobotObservation
      TaskState
      SceneSnapshot
      SceneEntityPose
      PlanningContext
      StateDelta
      TimedTrajectory
      RuntimeCommandPayload
      JointPositionPayload
      EndpointCommand
      RuntimeCommandFrame
      TimedCommandSequence
      TrajectorySegment
      PlannerDiagnostics
      PlanningFailure
      SkillPlan
      CompiledTrajectory

   .. rubric:: Articulation geometry adaptation

   .. autosummary::

      ArticulationAffordanceGeometry
      ArticulationGeometryProvider
      ArticulationJointGeometry
      create_rigidized_articulation_antipodal_affordance
      sample_initial_articulation_geometry

   .. rubric:: Semantic resource contracts

   .. autosummary::

      SkillDescriptor
      SkillBindingContract
      SkillResourceSlot
      SkillEndpointRequirement
      DisjointSlotEndpoints
      DisjointResourceSlots

   .. rubric:: Execution contracts

   .. autosummary::

      AtomicSkill
      AtomicSkillEngine
      PlanTransform
      ExecutionSession
      ExecutionRunner
      ExecutionRunnerCfg
      RunnerStep
      RunnerStatus
      ObservationProvider
      CommandSink
      EndpointCommandTransport
      EndpointCommandRouter
      CommandAcknowledgement
      CommandAckStatus
      CommandDispatch
      CommandOperation
      ExecutionClock
      create_simulation_atomic_skill_engine
      create_rigidized_articulation_antipodal_semantics
      SceneEntity
      SimulationExecutionAdapter
      ExecutionTick
      EffectVerificationRequest
      EffectVerificationRequirement
      EffectVerificationResult
      PhysicalEvidenceBatch
      PhysicalEvidenceFrame
      PhysicalEvidenceProvider
      PhysicalEvidenceRequest
      PhaseEffectGateRequest
      PhaseEffectGateResult
      PhysicalEffectVerifier
      PhysicalPhaseEffectGateVerifier
      HeldObjectGuardRequest
      HeldObjectGuardResult
      PhysicalHeldObjectGuardVerifier
      ExecutionPlanAttempt
      ExecutionEvent
      ExecutionEventKind
      ExecutionStatus

   .. rubric:: Built-in goals and actions

   .. autosummary::

      EndEffectorPoseGoal
      JointPositionGoal
      GraspGoal
      HandOverGoal
      AxisAlignGoal
      AxisAlignOptions
      AxisAlignAffordance
      HeldObjectPoseGoal
      PourGoal
      PourOptions
      PushObjectGoal
      PushObjectOptions
      PushObjectToolCalibration
      PlaceGoal
      AssembleGoal
      PressGoal
      PressOptions
      PressAffordance
      SlideGoal
      SlideOptions
      SlideAffordance
      OpenDoorGoal
      OpenDoorOptions
      OpenDoorAffordance
      TwistGoal
      TwistOptions
      TwistAffordance
      CoordinatedPickGoal
      CoordinatedPlacementGoal
      MoveEndEffector
      MoveJoints
      PickUp
      AxisAlign
      MoveHeldObject
      Pour
      PushObject
      Place
      Press
      Slide
      OpenDoor
      Twist
      CoordinatedPickment
      CoordinatedPlacement
      HandOver

.. toctree::
   :maxdepth: 1
   :hidden:

   embodichain.lab.sim.atomic_skills.primitives
   embodichain.lab.sim.atomic_skills.tracking

.. currentmodule:: embodichain.lab.sim.atomic_skills

Articulation geometry adaptation
--------------------------------

The adapter converts deterministic articulation meshes, FK, and parent-joint
topology into sampled Atomic Skill affordance geometry. Initial state and scale
are explicit inputs; the simulation object does not own sampling or semantic
geometry keys.

.. currentmodule:: embodichain.lab.sim.atomic_skills.articulation_geometry

.. autoclass:: ArticulationGeometryProvider
   :members:

.. autoclass:: ArticulationJointGeometry
   :members:

.. autoclass:: ArticulationAffordanceGeometry
   :members:

.. autofunction:: create_rigidized_articulation_antipodal_affordance

.. autofunction:: sample_initial_articulation_geometry

.. currentmodule:: embodichain.lab.sim.atomic_skills

Semantic resource contracts
---------------------------

.. autoclass:: SkillDescriptor
   :members:

.. autoclass:: SkillBindingContract
   :members:

.. autoclass:: SkillResourceSlot
   :members:

.. autoclass:: SkillEndpointRequirement
   :members:

.. autoclass:: DisjointSlotEndpoints
   :members:

.. autoclass:: DisjointResourceSlots
   :members:

Standard capability identifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: JOINT_POSITION_CAPABILITY

.. autodata:: CARTESIAN_POSE_CAPABILITY

.. autodata:: FORWARD_KINEMATICS_CAPABILITY

.. autodata:: INVERSE_KINEMATICS_CAPABILITY

.. autodata:: BATCH_INVERSE_KINEMATICS_CAPABILITY

.. autodata:: GRASP_CAPABILITY

Planning and state
------------------

.. autoclass:: SkillBinding
   :members:

.. autoclass:: EndpointBinding
   :members:

.. autoclass:: RuntimeEndpointTarget
   :members:

.. autoclass:: JointPositionTarget
   :members:

.. autoclass:: ControlCommand
   :members:

.. autoclass:: JointPositionCommand
   :members:

.. autoclass:: ControlPartCommandProfile
   :members:

.. autoclass:: SkillControlOverrides
   :members:

.. autoclass:: SkillInvocation
   :members:

.. autoclass:: ResolvedSkillRequest
   :members:

.. autoclass:: SkillOptions
   :members:

.. autoclass:: MotionPolicy
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: RecoveryPolicy
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: PlanningContext
   :members:

.. autoclass:: RobotObservation
   :members:

.. autoclass:: TaskState
   :members:

.. autoclass:: SceneSnapshot
   :members:

.. autoclass:: SceneEntityPose
   :members:

.. autoclass:: StateDelta
   :members:

.. autoclass:: TimedTrajectory
   :members:

.. autoclass:: PlanningFailure
   :members:

.. autoclass:: RuntimeCommandPayload
   :members:

.. autoclass:: JointPositionPayload
   :members:

.. autoclass:: EndpointCommand
   :members:

.. autoclass:: RuntimeCommandFrame
   :members:

.. autoclass:: TimedCommandSequence
   :members:

.. autoclass:: SkillPlan
   :members:

.. autoclass:: SkillPlanTemplateAdapter
   :members:

.. currentmodule:: embodichain.lab.sim.atomic_skills.trajectory_adapter

.. autosummary::
   :nosignatures:

   SkillPlanTemplateAdapter

.. currentmodule:: embodichain.lab.sim.atomic_skills

Engine and execution
--------------------

.. autoclass:: AtomicSkill
   :members:

.. autoclass:: AtomicSkillEngine
   :members:

.. autodata:: PlanTransform

.. autoclass:: ExecutionSession
   :members:

.. autoclass:: ExecutionRunner
   :members:

.. autoclass:: ExecutionRunnerCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: ObservationProvider
   :members:

.. autoclass:: PhysicalEvidenceProvider
   :members:

.. autoclass:: PhysicalEvidenceRequest
   :members:

.. autoclass:: PhysicalEvidenceBatch
   :members:

.. autoclass:: PhysicalEvidenceFrame
   :members:

.. autoclass:: CommandSink
   :members:

.. autoclass:: EndpointCommandTransport
   :members:

.. autoclass:: EndpointCommandRouter
   :members:

.. autoclass:: ExecutionClock
   :members:

.. autoclass:: MonotonicExecutionClock
   :members:

.. autofunction:: create_simulation_atomic_skill_engine

.. autofunction:: create_rigidized_articulation_antipodal_semantics

.. autoclass:: SceneEntity
   :members:

.. autoclass:: SimulationExecutionAdapter
   :members:

.. autoclass:: CommandAcknowledgement
   :members:

.. autoclass:: CommandAckStatus
   :members:

.. autoclass:: CommandDispatch
   :members:

.. autoclass:: CommandOperation
   :members:

.. autoclass:: RunnerStep
   :members:

.. autoclass:: RunnerStatus
   :members:

.. autoclass:: ExecutionTick
   :members:

.. autoclass:: ExecutionEvent
   :members:

.. autoclass:: ExecutionEventKind
   :members:

.. autoclass:: ExecutionStatus
   :members:

.. autoclass:: EffectVerificationRequest
   :members:

.. autoclass:: EffectVerificationRequirement
   :members:

.. autoclass:: EffectVerificationResult
   :members:

.. autodata:: PhysicalEffectVerifier

.. autodata:: PhysicalPhaseEffectGateVerifier

.. autodata:: PhysicalHeldObjectGuardVerifier

.. autoclass:: ExecutionPlanAttempt
   :members:

Semantic objects and helpers
----------------------------

.. autoclass:: ObjectSemantics
   :members:

.. autoclass:: HeldObjectState
   :members:

Verification implementation module
----------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_skills.evidence

.. autosummary::

   PhysicalEvidenceBatch
   PhysicalEvidenceFrame
   PhysicalEvidenceProvider
   PhysicalEvidenceRequest

.. currentmodule:: embodichain.lab.sim.atomic_skills.runner

.. autosummary::

   PhysicalEffectVerifier
   PhysicalPhaseEffectGateVerifier
   PhysicalHeldObjectGuardVerifier

Evidence-backed execution keeps acquisition separate from the existing
symbolic effect verifier. The runner collects one aligned frame after each
fresh observation and passes it only to the explicitly selected evidence-aware
verifier.

.. currentmodule:: embodichain.lab.sim.atomic_skills.verification

.. autosummary::

   EffectExpectationResult
   EffectVerificationRequest
   EffectVerificationResult
   HeldObjectGuardRequest
   HeldObjectGuardResult
   PhaseEffectGateRequest
   PhaseEffectGateResult
