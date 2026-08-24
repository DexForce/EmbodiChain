embodichain.lab.sim.atomic_actions
==================================

.. automodule:: embodichain.lab.sim.atomic_actions
   :members:
   :no-index:

   .. rubric:: Planning contracts

   .. autosummary::

      ActionBinding
      EndpointBinding
      RuntimeEndpointTarget
      JointPositionTarget
      ControlCommand
      JointPositionCommand
      ControlPartCommandProfile
      ActionControlOverrides
      ActionInvocation
      ResolvedActionRequest
      ActionOptions
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
      ExecutionFeedbackMode
      ActionPlan
      CompiledTrajectory

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

      AtomicAction
      AtomicActionEngine
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
      SimulationExecutionAdapter
      ExecutionTick
      EffectVerificationRequest
      EffectVerificationRequirement
      EffectVerificationResult
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
      PlaceGoal
      AssembleGoal
      PressGoal
      PressOptions
      PressAffordance
      SlideGoal
      SlideOptions
      SlideAffordance
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
      Place
      Press
      Slide
      Twist
      CoordinatedPickment
      CoordinatedPlacement
      HandOver

.. toctree::
   :maxdepth: 1
   :hidden:

   embodichain.lab.sim.atomic_actions.primitives

.. currentmodule:: embodichain.lab.sim.atomic_actions

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

.. autoclass:: ActionBinding
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

.. autoclass:: ActionControlOverrides
   :members:

.. autoclass:: ActionInvocation
   :members:

.. autoclass:: ResolvedActionRequest
   :members:

.. autoclass:: ActionOptions
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

.. autoclass:: ExecutionFeedbackMode
   :members:

.. autoclass:: ActionPlan
   :members:

Engine and execution
--------------------

.. autoclass:: AtomicAction
   :members:

.. autoclass:: AtomicActionEngine
   :members:

.. autoclass:: ExecutionSession
   :members:

.. autoclass:: ExecutionRunner
   :members:

.. autoclass:: ExecutionRunnerCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: ObservationProvider
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

.. autoclass:: ExecutionPlanAttempt
   :members:

Semantic objects and helpers
----------------------------

.. autoclass:: ObjectSemantics
   :members:

.. autoclass:: HeldObjectState
   :members:
