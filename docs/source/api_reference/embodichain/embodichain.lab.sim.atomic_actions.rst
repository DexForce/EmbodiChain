embodichain.lab.sim.atomic_actions
==================================

.. automodule:: embodichain.lab.sim.atomic_actions

   .. rubric:: Planning contracts

   .. autosummary::

      ActionGoal
      ActionBinding
      ActionInvocation
      MotionPolicy
      RecoveryPolicy
      RobotObservation
      TaskState
      SceneSnapshot
      SceneEntityPose
      PlanningContext
      StateDelta
      TimedTrajectory
      PhaseSpec
      PlannedPhase
      ActionPlan
      CompiledTrajectory

   .. rubric:: Execution contracts

   .. autosummary::

      AtomicAction
      AtomicActionEngine
      ExecutionSession
      ExecutionTick
      JointCommand
      ExecutionEvent

   .. rubric:: Built-in goals and actions

   .. autosummary::

      EndEffectorPoseGoal
      JointPositionGoal
      GraspGoal
      HeldObjectPoseGoal
      PlaceGoal
      AssembleGoal
      PressGoal
      CoordinatedPickGoal
      CoordinatedPlacementGoal
      MoveEndEffector
      MoveJoints
      PickUp
      MoveHeldObject
      Place
      Press
      CoordinatedPickment
      CoordinatedPlacement
      HandOver

.. toctree::
   :maxdepth: 1
   :hidden:

   embodichain.lab.sim.atomic_actions.primitives

.. currentmodule:: embodichain.lab.sim.atomic_actions

Planning and state
------------------

.. autoclass:: ActionBinding
   :members:

.. autoclass:: ActionInvocation
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

.. autoclass:: ExecutionTick
   :members:

.. autoclass:: JointCommand
   :members:

.. autoclass:: ExecutionEvent
   :members:

Semantic objects and helpers
----------------------------

.. autoclass:: ObjectSemantics
   :members:

.. autoclass:: HeldObjectState
   :members:

.. autoclass:: CoordinatedHeldObjectState
   :members:

.. autoclass:: TrajectoryBuilder
   :members:
