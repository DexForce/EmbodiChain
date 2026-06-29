embodichain.lab.sim.atomic_actions
==================================

.. automodule:: embodichain.lab.sim.atomic_actions

   .. rubric:: Classes

   .. autosummary::

      Affordance
      AntipodalAffordance
      InteractionPoints
      ObjectSemantics
      EndEffectorPoseTarget
      JointPositionTarget
      NamedJointPositionTarget
      GraspTarget
      HeldObjectPoseTarget
      CoordinatedPlacementTarget
      Target
      HeldObjectState
      WorldState
      ActionResult
      ActionCfg
      AtomicAction
      TrajectoryBuilder
      MoveEndEffectorCfg
      MoveEndEffector
      MoveJointsCfg
      MoveJoints
      PickUpCfg
      PickUp
      MoveHeldObjectCfg
      MoveHeldObject
      PlaceCfg
      Place
      PressCfg
      Press
      CoordinatedPlacementCfg
      CoordinatedPlacement
      AtomicActionEngine

.. currentmodule:: embodichain.lab.sim.atomic_actions

Core
----

.. autoclass:: Affordance
    :members:
    :show-inheritance:

.. autoclass:: AntipodalAffordance
    :members:
    :show-inheritance:

.. autoclass:: InteractionPoints
    :members:
    :show-inheritance:

.. autoclass:: ObjectSemantics
    :members:
    :show-inheritance:

.. autoclass:: EndEffectorPoseTarget
    :members:
    :show-inheritance:

.. autoclass:: JointPositionTarget
    :members:
    :show-inheritance:

.. autoclass:: NamedJointPositionTarget
    :members:
    :show-inheritance:

.. autoclass:: GraspTarget
    :members:
    :show-inheritance:

.. autoclass:: HeldObjectPoseTarget
    :members:
    :show-inheritance:

.. autoclass:: CoordinatedPlacementTarget
    :members:
    :show-inheritance:

.. autodata:: Target

.. autoclass:: HeldObjectState
    :members:
    :show-inheritance:

.. autoclass:: WorldState
    :members:
    :show-inheritance:

.. autoclass:: ActionResult
    :members:
    :show-inheritance:

.. autoclass:: ActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: AtomicAction
    :members:
    :show-inheritance:

Trajectory helpers
------------------

.. autoclass:: TrajectoryBuilder
    :members:
    :show-inheritance:

Actions
-------

.. autoclass:: MoveEndEffectorCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: MoveEndEffector
    :members:
    :show-inheritance:

.. autoclass:: MoveJointsCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: MoveJoints
    :members:
    :show-inheritance:

.. autoclass:: PickUpCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: PickUp
    :members:
    :show-inheritance:

.. autoclass:: MoveHeldObjectCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: MoveHeldObject
    :members:
    :show-inheritance:

.. autoclass:: PlaceCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: Place
    :members:
    :show-inheritance:

.. autoclass:: PressCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: Press
    :members:
    :show-inheritance:

.. autoclass:: CoordinatedPlacementCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: CoordinatedPlacement
    :members:
    :show-inheritance:

Engine & Registry
-----------------

.. autoclass:: AtomicActionEngine
    :members:
    :show-inheritance:

.. autofunction:: register_action

.. autofunction:: unregister_action

.. autofunction:: get_registered_actions
