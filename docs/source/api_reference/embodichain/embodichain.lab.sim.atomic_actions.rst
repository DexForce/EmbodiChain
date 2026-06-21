embodichain.lab.sim.atomic_actions
==================================

.. automodule:: embodichain.lab.sim.atomic_actions

   .. rubric:: Classes

   .. autosummary::

      Affordance
      InteractionPoints
      ObjectSemantics
      PoseTarget
      GraspTarget
      HeldObjectTarget
      Target
      HeldObjectState
      WorldState
      ActionResult
      ActionCfg
      AtomicAction
      TrajectoryBuilder
      MoveActionCfg
      MoveAction
      PickUpActionCfg
      PickUpAction
      MoveObjectActionCfg
      MoveObjectAction
      PlaceActionCfg
      PlaceAction
      AtomicActionEngine

.. currentmodule:: embodichain.lab.sim.atomic_actions

Core
----

.. autoclass:: Affordance
    :members:
    :show-inheritance:

.. autoclass:: InteractionPoints
    :members:
    :show-inheritance:

.. autoclass:: ObjectSemantics
    :members:
    :show-inheritance:

.. autoclass:: PoseTarget
    :members:
    :show-inheritance:

.. autoclass:: GraspTarget
    :members:
    :show-inheritance:

.. autoclass:: HeldObjectTarget
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

.. autoclass:: MoveActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: MoveAction
    :members:
    :show-inheritance:

.. autoclass:: PickUpActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: PickUpAction
    :members:
    :show-inheritance:

.. autoclass:: MoveObjectActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: MoveObjectAction
    :members:
    :show-inheritance:

.. autoclass:: PlaceActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict
    :show-inheritance:

.. autoclass:: PlaceAction
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
