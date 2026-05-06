embodichain.lab.sim.atomic_actions
==================================

.. automodule:: embodichain.lab.sim.atomic_actions

   .. rubric:: Classes

   .. autosummary::

      Affordance
      InteractionPoints
      ObjectSemantics
      ActionCfg
      AtomicAction
      MoveActionCfg
      MoveAction
      PickUpActionCfg
      PickUpAction
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

.. autoclass:: ActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: AtomicAction
    :members:
    :show-inheritance:

Actions
-------

.. autoclass:: MoveActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate
    :show-inheritance:

.. autoclass:: MoveAction
    :members:
    :show-inheritance:

.. autoclass:: PickUpActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate
    :show-inheritance:

.. autoclass:: PickUpAction
    :members:
    :show-inheritance:

.. autoclass:: PlaceActionCfg
    :members:
    :exclude-members: __init__, copy, replace, to_dict, validate
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
