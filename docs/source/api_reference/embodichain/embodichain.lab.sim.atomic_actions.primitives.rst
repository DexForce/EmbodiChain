embodichain.lab.sim.atomic_actions.primitives
============================================

.. automodule:: embodichain.lab.sim.atomic_actions.primitives

Overview
--------

Concrete implementations of the built-in atomic-action primitives. Each
primitive is an :class:`~embodichain.lab.sim.atomic_actions.AtomicAction` that
accepts a typed target and a
:class:`~embodichain.lab.sim.atomic_actions.WorldState`, plans a full-DoF
trajectory for all parallel environments, and returns an
:class:`~embodichain.lab.sim.atomic_actions.ActionResult`. The primitives are
chained by :class:`~embodichain.lab.sim.atomic_actions.AtomicActionEngine`,
which threads ``WorldState`` from one action to the next and concatenates the
resulting trajectories along the time axis.

   .. rubric:: Built-in Primitive Actions

   .. autosummary::

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
      CoordinatedPickmentCfg
      CoordinatedPickment
      CoordinatedPlacementCfg
      CoordinatedPlacement

   .. rubric:: Built-in Target Contracts

   .. autosummary::

      EndEffectorPoseTarget
      JointPositionTarget
      NamedJointPositionTarget
      GraspTarget
      HeldObjectPoseTarget
      PlaceTarget
      PressTarget
      CoordinatedPickTarget
      CoordinatedPickmentTarget
      CoordinatedPlacementTarget

.. currentmodule:: embodichain.lab.sim.atomic_actions.primitives

MoveEndEffector
---------------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.move_end_effector
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

MoveJoints
----------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.move_joints
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

PickUp
------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.pick_up
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

MoveHeldObject
--------------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.move_held_object
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Place
-----

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.place
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Press
-----

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.press
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

CoordinatedPickment
-------------------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.coordinated_pickment
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

CoordinatedPlacement
--------------------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.coordinated_placement
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict
