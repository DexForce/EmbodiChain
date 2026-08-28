embodichain.lab.sim.atomic_actions.primitives
=============================================

.. automodule:: embodichain.lab.sim.atomic_actions.primitives
   :members:
   :no-index:

Overview
--------

Concrete implementations of the built-in atomic-action primitives. Each
primitive is an :class:`~embodichain.lab.sim.atomic_actions.AtomicAction` that
accepts a :class:`~embodichain.lab.sim.atomic_actions.ResolvedActionRequest` and a
:class:`~embodichain.lab.sim.atomic_actions.PlanningContext`. Planning returns a
side-effect-free :class:`~embodichain.lab.sim.atomic_actions.ActionPlan` with a
full-robot timed trajectory and uncommitted expected effects.

   .. rubric:: Built-in Primitive Actions

   .. autosummary::

      MoveEndEffector
      MoveEndEffectorOptions
      MoveJoints
      MoveJointsOptions
      PickUp
      PickUpOptions
      AxisAlign
      AxisAlignOptions
      MoveHeldObject
      MoveHeldObjectOptions
      Pour
      PourOptions
      PushObject
      PushObjectOptions
      PushObjectToolCalibration
      Place
      PlaceOptions
      Press
      PressOptions
      Slide
      SlideOptions
      OpenDoor
      OpenDoorOptions
      Twist
      TwistOptions
      CoordinatedPickment
      CoordinatedPickmentOptions
      CoordinatedPlacement
      CoordinatedPlacementOptions
      HandOver
      HandOverGoal
      HandOverOptions

   .. rubric:: Built-in Goal Contracts

   .. autosummary::

      EndEffectorPoseGoal
      JointPositionGoal
      GraspGoal
      AxisAlignGoal
      HeldObjectPoseGoal
      PourGoal
      PushObjectGoal
      PlaceGoal
      AssembleGoal
      PressGoal
      SlideGoal
      OpenDoorGoal
      TwistGoal
      CoordinatedPickGoal
      CoordinatedPlacementGoal
      HandOverGoal

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

AxisAlign
---------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.axis_align
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

MoveHeldObject
--------------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.move_held_object
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Pour
----

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.pour
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

PushObject
----------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.push_object
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

Slide
-----

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.slide
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

OpenDoor
--------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.open_door
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Twist
-----

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.twist
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

HandOver
--------

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.hand_over
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict
