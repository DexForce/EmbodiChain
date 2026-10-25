embodichain.lab.sim.atomic_skills.primitives
=============================================

.. automodule:: embodichain.lab.sim.atomic_skills.primitives
   :members:
   :no-index:

Overview
--------

Concrete implementations of the built-in atomic-skill primitives. Each
primitive is an :class:`~embodichain.lab.sim.atomic_skills.AtomicSkill` that
accepts a :class:`~embodichain.lab.sim.atomic_skills.ResolvedSkillRequest` and a
:class:`~embodichain.lab.sim.atomic_skills.PlanningContext`. Planning returns a
side-effect-free :class:`~embodichain.lab.sim.atomic_skills.SkillPlan` with a
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

.. currentmodule:: embodichain.lab.sim.atomic_skills.primitives

MoveEndEffector
---------------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.move_end_effector
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

MoveJoints
----------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.move_joints
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

PickUp
------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.pick_up
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

AxisAlign
---------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.axis_align
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

MoveHeldObject
--------------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.move_held_object
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Pour
----

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.pour
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

PushObject
----------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.push_object
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Place
-----

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.place
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Press
-----

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.press
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Slide
-----

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.slide
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

OpenDoor
--------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.open_door
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

Twist
-----

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.twist
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

CoordinatedPickment
-------------------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.coordinated_pickment
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

CoordinatedPlacement
--------------------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.coordinated_placement
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict

HandOver
--------

.. automodule:: embodichain.lab.sim.atomic_skills.primitives.hand_over
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict
