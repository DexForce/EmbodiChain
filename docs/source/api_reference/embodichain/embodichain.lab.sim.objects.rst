embodichain.lab.sim.objects
==========================================

.. automodule:: embodichain.lab.sim.objects

Overview
--------

Scene-object classes spawned into the :class:`SimulationManager`. Every object
derives from :class:`~embodichain.lab.sim.common.BatchEntity` and pairs a
runtime class with a ``*Data`` buffer and a ``*Cfg`` config. The hierarchy
covers lights (``Light``), rigid bodies (``RigidObject`` and grouped
``RigidObjectGroup``), articulated chains (``Articulation``) and their robot
specialization (``Robot``), deformables (``SoftObject``, ``ClothObject``),
interactive ``Gizmo`` handles, and ``RigidConstraint`` attachments between
bodies.

  .. rubric:: Classes

  .. autosummary::

    Light
    LightCfg
    RigidObject
    RigidBodyData
    RigidObjectCfg
    RigidObjectGroup
    RigidBodyGroupData
    RigidObjectGroupCfg
    Articulation
    ArticulationJointKinematics
    ArticulationData
    ArticulationCfg
    SoftObject
    SoftBodyData
    SoftObjectCfg
    ClothObject
    ClothBodyData
    ClothObjectCfg
    Robot
    RobotCfg
    RobotWorkspaceCfg
    Gizmo
    GizmoCfg
    create_robot_ik_gizmo_controller
    RigidConstraint

.. currentmodule:: embodichain.lab.sim.objects

Light
-----

.. autoclass:: Light
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: LightCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Rigid Object
------------

.. autoclass:: RigidObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RigidBodyData
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RigidObjectCfg
    :members:       
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Rigid Object Group
-------------------

.. autoclass:: RigidObjectGroup
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RigidBodyGroupData
    :members:
    :inherited-members:
    :show-inheritance:  

.. autoclass:: RigidObjectGroupCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Articulation
------------

.. autoclass:: Articulation
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ArticulationJointKinematics
    :members:
    :show-inheritance:

.. autoclass:: ArticulationData
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ArticulationCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Soft Object
-----------

.. autoclass:: SoftObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: SoftBodyData
    :members:
    :show-inheritance:

.. autoclass:: SoftObjectCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Cloth Object
------------

.. autoclass:: ClothObject
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ClothBodyData
    :members:
    :show-inheritance:

.. autoclass:: ClothObjectCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Robot
-----

.. autoclass:: Robot
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RobotCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: RobotWorkspaceCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

Gizmo
-----

.. autoclass:: Gizmo
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: GizmoCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: create_robot_ik_gizmo_controller

SimulationManager automatically discovers robot control parts with configured
IK chain metadata. Native controllers activate on the first I press, while
Viser constructs its solver on the first drag. ``sim.update()`` owns updates
and cleanup; ordinary applications do not need the explicit factory.
Use ``SimulationManagerCfg(robot_ik_gizmo=None)`` to disable automatic setup.

The native controller defaults to DexSim Newton IK. With a ``PinkSolverCfg``
(or another EmbodiChain solver) configured for the robot's control part, pass
``GizmoCfg(ik_solver="embodichain")`` to select that solver for either a native
controller or a Viser gizmo. Its iteration limits and convergence settings
remain owned by the configured solver; ``ik_iterations`` applies to Newton IK.
Only the selected control part's drive targets are written, and failed
EmbodiChain IK solutions preserve the current joint positions.

The runnable example ``examples/sim/gizmo/gizmo_robot.py`` exposes
``--ik-solver dexsim|pytorch|pink`` for both the native window and ``--viser``.

Rigid Constraint
----------------

.. autoclass:: RigidConstraint
    :members:
    :inherited-members:
    :show-inheritance:
