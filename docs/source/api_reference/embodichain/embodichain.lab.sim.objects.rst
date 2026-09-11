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
specialization (``Robot``), deformables (``VolumeDeformableObject``, ``SurfaceDeformableObject``),
interactive ``Gizmo`` handles, and ``RigidConstraint`` attachments between
bodies.

  .. rubric:: Classes

  .. autosummary::

    Light
    LightCfg
    RigidObject
    CollisionShapeDesc
    RigidBodyData
    RigidObjectCfg
    RigidObjectGroup
    RigidBodyGroupData
    RigidObjectGroupCfg
    Articulation
    ArticulationJointKinematics
    ArticulationData
    ArticulationCfg
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

.. autoclass:: CollisionShapeDesc
    :members:

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
IK chain metadata. Native controllers activate on the first I press by default, while
Viser constructs its solver on the first drag. ``sim.update()`` owns updates
and cleanup; ordinary applications do not need the explicit factory.
Use ``SimulationManagerCfg(robot_ik_gizmo=None)`` to disable automatic setup.
Set ``GizmoCfg(ik_start_enabled=True)`` to activate native IK on the first update
with an open window; subsequent visibility toggles and reopening are preserved.

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

Backend Views
-------------

Backend views normalize tensor layouts and row selection over backend-neutral
DexSim Scene batches. The package import path exposes the common contracts,
Scene adapters, and the Newton Scene predicate. ``Scene*View.from_entities()``
owns Scene batch creation so object facades do not depend directly on DexSim's
batch-factory signatures.

.. currentmodule:: embodichain.lab.sim.objects.backends

.. autosummary::

    ArticulationViewBase
    RigidBodyViewBase
    is_newton_scene
    SceneArticulationView
    SceneRigidBodyView

.. autoclass:: ArticulationViewBase
    :members:

.. autoclass:: RigidBodyViewBase
    :members:

.. autoclass:: SceneArticulationView
    :members:
    :show-inheritance:

.. autoclass:: SceneRigidBodyView
    :members:
    :show-inheritance:

.. autofunction:: is_newton_scene

Backend implementation import paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.lab.sim.objects.backends.base

.. autosummary::

    RigidBodyViewBase
    ArticulationViewBase

.. currentmodule:: embodichain.lab.sim.objects.backends.newton

.. autosummary::

    is_newton_scene

.. currentmodule:: embodichain.lab.sim.objects.backends.scene

.. autosummary::

    SceneArticulationView
    SceneRigidBodyView

Unified Deformable Objects
--------------------------

Surface and volume objects share one concrete ``DeformableObjectData`` backed
by DexSim Newton particle batches. After ``sim.prepare()``, prefer ``obj.data``
for state reads: ``n_nodes`` gives the particle count without fetching state,
``nodal_pos_w`` and ``nodal_vel_w`` return world-frame tensors of shape
``(num_instances, n_nodes, 3)``, and ``nodal_state_w`` concatenates the two.
State reads return independent snapshots. ``default_nodal_state_w`` retains
the state captured at Spawn binding; ``root_pos_w`` is the mean node position,
not a mass-weighted center of mass.

Use ``obj.deformable_type`` to distinguish physical topology. Read simulation
nodes through ``data``. Render vertices and triangles are available through
``get_surface_vertices()`` and ``get_surface_triangles()``; volume objects also
expose tetrahedral boundary triangles through ``get_collision_surface_triangles()``.
Use ``SimulationManager.add_deformable_object()`` and ``get_deformable_object()``
to manage both topologies.

.. currentmodule:: embodichain.lab.sim.objects.deformable

.. autosummary::

    DeformableObject
    DeformableObjectData
    SurfaceDeformableObject
    VolumeDeformableObject

.. autoclass:: DeformableObject
    :members:
    :show-inheritance:

.. autoclass:: DeformableObjectData
    :members:

.. autoclass:: SurfaceDeformableObject
    :members:
    :show-inheritance:

.. autoclass:: VolumeDeformableObject
    :members:
    :show-inheritance:

Deformable implementation import paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.lab.sim.objects.deformable.base

.. autosummary::

    DeformableObject

.. currentmodule:: embodichain.lab.sim.objects.deformable.data

.. autosummary::

    DeformableObjectData

.. currentmodule:: embodichain.lab.sim.objects.deformable.surface

.. autosummary::

    SurfaceDeformableObject

.. currentmodule:: embodichain.lab.sim.objects.deformable.volume

.. autosummary::

    VolumeDeformableObject
