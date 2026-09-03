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

The deformable package provides a backend-neutral nodal-state contract and
canonical surface/volume names. ``Cloth*`` and ``Soft*`` remain compatibility
aliases for existing environments and tutorials.

.. currentmodule:: embodichain.lab.sim.objects.deformable

.. autosummary::

    ClothBodyData
    ClothObject
    DeformableObject
    DeformableObjectData
    SoftBodyData
    SoftObject
    SurfaceDeformableData
    SurfaceDeformableObject
    VolumeDeformableData
    VolumeDeformableObject

.. autoclass:: DeformableObject
    :members:
    :show-inheritance:

.. autoclass:: DeformableObjectData
    :members:

.. autoclass:: SurfaceDeformableData
    :members:
    :show-inheritance:

.. autoclass:: SurfaceDeformableObject
    :members:
    :show-inheritance:

.. autoclass:: VolumeDeformableData
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

    ClothBodyData
    ClothObject
    SurfaceDeformableData
    SurfaceDeformableObject

.. currentmodule:: embodichain.lab.sim.objects.deformable.volume

.. autosummary::

    SoftBodyData
    SoftObject
    VolumeDeformableData
    VolumeDeformableObject
