embodichain.lab.sim.cfg
===================================

.. automodule:: embodichain.lab.sim.cfg

Overview
--------

This module collects the ``@configclass`` configuration objects for everything
that can be spawned into a simulation scene. It covers global simulation
settings (rendering, physics, GPU memory, markers, window recording/camera),
rigid-body / soft-body / cloth physical attributes and their overrides, joint
drive properties, and the per-entity configs consumed by
:class:`~embodichain.lab.sim.sim_manager.SimulationManager` and the object
factory in :mod:`embodichain.lab.sim.utility.sim_utils`.

Entity configs form a small inheritance hierarchy rooted at ``ObjectBaseCfg``
(``LightCfg``, ``RigidObjectCfg``, ``VolumeDeformableObjectCfg``, ``SurfaceDeformableObjectCfg``,
``ArticulationCfg`` and its ``RobotCfg`` subclass), while ``URDFCfg`` and
``RigidConstraintCfg`` describe multi-component assembly and constraints.
``RobotPresetCfg`` provides replace-only complete robot alternatives when a
backend-specific asset or actuator definition is unavoidable.
Public backend selectors use only ``default`` and ``newton``. Nested physical
property groups may additionally use ``common`` for backend-neutral intent;
DexSim names belong to the runtime and Spawn SDK adapter boundary.
Surface deformables may keep a stable low-resolution simulation topology in
``shape`` while binding an independently indexed ``visual_shape`` for authored
UV seams and render detail.

.. rubric:: Type aliases

.. autosummary::

   AssetPhysicsMode
   MeshCollisionApproximation

.. rubric:: Classes

.. autosummary::

   RenderCfg
   PhysicsBackendCfg
   DefaultPhysicsCfg
   NewtonPhysicsCfg
   NewtonCollisionPipelineCfg
   MarkerCfg
   WindowRecordCfg
   WindowCameraPoseCfg
   GPUMemoryCfg
   MassPropertiesCfg
   DefaultRigidBodyPropertiesCfg
   CollisionPropertiesCfg
   DefaultCollisionPropertiesCfg
   NewtonCollisionPropertiesCfg
   RigidBodyMaterialCfg
   NewtonRigidBodyMaterialCfg
   MeshCollisionCfg
   RigidBodyPhysicsCfg
   ArticulationRootPropertiesCfg
   LinkPhysicsOverrideCfg
   VolumeDeformableMeshingCfg
   VolumeDeformablePhysicsCfg
   SurfaceDeformablePhysicsCfg
   SurfaceElementPropertiesCfg
   JointDrivePropertiesCfg
   NewtonJointDrivePropertiesCfg
   ObjectBaseCfg
   LightCfg
   RigidObjectCfg
   RigidObjectGroupCfg
   RigidConstraintCfg
   URDFCfg
   ArticulationCfg
   RobotCfg
   RobotPresetCfg

Deformable physics and meshing
-----------------------------

Like ``RigidObjectCfg.attrs``, deformable objects group physical intent under
``attrs``. Volume objects keep mesh generation under ``meshing``. Both physics
configs share the seven Newton surface-element parameters through
``attrs.surface_props``; volume coefficients default to zero, while cloth
coefficients default to ``None`` (Newton defaults). Density remains topology
specific: kg/m³ for volumes and kg/m² for surfaces.

.. autoclass:: VolumeDeformablePhysicsCfg
   :members:
   :no-index:

.. autoclass:: SurfaceDeformablePhysicsCfg
   :members:
   :no-index:

.. autoclass:: SurfaceElementPropertiesCfg
   :members:
   :no-index:

.. autoclass:: VolumeDeformableMeshingCfg
   :members:
   :no-index:

Constructors and ``from_dict()`` accept only the current fields. Surface-element
properties are grouped under ``attrs.surface_props``. Volume elasticity uses
``youngs`` and ``poissons``. Configuration dictionaries serialize this same
schema; no legacy aliases or field migration are provided.
