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
   DenoisingMode
   MeshCollisionApproximation

.. rubric:: Classes

.. autosummary::

   RenderCfg
   DenoisingCfg
   DLSSCfg
   NRDCfg
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
   UsdSourceCfg
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

USD source selection and placement
---------------------------------

``RigidObjectCfg.usd`` and ``ArticulationCfg.usd`` select an original USD prim
subtree, variant choices, and an optional namespace. Selection is performed by
DexSim in a private session layer; source prim paths remain stable for
``SimulationManager.add_usd(asset_cfgs=...)`` mappings. Root placement uses
``pose_mode`` independently of ``asset_physics_mode``: ``auto`` preserves the
source pose until an initial pose is configured, ``config`` replaces it,
``preserve`` retains it, and ``offset`` composes a scene-space transform.

Omitted USD actor types and overlay scales retain source intent. Articulation
root defaults retain USD fixation and self-collision, while URDF defaults remain
fixed with self-collision disabled. Explicit root fields still override the
asset in either physics mode. Imported geometry keeps its authored collision
representation.

.. autodata:: embodichain.lab.sim.cfg.asset.AssetPhysicsMode

.. autoclass:: embodichain.lab.sim.cfg.asset.ObjectBaseCfg
   :members:

.. autoclass:: embodichain.lab.sim.cfg.asset.UsdSourceCfg
   :members:

Rendering configuration module
------------------------------

Rendering types live in ``embodichain.lab.sim.cfg.rendering``. The public
``dlss`` path maps to DLSS Ray Reconstruction and ``nrd`` maps to standalone
NRD RELAX.

.. currentmodule:: embodichain.lab.sim.cfg.rendering

.. autosummary::

   DenoisingMode
   DenoisingCfg
   DLSSCfg
   NRDCfg
   RenderCfg

Physics configuration module
----------------------------

Physics backend types live in ``embodichain.lab.sim.cfg.physics``. The legacy
``embodichain.lab.sim.cfg.simulation`` module remains a compatibility facade.
For manager-owned Newton steps, ``sync_to_renderer=None`` or ``False`` publishes
only for visual consumers; ``True`` additionally requests publication on steps
without consumers.

.. currentmodule:: embodichain.lab.sim.cfg.physics

.. autosummary::

   GPUMemoryCfg
   PhysicsBackendCfg
   DefaultPhysicsCfg
   NewtonCollisionPipelineCfg
   NewtonPhysicsCfg
   physics_cfg_for_backend
   physics_backend_from_cfg
   validate_physics_cfg

Compatibility facade
--------------------

The historical ``embodichain.lab.sim.cfg.simulation`` module re-exports the
rendering and physics configuration types for callers that have not migrated
to the split modules.

.. currentmodule:: embodichain.lab.sim.cfg.simulation

.. autosummary::

   DenoisingMode
   DenoisingCfg
   DLSSCfg
   NRDCfg
   RenderCfg
   GPUMemoryCfg
   PhysicsBackendCfg
   DefaultPhysicsCfg
   NewtonCollisionPipelineCfg
   NewtonPhysicsCfg
   physics_cfg_for_backend
   physics_backend_from_cfg
   validate_physics_cfg

Rigid-body property module
--------------------------

The rigid-body configuration types are also available from
``embodichain.lab.sim.cfg.rigid``. They describe mass, collision properties,
materials, and backend-specific overrides. ``NewtonCollisionPropertiesCfg``
accepts an optional ``priority`` for MuJoCo contact-parameter selection;
``None`` preserves the source value.

.. currentmodule:: embodichain.lab.sim.cfg.rigid

.. autosummary::

   MassPropertiesCfg
   DefaultRigidBodyPropertiesCfg
   CollisionPropertiesCfg
   DefaultCollisionPropertiesCfg
   NewtonCollisionPropertiesCfg
   RigidBodyMaterialCfg
   NewtonRigidBodyMaterialCfg
   RigidBodyPhysicsCfg

USD dependency and constraint support
-------------------------------------

USD loading requires the companion DexSim import update: SI/Z-up normalization,
source selection, and ``SceneDesc.constraints``. A wheel predating that contract
raises an actionable error before source loading. The dependency range permits
compatible 0.5-series release and development wheels; install the companion
DexSim build before using these USD APIs.

Whole-scene imports retain standalone fixed, revolute, prismatic, spherical and
representable D6 constraints on the Default backend. Retrieve their live handles
through ``sim.spawn_result.get_constraint(path)`` after ``prepare()``. Explicit
asset ``uid`` overrides also remap their endpoints. A scene-root pose transforms
world anchors together with the imported objects. Singleton loaders reject
sources containing standalone relationships and direct callers to ``add_usd``.

Newton standalone constraints, closed loops, new deformable schemas, unsupported
D6 drive combinations, and finite Direct GPU break thresholds fail explicitly.
No deformable reader or closed-loop implementation is included in this update.
Per-asset ``usd`` overrides control placement; pass subtree, variant and namespace
selection through the outer ``add_usd(usd=...)`` argument.

Configuration implementation modules
------------------------------------

The implementation modules expose the same configuration types as the public
``cfg`` namespace, plus the link-override dictionary loader.

.. autoclass:: embodichain.lab.sim.cfg.articulation.ArticulationRootPropertiesCfg
.. autoclass:: embodichain.lab.sim.cfg.articulation.LinkPhysicsOverrideCfg
.. autofunction:: embodichain.lab.sim.cfg.articulation.link_attrs_from_dict
.. autoclass:: embodichain.lab.sim.cfg.articulation.JointDrivePropertiesCfg
.. autoclass:: embodichain.lab.sim.cfg.articulation.NewtonJointDrivePropertiesCfg
.. autoclass:: embodichain.lab.sim.cfg.articulation.ArticulationCfg
.. autoclass:: embodichain.lab.sim.cfg.rigid_object.RigidObjectCfg
.. autoclass:: embodichain.lab.sim.cfg.rigid_object.RigidObjectGroupCfg
