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
(``LightCfg``, ``VisualObjectCfg``, ``RigidObjectCfg``, ``VolumeDeformableObjectCfg``, ``SurfaceDeformableObjectCfg``,
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
   VisualObjectCfg
   RigidObjectCfg
   RigidObjectGroupCfg
   RigidConstraintCfg
   URDFCfg
   ArticulationCfg
   RobotCfg
   RobotPresetCfg

USD source selection
--------------------

``UsdSourceCfg`` groups the source file and its optional exact
``source_prim_path``. The path addresses an object inside the source USD stage;
``uid`` identifies the resulting runtime instance. Source selection does not
set a destination path or attach the instance to a parent/link.

.. code-block:: python

   from embodichain.lab.sim.cfg import UsdSourceCfg, VisualObjectCfg

   mug = VisualObjectCfg(
       uid="mug_visual",
       source=UsdSourceCfg(
           file_path="scene.usda",
           source_prim_path="/World/Props/Mug",
       ),
   )

The same ``source`` configuration is supported by ``RigidObjectCfg`` and
``ArticulationCfg`` (including ``RobotCfg``). Supply either ``source`` or the
existing ``shape``/``fpath`` input. Ordinary mesh files continue to use
``MeshCfg(fpath=...)``. Without a selector, USD loading requires one candidate
of the requested type; ambiguous sources fail instead of selecting the first
object. Whole-scene imports preserve this source address on their facades.

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
