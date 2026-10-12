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

DexUni accepts ``coupling="two_way"`` with ``joint_mode="dynamic"`` to feed
contact reactions back to articulations. Keep ``collision_cfg=None`` when
DexUni owns collision detection. ``coupling_options`` configures the native
feedback policy; full-surface rigid-soft contacts are unsupported in this mode.
Articulation gravity compensation and passive joint damping are configured
separately below.

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

Articulation configuration module
---------------------------------

Articulation types live in ``embodichain.lab.sim.cfg.articulation`` and remain
available from the ``embodichain.lab.sim.cfg`` facade. Root creation settings
are independent of sparse link and joint physics overlays.
``root_props.newton_gravity_compensation`` compensates articulation link
weight on MuJoCo-backed Newton solvers while retaining payload gravity and
contact reactions. Omission preserves source intent.

``NewtonJointDrivePropertiesCfg.passive_damping`` selects passive DOF damping
through the same exact-name, regex and control-part rules as the drive gains.
It is independent of the drive velocity gain ``damping`` and stays active for
passive and effort target modes. In dictionaries, select this subtype with
``joint_drive_props.backend="newton"``. Both settings require Newton.

.. currentmodule:: embodichain.lab.sim.cfg.articulation

.. autosummary::

   ArticulationRootPropertiesCfg
   LinkPhysicsOverrideCfg
   link_attrs_from_dict
   JointDrivePropertiesCfg
   NewtonJointDrivePropertiesCfg
   ArticulationCfg

.. autoclass:: ArticulationRootPropertiesCfg
   :members:
   :no-index:

.. autoclass:: NewtonJointDrivePropertiesCfg
   :members:
   :no-index:

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
