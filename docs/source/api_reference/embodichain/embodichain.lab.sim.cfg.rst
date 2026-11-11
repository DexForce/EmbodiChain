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
   GaussianRenderCfg
   GaussianSplatCfg
   VisualElementCfg
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
   UsdImportCfg
   LightCfg
   VisualObjectCfg
   RigidObjectCfg
   CableObjectCfg
   CablePhysicsCfg
   CableAttachmentCfg
   RigidObjectGroupCfg
   RigidConstraintCfg
   URDFCfg
   ArticulationCfg
   RobotCfg
   RobotPresetCfg

Gaussian visual composition
---------------------------

``VisualObjectCfg.visuals`` and ``RigidObjectCfg.visuals`` replace the owner's
appearance. ``ArticulationCfg.link_visuals`` (also available on ``RobotCfg``)
replaces only the named links. ``None`` preserves existing appearance; an empty
list removes it. Each ``VisualElementCfg`` has an owner-local rigid transform,
a positive uniform scale and a stable ID. Mesh/primitive and Gaussian elements
can coexist. Physical geometry, mass, inertia, joints and control remain owned
by the existing physical entity. Gaussian opacity is never physical density.

.. code-block:: python

   from embodichain.lab.sim.cfg import (
       GaussianRenderCfg, GaussianSplatCfg, RenderCfg,
       VisualElementCfg, VisualObjectCfg,
   )

   render = RenderCfg(gaussian=GaussianRenderCfg(enabled=True))
   background = VisualObjectCfg(
       uid="room",
       visuals=[VisualElementCfg(
           id="scan", representation=GaussianSplatCfg(file_path="room.ply"),
       )],
   )

For YAML/JSON, use ``representation: {type: gaussian_splat, file_path: room.ply}``.
Mesh representations retain ``shape_type`` and the existing shape fields.
The Gym ``visual_object`` list uses the same decoder. Keep ``background`` for
its existing physical-object meaning.

This feature requires a qualified DexSim integration build exposing Gaussian
Spawn/Scene batching, ``Scene.clear_visual_meshes`` and shared template
replication. The pinned legacy wheel is not assumed to provide these APIs;
ordinary Mesh scenes do not require them. The default physics backend is the
initial supported backend. PLY and compressed GSQ are supported; GSQ cannot
use a dynamic LoD sidecar. ``.dgslod`` is optional for PLY. Dynamic hierarchy
requires both the global render mode and the instance setting. Geometric
scale and ``lod_detail_scale`` are independent. Gaussian owners must have
positive uniform ``body_scale``.

Gaussian instance randomization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GaussianSplatCfg.color_gain`` and ``opacity_gain`` initialize independent
instance multipliers. Equal RGB gains change brightness; unequal gains tint
captured radiance after SH evaluation and before the existing tone mapper.
Opacity gain applies before native alpha culling/compositing. These operations
retain shared source buffers and do not represent physical relighting.

After adding an entity, stage selected-environment changes before rendering:

.. code-block:: python

   room.set_gaussian_visual_state(
       "scan", env_ids=[1, 3],
       color_gain=[[0.8, 1.0, 1.2], [1.2, 0.9, 0.7]],
       opacity_gain=[0.7, 1.0], scale=[0.9, 1.1],
   )

Scalar parameters accept a scalar or one value per selected environment;
RGB accepts ``(3,)`` or ``(N,3)``. Scale is a positive uniform world geometry
scale including owner ``body_scale``; it does not scale physics or the local
translation. Articulation calls supply ``link_name``. Reset restores the
initial values only for selected environments; topology rebuild retains live
values and the separate reset baseline. Invalid requests change no staged state.
Runtime publication errors invalidate the capture; the next render resubmits
all authoritative instance state.

For per-arena asset choices, declare multiple elements and select their
``visible`` state. Each distinct asset/variant remains resident even when hidden;
sharing saves copies across environments, not the cost of the variant bank.
``point_opacities`` (activated values in [0,1], shape ``(N,)``) and
``sh_coefficients`` (basis-major ``(N,K,3)``, K=1/4/9/16) create independent,
immutable flat-PLY variants. Shapes must match the loaded source. Their clones
share that variant, while the original asset remains unchanged. These arrays
are consumed at creation, never by per-frame state batches. Select among
prepared variants for runtime randomization. Compressed GSQ and existing LoD
sidecars reject point overrides; prepare an updated asset and hierarchy offline.
This is not a mutable/deformable point-buffer API.

Declare assets, call ``prepare()``, then consume frames through the existing
``render_frame()``/CameraGroup path. Visual bindings publish transforms without
stepping physics. Selected visibility, reset and topology changes retain the
existing owner lifecycle. ``get_visual_elements()`` returns metadata, not a
copy of the Gaussian point arrays. Mesh material assignments affect companion
meshes and reject Gaussian-only selections. ``get_semantic_mesh()`` retains
the source surface across appearance replacement; ``semantic_shape`` supplies
an explicit independent proxy. Layout and antipodal-grasp consumers use this
surface instead of assuming that rendered appearance is a triangle mesh.

Camera ``color`` retains its uint8 layout. ``capture_metadata`` records the
completed group/frame, simulation time, calibration, instance-state snapshots
and native output contract;
Gym exposes it separately as ``sensor_capture_metadata``. Depth is the first
accepted Gaussian contribution's view-Z with mesh fallback and invalid
background, not selected-surface depth. Display alpha is not Gaussian coverage.
Capture errors propagate before returning observations. Captured SH is not
relit by ordinary light randomization. The existing native postprocess is the
sole final exposure/tone-mapping owner.

Current unsupported combinations are Newton, native viewer recording (use a
managed Camera recorder instead), StereoCamera Gaussian metadata,
complete Gaussian mask/normal/position/motion outputs, Hybrid standalone NRD,
and browser visual-composition rendering. RigidObjectGroup composition and
whole-scene ``add_usd(robot_cfgs=...)`` appearance overrides are not supported;
use individual objects or selected ``UsdSourceCfg`` robot declarations. NoRender retains declarations without creating
Gaussian resources. Existing physical USD sources may receive external PLY
appearance replacements; this does not provide Gaussian USD import.

Gaussian USD import remains deferred until the qualified DexSim 0.6.0 release.
The planned import-result/selected-binding path will preserve source identity
and avoid duplicate creation, using the native importer rather than an
EmbodiChain USD-to-PLY conversion. Import-only static assets, Newton, true HDR
and optional ISP, browser splats and deformable Gaussian mappings require
separate implementation and acceptance. No placeholder APIs advertise them.

Run ``examples/sim/gaussian_visuals.py --background /assets/room.ply`` for a
scanned-background, rigid and link composition example. The benchmark entry
``scripts/benchmark/scene/gaussian_visuals.py`` compares native publication and
EmbodiChain publication on the same prepared Scene and CameraGroups.

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

USD import recipes
------------------

``SimulationManager.add_usd(UsdImportCfg(...))`` declares a complete scene
selection. ``source.root_prim_path`` selects a source subtree and ``variants``
selects source variants. The import recipe separately owns runtime namespace,
the optional left-multiplied scene pose and overrides keyed by unchanged source
prim paths. Unset source properties are retained. Source and override validation
precedes native declaration; returned facades bind in place at ``prepare()``.

.. code-block:: python

   from embodichain.lab.sim.cfg import UsdImportCfg, UsdSourceCfg

   imported = sim.add_usd(
       UsdImportCfg(
           name="workcell",
           source=UsdSourceCfg(
               file_path="workcell.usda",
               root_prim_path="/World/Cell",
           ),
           namespace="cell_1",
       )
   )
   sim.prepare()
   robot = imported["/World/Cell/Robot"]

.. autoclass:: embodichain.lab.sim.cfg.usd.UsdImportCfg
   :members:

Single-asset configurations retain their exact ``source_prim_path`` selector.
Their ``source_pose_mode`` chooses automatic, replacement, preservation or
offset placement. Legacy source-option spellings remain accepted by the
compatibility adapter. New scene recipes keep namespace and placement on
``UsdImportCfg``.

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

Cable authoring
---------------

``CableObjectCfg`` authors an ordered centreline, radius and optional material
frames in EmbodiChain's ``xyzw`` convention. ``CablePhysicsCfg`` preserves
unspecified four-mode coefficients as ``None`` and reuses existing collision
and contact-material schemas. ``CableAttachmentCfg`` describes permanent
creation-time endpoint welds. Declare cables before ``prepare()`` with explicit
DexUni on CUDA. Segment writes and complete environment reset require a DexSim
build exposing ``CableBatch.apply_state`` and ``Scene.reset_arenas``.

.. currentmodule:: embodichain.lab.sim.cfg.cable

.. autosummary::

   CablePhysicsCfg
   CableAttachmentCfg
   CableObjectCfg

.. autoclass:: CablePhysicsCfg
   :members:

.. autoclass:: CableAttachmentCfg
   :members:

.. autoclass:: CableObjectCfg
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

DexUni accepts ``coupling="two_way"`` with ``joint_mode="dynamic"`` to feed
contact reactions back to articulations. Keep ``collision_cfg=None`` when
DexUni owns collision detection. ``coupling_options`` configures the native
feedback policy. Full-surface rigid-soft feedback is supported; native validation
owns the compatible contact modes and option combinations.
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

``JointDrivePropertiesCfg.limit_stiffness`` and ``limit_damping`` separate
joint-limit gains from drive gains. Each accepts a scalar or joint-rule mapping;
``None`` preserves the source/backend value. With Newton's MuJoCo solver,
setting both to zero selects the default MuJoCo limit response while retaining
joint bounds.

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
