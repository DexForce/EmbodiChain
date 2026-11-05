embodichain.lab.sim.scene
=========================

Scene services share composition, geometric layout, physical state operations
and measured validation between GenSim, explicit recipes and execution hosts.
They consume existing entity configurations and UIDs; Spawn and
``SimulationManager`` remain the owners of native resources and topology.

Public poses are arena-local Z-up transforms in metres. Seven-component
physical poses use position followed by ``xyzw`` quaternion. Layout array geometry
already includes asset scale and orientation. Input-mesh snapshots retain scaled
owning-frame vertices until their separately observed poses are projected.
GenSim adapts its Y-up editing
model at its existing boundary; shared services never load prompts or assets
from generation directories.

Observed geometry and collision sources
---------------------------------------

Geometry capture accepts plain logical/native UID maps and explicit link bindings.
Rigid visual envelopes and articulation collision-input components remain distinct,
with ordered topology, neutral role metadata and explicit unavailable results.
The capture reads no live poses. ``NONE`` planner metadata does not remove a
selected collider input. These are materialization/input envelopes, not native
cooked-shape or contact evidence.

``snapshot_collision_sources`` projects explicitly supplied shape/pose values
through one ``world_to_frame`` transform. Its owned providers reuse
``CollisionShapeDesc`` and can be passed to ``CuroboWorldCfg.rigid_objects``.
The link-box reader preserves separate connected components and applies scale once.

.. currentmodule:: embodichain.lab.sim.scene

.. autosummary::

   SceneMeshComponent
   SceneMeshSnapshot
   capture_scene_meshes
   iter_scene_components
   CollisionObjectSnapshot
   read_articulation_link_boxes
   snapshot_collision_sources

.. autoclass:: SceneMeshComponent
   :members:

.. autoclass:: SceneMeshSnapshot
   :members:

.. autoclass:: CollisionObjectSnapshot
   :members:

.. autofunction:: capture_scene_meshes

.. autofunction:: iter_scene_components

.. autofunction:: read_articulation_link_boxes

.. autofunction:: snapshot_collision_sources

Robot-aware workspace proposals
------------------------------

The reusable workspace adapter combines Robot cache sampling with object-local
grasp geometry. ``sample_object_poses`` returns proposed object poses and joint
seeds; ``check_object_pose`` independently checks actual pose IK, joint limits
and complete FK position/orientation residuals. It returns ``SceneCheckReport``.
These checks do not establish support, collision-free motion or task success.
The legacy Task Program import converts reports to ``ValidationResult``.

.. currentmodule:: embodichain.lab.sim.scene

.. autosummary::

   RobotSceneWorkspace
   SceneWorkspaceCandidate

.. autoclass:: RobotSceneWorkspace
   :members:

.. autoclass:: SceneWorkspaceCandidate
   :members:

Implementation module: ``embodichain.lab.sim.scene.workspace``

.. currentmodule:: embodichain.lab.sim.scene.workspace

.. autosummary::

   RobotSceneWorkspace
   SceneWorkspaceCandidate

The implemented layout is fixed-orientation XY placement using AABB proxies.
It does not establish exact mesh collision freedom, vertical support, robot
reachability or task success. Physical state services currently reject worlds
other than B=1 and selections other than ``env_ids=(0,)`` before writing.

Installation and runnable recipe
--------------------------------

Use the normal EmbodiChain installation, then run the model-free example.
DexSim supplies SciPy and Trimesh; EmbodiChain requires Shapely 2 or newer:

.. code-block:: bash

   pip install -e .
   PYTHONPATH=. python scripts/tutorials/scene/tabletop_recipe.py --seed 7

The example declares a workbench, two containers built from five panels each,
six parts and a keep-out region. It generates a proposal, prepares the existing
manager, settles through the caller-owned physics loop, checks measured layout
and restores a physical snapshot. It requires a working native simulator.
Browser visualization is explicitly disabled with
``VisualizationCfg(backend="none")``, and the renderer is ``"no-render"``.
Use ``--output /path/to/report.json`` to save the measured checks. Successful
execution reports ``layout_status="solved"``, ``stable=true``,
``measured_xy_layout_valid=true`` and ``state_restored=true``. These checks
qualify this tabletop preparation; they do not establish robot reachability
or task success.

GenSim integration and current scope
------------------------------------

GenSim's established entry points adapt the shared services while retaining
their editing conventions and lifecycle. The following capabilities already
use a single shared implementation:

.. list-table::
   :header-rows: 1
   :widths: 28 35 37

   * - Capability
     - GenSim entry point
     - Shared implementation
   * - Main horizontal support detection
     - ``TableSupportSurfaceDetector``
     - Scene CPU support detection, also used by ``detect_support_region``.
   * - Group height alignment and planar placement
     - ``AssetsGroupTableAligner`` and ``AssetsGroupSupportClamp``
     - Scene CPU group-offset and support-containment strategies.
   * - Independent AABB separation
     - ``AssetsSupportLayoutOptimizer``
     - Scene CPU packing strategy, also used by ``separate_aabbs``.
   * - Tabletop and parent-surface constraints
     - ``TableSurfaceLayoutOptimizer`` and ``ParentSurfaceLayoutOptimizer``
     - ``LayoutProblem`` and ``solve_layout`` with the existing GenSim policy.
   * - Physical settling and pose readback
     - ``GravitySettler``
     - ``settle_scene`` and ``SceneView.pose``; convergence is explicit opt-in.

See :doc:`embodichain.gen_sim.scene_adapters` for the adapter APIs.
GenSim owns image/text understanding, model clients, asset preparation, visual
orientation decisions, the editable scene graph, hierarchy traversal, Y-up
conversion, diagnostics and export. Its temporary simulation remains
caller-owned. ``SceneFragment`` and ``SceneRecipe`` provide a separate explicit
construction path; the existing image/edit pipelines do not run through that
recipe interface or use the physical snapshot-restoration workflow.

Physical mutation and restoration currently support B=1 with fixed topology.
Parallel tensor geometry does not extend that runtime scope. Mesh support
detection selects one main horizontal outer contour, and layout uses fixed-
orientation XY AABBs. Changed orientations require measured footprints for
post-settle layout checks. General room construction, batched physical writes,
exact mesh collision and full solver checkpoints remain future work.

CPU strategies and tensor kernels
---------------------------------

Scene layout, support selection and placement retain their CPU algorithms and
Shapely/Trimesh/SciPy dependencies. They call reusable Torch geometry kernels;
this does not move SLSQP, polygon search or physical hosts to GPU.

Kernel batching distinguishes independent leading layouts ``B`` from the
objects ``N`` within each layout. For example, ``(B, N, 2, 3)`` contains B sets
of N 3D boxes. Object masks belong to that object axis; masking padding does not
select simulation environments. Scene runtime services still require B=1.

See :doc:`embodichain.compute.scene_geometry` for tensor shape, mask, dtype and
device contracts. The :download:`geometry kernel benchmark
<../../../../scripts/benchmark/scene/geometry_kernels.py>` measures kernels
separately from CPU strategy solving and simulator execution. Its results must
be reported with the actual device, shapes and synchronization policy.

Public facade
-------------

The facade resolves implemented services lazily. Expansion contracts have a
separate ``scene.expansion`` namespace. Importing through ``lab.sim`` retains
that package's normal simulation dependency requirements.

.. currentmodule:: embodichain.lab.sim.scene

.. autosummary::
   :nosignatures:

   SceneFragment
   compose
   declare_fragment
   SceneRecipe
   GeneratedScene
   generate
   check_layout
   SupportSurfaceConfig
   SupportRegion
   detect_support_region
   AABBLayoutConfig
   GroupSupportClampConfig
   clamp_group
   separate_aabbs
   align_group_height
   LayoutCfg
   PlanarRelation
   LayoutProblem
   LayoutSolution
   solve_layout
   SceneView
   EntityState
   SceneState
   capture_state
   restore_state
   SceneChangeSet
   SceneChangeResult
   apply_changes
   SceneCheck
   SceneCheckReport
   SettlingCfg
   validate_scene
   compare_states
   settle_scene

embodichain.lab.sim.scene.composition
-------------------------------------

Fragments copy existing configurations and optional anchor transforms. Composition transforms poses and namespaces UIDs without changing body scales or physics ownership. ``declare_fragment`` declares through an existing manager; the caller still invokes ``prepare()``. Native declaration failures do not promise rollback.

.. currentmodule:: embodichain.lab.sim.scene.composition

.. autosummary::
   :nosignatures:

   SceneFragment
   compose
   declare_fragment

.. automodule:: embodichain.lab.sim.scene.composition
   :members:
   :imported-members:

embodichain.lab.sim.scene.procedural
------------------------------------

Recipes consume resolved assets, a versioned recipe identity and explicit geometry. ``generate`` derives candidates from seed and logical ordinal, independently of physical slots. Failed attempts return no fragment or variant. ``check_layout`` remeasures XY constraints from actual poses. Callers may supply ``actual_aabbs_xy_by_uid`` measured in arena coordinates to update proxies after rotation. Without updated geometry, changed orientation is unverified; changed support remains unverified. Moving a fixed entity fails validation.

.. currentmodule:: embodichain.lab.sim.scene.procedural

.. autosummary::
   :nosignatures:

   SceneRecipe
   GeneratedScene
   generate
   check_layout

.. automodule:: embodichain.lab.sim.scene.procedural
   :members:
   :imported-members:

embodichain.lab.sim.scene.geometry
----------------------------------

Support regions identify a supporting entity and its arena-local polygon. Direct polygon inputs may include holes and disconnected components. Mesh detection currently selects one main horizontal outer contour and its inscribed optimization rectangle, preserving the legacy GenSim selection policy.

.. currentmodule:: embodichain.lab.sim.scene.geometry

.. autosummary::
   :nosignatures:

   SupportSurfaceConfig
   SupportRegion
   detect_support_region

.. automodule:: embodichain.lab.sim.scene.geometry
   :members:
   :imported-members:

embodichain.lab.sim.scene.placement
-----------------------------------

Group clamping preserves relative positions; AABB separation moves entities independently; height alignment returns one shared Z offset. These operations only propose numeric changes. A failed bounded search is not a proof of infeasibility.

.. currentmodule:: embodichain.lab.sim.scene.placement

.. autosummary::
   :nosignatures:

   AABBLayoutConfig
   GroupSupportClampConfig
   clamp_group
   separate_aabbs
   align_group_height

.. automodule:: embodichain.lab.sim.scene.placement
   :members:
   :imported-members:

embodichain.lab.sim.scene.layout
--------------------------------

Layout problems name fixed and movable entities, full-box bounds, displacement preferences and arena-frame planar relations. Solutions distinguish solved, no_solution, budget_exhausted and unsupported. Residuals describe the checked linear, fixed-position and AABB constraints; failed proposals must not be applied.

.. currentmodule:: embodichain.lab.sim.scene.layout

.. autosummary::
   :nosignatures:

   LayoutCfg
   PlanarRelation
   LayoutProblem
   LayoutSolution
   solve_layout

.. automodule:: embodichain.lab.sim.scene.layout
   :members:
   :imported-members:

embodichain.lab.sim.scene.query
-------------------------------

A view reads existing physical bindings and cloned poses while retaining the environment dimension. It rejects unsupported entities and changed bindings. It is not a second scene registry or a general raycast/contact-query API.

.. currentmodule:: embodichain.lab.sim.scene.query

.. autosummary::
   :nosignatures:

   SceneView

.. automodule:: embodichain.lab.sim.scene.query
   :members:
   :imported-members:

embodichain.lab.sim.scene.state
-------------------------------

Snapshots own immutable arena-local physical state, including velocities, joints, controller targets and efforts. Restore preflights ownership, backend, binding, topology and current joint limits before writes. By default saved joint positions or targets outside those limits are rejected. Callers may explicitly pass ``qpos_tolerance`` to allow small measured-position excursions to be clipped and then verify the actual round trip; controller targets remain strict. Restore does not step and cannot be used as a full backend checkpoint. Native failures may leave partial writes.

.. currentmodule:: embodichain.lab.sim.scene.state

.. autosummary::
   :nosignatures:

   EntityState
   SceneState
   capture_state
   restore_state

.. automodule:: embodichain.lab.sim.scene.state
   :members:
   :imported-members:

embodichain.lab.sim.scene.operations
------------------------------------

Change sets require an explicit movable-entity allowlist. Application preflights all rigid-body changes, clears their dynamics and publishes render state. Returned invalidations must be handled by the host; observations and planner bindings cannot be reused solely because simulation time is unchanged.

.. currentmodule:: embodichain.lab.sim.scene.operations

.. autosummary::
   :nosignatures:

   SceneChangeSet
   SceneChangeResult
   apply_changes

.. automodule:: embodichain.lab.sim.scene.operations
   :members:
   :imported-members:

embodichain.lab.sim.scene.validation
------------------------------------

A report passes only when its nonempty required checks all pass. Fixed-step settling returns unverified stability; optional convergence mode measures velocity thresholds over consecutive samples. The caller owns stepping and advances the entire world. State comparison treats quaternion signs equivalently and retains the existing restore tolerances.

.. currentmodule:: embodichain.lab.sim.scene.validation

.. autosummary::
   :nosignatures:

   SceneCheck
   SceneCheckReport
   SettlingCfg
   validate_scene
   compare_states
   settle_scene

.. automodule:: embodichain.lab.sim.scene.validation
   :members:
   :imported-members:

embodichain.lab.sim.scene.native_hand
--------------------------------------

These read-only services capture Default cooked convex/box shapes, native
rest/contact offsets and the configured hand stroke through named FK. They reuse
object-backend physical geometry/offset capture and the existing hand-stroke sampler. Capture
requires the exact simulation-owned Robot, registered native UIDs and B1/env0;
subsequent rigid reads reject replaced owners. No query, grasp policy or
experiment revision is selected here.

GraspKit owns discrete geometric qualification through an injected collision
backend. ``ActionPlanningServices`` binds an explicitly supplied geometric policy
and owns query lifetimes. Captured geometry does not establish physical acquisition or release.

.. currentmodule:: embodichain.lab.sim.scene.native_hand

.. autosummary::
   :nosignatures:

   NativeHandGeometry
   capture_native_hand_geometry
   capture_native_rigid_geometry
   capture_native_rigid_obstacles

.. automodule:: embodichain.lab.sim.scene.native_hand
   :members:


Related integrations
--------------------

The Task Program host keeps Gym reset ordering, sensor/observation refresh,
semantic binding recreation, measured task acceptance and persistence gating.
The generic scene services own no task lifecycle or dataset writer.

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.scene.expansion
   embodichain.gen_sim.scene_adapters

See :mod:`embodichain.lab.task_program.integrations.scene_expansion` for task
execution and :mod:`embodichain.compute.geometry` for reusable tensor kernels.
The support, polygon-placement, greedy-packing and SLSQP policies remain private
CPU implementations in ``sim.scene``.

Joint task feasibility
----------------------

See :doc:`embodichain.lab.sim.scene.feasibility` for task-conditioned scene and
robot installation search, independent validation and the configured entry point.

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.scene.feasibility

Authored collider inputs
-----------------------

``LinkMesh`` carries owned link-local source vertices and triangle indices.
Its implementation belongs to ``spawn.source_geometry``; the Scene import
remains available as the same class.
It preserves authored order and dtype, and explicitly permits vertex-only
inputs with an empty triangle array. Native cooked collision geometry and
runtime poses remain separate observations; this value proves neither contact
nor collision freedom.

.. currentmodule:: embodichain.lab.sim.scene.source_geometry

.. autoclass:: LinkMesh
   :members:

Static articulation and control inputs
--------------------------------------

These readers inspect self-contained USD sources without creating a simulator.
``ArticulationGeometry`` measures collider vertices relative to the imported
root body; ``fit_articulation_to_proxy`` explicitly returns a revised physical
declaration. Prismatic discovery preserves stable part-identity order and
requires explicit part selection when several mechanisms exist. Press inspection measures a
selected joint, endpoint and collider; it does not evaluate a press event.

Static readers share the Object/Spawn source reader. Config-based inspections
accept an optional ``source=UsdSourceSnapshot``; path-based inspections also
accept that snapshot directly. This reuses the selected source parse within an assembly while retaining
source-byte and annotation checks at the reader boundaries. Mass lineage
qualification keeps its own original/derived source checks::

    from embodichain.lab.sim.spawn.source_geometry import read_usd_source
    from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
    from embodichain.lab.sim.scene.revolute_geometry import load_twist_geometry
    from embodichain.lab.sim.scene.control_source import capture_revolute_source_seal

    source = read_usd_source(config["fpath"])
    binding = inspect_revolute_control(config, 90, source=source)
    geometry = load_twist_geometry(source, binding)
    seal = capture_revolute_source_seal(source, binding)

Revolute inspection consumes authored joint, grip, pointer and printed-setting
roles. The existing ``gen_sim.twist-control/v1`` annotation format is retained
for frozen assets; no roles are inferred from names. The physical binding does
not select execution chunk counts, recovery rules or acceptance tolerances.
``TwistGeometry`` keeps each body's input mesh and cooking approximation separate,
with scale applied once. Native cooked shapes require their own observation.

``create_mass_source_copy`` performs the existing Default-only compensation:
center of mass scales by the uniform body scale and diagonal inertia by its
square, while mass and all other authored fields remain unchanged. It writes a
content-addressed copy and lineage manifest, leaving the original read-only.
Qualification reconstructs that exact operation and rejects altered artifacts.
``RevoluteSourceSeal`` pins asset, calibration and optional mass-source bytes,
including required sidecar absence. Freshness checks reject drift rather than
re-signing it. The circular-prism proof requires two complete uniform rings and
closed convex topology; it does not establish dynamic path clearance.

.. currentmodule:: embodichain.lab.sim.scene.articulation_geometry

.. autoclass:: ArticulationGeometry
   :members:

.. autofunction:: read_articulation_geometry

.. autofunction:: fit_articulation_to_proxy

.. currentmodule:: embodichain.lab.sim.scene.articulation_inputs

.. autoclass:: PrismaticBinding
   :members:

.. autoclass:: ArticulationPartCandidate
   :members:

.. autofunction:: discover_prismatic_parts

.. autofunction:: inspect_prismatic

.. autofunction:: inspect_prismatic_part

.. autofunction:: handle_mesh

.. currentmodule:: embodichain.lab.sim.scene.control_semantics

.. autoclass:: TwistSemantics
   :members:

.. autoexception:: MissingTwistSemanticsError

.. autoexception:: InvalidTwistSemanticsError

.. autofunction:: resolve_twist_semantics

.. currentmodule:: embodichain.lab.sim.scene.revolute_inputs

.. autoclass:: RevoluteControlBinding
   :members:

.. autofunction:: inspect_revolute_control

.. currentmodule:: embodichain.lab.sim.scene.revolute_geometry

.. autoclass:: TwistGeometry
   :members:

.. autofunction:: load_twist_geometry

.. currentmodule:: embodichain.lab.sim.scene.press_inputs

.. autoclass:: PressBinding
   :members:

.. autofunction:: inspect_press

.. currentmodule:: embodichain.lab.sim.scene.control_mass_source

.. autoclass:: MassSourceCopy
   :members:

.. autofunction:: adjacent_manifest_path

.. autofunction:: qualify_mass_source_copy

.. autofunction:: create_mass_source_copy

.. currentmodule:: embodichain.lab.sim.scene.control_source

.. autoclass:: RevoluteSourceSeal
   :members:

.. autofunction:: capture_revolute_source_seal

.. autofunction:: qualify_revolute_grip_shell

Native planar observations
--------------------------

.. currentmodule:: embodichain.lab.sim.scene.planar_observation

Capture scaled native geometry using exact object UIDs and measure supplied
observation poses. Object replacement invalidates the observer. Unsupported or
missing geometry raises an acquisition error; measured footprint containment
is not evidence of contact support, release, stability or task completion.

.. autosummary::

   NativePlanarPlacement

.. autoclass:: NativePlanarPlacement
   :members:

Exact batched rigid mesh inputs
-------------------------------

.. currentmodule:: embodichain.lab.sim.scene.rigid_mesh

Capture scaled vertices and their original triangle indices without reading a
pose, changing winding, processing a mesh or merging environment rows. The owned
CPU data supports exact silhouette and staged Pour geometry. Input meshes are
not cooked collision shapes or physical contact evidence; consumers retain the
observation pose and native-object lifetime separately. The lightweight scene
mesh-envelope path does not acquire triangles unless explicitly requested by
this separate API.

.. autosummary::

   RigidMeshSnapshot
   capture_rigid_mesh

.. autoclass:: RigidMeshSnapshot
   :members:

.. autofunction:: capture_rigid_mesh


Authored source mesh inputs
---------------------------

These readers retain authored scene-node transforms and face order, convert the
source glTF Y-up basis to object-local Z-up and apply body scale once. They read
files without native simulation. The top estimator preserves unavailable input
as None; it is not a cooked-collider observation or physical clearance proof.

.. currentmodule:: embodichain.lab.sim.scene.source_geometry

.. autosummary::

   read_gltf_z_up_mesh
   estimate_source_mesh_top_z

.. autofunction:: read_gltf_z_up_mesh

.. autofunction:: estimate_source_mesh_top_z


Source scene ingress declarations
--------------------------------

These explicit adapters normalize owned source mappings without constructing a
simulation or changing shared physical defaults. The source physics adapter
retains authored/grouped property precedence and legacy collider declarations.
Pose/scale adapters preserve matrix authority, intrinsic XYZ degree rotations
and original world-Z rounding; callers supply owned declarations.

.. currentmodule:: embodichain.lab.sim.scene.source_physics

.. autosummary::

   source_scene_object_declaration

.. autofunction:: source_scene_object_declaration

.. currentmodule:: embodichain.lab.sim.scene.source_transforms

.. autosummary::

   normalize_source_pose
   apply_source_body_scale
   rotate_source_world_z

.. autofunction:: normalize_source_pose

.. autofunction:: apply_source_body_scale

.. autofunction:: rotate_source_world_z
