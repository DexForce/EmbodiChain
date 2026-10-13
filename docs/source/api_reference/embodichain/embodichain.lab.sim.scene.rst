embodichain.lab.sim.scene
=========================

Scene services share composition, geometric layout, physical state operations
and measured validation between GenSim, explicit recipes and execution hosts.
They consume existing entity configurations and UIDs; Spawn and
``SimulationManager`` remain the owners of native resources and topology.

Public poses are arena-local Z-up transforms in metres. Seven-component
physical poses use position followed by ``xyzw`` quaternion. Array geometry
already includes asset scale and orientation. GenSim adapts its Y-up editing
model at its existing boundary; shared services never load prompts or assets
from generation directories.

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

Snapshots own immutable arena-local physical state, including velocities, joints, controller targets and efforts. Restore preflights ownership, backend, binding, topology and current joint limits before writes. Saved joint positions or targets outside those limits are rejected rather than silently clipped. Restore does not step and cannot be used as a full backend checkpoint. Native failures may leave partial writes.

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
