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

Install the numerical scene dependencies alongside the normal simulation
runtime, then run the model-free example:

.. code-block:: bash

   pip install -e '.[scene]'
   PYTHONPATH=. python scripts/tutorials/scene/tabletop_recipe.py --seed 7

The example declares a workbench, two containers built from five panels each,
six parts and a keep-out region. It generates a proposal, prepares the existing
manager, settles through the caller-owned physics loop, checks measured layout
and restores a physical snapshot. It requires a working native simulator;
installing the numerical extra alone does not provide that runtime.

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
   SupportRegion
   detect_support_region
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

Snapshots own immutable arena-local physical state, including velocities, joints, controller targets and efforts. Restore preflights ownership, backend, binding and topology before writes, does not step, and cannot be used as a full backend checkpoint. Native failures may leave partial writes.

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
execution and :mod:`embodichain.compute.geometry` and
:mod:`embodichain.compute.layout` for numerical implementations.
