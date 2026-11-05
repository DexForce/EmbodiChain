embodichain.toolkits
====================

The :mod:`embodichain.toolkits` package contains asset-preparation and
manipulation utilities that can be used independently of the simulation loop.

.. automodule:: embodichain.toolkits

   .. rubric:: Submodules

   .. autosummary::

      acd
      graspkit
      urdf_assembly


GraspKit — Parallel-Gripper Grasp Sampling
-------------------------------------------

The :mod:`embodichain.toolkits.graspkit` package owns the standalone
grasp-pose service contracts. The toolkit does not import
:mod:`embodichain.lab`, so the same generator instance can be called directly
or installed in a higher-level planning runtime.

.. currentmodule:: embodichain.toolkits.graspkit

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
   get_parallel_jaw_gripper_model

.. autoclass:: GraspPoseGenerator
   :members:

.. autoclass:: ParallelJawGraspPoseGenerator
   :members:

.. autoclass:: ParallelJawGripperModelCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: get_parallel_jaw_gripper_model

.. currentmodule:: embodichain.toolkits.graspkit.pose_generator

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
   get_parallel_jaw_gripper_model

Box-side grasp regions
~~~~~~~~~~~~~~~~~~~~~

Rectangular cartons can declare opposed object-Y faces, an inset along object-X
and a height along object-Z. Projection selects the nearest legal longitudinal
position and jaw-symmetric orientation to an input anchor. It does not establish
IK, aperture, collision or physical grasp feasibility; downstream planning must
check those constraints against the declared gripper geometry.

.. currentmodule:: embodichain.toolkits.graspkit.box_region

.. autosummary::
   :nosignatures:

   BoxSideGraspRegionCfg
   project_box_side_grasp

.. autoclass:: BoxSideGraspRegionCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: project_box_side_grasp

The :mod:`embodichain.toolkits.graspkit.pg_grasp` module provides a reusable
antipodal implementation of these contracts. The pipeline consists of three
stages:

1. **Antipodal sampling** — Surface points are uniformly sampled on the mesh and rays are cast to find antipodal point pairs on opposite sides.
2. **Pose construction** — For each antipodal pair, a 6-DoF grasp frame is built aligned with the approach direction.
3. **Filtering & ranking** — Grasp candidates that cause the gripper to collide with the object are discarded; survivors are scored by a weighted cost.

.. rubric:: Public API

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp

The application-facing entry point is :class:`AntipodalGraspPoseGenerator`.
It requires explicit collision geometry at the physical pre-grasp opening.
Contact-aperture bounds, grasp algorithm, collision policy and annotation/cache
policy have separate owners. Internal candidates use the canonical contact
frame; public pose outputs use the geometry's declared runtime TCP.

``get_dual_arm_grasp_proposals()`` exposes unqualified paired proposals for an
explicit downstream refinement stage. It retains angle eligibility, NMS,
ranking, population budgets and runtime-TCP conversion, while constructing no
fixed-hand collision checker. Ordinary validated queries retain their separate
checked backend and existing behavior.

The optional ``role="left"`` or ``role="right"`` returns only that side.
Both sides' eligibility-dependent direction draws remain unchanged, preserving
the selected population and final Torch RNG; unrequested pose construction,
NMS and ranking are skipped. Omitting ``role`` retains both sides.
Raw and validated backends share read-only mesh and antipodal annotations,
while collision state remains separate. Refresh applies once per mesh's first
preparation. ``clear_cache()`` releases both modes' mesh backends and retains
configuration/hand geometry; subsequent queries prepare lazily again.

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
   AntipodalGraspPoseGenerator
   AntipodalGraspPoseGeneratorCfg
   GraspRegionSelector
   ParallelJawGraspCollisionCfg
   GraspAnnotationCfg
   FixedGripperCollisionGeometry
   get_parallel_jaw_collision_geometry
   AntipodalSampler
   AntipodalSamplerCfg
   GripperCollisionChecker
   GripperCollisionCfg
   ConvexCollisionChecker
   ConvexCollisionCheckerCfg



Collision-qualified region selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``select_supported_end`` selects only existing collision-qualified fallback
contacts before ranking and top-K. Duplicate orientation proposals count as one
contact pair. ``for_axis_region`` binds this policy for one request without
mutating a cached grasp generator; ``mesh_extent`` retains the original service.
Neither function provides IK, trajectory or physical-success qualification.

.. autosummary::
   :nosignatures:

   for_axis_region
   select_supported_end

.. autoclass:: GraspRegionSelector

.. autofunction:: for_axis_region

.. autofunction:: select_supported_end


AntipodalGraspPoseGenerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalGraspPoseGenerator
   :members:
   :show-inheritance:

AntipodalGraspPoseGeneratorCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalGraspPoseGeneratorCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

ParallelJawGraspCollisionCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ParallelJawGraspCollisionCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

FixedGripperCollisionGeometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This required geometry represents a declared pre-grasp joint state from actual
URDF collision meshes. The loader preserves
mesh origins and scales, applies the explicit canonical contact/TCP transform,
and records asset hashes. It samples finite surfaces; it does not certify
continuous intersections, closing contact, or full robot/path safety.

.. autoclass:: FixedGripperCollisionGeometry
   :members:

get_parallel_jaw_collision_geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resolve a registered hand's actual meshes, open joints and calibrated TCP.
Asymmetric hands require an explicit side; a TCP override preserves the
physical geometry while changing returned poses. Custom hands supply their
own :class:`FixedGripperCollisionGeometry`.

.. autofunction:: get_parallel_jaw_collision_geometry

GraspAnnotationCfg
~~~~~~~~~~~~~~~~~~

.. autoclass:: GraspAnnotationCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

AntipodalSampler
~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalSampler
   :members: sample
   :show-inheritance:

AntipodalSamplerCfg
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalSamplerCfg
   :members:
   :show-inheritance:

GripperCollisionChecker
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GripperCollisionChecker
   :members: query
   :show-inheritance:

GripperCollisionCfg
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GripperCollisionCfg
   :members:
   :show-inheritance:

ConvexCollisionChecker
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvexCollisionChecker
   :members: query, query_batch_points
   :show-inheritance:

ConvexCollisionCheckerCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvexCollisionCheckerCfg
   :members:
   :show-inheritance:

Implementation Module
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.pose_generator

.. autosummary::
   :nosignatures:

   AntipodalGraspPoseGenerator
   AntipodalGraspPoseGeneratorCfg
   GraspRegionSelector
   ParallelJawGraspCollisionCfg
   GraspAnnotationCfg



Region policy implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.regions

.. autosummary::
   :nosignatures:

   select_supported_end
   for_axis_region



Sampled hand contact qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.contact_fit

These services fit finite pad surfaces and check discrete closing samples against
an object-frame signed surface query. Robot/FK geometry construction and physical
contact evidence remain outside GraspKit. Qualification tolerances and occupancy
probe distance are explicit inputs; the checker does not infer task semantics.
``SampledHandGeometry.at_width()`` interpolates the hand surfaces at requested
pad gaps and clamps out-of-range gaps to the sampled endpoints.
``HandContactChecker.refine()`` combines fitting and target-only closing checks,
returning corrected candidates and a valid mask without changing their order.

.. autosummary::
   :nosignatures:

   SampledHandGeometry
   GraspSurfaceQuery
   TriangleSurfaceQuery
   HandContactChecker

.. autoclass:: SampledHandGeometry
   :members: at_width

.. autoclass:: GraspSurfaceQuery
   :members:

.. autoclass:: TriangleSurfaceQuery
   :members: query_batch_points, query_surface_normals

.. autoclass:: HandContactChecker
   :members: refine, fit_contacts, query


Paired grasp sampling
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.coordinated

Paired refinement composes the public proposal API with
``HandContactChecker.refine()``; it does not overwrite generator internals.
One immutable target-surface query is shared by the two hand refiners for each
mesh identity/version. Each hand retains independent candidate and diagnostic
state. Mesh mutation creates a new query and new refiners.

Superseded versions of the same tensor pair release their old queries,
refiners and generator backends, including when tensor storage or shape changes.
Distinct tensor pairs remain cached until ``clear_cache()`` releases all target
queries/refiners and both bound generators' mesh backends. Sampled hand geometry
remains available; runtime hosts own cleanup between unrelated asset cohorts.

The data-only sampler combines two existing antipodal generators with explicit
sampled hand geometry and contact settings. It preserves paired proposal RNG
scope and applies an optional clearance preference after candidate selection.
The facade forwards standalone calls unchanged; its numeric paired-provider
port allows a runtime owner to defer robot geometry capture outside GraspKit.
Scene observation, cargo membership and physical execution remain caller-owned.

.. autosummary::
   :nosignatures:

   CoordinatedGraspCfg
   PairedGraspProvider
   CoordinatedGraspSampler
   CoordinatedGraspPoseGenerator
   grasp_clearance_costs

.. autoclass:: CoordinatedGraspCfg
   :members:

.. autoclass:: PairedGraspProvider
   :members:

.. autoclass:: CoordinatedGraspSampler
   :members:

.. autoclass:: CoordinatedGraspPoseGenerator
   :members:

.. autofunction:: grasp_clearance_costs

Explicit URDF collision surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.urdf_surfaces

Selected mesh/box collision links are subdivided in their authored local frames.
Callers supply link names and sampling bounds; no robot, control-group naming
or simulator state is consulted.
Shape decoding and collision-origin handling are shared with fixed-opening
geometry. Its barycentric sampling and this service's subdivision sampling
retain their existing point populations.

.. autosummary::
   :nosignatures:

   sample_urdf_collision_surfaces

.. autofunction:: sample_urdf_collision_surfaces


URDF Convex Decomposition
-------------------------

The :mod:`embodichain.toolkits.acd.urdf_modifider` module converts concave URDF
collision meshes into CoACD-generated convex hulls. The high-level function can
also scale the model and recompute inertial properties.

.. currentmodule:: embodichain.toolkits.acd.urdf_modifider

.. autofunction:: generate_urdf_collision_convexes


URDF Assembly
-------------

.. automodule:: embodichain.toolkits.urdf_assembly
   :members:
   :undoc-members:
   :show-inheritance:


Captured convex hand qualification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pg_grasp.native_contact`` qualifies discrete closing, release-opening and
rigid-clearance samples using caller-supplied convex collision groups. It imports
no Robot, simulator or Lab backend. ``ClosingContactPolicy`` declares geometric
rules directly; source experiment revisions are selected by Atomic Actions.
The existing sampled-surface ``HandContactChecker`` remains independently usable.
Native offset/order semantics are preserved on this convex route. An ambiguous
clearance interval remains unavailable, and geometric acceptance does not certify
physical acquisition, release or continuous collision freedom. Close owned
queries explicitly or through the closing query's context manager.

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.native_contact

.. autosummary::
   :nosignatures:

   ClosingContactPolicy
   ConvexMeshQuery
   NativeQueryScope
   NativeContactPart
   NativeClosingQuery
   NativeOpeningQuery
   NativeRigidRestScene
   NativeRigidObstacle

.. automodule:: embodichain.toolkits.graspkit.pg_grasp.native_contact
   :members:
