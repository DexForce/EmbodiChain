Geometry arrays and tensor kernels
==================================

``embodichain.compute.geometry`` provides reusable Torch operations on dense
float32/float64 tensors. Inputs share dtype and device; outputs stay on that
CPU or CUDA device. The kernels do not accept entity UIDs, asset paths, mesh
library objects or simulator handles, and do not import scene services.

For the batched AABB and triangle APIs, shape, dtype, device and scalar-argument checks use host
metadata. Numeric
validity is computed on the tensor device and returned through masks or NaN
outputs; no tensor value is copied to the host to decide validity.
The CPU contact-patch API below instead consumes NumPy arrays and reports
finite-qualification errors explicitly.

Independent batches and object axes
-----------------------------------

Leading dimensions broadcast according to Torch rules. ``B`` denotes
independent layouts and ``N`` denotes objects within one layout. A collection
of N objects is not N independent layouts. Pairwise comparisons require
explicit singleton axes; there is no implicit all-pairs expansion:

.. code-block:: python

   # boxes: (B, N, 2, 3), with min/max corners in the penultimate axis.
   from embodichain.compute.geometry import aabb_overlap_depth

   depths = aabb_overlap_depth(boxes[:, :, None], boxes[:, None, :])
   # depths: (B, N, N, 3), with comparisons confined to each batch.

``group_height_offset`` reduces only the N object axis of ``(..., N, 2, 3)``
boxes. Its boolean ``valid_mask`` has shape ``(..., N)`` and ignores padded
objects, including their invalid numeric values. Support heights and mask
leading axes may broadcast with the independent batch dimensions. Empty or
all-padding groups return zero offsets and false validity. This object mask
is independent of simulator environment selection.

Operations and invalid values
-----------------------------

.. list-table:: Tensor contracts
   :header-rows: 1
   :widths: 25 30 45

   * - Operation
     - Output
     - Numeric invalidity
   * - ``aabb_overlap_depth``
     - Per-axis intersection lengths, ``(..., D)``
     - Nonfinite/reversed boxes or nonfinite results produce all-NaN rows.
   * - ``aabb_containment_violation``
     - Per-axis boundary violations, ``(..., D)``
     - Invalid boxes or regions produce all-NaN rows; boundary contact is valid.
   * - ``aabb_separation``
     - Euclidean box distance in the broadcast batch shape
     - Invalid boxes produce NaN; overlapping or touching boxes have zero distance.
   * - ``group_height_offset``
     - ``(offset, valid)`` in the broadcast batch shape
     - An unmasked invalid box or invalid support/result invalidates that group.
   * - ``transform_aabbs``
     - Affine-transformed bounds, ``(..., 2, 3)``
     - Invalid boxes, transforms, non-affine last rows or overflow produce NaN boxes.
   * - ``triangle_geometry``
     - Normals ``(..., 3)``, areas ``(...)``, validity ``(...)``
     - Degenerate, nonfinite or unrepresentable triangles return zero measurements and false validity.

AABB overlap is positive volume only when all axis depths are positive;
touching faces produce zero on the touching axis. Affine transforms support
rotation, nonuniform scale, shear and reflection, with an exact final row of
``[0, 0, 0, 1]``. They bound transformed input boxes, not the original mesh.
Triangle winding controls normal sign; callers own slope and area thresholds.

CPU strategies and benchmark scope
----------------------------------

Scene support selection, polygon placement, greedy packing and constrained
SLSQP solving remain CPU strategies under ``sim.scene``. They may invoke these
kernels using CPU tensors. Kernel CUDA support does not move those strategies
to GPU or enable B>1 scene-state operations and physical hosts.

The :download:`geometry kernel benchmark
<../../../../scripts/benchmark/scene/geometry_kernels.py>` measures kernels
separately from scene solver and simulator work. Compare recorded shapes,
dtype, device, warmup and synchronization policy when interpreting results.

Public kernel namespace
-----------------------

.. currentmodule:: embodichain.compute.geometry

.. autosummary::
   :nosignatures:

   aabb_overlap_depth
   aabb_containment_violation
   aabb_separation
   convex_projection_distance_to_origin
   axis_contact_pose
   group_height_offset
   transform_aabbs
   triangle_geometry
   nearest_half_turn_pose
   rotate_rigid_grasp
   transform_vertex_bounds
   aabb_sphere_cover

AABB arithmetic and group reductions
------------------------------------

.. currentmodule:: embodichain.compute.geometry.aabb

.. autosummary::
   :nosignatures:

   aabb_overlap_depth
   aabb_containment_violation
   group_height_offset
   aabb_separation

.. automodule:: embodichain.compute.geometry.aabb
   :members:

Affine box transforms
---------------------

.. currentmodule:: embodichain.compute.geometry.transforms

.. autosummary::
   :nosignatures:

   transform_aabbs

.. automodule:: embodichain.compute.geometry.transforms
   :members:

Mesh envelopes
--------------

``transform_vertex_bounds`` projects the actual B=1 vertices before reducing
bounds, rather than rotating a larger input AABB. ``aabb_sphere_cover`` covers
the input AABB using the existing 27 circumscribed cell spheres. These pure
tensor operations carry no simulation identity and do not certify cooked shapes
or continuous collision freedom.

``input_mesh_bounds`` and ``input_mesh_radius`` consume explicit ordered
Trimesh inputs. Install ``embodichain[geometry]`` for these CPU mesh routines;
importing the tensor-only mesh-bound functions does not load Trimesh.
``maximum_halfspace_violation`` measures all supplied points in bounded
256-row multiplication chunks. ``iter_clearance_translations`` yields the
original horizontal positive/negative and upward candidates lazily, rounding
clearances outward in the caller's tensor dtype.

.. currentmodule:: embodichain.compute.geometry.mesh_bounds

.. autosummary::

   transform_vertex_bounds
   aabb_sphere_cover
   input_mesh_bounds
   input_mesh_radius
   maximum_halfspace_violation
   iter_clearance_translations

.. autofunction:: transform_vertex_bounds

.. autofunction:: aabb_sphere_cover

.. autofunction:: input_mesh_bounds

.. autofunction:: input_mesh_radius

.. autofunction:: maximum_halfspace_violation

.. autofunction:: iter_clearance_translations

Triangle measurements
---------------------

.. currentmodule:: embodichain.compute.geometry.mesh

.. autosummary::
   :nosignatures:

   triangle_geometry

.. automodule:: embodichain.compute.geometry.mesh
   :members:

Rigid grasp transforms
----------------------

These pure tensor operations select a half-turn grasp orientation and rotate
a link with its rigidly attached TCP about a link-local axis. Atomic motion
primitives and benchmark case generation share this algebra; physical
validation and execution remain in their consuming layers.
``rigid_pose_displacement`` returns corresponding translation distances and
rotation angles without selecting elapsed time or stability thresholds.

.. currentmodule:: embodichain.compute.geometry.rigid

.. autosummary::
   :nosignatures:

   nearest_half_turn_pose
   rotate_rigid_grasp
   rigid_pose_displacement
   pose_axis_alignment
   axis_contact_pose
   validate_rigid_pose_batch
   rotate_held_object_world_z
   rotate_final_pose_world_z


.. automodule:: embodichain.compute.geometry.rigid
   :members:

Circular-prism qualification
----------------------------

The CPU source-mesh proof checks two complete radial rings, convex closed
triangle topology and strict containment of additional rotor colliders. It
preserves the existing qualification tolerances and returns a radius, without
simulation state or task-specific success evaluation.

.. currentmodule:: embodichain.compute.geometry.cylinder

.. autofunction:: circular_prism_radius

Native support patch measurements
---------------------------------

This CPU NumPy/Shapely calculation consumes numeric contact arrays and explicit
actor IDs. Positive impulses with normalized upward normals at least 0.9 form
the measured XY patch. The reported COM distance is measured against its convex
hull; no silhouette containment, stability window or task verdict is inferred.

.. currentmodule:: embodichain.compute.geometry.contact_support

.. autosummary::
   :nosignatures:

   support_from_contacts
   contact_load_path

.. autofunction:: support_from_contacts

``contact_load_path`` checks one sampled contact graph using positive upward
loads and explicit dynamic cargo IDs. It does not join paths across substeps or
choose a stable-window duration. That qualification belongs to the consumer.

.. autofunction:: contact_load_path

Pour arcs and grasp projections
-------------------------------

These B=1 numerical routines retain an observed staging orientation, generate
an ordered-jaw fixed-pivot arc, and project five downstream targets for each
candidate grasp before IK. ``PourGraspProjection`` owns its numeric inputs and
retains invalid candidates' staging fallbacks. Its angle and sample count are
resolved query operands; policies, scene observations and physical acceptance
remain with callers.

.. currentmodule:: embodichain.compute.geometry.pouring

.. autosummary::

   PourFrameSet
   JawPourFrameSet
   PourGraspProjection
   gravity_top_frame
   pour_staging_pose
   pour_frames
   jaw_tilt_axis
   jaw_pour_frames
   source_pour_staging_frame

.. automodule:: embodichain.compute.geometry.pouring
   :members:

Triangle mesh XY projection
---------------------------

The complete positive-area triangle union preserves concavities, holes and
disconnected components. It takes plain native vertex/face arrays and retains
source float64 projection arithmetic and face order. It does not select floor
regions, choose placements, inspect simulation handles or compute task success.

.. currentmodule:: embodichain.compute.geometry.projection

.. autosummary::

   triangle_mesh_xy_projection

.. autofunction:: triangle_mesh_xy_projection

Projected clearance and envelope gaps
-------------------------------------

The convex-projection routine returns a lower bound to the origin distance
for explicit planar points. Signed envelope gaps retain their supplied
transforms and reduction order. Neither query acquires scene geometry or
selects a collision tolerance.

.. automodule:: embodichain.compute.geometry.convex_projection
   :members:

.. automodule:: embodichain.compute.geometry.envelope
   :members:

Grasp samples and release clearance
-----------------------------------

These routines consume measured numeric hand samples, object vertices and
explicit geometric axes. They retain candidate order and expose masks or
proposals; a returned direction or clearance does not establish acquisition,
release or a collision-free motion. Uniform grasp fitting preserves the
source support bottom while reporting the original pad-span margins.

.. automodule:: embodichain.compute.geometry.grasp_fit
   :members:

.. automodule:: embodichain.compute.geometry.gripper
   :members:

.. automodule:: embodichain.compute.geometry.hand_clearance
   :members:

Placement dimensions and ordered proposals
------------------------------------------

Mesh-axis and placement dimensions use explicit input geometry and target
orientation. The CPU proposal routines preserve complete projected footprints,
contact-patch COM coverage and the first qualified candidate's original order.
They consume numerical arrays and polygons; object ownership, observed contact
and task acceptance stay with their callers.

.. currentmodule:: embodichain.compute.geometry.landing

.. autofunction:: select_table_landing

.. automodule:: embodichain.compute.geometry.placement_dimensions
   :members:

.. automodule:: embodichain.compute.geometry.placement_proposals
   :members:

Measured mesh projections
--------------------------

``PadProjectionStroke`` owns explicit captured mesh and pose samples without
retaining runtime objects. Preshape fitting uses the full target projection;
floor-content projection and occupied-floor proposals keep source order and
reservation measurements. These results provide geometric facts and proposed
displacements, without interpreting physical effect evidence.

.. automodule:: embodichain.compute.geometry.measured_manipulation
   :members:

Source geometry adaptation
---------------------------

Source relation helpers retain the original orientation labels, AABB support
tie order and confinement measurements on explicit arrays. Target helpers
derive declared approach directions, upright orientations, release heights
and bounded carry distances without reading semantic graphs or choosing
robot defaults. ``region_staging_pose`` changes only staging XY within an
explicitly projected and inset region.

.. automodule:: embodichain.compute.geometry.source_relations
   :members:

.. automodule:: embodichain.compute.geometry.source_targets
   :members:

.. automodule:: embodichain.compute.geometry.pour_region
   :members:

Rigid staging inputs
---------------------

``as_rigid_matrix`` checks the original finite, proper-transform tolerances
without repairing or reducing input precision. ``distal_staging_variants``
uses explicit TCP/root geometry to produce the ordered outward translations;
rejected geometry or bounds remain unavailable proposals. Robot kinematics,
collision queries and execution policy are outside these numeric operations.

.. automodule:: embodichain.compute.geometry.pose_checks
   :members:

.. automodule:: embodichain.compute.geometry.staging
   :members:

Qualified contact heights
-------------------------

``contact_support_heights`` selects the highest qualified patch point without
averaging penetration away. Missing load, COM support or finite coordinates
produce NaN and preserve the caller's conservative envelope. The sensor-facing
``qualified_support_heights`` API delegates to this shared calculation; TaskSpec
uses it without importing Lab or acquiring physical state.

.. autofunction:: embodichain.compute.geometry.contact_support.contact_support_heights

Recorded rigid Pour measurements
--------------------------------

These pure numeric functions measure a recorded rigid tilt/restore window and
receiver projection. They take explicit source vertices and poses, preserve
input precision and first-maximum order, and set no acceptance thresholds or
fluid-transfer conclusions.

.. currentmodule:: embodichain.compute.geometry.pouring

.. autofunction:: measure_pour_tilt
   :no-index:

.. autofunction:: measure_pour_receiver
   :no-index:

Control source preparation numbers
----------------------------------

These routines consume explicit dimensions, bounds and calibration values.
They retain original bounded candidates, XY tie order and coefficient rules,
and do not acquire assets, robots, scenes or physical acceptance evidence.

.. currentmodule:: embodichain.compute.geometry.control_grasp_scale

.. autoclass:: TwistGripDimensionError

.. autofunction:: select_control_grasp_scale

.. currentmodule:: embodichain.compute.geometry.control_resize

.. autofunction:: control_bounds_overlap

.. autofunction:: control_clearance_translations

.. currentmodule:: embodichain.compute.geometry.press_calibration

.. autofunction:: source_press_release_bias
