Geometry tensor kernels
=======================

``embodichain.compute.geometry`` provides reusable Torch operations on dense
float32/float64 tensors. Inputs share dtype and device; outputs stay on that
CPU or CUDA device. The kernels do not accept entity UIDs, asset paths, mesh
library objects or simulator handles, and do not import scene services.

Shape, dtype, device and scalar-argument checks use host metadata. Numeric
validity is computed on the tensor device and returned through masks or NaN
outputs; no tensor value is copied to the host to decide validity.

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
   group_height_offset
   transform_aabbs
   triangle_geometry
   nearest_half_turn_pose
   rotate_rigid_grasp

AABB arithmetic and group reductions
------------------------------------

.. currentmodule:: embodichain.compute.geometry.aabb

.. autosummary::
   :nosignatures:

   aabb_overlap_depth
   aabb_containment_violation
   group_height_offset

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

.. currentmodule:: embodichain.compute.geometry.rigid

.. autosummary::
   :nosignatures:

   nearest_half_turn_pose
   rotate_rigid_grasp

.. automodule:: embodichain.compute.geometry.rigid
   :members:
