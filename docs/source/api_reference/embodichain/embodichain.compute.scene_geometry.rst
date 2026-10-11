Scene geometry and layout
=========================

These numerical packages accept arrays and polygonal geometry, never
simulation objects. They own no world, resource registry, task semantics or
file-based GenSim model. Scene and GenSim adapters reuse these implementations.

Support detection uses already transformed Z-up mesh vertices in metres.
The initial detector selects the largest outer contour at the chosen main
horizontal level. Group placement accepts holes and disconnected polygonal
regions, uses raster proposals, then checks exact polygon containment.
The greedy AABB optimizer is distinct from the constrained SLSQP layout solver:
the former preserves the existing generation strategy, while the latter
expresses explicit fixed positions, bounds, planar relations and preferences.
Neither is an exact 3D collision or physical stability solver.

embodichain.compute.geometry
----------------------------

Package-level geometry contracts and reusable placement algorithms.

.. currentmodule:: embodichain.compute.geometry

.. autosummary::
   :nosignatures:

   SupportSurfaceConfig
   DetectedSupportSurface
   SupportSurfaceDetector
   SupportGeometry
   GroupSupportClampConfig
   GroupSupportClamp
   group_height_offset
   AABBLayoutConfig
   AABBLayoutOptimizer

embodichain.compute.geometry.support
------------------------------------

Array-based mesh support extraction. Normal repair and seam merging are internal geometric preparation; input arrays are not modified.

.. currentmodule:: embodichain.compute.geometry.support

.. autosummary::
   :nosignatures:

   SupportSurfaceConfig
   DetectedSupportSurface
   SupportSurfaceDetector

.. automodule:: embodichain.compute.geometry.support
   :members:
   :imported-members:

embodichain.compute.geometry.placement
--------------------------------------

Rigid-group XY translation and vertical alignment. ``find_translation`` returns ``None`` when its bounded search finds no valid translation.

.. currentmodule:: embodichain.compute.geometry.placement

.. autosummary::
   :nosignatures:

   SupportGeometry
   GroupSupportClampConfig
   GroupSupportClamp
   group_height_offset

.. automodule:: embodichain.compute.geometry.placement
   :members:
   :imported-members:

embodichain.compute.geometry.aabb
---------------------------------

Greedy pairwise separation with support containment and a finite round budget. Unsuccessful search raises ``ValueError`` without establishing mathematical infeasibility.

.. currentmodule:: embodichain.compute.geometry.aabb

.. autosummary::
   :nosignatures:

   AABBLayoutConfig
   AABBLayoutOptimizer

.. automodule:: embodichain.compute.geometry.aabb
   :members:
   :imported-members:

embodichain.compute.layout
--------------------------

Constrained fixed-orientation AABB layout. Results include candidate coordinates, status and measured residuals even on numerical failure.

.. currentmodule:: embodichain.compute.layout

.. autosummary::
   :nosignatures:

   AABBLayoutResult
   solve_aabb_layout

embodichain.compute.layout.aabb
-------------------------------

The SLSQP solver eliminates fixed variables, adds separating constraints for detected pairs, and reports budget exhaustion separately from a failed solve. Bounds constrain complete boxes rather than only their centers.

.. currentmodule:: embodichain.compute.layout.aabb

.. autosummary::
   :nosignatures:

   AABBLayoutResult
   solve_aabb_layout

.. automodule:: embodichain.compute.layout.aabb
   :members:
   :imported-members:
