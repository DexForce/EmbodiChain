embodichain.lab.sim.shapes
======================================

.. automodule:: embodichain.lab.sim.shapes

Overview
--------

Geometry configuration objects used to build collision, visual, and deformable
simulation shapes. :class:`ShapeCfg` is the common base; :class:`MeshCfg`,
:class:`CubeCfg`, and :class:`SphereCfg` describe triangle-mesh, box, and
sphere primitives respectively. :class:`MeshCollisionCfg` explicitly selects
the collision representation and its cooking settings, while
:class:`LoadOption` controls mesh loading.

``MeshCfg`` accepts either a file path or explicit vertex/triangle arrays; for
surface deformables, the array-backed form preserves node order for particle
flags and kinematic trajectories. Volume deformables generate a separate
tetrahedral simulation mesh during voxelization.

``CompoundShapeCfg`` attaches body-local ``ShapePartCfg`` pieces to one rigid
body. Collision meshes stay separate convex hulls, preserving apertures and
clearance; optional independent visual parts do not participate in collision.
``CylinderCfg`` retains an analytic cylinder descriptor. Backend cooking may
approximate that primitive, so precision experiments qualify each backend.

.. rubric:: Type aliases

.. autosummary::

   MeshCollisionApproximation

.. rubric:: Classes

.. autosummary::

   CubeCfg
   CylinderCfg
   ShapePartCfg
   CompoundShapeCfg
   LoadOption
   MeshCollisionCfg
   MeshCfg
   ShapeCfg
   SphereCfg

.. currentmodule:: embodichain.lab.sim.shapes

.. autoclass:: ShapeCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: MeshCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: MeshCollisionCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: CubeCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: SphereCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: LoadOption
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CylinderCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: ShapePartCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: CompoundShapeCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict, validate
