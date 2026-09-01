embodichain.lab.sim.shapes
======================================

.. automodule:: embodichain.lab.sim.shapes

Overview
--------

Geometry configuration objects used to build collision, visual, and deformable
simulation shapes. :class:`ShapeCfg` is the common base; :class:`MeshCfg`,
:class:`CubeCfg`, and :class:`SphereCfg` describe triangle-mesh, box, and
sphere primitives respectively, and :class:`LoadOption` controls how mesh
assets are loaded and decomposed. ``MeshCfg`` accepts either a file path or
explicit vertex/triangle arrays; for surface deformables, the array-backed form
preserves node order for particle flags and kinematic trajectories. Volume
deformables generate a separate tetrahedral simulation mesh during voxelization.

.. rubric:: Classes

.. autosummary::

   CubeCfg
   LoadOption
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
