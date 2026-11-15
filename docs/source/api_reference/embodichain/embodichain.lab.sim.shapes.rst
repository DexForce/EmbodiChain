embodichain.lab.sim.shapes
======================================

.. automodule:: embodichain.lab.sim.shapes

Overview
--------

Geometry configuration objects used to build collision, visual, and deformable
simulation shapes. :class:`ShapeCfg` is the common base; :class:`MeshCfg`,
:class:`CubeCfg`, :class:`PlaneCfg`, and :class:`SphereCfg` describe triangle-mesh,
box, plane, and sphere primitives respectively. :class:`MeshCollisionCfg` explicitly selects
the collision representation and its cooking settings, while
:class:`LoadOption` controls mesh loading.

``MeshCfg`` accepts either a file path or explicit vertex/triangle arrays; for
surface deformables, the array-backed form preserves node order for particle
flags and kinematic trajectories. Volume deformables generate a separate
tetrahedral simulation mesh during voxelization.

``PlaneCfg`` creates an infinite collision plane on a static rigid object.
Its ``size`` controls only the visual width and length. The plane's local normal
is +Z; the rigid object's initial pose sets its world position and orientation.

.. rubric:: Type aliases

.. autosummary::

   MeshCollisionApproximation

.. rubric:: Classes

.. autosummary::

   CubeCfg
   LoadOption
   MeshCollisionCfg
   MeshCfg
   PlaneCfg
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

.. autoclass:: PlaneCfg
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
