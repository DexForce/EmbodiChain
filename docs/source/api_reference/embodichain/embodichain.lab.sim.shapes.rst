embodichain.lab.sim.shapes
======================================

.. automodule:: embodichain.lab.sim.shapes

Overview
--------

Geometry configuration objects used to build the collision and visual shapes of
rigid bodies. :class:`ShapeCfg` is the common base; :class:`MeshCfg`,
:class:`CubeCfg`, and :class:`SphereCfg` describe triangle-mesh, box, and
sphere primitives respectively. :class:`MeshCollisionCfg` explicitly selects
the collision representation and its cooking settings, while
:class:`LoadOption` controls mesh loading.

.. rubric:: Type aliases

.. autosummary::

   MeshCollisionApproximation

.. rubric:: Classes

.. autosummary::

   CubeCfg
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
