embodichain.lab.sim.material
========================================

.. automodule:: embodichain.lab.sim.material

Overview
--------

Visual materials control how scene objects are rendered. A
:class:`VisualMaterialCfg` describes PBR appearance properties (base color,
metallic, roughness, emissive, ...) used by both rasterization and ray-tracing
passes; :class:`VisualMaterial` is the resolved material definition and
:class:`VisualMaterialInst` is a per-object instance bound into the scene.
:class:`ReuseSegmentState` tracks per-render-body segment reuse so that
segmentation masks stay consistent across replicated environments.

.. rubric:: Classes

.. autosummary::

   VisualMaterialCfg
   VisualMaterial
   VisualMaterialInst
   ReuseSegmentState

.. currentmodule:: embodichain.lab.sim.material

.. autoclass:: VisualMaterialCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autoclass:: VisualMaterial
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: VisualMaterialInst
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ReuseSegmentState
   :members:
   :undoc-members:
   :show-inheritance:
