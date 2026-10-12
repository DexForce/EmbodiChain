GenSim scene adapters
=====================

These existing entry points adapt GenSim's editable Y-up layouts and asset
files to shared numerical geometry and arena-local scene services. They retain
diagnostic output and generation/edit orchestration. GenSim still owns model
clients, asset preparation, temporary simulator lifecycle and export formats.
Support selection, polygon search, greedy packing and SLSQP are CPU strategies
owned by ``sim.scene``. Their reusable tensor operations use
``compute.geometry``; kernel CUDA support does not migrate these pipelines to GPU.

embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_layout_optimizer
-----------------------------------------------------------------------------

Preserves the generation path's greedy AABB separation strategy and Y-up output; numeric offsets come from the shared scene CPU packing strategy.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_layout_optimizer

.. autosummary::
   :nosignatures:

   AssetsSupportLayoutOptimizerConfig
   AssetsSupportLayoutOptimizer

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_layout_optimizer
   :members:
   :imported-members:

embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp
--------------------------------------------------------------------------

Applies a shared support-valid group translation to Y-up layouts. Existing Polygon/MultiPolygon, hole and diagnostic-image behavior remains available.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp

.. autosummary::
   :nosignatures:

   SupportGeometry
   AssetsGroupSupportClampConfig
   AssetsGroupSupportClamp

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp
   :members:
   :imported-members:

embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_table_aligner
--------------------------------------------------------------------------

Measures transformed geometry, uses the shared group-height offset and converts the resulting layout back to Y-up without changing relative poses.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_table_aligner

.. autosummary::
   :nosignatures:

   AssetsGroupTableAlignerConfig
   AssetsGroupTableAligner

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_table_aligner
   :members:
   :imported-members:

embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler
---------------------------------------------------------------

Owns its temporary simulation and asset materialization. Fixed-step settling remains the default; optional convergence settings request measured stability through shared scene validation.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler

.. autosummary::
   :nosignatures:

   GravitySettlerConfig
   GravitySettleBody
   GravitySettler

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler
   :members:
   :imported-members:

embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer
-------------------------------------------------------------------------------

Translates per-parent support problems into shared scene layout inputs. The Scene Engine caller retains hierarchy order, edit selection and pose propagation.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer

.. autosummary::
   :nosignatures:

   ParentSurfaceLayoutProblem
   ParentSurfaceLayoutOptimizerConfig
   ParentSurfaceLayoutOptimizer

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer
   :members:
   :imported-members:

embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface
---------------------------------------------------------------------

Keeps file-independent mesh input and optional support-detection plots; the shared scene CPU detector owns surface selection.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface

.. autosummary::
   :nosignatures:

   SupportSurfaceConfig
   TableSupportRegion
   TableSupportSurfaceDetector

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface
   :members:
   :imported-members:

embodichain.gen_sim.scene_engine.pipeline.utils.table_surface_layout_optimizer
------------------------------------------------------------------------------

Translates the existing tabletop problem into shared layout constraints while retaining its compatibility settings and debug output.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.table_surface_layout_optimizer

.. autosummary::
   :nosignatures:

   TableSurfaceLayoutProblem
   TableSurfaceLayoutOptimizerConfig
   TableSurfaceLayoutOptimizer

.. automodule:: embodichain.gen_sim.scene_engine.pipeline.utils.table_surface_layout_optimizer
   :members:
   :imported-members:
