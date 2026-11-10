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

Default material presets
------------------------

The 24-preset catalog contains six built-in parameter definitions and 18
downloadable resources. See the
:doc:`material guide </overview/sim/sim_manager/rendering/materials>` for
parameter tables, individual previews and OfflineRT galleries.

Handwritten and asset-based materials use the same configuration type::

   from embodichain.data import list_default_materials, get_default_material_info
   from embodichain.lab.sim import VisualMaterialCfg
   from embodichain.lab.sim.shapes import SphereCfg

   cfg = VisualMaterialCfg.from_preset(
       "GlassClear", uid="glass", ior=1.6,
       parameters={"volume_attenuation_distance": 8.0},
   )
   material = sim.create_visual_material(cfg)
   shape = SphereCfg(radius=0.25, visual_material=cfg)

The equivalent YAML is:

.. code-block:: yaml

   visual_material:
     preset: GlassClear
     uid: glass
     ior: 1.6
     parameters:
       volume_attenuation_distance: 8.0

With a preset, unspecified common fields remain ``None`` and inherit authored
values. GlassClear therefore retains IOR 1.52 unless an override is supplied;
explicit ``ior=1.5`` sets 1.5. Zero and false values are explicit overrides.
``cfg.replace(ior=None)`` restores inheritance. ``copy()``, ``replace()`` and
dictionary/YAML round trips preserve the other explicit fields. Construct a
new source with ``from_preset()``; ``replace()`` retains the original fields.
Handwritten configurations still fill the historical defaults at construction.

Common fields stay flat; native schema-v1 advanced parameters belong in
``parameters``. ``emissive_intensity`` normalizes to ``emission_intensity``.
Common properties supplied through ``parameters`` are moved to their flat
fields during construction, so ``replace(ior=None)`` also clears that form of
override. The remaining advanced properties stay in ``parameters``.
Specifying the same native property in both places is an error. Unknown,
nonfinite, incorrectly typed and out-of-range parameters fail before native
loading. Presets reject texture replacement. Both handwritten and preset
materials select scattering from optical parameters; ``material_type`` and
the ``VisualMaterial.RT_MATERIAL_TYPES`` / ``MAT_TYPE_MAPPING`` constants have
been removed. No default BRDF is explicitly written to native instances.
Manual configurations with advanced parameters use a complete native manifest::

   ice = VisualMaterialCfg(
       base_color=[1, 1, 1, 1], roughness=0.2, ior=1.31,
       parameters={"transmission": 1.0},
   )

Transmission and other advanced handwritten parameters require OfflineRT.
Remove ``material_type`` from stored Python/YAML configurations; ``from_dict``
rejects the removed key even if its value is ``None``. Legacy BRDF/BTDF/BSDF
labels are not silently converted because their scattering semantics are not
interchangeable. See the
:doc:`migration example </overview/sim/sim_manager/rendering/materials>`.

``BrickRed``, ``WoodOak``, ``CeramicTileWhite``, ``MarbleWhite``, ``AsphaltDark``,
``FabricGray``, ``LeatherBrown``, ``PlasticABSBlack``, ``RubberBlack`` and
``SteelBrushed`` support ``hybrid``, ``fast-rt`` and ``rt``. ``ChromePolished``,
``GoldPolished``, ``CopperPolished``, ``GlassClear``, ``GlassFrosted``,
``GlassSmoke``, ``SkinHumanLight`` and ``SoapBubble`` require ``rt`` (OfflineRT).
The built-ins ``PlasticMatteBlue``, ``PlasticGlossyRed``, ``GoldMetal`` and
``MirrorMetal`` support all three renderers. ``GlassClearSurface`` and
``GlassFrostedSurface`` require ``rt`` and configure surface transmission without
volume absorption.
Advanced parameter variations of a basic preset also require ``rt``; the
factory rejects incompatible renderers before downloading or loading assets.
NoRender stores declarations without downloading or creating native materials.
Render-only VisualObject facades remain unbound in that mode; their declarations
are retained, while physics assets still prepare normally.

Variants stage only JSON before binding. Textures stay in the shared resource
cache. Equivalent definitions reuse immutable native templates, while each UID
has a separate child instance. Explicit resets restore the entire template,
including advanced parameters and textures. Repeated creation preserves state.
Native shader-model changes after binding are outside this interface's scope;
pre-create variants and bind their instances instead.

Catalog queries do not require a World or asset download. The six built-ins
include native parameter defaults in the wheel and do not request the texture
archive. Their metadata exposes ``source="builtin"`` and
``requires_download=False``. Resource metadata uses ``source="resource"``.
The archive is lazy,
revisioned and checksum-verified. Pre-download with
``embodichain data download --name DefaultMaterials``. The current upstream
archive is hosted on the DexSim internal resource server. To use a mirror, set
``EMBODICHAIN_DOWNLOAD_PREFIX`` before importing EmbodiChain and place the same
archive at
``materials/default-materials/ec9f748242c50b85/offlinert_materials.zip`` below
that prefix. The wheel contains metadata, not PNGs. The downloaded archive
retains the upstream README and manifests; source attribution and file digests
are recorded in ``embodichain/data/default_materials.json``.

The optional ``SoapBubbleInterior`` auxiliary material references SoapBubble's
textures. Hollow bubble shells and segmented skin meshes belong to preview
geometry. The factory neither creates those shapes nor changes UVs.
``SkinHumanLight`` approximates diffuse transmission rather than random-walk
subsurface scattering. Use ``embodichain preview-material`` for previews,
optional segmented-face geometry and direct-native image comparisons; see the
:doc:`CLI guide </guides/preview_material>`.

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
