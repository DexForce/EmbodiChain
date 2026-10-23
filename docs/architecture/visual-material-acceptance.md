# Visual material acceptance

Target `dexsim_engine==0.5.1rc1` with its matching Python wheel, Warp `1.17.0`
and Newton `1.6.0`. Install the repository with its bundled extensions before
running the checks. See [material contracts](../../agent_context/topics/simulation-system/materials.md)
for implementation ownership and lifecycle rules.

## Acceptance criteria

| Boundary | Required behavior |
| --- | --- |
| Identity and aliases | Unnamed materials keep independent appearances. Explicit UID conflicts fail across runtime and Spawn creation in either order. Equivalent aliases and repeated instance creation preserve native state. |
| PBR conversion | Both creation paths agree for BRDF/BTDF/BSDF, emission intensities 0/1/10 and nondefault scalar properties. Invalid values fail before native writes. RGBA writes retain alpha; RGB getters cannot verify it. |
| Native handles | Imported and working instances update their exact native handles. Material IDs distinguish different instances with identical names. |
| Selection and reset | Empty selections do nothing; invalid indices and partial shared selections fail before writes. Independent updates affect selected environments only. Deterministic rigid-object and group resets restore selected properties. |
| Randomization resources | Solid and library tiers honor scalar PBR ranges. Shared instances are mutated once per call. Different source directories remain separate; warmed texture pools and instance registries do not grow on repeated calls. |
| Rebuilds and replacement | Current per-segment assignments survive reset and topology rebuilds. Failed replay can retry; removal discards old snapshots before same-UID replacement. Equal replacement definitions reuse registration. |
| Native qualification | Hybrid scenes with two environments pass on Default and Newton. After 1,025 partial randomizations per backend, native instance counts remain stable, requested properties hold, and the other environment is unchanged. |

Use fixed sampling ranges to test requested properties deterministically.
Use floating-point comparisons with an absolute tolerance of `1e-5` for native
color round trips. Solid palette colors include uint8 quantization and finite
palette sampling; they do not represent arbitrary continuous colors.

## Commands

Verify the actual runtime, then run CPU and native checks separately. Keep
native resource tests serial.

```bash
python -c 'from importlib.metadata import version; import dexsim; assert version("dexsim-engine") == "0.5.1rc1"; print(version("warp-lang"), version("newton"))'

python -m pytest -q -m 'not requires_sim and not gpu' \
  tests/sim/test_material.py tests/sim/test_material_texture_obj.py \
  tests/sim/test_material_cache_config.py \
  tests/sim/objects/test_asset_material_initialization.py \
  tests/sim/objects/test_rigid_object_reuse_material.py \
  tests/sim/objects/test_articulation_reuse_material.py \
  tests/gym/envs/managers/test_randomize_visual_material.py \
  tests/sim/objects/test_visual_object.py \
  tests/sim/spawn tests/sim/test_cfg.py tests/gym/utils/test_gym_utils.py

python -m pytest -q --run-gpu \
  tests/sim/test_material_integration.py \
  tests/sim/objects/test_visual_object_integration.py::test_render_only_object_native_lifecycle

python -m black --check .
python docs/scripts/check_api_docs.py
python .agents/skills/project-dev-context/scripts/context.py check
```

## Default material package

In addition to the legacy handwritten-material checks above, qualify the unified
`VisualMaterialCfg.from_preset()` route:

| Boundary | Acceptance |
| --- | --- |
| Inheritance | GlassClear inherits IOR 1.52; explicit 1.5 overrides it. Zero/false are explicit; `replace(ior=None)` restores inheritance. No flat/native duplicate properties or invalid parameters reach native loading. |
| Round trips | Constructor, factory, Python/YAML, copy, replace and dictionary round trips retain the preset and unspecified state. Manual construction retains historical concrete defaults. |
| Distribution | Offline queries list 18 public presets, with SoapBubbleInterior available explicitly. Build/install a wheel away from the source checkout. With an empty data cache, download the pinned archive; repeat without network, repair a missing/corrupt member, and verify auxiliary texture references. |
| Renderer preflight | Ten basic presets load on Hybrid/FastRT/OfflineRT. Eight advanced presets, the helper and advanced parameter variations require OfflineRT. Reject before download/native loading. NoRender preserves declarations and does neither. |
| Native resources | Runtime-first and inline-first aliases share exact handles. Different UIDs use separate children of immutable templates. Repeated creation preserves state and bounded native instance counts. Variants stage JSON only, never duplicate PNGs. |
| Lifecycle | Complete native template reset restores properties beyond the flattened fields. Default/Newton selection, reset, topology rebuild and cleanup preserve identity and expected assignments. |
| Optical parity | Render all 18 on OfflineRT and the ten basic presets on Hybrid and FastRT. Compare the package route to direct native loading with identical geometry/camera/lights, plus a native-repeat noise control. Use closed glass meshes, explicit double shells for bubbles and optional segmented face geometry. |

```bash
python -m pytest -q tests/data/test_default_materials.py tests/sim/test_material.py
python -m pytest -q --run-gpu tests/sim/test_material_integration.py

# Each renderer requires a separate process. Run native checks serially.
python examples/sim/rendering/default_materials.py --all --compare-native --renderer rt
python examples/sim/rendering/default_materials.py --all --compare-native --renderer hybrid
python examples/sim/rendering/default_materials.py --all --compare-native --renderer fast-rt
# Optionally add --model /path/to/skin_face.glb for SkinHumanLight.
```

The preview writes candidate/native/repeat images and `results.json`. Its center
crop RMSE tolerance is max(3/255, three times native-repeat RMSE plus 0.5/255).
Inspect the images as well as the metric: similar dark or empty frames alone do
not establish material correctness. The first material load includes data-cache
verification; creation timings are descriptive, not a render-throughput claim.
The current pinned upstream address is internal; public mirrors must provide
exactly the archive and versioned path documented in the material API. Asset
redistribution/provenance must be retained when deploying a mirror.

Parameter variants must be created before binding. Qualification does not cover
cross-model hot shader mutation, NRD packaging fixes, automatic geometry/UV
changes or a global removal of the legacy manual `material_type` interface.

## Compatibility

- `base_color` reads return native display-sRGB RGB; configuration and writes
  still accept RGBA.
- Unnamed configurations use `uid=None`. Remove an old explicit
  `uid: default_mat` or assign distinct UIDs when independent appearances are required.
- Repeated `create_instance(uid)` preserves identity and state. Use
  `set_default_properties()` for an explicit property reset.
- Explicit definitions remain reserved for the World lifetime, including after
  `clean_materials()` releases runtime wrappers and unused native resources.
- Shared material randomization requires all environments; partial selections
  are rejected.

## Release qualification

State and lifecycle checks do not establish optical correctness. Before release:

- Compare BRDF/BTDF/BSDF reflection, transmission, alpha and emission intensities
  0/1/10 in a separate FastRT process with controlled lighting, camera and exposure.
- Check multi-material meshes, URDF articulations and deformables across partial
  reset and rebuilds. Compare file, Tensor and native Texture inputs for color
  space, normal-map direction and rebinding behavior.
- Run 10,000 resets while recording native instance counts, texture counts and
  VRAM. Distinguish warmup caches from sustained growth; Torch allocator statistics
  do not cover all Vulkan/OptiX memory.

Record run-specific environment details, timings, results and unexecuted checks
in the PR description.
