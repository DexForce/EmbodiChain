# Preview default materials

`embodichain preview-material` previews a default material in a native DexSim
window or exports images from a fixed studio scene. It uses the same
`VisualMaterialCfg.from_preset()` resolver as runtime and Spawn material creation.
`python -m embodichain preview-material` is equivalent and works outside the
source checkout after installing the wheel.

See the {doc}`material catalog </overview/sim/sim_manager/rendering/materials>`
for all 24 public presets, their parameters, OfflineRT images and download setup.

## Query and preview

```bash
# Metadata only: no simulator imports, GPU initialization or material download.
embodichain preview-material --list
embodichain preview-material --list --source builtin

# Open the native window after saving a reference capture.
embodichain preview-material --material GlassClear

# Export one image and exit without a window.
embodichain preview-material --material PlasticMatteBlue --headless \
  --output outputs/blue
```

The default material is `GlassClear` and the default renderer is `rt`
(**OfflineRT**), which supports the complete catalog. A single preview stays
open until Ctrl+C; `--max-steps 100` bounds its preview updates. Physics is
disabled. Use `--gpu-id` to select the render GPU; `--size`, `--frames` and
`--spp` control capture resolution, accumulation frames and DexSim render
iterations per frame. The defaults are 320 pixels, four frames and 16 iterations.

## Override a preset

Pass a JSON object containing `VisualMaterialCfg` fields. Common fields stay
flat, while advanced fields belong inside `parameters`:

```bash
embodichain preview-material --material GlassClear --headless \
  --material-params '{"ior":1.6,"parameters":{"volume_attenuation_distance":8.0}}' \
  --output outputs/glass-variant

embodichain preview-material --material PlasticMatteBlue --headless \
  --material-params '{"roughness":0}' --output outputs/smooth-blue
```

Omitted fields and JSON `null` inherit the preset. Explicit `0` and `false`
remain overrides. Duplicate flat/native properties, invalid values,
the removed `material_type` field and texture edits fail before creating a World
or loading assets. Both handwritten and preset configurations now select
scattering from optical parameters; see the catalog's migration example.
The command owns material UIDs; select the preset using `--material`.

Variants are created before binding. The preview does not depend on native
cross-model hot mutation or change the material factory's geometry/UV behavior.

## Batch export and renderer requirements

```bash
# All 24 presets, individual WebPs and grouped galleries. Batch mode has no window.
embodichain preview-material --all --headless --gallery \
  --size 640 --frames 6 --spp 12 --output outputs/material-catalog

# Only the six built-in parameter definitions; no material-archive download.
embodichain preview-material --all --source builtin --gallery

# Explicitly export the 14 presets supported by Hybrid.
embodichain preview-material --all --renderer hybrid --compatible-only \
  --output outputs/materials-hybrid
```

Hybrid and FastRT support ten basic resource presets and four built-in
parameter presets. The remaining ten public presets and advanced parameter
variations require OfflineRT. An incompatible request fails before resource
download or native loading; the command never changes the renderer or downgrades
material parameters. With `--all --compatible-only`, excluded names and reasons
are printed to stderr and recorded in `results.json`. If no presets remain,
the command fails. `--source` filters `--list` and `--all`; `--material-params`
applies the supplied overrides to every selected preset.

Each renderer must run in a separate process. `--gallery` requires Pillow and
matplotlib; install those packages if they are absent. Gallery composition runs
after native renderer cleanup and uses the saved captures.

## Outputs and direct-native verification

Every render writes `<PresetName>_preset.png` and `results.json` under
`--output` (default `outputs/default-materials`). Use a separate directory per
variant or renderer to retain previous results. The report records the scene
recipe, renderer and DexSim version, capture settings, material source/revision,
serialized configuration, geometry choice, parameter readback and exclusions.
Unspecified configuration fields stay `null`; authored resource defaults are
not expanded into explicit overrides.

`--gallery` additionally writes `<PresetName>.webp`, `default-materials.webp`
and nonempty built-in, basic-resource and advanced-resource galleries.

```bash
# Compare the preset resolver with the original native definitions.
embodichain preview-material --all --headless --compare-native \
  --renderer rt --output outputs/material-parity
```

`--compare-native` also saves `*_native.png` and `*_native_repeat.png`, checks
parameter readback and compares the center crop with a repeat-render noise
control. A mismatch fails the command. This mode verifies authored presets and
cannot be combined with `--material-params`.

The scene uses fixed colored cards, a checkerboard, rectangular lights and
exposure. Architecture/wood/stone/ground presets use a slab; other presets use
a sphere. `SoapBubble` has an explicit inward-facing inner shell. An optional
`--model /path/to/skin_face.glb` supplies a segmented face for `SkinHumanLight`
or a batch containing it; that model is not downloaded with the materials.

The existing `examples/sim/rendering/default_materials.py` delegates to this
command and retains its headless default. Use the production CLI for window
preview or add `--no-headless` to the example.
