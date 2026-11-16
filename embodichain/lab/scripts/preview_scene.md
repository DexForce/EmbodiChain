# Standard USD scene preview

`embodichain preview-scene` belongs to Lab and accepts ordinary `.usd`, `.usda`,
`.usdc` and relocatable `.usdz` files. `SimulationManager.add_usd()` resolves
source structure, supported entity types, transforms, units, materials and state.
The manager-owned colliding ground is disabled, so preview adds no synthetic
collider to the source. Existing asset/robot/Gym defaults remain unchanged.
Renderer `auto` reserves Hybrid before source discovery so source cameras and
screenshots can be created. Explicit no-render configuration is still honored
for load-only workflows.
Producer-specific custom metadata is opaque: no GenSim manifest, private UID or
`initial_qpos` adapter is used, and no model-service configuration is loaded.

```bash
embodichain preview-scene --usd-file scene.usdz
embodichain preview-scene --usd-file scene.usda --viser
embodichain preview-scene --usd-file scene.usdc --simulate
embodichain preview-scene --usd-file scene.usdz --headless --report inspection.json
embodichain preview-scene --usd-file scene.usdz --headless --screenshot view.png
embodichain preview-scene --usd-file scene.usdz --headless --simulate --steps 100
```

The default displays imported state without physics steps. Camera interaction and
explicit paused joint edits still work. `--simulate` starts physics; `--steps`
requires it and provides a finite step budget. Headless simulation requires
`--steps`. `--headless --viser` keeps the browser session alive. The native window
and Viser share the display loop, stepping policy and teardown owner.

Native shortcuts are **Space** (play/pause), **N** (one physics step), **Home**
(reload source), and **Escape** (stop). Viser exposes the corresponding buttons.
Joint sliders are available when paused and only edit commanded joints. They are
hidden while physics advances so preview does not overwrite evolving qpos or
hold every joint at its initial pose. Asset and robot previews keep their own
existing pose-holding default.

Reset recreates the owned scene rather than restoring a partial solver checkpoint.
It starts paused and aborts if the source entry bytes changed during the session.
There is no implicit file watching, geometry repair, generation or conversion.

Reports contain source identity, source prim/resource keys, imported entity types,
backend/frontend, committed physics-step count, simulation time and staged error
diagnostics. A successful load is not an asset certification. Native importer and
renderer limitations remain explicit failures. Screenshots are final offscreen
PNG camera images, and render warmup never advances physics. `--resolution WIDTH
HEIGHT` controls their dimensions. Outputs are written only when requested and
cannot replace the input entry.

The Python `preview_scene()` function accepts a `SimulationManagerCfg`, owns its
simulation and returns a portable report after teardown. Existing hosts can
continue using the borrowed `_preview.py` loop for loaded assets and playback.

## GenSim migration

Use the newly exported USD file directly:

```bash
embodichain gensim scene preview --usd-file outputs/scene/scene_usd/scene.usda --viser
```

This is a thin alias to the Lab CLI. The old `--scene`, `--output_root`,
`--output-root` and `--usd` preview arguments and manifest/native-asset loaders are
removed. Existing scripts that expected automatic physics stepping must add
`--simulate`. Source-specific conversion and state authoring belong to the
producer and native importer; preview does not restore old GenSim custom fields.

Scene Engine, Gradio and saved-layout playback use the same standard-file path.
Initial view fitting prefers volumetric geometry when present so a large flat
context floor does not hide the objects of interest.
A workspace must already contain its exported USD; preview never builds a missing
USD as a side effect.
