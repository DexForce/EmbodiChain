# Generation, ingest and UI details

[Topic overview](gen-sim.md). Read this for the selected detailed flow.

### Lifecycle and configuration path

1. `scene-engine` CLI validates that an input image exists and has an accepted suffix, and
   that edit-only mode has an existing output root (`scene_engine/cli/start.py`). With both
   image and edit prompt it generates first, then edits.
2. Generation creates VLM, segmentation, geometry, and conditionally articulation clients.
   Segmentation, geometry, and articulation sessions are health-checked and closed in
   `finally` blocks; the VLM client has no explicit close in this function (`generate.py`).
3. Scene understanding produces the table/object model and scene graph; geometry and optional
   articulation stages materialize assets; refinement updates placement.
   `SceneExporter.export()` is the durable generation boundary.
4. Editing imports `scene_export`, validates the scene and graph, asks the VLM for a typed
   edit plan, generates assets only for additions, applies layout changes, then exports over
   the same portable directory (`edit.py`).
5. The portable export is `scene_export/` with `scene.json`, `scene_config.json`,
   `scene_graph.json`, `mesh_assets/`, and `articulated_assets/`. It represents a scene, not a
   complete `EmbodiedEnv` deployment.
6. The general SimReady CLI is a separate ingest path. `simready_pipeline/cli/start.py` sets
   `PYOPENGL_PLATFORM=egl`, builds `JsonStore` and `ParserManager`, then calls
   `ingest_one_asset()`.
7. Ingest accepts mesh formats for canonicalization and `.urdf`/`.usd` for direct copy. Both
   branches hash and archive the source in a UUID asset directory. The direct-copy branch
   copies raw input and saves immediately; only the unified mesh branch normalizes the mesh,
   injects semantic/user extras, runs parser capabilities, and then saves asset/registry JSON
   (`ingest.py`).
8. Gradio copies only existing uploaded files into a UUID run root using basename-safe
   destinations. A session-token-owned subprocess performs conversion; its merged output is
   streamed through a queue; success requires exit code zero and a produced
   `asset_simready.glb` or `.obj` (`app_asset_engine.py`).

### Invariants to preserve

- Portable export node IDs must exactly equal scene object IDs. Both importer and exporter
  reject missing or extra graph nodes.
- Editable scene state is Y-up; portable runtime output is Z-up. Export converts world
  position/rotation only. `body_scale` stays on the object instead of being baked into the
  GLB.
- Every generated scene must contain a rigid table. Export gives articulated objects a runtime
  USDC plus a GLB edit proxy.
- Export validates and writes the new scene before removing stale assets during overwrite. Do
  not make deletion the first mutation.
- The Scene Engine's internal SimReady processor fixes table physics as kinematic and object
  physics as dynamic; it is distinct from the general asset-ingest pipeline.
- Parser capabilities should be idempotent. `ParserManager` records completed stages; new
  parsers must respect that resume model.
- Gradio remote access requires complete basic auth. Allowed paths are explicit; `.git` and
  dotenv files are blocked; the artifact root cannot be the repository or its ancestor
  (`app_env.py`).
- One `SessionProcessRegistry` slot belongs to one session. Starting a replacement terminates
  the previous process; reset affects only that session. App shutdown terminates process
  groups and discovered descendants (`app_processes.py`).
- Environment loading is non-destructive: values already present in the process environment win.

### Common failures and recommended change sites

| Symptom | Likely cause / change site |
|---|---|
| Image/edit command rejected before provider calls | CLI mode, input path, or suffix validation in `scene_engine/cli/start.py`. |
| Import says graph IDs differ from scene IDs | Repair the portable producer or edit plan; do not weaken `SceneExportImporter`/`SceneExporter` set equality. |
| Asset looks rotated or scaled twice | Coordinate conversion belongs in exporter/importer world pose handling; GLB geometry and `body_scale` have separate ownership. |
| Articulated asset works while generating but not after import | Check both exported USDC runtime asset and GLB edit proxy creation/copy. |
| Ingest reports a supported file but no canonical asset | Inspect `ingest.py` format branch and the selected parser capability; direct-copy and normalized meshes follow different paths. |
| Gradio conversion finishes without an artifact | Treat nonzero exit or absent `asset_simready.{glb,obj}` as failure; inspect the session's streamed subprocess output. |
| Remote UI exposes or cannot serve a path | Change `app_env.py` allowed/blocked-root policy, not component callbacks. |
| A new run kills an older run | Expected only within the same session token; otherwise inspect session-token propagation into `SessionProcessRegistry`. |
