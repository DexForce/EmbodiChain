# Generative simulation

Scene Engine turns images into editable scenes and portable exports; the
general SimReady pipeline ingests individual assets. Neither output is by
itself a complete runnable Gym task deployment.

## Entry points and resolution

Paths below are relative to `embodichain/gen_sim/` unless qualified.

| Request | Owner |
|---|---|
| Unified command dispatch | `embodichain/cli/main.py`: `scene-engine`, `scene-preview`, `simready` |
| Generate/edit CLI modes | `scene_engine/cli/start.py` |
| Image → scene | `scene_engine/pipeline/generate.py`: `generate_scene_from_image()` |
| Edit portable scene | `scene_engine/pipeline/edit.py`: `edit_scene()` |
| Portable scene contract | `scene_engine/pipeline/utils/scene_exporter.py`, `scene_importer.py` |
| Whole-scene USD package | `scene_engine/pipeline/utils/scene_usd.py`: `build_scene_usd()`, direct preview loading; `usd_scene.py`: entity index |
| General asset ingest | `simready_pipeline/pipeline/ingest.py`: `ingest_one_asset()` |
| Web app configuration | `gradio_ui/gradio_app.py`, `app_env.py` |
| Session-owned subprocesses | `gradio_ui/app_processes.py`: `SessionProcessRegistry` |

Generate: validate input → VLM understanding → geometry/articulation generation
→ placement refinement → export. Segmentation, geometry and articulation
clients are closed in `finally` blocks; VLM lifetime is separate.
Edit: import export → validate graph/typed edit plan → generate additions
→ change layout → overwrite export. Combined image/edit mode generates first.

Read [pipeline details](pipeline-details.md) for stage contracts, parser resume
behavior, Gradio artifact ownership and focused failure diagnosis.

## Durable scene boundary

The `scene_export/` directory contains `scene.json`, `scene_config.json`,
`scene_graph.json`, `mesh_assets/` and `articulated_assets/`.
Generation and editing also produce an optional `scene_usd/` delivery package
with a schema-v2 `scene.usda` carrying EmbodiChain entity metadata and a
relocatable `scene.usdz` single-file package. DexSim preview loads schema-v2
stages directly through `preview-scene --usd-file`; the manifest and native GLTF/USDC assets remain legacy
compatibility data. Build it only after the complete simulation scene is
prepared; validate UIDs and keep every packaged asset path inside the output
root.

glTF image-origin conversion is baked into the copied USD textures. Mesh UVs
remain authored as exported, and materials use ordinary USD texture nodes so
loading does not require development-only DexSim texture metadata or UV-transform
support. The packaged native glTF textures retain their original pixels.

- Scene object IDs and graph node IDs must be equal sets on import and export.
- Editable scene state is Y-up; portable runtime output is Z-up. Convert world
  pose at this boundary, and preserve `body_scale` separately from mesh geometry.
- A generated scene includes a rigid table. Articulated assets need both a
  runtime USDC and a GLB proxy for editing.
- Overwrite validates/writes new scene state before deleting stale assets.
- The internal Scene Engine SimReady processor and general asset-ingest CLI
  are separate flows; change the selected owner rather than assuming one pipeline.

## Shared scene services

Geometric support detection, group translation/alignment and AABB separation
use private CPU strategies under `sim.scene`; tabletop and parent-surface
optimization adapt `sim.scene.layout`. These strategies reuse Torch geometry
kernels from `compute.geometry`, while Shapely, Trimesh and SLSQP stay on CPU. Existing adapters keep Y-up conversion and debug output;
`SceneLayoutConstructor` retains hierarchy traversal and edit-specific pose
propagation. `GravitySettler` owns its temporary simulator and fixed-step default;
optional convergence checks use shared scene validation. See
[scene services](../scene-services/scene-services.md) for frames, supported
geometry and physical-state boundaries.

GenSim owns scene understanding, asset generation, VLM orientation, its editable
scene graph and file formats. General services consume explicit UID/geometry
inputs rather than `SceneObject`, image masks or generation output directories.

## Runtime and extension boundaries

Keep parser capabilities idempotent so recorded stages support resume.
Provider configuration belongs to its loader/client, while coordinate and graph
contracts belong to importer/exporter. Task registration and physical/semantic
composition belong to [env-framework](../env-framework/env-framework.md).

Gradio uses explicit allowed roots and per-session process ownership. A
replacement run terminates the previous process for that session. Remote
access requires complete basic auth; repository/dotenv path restrictions belong
to `app_env.py`. Existing process environment values take precedence over loaded
environment files.

## Focused validation

| Change | Tests |
|---|---|
| Portable scene/pose/overwrite contract | `tests/gen_sim/scene_engine/test_scene_core_and_export.py` |
| Edit plans and graph | `tests/gen_sim/scene_engine/test_scene_edit.py`, `test_scene_edit_plan.py`, `test_scene_graph.py` |
| Ingest formats and metadata | `tests/gen_sim/simready_pipeline/` |
| UI roots/auth/session workflow | `tests/gen_sim/gradio_ui/` |

Select provider-backed or simulator conversion tests only when that runtime
boundary changes; source/unit checks do not establish remote service quality.
