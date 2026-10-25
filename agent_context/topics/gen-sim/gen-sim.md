# Generative simulation

Scene Engine turns images into editable scenes and portable exports; the
general SimReady pipeline ingests individual assets. Neither output is by
itself a complete runnable Gym task deployment.

## Entry points and resolution

Paths below are relative to `embodichain/gen_sim/` unless qualified.

| Request | Owner |
|---|---|
| Unified command dispatch | `embodichain/cli/main.py`: `scene-engine`, `scene-randomization`, `preview-scene`, `simready` |
| Generate/edit CLI modes | `scene_engine/cli/start.py` |
| Image → scene | `scene_engine/pipeline/generate.py`: `generate_scene_from_image()` |
| Edit portable scene | `scene_engine/pipeline/edit.py`: `edit_scene()` |
| Portable scene contract | `scene_engine/pipeline/utils/scene_exporter.py`, `scene_importer.py` |
| Grouped layouts from an existing export | `scene_engine/cli/randomization.py`, `scene_engine/randomization/generation.py` |
| Baseline cache, saved runs and pose playback | `scene_engine/randomization/baseline.py`, `persistence.py`, `loading.py` |
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

Grouped randomization consumes one rigid-GLB table export and its explicit
support graph. The default SLSQP path retains the shared procedural solver;
the optional batched backend uses CPU/CUDA counter-addressed rejection sampling
from `compute.geometry._sampling`. Rejection has a different distribution and
acceptance rate from SLSQP repair. Batch boundaries do not change logical
proposal ordinals or the consumed attempt budget. CUDA selection is explicit.

The adapter owns group/member transforms, source/configuration snapshots and
canonical CPU float64 pose expansion for persisted identities. Version-3 policies
add explicit non-root XY/yaw targets and optional fixed group roots; empty targets
retain the original group path. `_intra.py` composes numeric postorder subtree
proposals with dynamic outer AABBs. UID-addressed streams retain the logical
prefix across batch sizes, and each failed layout consumes one whole attempt.
Hierarchical generator v2 samples outer roots from free center rectangles rather
than retrying collisions; version-1 hierarchical runs remain replayable.
Free-region temporaries use obstacle chunks and an adaptive batch ceiling;
resolved policies record the effective batch size without changing the prefix.

`_intra_geometry.py` adapts scaled assets and graph topology to the shared
parent-local support-grid strategy. Child bottom geometry guides plane selection;
ambiguous layers require evidence-checked overrides. Immutable content-addressed
caches include source/extractor/settings identities and leave the original
baseline unchanged. Cache v2 records the upper height deviation of the selected
layer so shallow bumps do not consume the support gap. Preserve earlier cache
height semantics when replaying old runs. Do mesh loading, plane extraction and polygon rasterization
only during preparation; do not call scene construction or SLSQP per candidate.

Batch candidates and loaded variants own lazy numeric sequences; consuming an
item constructs typed poses/configurations while saving validates numeric evidence.
Runtime collision settings are preserved. V3 layouts store root XY translations,
selected parent-relative full poses and cache references, then validate all source,
geometry, accounting and identities before exposing variants. Playback reconstructs
saved results without sampling or stepping physics. Existing v1 individual and
v2 packed group runs remain readable; complete resolved policies can be reused.

JSON remains the default storage; packed writes one atomic candidate container.
Both revalidate source/baseline and the exact cache snapshot before publishing
the completion manifest last. Concurrent cache writers use distinct staging files. Geometric
support/AABB acceptance does not establish physical or task success. See
[`scripts/tutorials/scene/README.md`](../../../scripts/tutorials/scene/README.md)
for policy settings, demo and end-to-end CPU/CUDA benchmarks and the
[public API supplement](../../../docs/source/api_reference/public_api.rst).

`randomization/adapter.py` projects rigid exports into the shared scene-layout
input. `generate_scene_from_image` and `edit_scene` accept an optional post-export
job; the CLI's combined generation/edit flow triggers it once after the final
export. New runs use source-independent v4 storage and retain the original
export for preview. `convert_layouts` preserves validated legacy poses and IDs
without sampling. File/model coordinate adaptation remains owned here.

## Runtime and extension boundaries

Procedural precision assembly has a separate owner under
`embodichain/gen_sim/assembly/`; see [assembly generation and precision](assembly.md)
for pinned mjfit adaptation, portable manifests and native/USD construction.
It does not use Scene Engine inference or the general SimReady decomposition
pipeline, because approximate recooking can consume an assembly's clearance.

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
| Grouped layout generation, provenance and playback | `tests/gen_sim/scene_engine/test_randomization_*.py` |
| Edit plans and graph | `tests/gen_sim/scene_engine/test_scene_edit.py`, `test_scene_edit_plan.py`, `test_scene_graph.py` |
| Ingest formats and metadata | `tests/gen_sim/simready_pipeline/` |
| UI roots/auth/session workflow | `tests/gen_sim/gradio_ui/` |

Select provider-backed or simulator conversion tests only when that runtime
boundary changes; source/unit checks do not establish remote service quality.
