# Generative simulation

Scene Engine turns images into editable scenes and portable exports. The
Asset Engine ingests existing assets; the asset package also owns a
separate articulated-generation service. Neither output is by itself a complete
runnable Gym task deployment.

## Entry points and resolution

Paths below are relative to `embodichain/gen_sim/` unless qualified.

| Request | Owner |
|---|---|
| Model service configuration, adapters and ownership | `services/`; [service guide](../../../embodichain/gen_sim/services/README.md) |
| External scene-provider lifecycle | `scene_providers.py`: request/job/cancellation contracts and selected entry-point loading |
| Public staged workflow | `workflow.py`: Task Plan loading, provider job operations, generated-scene validation, qualification and deterministic deployment preparation |
| Unified command dispatch | `embodichain/cli/gensim.py`: `gensim asset ingest`, `scene generate/edit/expand/preview`, `task plan`, `run`, `ui`; legacy top-level commands remain in `embodichain/cli/main.py` |
| Image to native execution | `_workflow.py`, `_run_config.py`, `_run_binding.py`, `_run_execution.py`; [command and run guide](../../../embodichain/gen_sim/README.md) |
| Generate/edit CLI modes | `scene_engine/cli/start.py` |
| Image → scene | `scene_engine/pipeline/generate.py`: `generate_scene_from_image()` |
| Edit portable scene | `scene_engine/pipeline/edit.py`: `edit_scene()` |
| Audited image/edit stages | `scene_engine/pipeline/api.py`: `analyze_image()`, `analyze_edit()`, and their materialization functions |
| Prepare source GLB geometry before publication | `scene_engine/asset_normalization.py`: `normalize_source_scene_assets()`; owned declarations and content-addressed output, no simulator |
| Portable scene contract | `scene_engine/pipeline/utils/scene_exporter.py`, `scene_importer.py` |
| Grouped layouts from an existing export | `scene_engine/cli/randomization.py`, `scene_engine/randomization/generation.py` |
| Baseline cache, saved runs and pose playback | `scene_engine/randomization/baseline.py`, `persistence.py`, `loading.py` |
| Whole-scene USD package | `scene_engine/pipeline/utils/scene_usd.py`: `build_scene_usd()`, direct preview loading; `usd_scene.py`: entity index |
| DexSim AssetRevision delivery | `scene_engine/pipeline/utils/scene_package.py`: SDK-owned scheme-one packaging and dependency closure; [output contract](../../../embodichain/gen_sim/README.md) |
| General asset ingest | `asset_engine/pipeline/ingest.py`: `ingest_one_asset()` |
| Articulated asset generation | `asset_engine/pipeline/articulation.py`, `asset_engine/clients/articulated_generation.py` |
| Web app launch/configuration | `gensim ui` replaces its process with `gradio_ui/gradio_app.py`; `app_env.py` retains auth and allowed-root rules |
| Session-owned subprocesses | `gradio_ui/app_processes.py`: `SessionProcessRegistry` |
| Frozen placement declaration → normative evidence binding | `task_engine/evidence.py` → [TaskSpec evidence boundary](../task-spec/task-spec.md#physical-evidence-boundary) |

Generate: validate input → VLM understanding → geometry/articulation generation
→ placement refinement → export. Model capabilities resolve lazily through one run-scoped `GenSimServices`.
`gensim run` additionally composes a trusted embodiment and verified execution
policy, binds generated native IDs/resources, preflights a configured Task
Program and executes it through the ordinary Gym demo owner. Its automatic
placement uses measured rigid planar geometry. Unsupported/ambiguous references
reject; native execution success and normative TaskSpec certification remain
separate. Failures retain stage artifacts and execution reports. The source graph
stays unbound; native deployment and bound graph have separate run-owned paths.
Normal wheel installs discover external profiles through
`EMBODICHAIN_MODEL_SERVICES_CONFIG`, then the XDG/home user configuration, before
bundled JSON and legacy rules. Python configuration overrides discovery. Selected
external files fail explicitly when invalid; no installed file needs editing.
`EMBODICHAIN_ENV_FILE` independently supplies referenced variables.
Bound adapter IDs resolve shipped factories first, then only an installed
`embodichain.gensim_service_adapters` entry point with the selected ID; unrelated
plugins are not imported. `gensim services validate` resolves profiles and
constructs adapters without provider requests, while `services doctor` also
initializes selected roles and runs their available health checks. The versioned
wire protocol remains model, port and deployment-framework independent.

Full scene providers are separate from model-service roles. The selected
`embodichain.gensim_scene_providers` adapter owns its external job transport and
materialization, while GenSim owns task requirements, qualification and deployment
validity. Task plans, generated scenes, qualification reports and prepared
deployments use separate immutable manifests. Expected scene incompatibility is a
structured qualification result; artifact corruption and transport failure remain
errors. These operations reserve the durable identity and feedback boundary needed
by a later bounded agent loop, but do not implement an agent runtime, MCP transport
or automatic retry loop.

Owned transports close when the run exits; explicit legacy clients and supplied
service runs remain caller-owned. Chat requests own their stateless transport.
Edit: import export → validate graph/typed edit plan → generate additions
→ change layout → overwrite export. Combined image/edit mode generates first.

Articulated USDC admission allows absent `gen_sim:closedPosition` metadata;
present declarations must identify a finite joint-limit endpoint. Missing
endpoint semantics remain unknown. Operation-specific target qualification
belongs to the [source control bindings](../scene-services/scene-services.md).

Read [pipeline details](pipeline-details.md) for stage contracts, parser resume
behavior, Gradio artifact ownership and focused failure diagnosis.
Use the [scripted reproduction](../../../examples/gensim/scripted_pick_place/README.md)
for a provider-free input fixture, native execution receipts, offline graph
display and explicitly scoped wall-clock measurements. Its scene is procedural;
it does not qualify image reconstruction or a successful expert trajectory.

The staged API persists a versioned image or edit blueprint before generating
assets. Image materialization verifies the persisted blueprint and source/mask
hashes before generation; changed workspaces require a new analysis. It copies
the supplied scene/graph and closes only clients it creates. Move/delete-only
edit materialization uses no generation clients or VLM configuration. Pipeline
exports load lazily so utility imports do not require generation dependencies.
Blueprint objects are in-process values with persisted audit documents; this API
does not load them as a complete task deployment. Whole-scene rotations also
rotate planar graph relations and table regions; directional graphs require
quarter turns because their vocabulary is axis-aligned.
The Asset Engine uses the `asset-engine` CLI and `asset_engine` Python
namespace. Its [user guide](../../../docs/source/features/generative_sim/asset_engine.md)
owns input selection, processing settings, artifacts and simulation handoff.
Instruction/image semantic planning belongs to
[Task Engine](../task-engine/task-engine.md); its unbound graph output retains
pending runtime bindings rather than creating a scene or executable deployment.

## Durable scene boundary

The `scene_export/` directory contains `scene.json`, `scene_config.json`,
`scene_graph.json`, `mesh_assets/` and `articulated_assets/`.
The GenSim execution handoff bakes Y-up GLB scale and axis conversion into
run-owned Z-up OBJ copies, with isolated texture/material sidecars. Native vertex
coordinates carry the conversion; GLB scene-node transforms are insufficient
across graphics/collision loaders. The editable source export stays unchanged.
Generation, editing and staged materialization also produce a `scene_usd/` delivery package
with a `scene.usda` carrying generation provenance metadata and a
relocatable `scene.usdz` single-file package, plus `scene_package.zip` using
DexSim's AssetRevision `2.0-draft` contract. The SDK owns canonical `/Asset`
packaging, schema validation, file digests and provenance; it currently exposes
processed-USD export through a private entry isolated in `scene_package.py`.
Packaging leaves classification and unmeasured quality facts unknown and grants
no runtime capability or certification. Export failures propagate; stale delivery
files are invalidated when an editable export is replaced. The Lab-owned
`preview-scene --usd-file` loads standard USD through `SimulationManager.add_usd`
without interpreting GenSim identity or joint-state tags. Manifest and packaged
native-asset preview adapters have been removed. Build USD only after the complete simulation scene is
prepared; validate UIDs and keep every packaged asset path inside the output
root.

glTF image-origin conversion is baked into the copied USD textures. Mesh UVs
remain authored as exported, and materials use ordinary USD texture nodes so
loading does not require development-only DexSim texture metadata or UV-transform
support. The packaged native glTF textures retain their original pixels.

- Scene object IDs and graph node IDs must be equal sets on import and export.
- Editable scene state is Y-up; portable runtime output is Z-up. Convert world
  pose at this boundary, and preserve `body_scale` separately from mesh geometry.
- Articulation export measures the native root collision geometry against its
  support surface. `init_pos` belongs to the runtime root; `proxy_init_pos`
  preserves the editable GLB placement independently.
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

Live scene generation and execution share a process. The live example keeps
`hybrid` throughout because DexSim's renderer selection is process-wide and
closing a temporary World does not reset it.

Blender operations use `gen_sim/_blender_runtime.py` to launch the standalone
worker. `EMBODICHAIN_BLENDER_EXECUTABLE` selects the Blender application when
Eevee needs application graphics initialization; otherwise
`EMBODICHAIN_BLENDER_PYTHON` selects the isolated interpreter. These selectors
follow shared-dotenv/process precedence. The worker imports
only Blender operations and their direct dependencies; it must not import the
business framework, Torch or DexSim. Input/output files cross this process boundary;
Blender objects and business-environment Python/native-library paths do not.
`docker/` owns the py312 public runtime, py311/py312 internal profiles and the
separate Python 3.11/NumPy 1.26 Blender environment. See
[container setup](../../../docker/README.md) for building and qualification.

The parser interface requires idempotent capabilities, but the current CLI
allocates a new UUID and reruns the selected sequence on every invocation.
`status.parsed` is completion metadata, not per-stage caching or validation.
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
| Ingest formats and metadata | `tests/gen_sim/asset_engine/` |
| UI roots/auth/session workflow | `tests/gen_sim/gradio_ui/` |
| DexSim package schema, bytes, dependency closure and namespace preservation | `tests/gen_sim/scene_engine/test_scene_package.py` |
| Unified/legacy CLI grammar and lightweight help | `tests/cli/test_gensim.py`, `tests/test_main.py` |
| Cross-engine binding, geometry handoff and execution outcomes | `tests/gen_sim/test_workflow.py` |

Select provider-backed or simulator conversion tests only when that runtime
boundary changes; source/unit checks do not establish remote service quality.

Placement declaration projection for new bundles uses
`project_configured_placement_evidence_binding` with the public configured owner's
resolved condition and compiled occurrence descriptors. The final builder must
not load or emit `constraints.json`. The original `project_placement_evidence_binding`
only validates the frozen B0 constraints payload for source-comparison fixtures;
it delegates the semantic projection and is not a second runtime policy owner.

Task Engine's `project_condition_evidence_binding` projects frozen role/lineage,
explicit compiled occurrences and already-resolved physical criteria into the
TaskSpec state-evidence contract. It reads no physical state and chooses no
acceptance thresholds; the Lab observation owner supplies facts.
