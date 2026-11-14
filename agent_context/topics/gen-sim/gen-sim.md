# Generative simulation

GenSim owns image/text understanding, asset generation, editable scene graphs
and portable outputs. Generated scenes are not complete Gym deployments.

## Entry points and resolution

Paths are relative to `embodichain/gen_sim/` unless qualified.

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
| General asset ingest | `asset_engine/pipeline/ingest.py`: `ingest_one_asset()` |
| Asset orientation and scaling orchestration | `asset_engine/utils/_simready_pipeline.py`; `simready_utils.py` and `full_blender.py` retain geometry/render/export adapters and entry points |
| Articulated asset generation | `asset_engine/pipeline/articulation.py`, `asset_engine/clients/articulated_generation.py` |
| Web app launch/configuration | `gensim ui` replaces its process with `gradio_ui/gradio_app.py`; `app_env.py` retains auth and allowed-root rules |
| Shared dotenv syntax | `env.py`; environment and service readers retain key selection, missing-file and precedence policies |
| Session-owned subprocesses | `gradio_ui/app_processes.py`: `SessionProcessRegistry` |
| Frozen placement declaration → normative evidence binding | `task_engine/evidence.py` → [TaskSpec evidence boundary](../task-spec/task-spec.md#physical-evidence-boundary) |

Generation validates input, understands the scene, materializes geometry and
articulation, refines placement and exports. Editing imports an existing scene,
validates a typed plan, generates additions and overwrites the export. Combined
image/edit mode generates first. [Pipeline details](pipeline-details.md) own
stage, ingest, resume, UI and grouped-randomization contracts.

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

`gensim run` binds a generated scene to a trusted embodiment/execution policy,
preflights Task Program and uses ordinary Gym demo execution. Its source graph
stays unbound; bound graph/deployment have separate run-owned paths. Ambiguous
references reject and failures retain stage artifacts. Native success and
normative TaskSpec certification remain distinct. See the
[command guide](../../../embodichain/gen_sim/README.md) and provider-free
[scripted reproduction](../../../examples/gensim/scripted_pick_place/README.md).

One `GenSimServices` run lazily resolves selected model capabilities and closes
owned transports; injected clients/runs remain borrowed. Python configuration
overrides external-file discovery, then environment/user/bundled/legacy sources
follow their declared priority. Invalid selected files fail instead of falling
through. Only the selected installed adapter entry point loads. `services validate`
constructs without provider requests; `doctor` also initializes roles and health
checks. The [service guide](../../../embodichain/gen_sim/services/README.md) owns
configuration, adapter and wire contracts. Dotenv syntax is shared; caller
selection and process-environment precedence stay separate.

Staged analysis persists versioned blueprints before generation. Materialization
checks source/mask hashes, copies supplied scene/graph state and closes only
owned clients. Move/delete-only edits need no generation client. Blueprint audit
documents are not runnable deployments. Whole-scene rotation also rotates graph
relations/table regions; directional vocabularies require quarter turns.
[Task Engine](../task-engine/task-engine.md) owns unbound instruction planning.

## Durable scene boundary

- `scene_export/` contains scene/config/graph JSON and mesh/articulation assets;
  imported/exported graph node IDs must equal object IDs.
- Rigid exports carry conservatively measured uniform GLB color properties and
  content-bound provenance. Opacity follows the material's alpha mode; unsupported
  geometry, textures and mixed appearance leave the measurement unverified.
  Role checks verify measured proofs against the mesh the deployment consumes,
  including the legacy `shape.fpath` selection. Proxy GLB proofs cannot certify
  the appearance of packaged USD prims; a runtime-material measurement is required.
- Editable state is Y-up and runtime output Z-up. Preserve `body_scale` separately.
  Native execution bakes scale/axis conversion into run-owned OBJ vertices and
  isolated material sidecars; GLB node transforms alone are insufficient.
- Every generated scene contains a rigid table. Articulations retain runtime
  USDC plus edit-proxy GLB, with distinct native-root and proxy placement.
- Optional `scene_usd/` carries schema-v2 entity metadata; `scene.usdz` is its
  relocatable package. Preview loads the stage directly. Build after preparation,
  validate UIDs and keep packaged paths inside the output root.
- USD texture copies bake image-origin conversion while retaining authored UVs;
  native glTF textures keep original pixels. Use ordinary USD texture nodes.
- Overwrite validates/writes before stale deletion. Scene Engine's internal
  SimReady and general asset ingest remain separate flows.

## Shared scene services

GenSim adapters retain Y-up conversion, hierarchy traversal and edit propagation.
`sim.scene` owns shared support/group/layout/state operations and their CPU
strategies; `compute.geometry` owns pure tensor proposals. Shapely/Trimesh/SLSQP
remain CPU algorithms. `GravitySettler` owns its temporary simulator and optional
convergence checks. General services consume explicit UID/geometry, not GenSim
models, masks or output directories.

Grouped layouts retain explicit support graphs, source snapshots, logical
proposal identity and attempt accounting. Rejection sampling differs from SLSQP
repair; CUDA selection is explicit and batch boundaries cannot change prefixes.
See [pipeline details](pipeline-details.md#grouped-layouts) for hierarchy,
canonicalization and pose-only playback boundaries.

## Runtime and extension boundaries

Live generation/execution share a process and retain `hybrid`: renderer
selection is process-wide and closing a temporary World does not reset it.

`EMBODICHAIN_BLENDER_EXECUTABLE` selects the Blender application for Eevee
graphics initialization; otherwise `EMBODICHAIN_BLENDER_PYTHON` selects the
isolated interpreter. Both follow shared-dotenv/process precedence.
Blender operations run through `_blender_runtime.py` in the separately selected
worker environment. The worker loads operations/direct dependencies, without
Torch, DexSim or business-framework initialization. Files cross the boundary;
Blender objects and business native-library paths do not. Runtime profiles belong
to [docker](../../../docker/README.md).

Gradio owns allowed roots, auth, UUID artifacts and session-owned subprocesses.
Replacement runs terminate the prior process for that session. Remote access
requires complete basic auth; existing process env values retain priority.
Asset/source registry and task composition remain owned by their existing
subsystems. Frozen placement/control evidence normalizes through Task Program;
TaskSpec separately applies normative criteria.

## Focused validation

| Boundary | Coverage |
|---|---|
| Scene/graph/pose/overwrite and staged API | `tests/gen_sim/scene_engine/` |
| Ingest metadata and backend orchestration | `tests/gen_sim/asset_engine/` |
| Service discovery, parsing and lifetime | `tests/gen_sim/services/`, `tests/gen_sim/test_gen_sim_env.py` |
| UI roots, auth and process ownership | `tests/gen_sim/gradio_ui/` |
| CLI and execution handoff | `tests/cli/test_gensim.py`, `tests/test_main.py`, `tests/gen_sim/test_workflow.py` |

Choose provider/simulator tests for the changed boundary; unit checks alone do
not establish remote model quality or native task success.
