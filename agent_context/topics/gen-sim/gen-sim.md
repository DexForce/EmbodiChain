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

- Scene object IDs and graph node IDs must be equal sets on import and export.
- Editable scene state is Y-up; portable runtime output is Z-up. Convert world
  pose at this boundary, and preserve `body_scale` separately from mesh geometry.
- A generated scene includes a rigid table. Articulated assets need both a
  runtime USDC and a GLB proxy for editing.
- Overwrite validates/writes new scene state before deleting stale assets.
- The internal Scene Engine SimReady processor and general asset-ingest CLI
  are separate flows; change the selected owner rather than assuming one pipeline.

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

## Task Engine semantics and execution scope

`task_engine/ontology.py` owns scene-independent task meaning and capability
requirements; `task_engine/interpretation.py` owns strict intent validation and
model guidance. E6 is a prismatic part opened or closed (`target_state=open` or
`closed`, `slideable`); E7 is a revolute door opened (`target_state=open`,
`openable`). Closing a drawer remains E6 regardless of the selected arm.
Closing a hinged door is not represented by either contract.

`task_engine/agent.py` derives SceneRequest requirements from that ontology;
`task_engine/orchestration/scene_adapter.py` checks declared capabilities without
aliasing legacy `pullable`/`pushable` labels. Empty capability metadata remains
unknown, not proof of native joint type. Native joint qualification is deferred
until execution integration.

Persisted candidates must match the current intent and exactly derived scene
request. Regenerate legacy E7 closing candidates and E6/E7 candidates with old
capability declarations; do not silently relabel them. The serialized field
layout is unchanged. E6-E9 execution is still rejected by Task Engine's
`workflow.py`, `semantic_planner.py`, and `task_program_bundle.py`; corrected
interpretation does not enable a runtime route or establish physical qualification.

## Focused validation

| Change | Tests |
|---|---|
| Portable scene/pose/overwrite contract | `tests/gen_sim/scene_engine/test_scene_core_and_export.py` |
| Edit plans and graph | `tests/gen_sim/scene_engine/test_scene_edit.py`, `test_scene_edit_plan.py`, `test_scene_graph.py` |
| Ingest formats and metadata | `tests/gen_sim/simready_pipeline/` |
| UI roots/auth/session workflow | `tests/gen_sim/gradio_ui/` |
| Task intent and scene capability contracts | `tests/gen_sim/task_engine/test_agent.py`, `test_interpretation.py`, `orchestration/test_scene_adapter.py` |

Select provider-backed or simulator conversion tests only when that runtime
boundary changes; source/unit checks do not establish remote service quality.
