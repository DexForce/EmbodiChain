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
layout is unchanged. Execution admits E1-E6; E7-E9 remain rejected before graph
generation and bundle publication. E6 uses the explicit registered
Slide/withdraw/Park recipe in `_task_program/articulation_binding.py` and
`articulation_slide.py`. Its first supported binding is one fixed-base,
single-prismatic, self-contained metre-authored USD with uniform scale and an
unambiguous handle mesh, in one simulation environment. Asset hashes, joint
ownership, and declared limits are checked again at runtime. Public Slide owns
planning; public Task Program and Gym own execution. Joint-target retention is
checked after every recipe call; this is not in-flight contact qualification.

`task_engine/scene/articulation_geometry.py` measures Z-up, metre-authored USD
collision meshes in the native base-link frame, not the default prim's world
frame that runtime reset replaces. Final inspection now measures articulated
geometry. E6 rejects an unmeasured tabletop, more than 2 mm initial table
penetration, or a buried handle before bundle publication and at runtime binding.
An explicit proxy-fit repair returns a new configuration using uniform scale and
a 1 mm placement clearance, levelling only bases within one degree while preserving
yaw; normal loading never silently applies that repair.
Publish repaired exports separately with asset provenance. GenSim assembly and
E6 binding check fresh native joint limits and reapply the same values only when the
public pre-scale cache is stale; it does not enlarge the physical travel range.

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
