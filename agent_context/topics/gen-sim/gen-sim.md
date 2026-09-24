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
- Articulation export measures collision meshes relative to the native root
  rigid body, respecting USD up-axis, scale and deployment rotation, and sets
  root Z to leave 1 mm above the support. `proxy_init_pos` retains the independent
  GLB placement; import/edit must use it (legacy exports fall back to `init_pos`).
  An authored USD root translation is not a runtime rigid-body origin rebase.
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

Prepared scenes use intrinsic `XYZ` degree rotations. Bundle scene payloads
serialize explicit `init_local_pose` matrices so generic configuration decoders
cannot reinterpret those angles as extrinsic `xyz`; keep this boundary distinct
from the Task Program's `quaternion_xyzw` serialization. E6 runtime binding,
geometry bounds, rail direction and handle clearance use that same matrix;
never reinterpret a decoder's derived extrinsic angles as source `XYZ` angles.
Explicit proxy fitting updates an existing matrix together with its pose fields.
Generated deployments retain the measured settled layout instead of teleporting
objects back to configured X/Y after settling.
Task motion policy samples even single-EEF-target transports in Cartesian space;
joint targets retain their joint-space behavior. E2 alignment preserves the
acquired orientation during the staging lift, then turns the object aloft.
Only yaw-free alignment calls enable alternative final headings; exact-pose
transport retains its orientation contract. All candidates use the same motion
and velocity checks.
E2 release preserves the observed aligned heading instead of adding another
fixed yaw during descent; an upright instruction does not require that rotation.
Generated E1/E2-only deployments cap the used Robotiq gripper `max_effort` at 0.5
when every picked rigid body has an explicitly declared mass of at most 10 g.
The full closing range is retained; smaller existing effort limits are not
increased. Unknown/heavier loads and other recipe families retain their declared
controls. Imported mimic behavior and shared robot defaults are unchanged.
E1 placement on another object retains its original position tolerance and adds
the existing stack stability checks for vertical orientation, support gap and
reference stability. Source pose matrices take precedence over Euler fields.
After UID grounding, independent E1 placement, E2 upright, and E5
lift-and-return `count`/`all` sets lower to ordered single-object steps;
line layout, E5 terminal hold, other task routes, and a downstream
`step_result` consumer still reject implicit expansion.
For an unbound existing Gym scene, Task Engine automatically renders the
current GLB assets into UID-labeled overviews, segmentation masks and crops,
then retries grounding with the same LLM transport configured for text binding
in `gen_sim/.env` or the process environment;
`--reference-image` is not this binding input. A deterministic tenth of already
bound scenes receives an audit-only visual spot check. Evidence records the
scene revision and image hashes; the VLM may propose only visible inventory
UIDs, while native semantic compatibility remains authoritative. Articulation
proxy GLBs do not represent live joint state, so E6-E9 binding does not use
this visual path. Without usable Task Engine LLM configuration, text-only
binding and its existing conflict behavior remain unchanged.
E4 may defer unnamed transfer/receiver arms as `auto`; the planner binds the
source from the held state or nearest scene side and the other arm as receiver,
while preserving explicitly named arms. Multiple children
placed inside the same measured container reserve each prior landing footprint
and margin; insufficient space or an unmeasured shared floor fails preflight
instead of assigning overlapping destinations. These are planning contracts,
not physical placement acceptance.
Generated non-drawer held-object transports declare a measured attachment postcondition.
For deployments without drawer routes, GenSim installs wrappers around MoveHeldObject and Pour
descriptors after shared engine validation; shared Lab action classes and
options remain unchanged. The transport wrapper stages above the target and
returns the input attachment as its expected effect, so the existing composite
monitor verifies retention before observed-transform reconciliation. This is
terminal verification, not in-flight slip prevention. Regenerate older bundles
to include the wrapper registration and revised lowerer fingerprint.
E3 now uses the vessel's geometric upright axis only as an upright reference,
not as its rotation axis. The selected grasp is checked at runtime against the
actual pad line before the tilt arc is generated;
The GenSim Pour wrapper reads the two native finger-pad link poses (not TCP-X)
and derives the projected pad-line as its rotation axis. Its visible object
tilt is the cross product with upright, hence perpendicular to the actual jaw
line. It samples the full tilt-and-return arc through the shared motion
generator.
No spout annotation is required for this tilt-and-restore contract. Receiving
vessel position is late-bound while the source's upright heading is preserved
independently of receiver yaw. Rim height, vessel radius and the declared finger
envelope bound the clearance; E3 transports stage above the target before
descending. E3 Place uses 0.12 m retreat/approach height without relaxing release
acceptance. This is geometric/kinematic screening, not global collision planning
or proof of fluid transfer; ambiguous upright geometry still needs qualification.
For level containers with a dominant planar floor, inside placement selects an
arm-side point from the actual floor triangles, eroded by the projected child
footprint and a margin. Holes are retained. Unsupported floor geometry retains
the existing target policy; an oversized footprint on a measured floor fails
explicitly. This geometric preference still requires downstream IK and physical
release/containment qualification.
Opt-in grasp fitting reserves the declared/default robot and object contact
envelopes before selecting a uniform scale. The adaptation report records the
effective per-jaw clearance; the 0.25 minimum scale and source-preservation
boundary remain unchanged.

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
Generated E6 bundles apply the calibrated 1 kg mass only to discovered
prismatic moving links. Fixed cabinet/root links retain source-authored mass,
while contact offsets and materials remain asset-owned; this preserves the
Slide grasp response across the legacy loader and Spawn without restoring the
old global articulation-physics override.

E6-open / E1-inside / E6-close composition is owned by
`_task_program/drawer_binding.py`, `drawer_geometry.py`, `drawer_runtime.py`,
and `drawer_curobo.py`.
When drawer routes exist, the adapter creates `DrawerPlacementEngine` and does
not install the ordinary GenSim transport/pour wrappers. The local
`drawer_transport` lowerer keeps the v36 effectless transport contract; its
motion command and later Place/E6 acceptance are not a new transport retention
certificate. Engine selection and lowerer/monitor selection must change together.
Adapter contract v4 rejects older bundles; regenerate after this merge repair.
Inside targets share E6's asset-qualified part/link identity, never a duplicate
rigid root. Geometry supplies a floor/center hint, not an interior certificate.
The GenSim adapter refreshes the placement target from actual link/object poses
on every Place plan/replan while preserving the declared scene manifest.
No cavity-fit or separate FCL rejection gate is installed. Drawer payload pickup
does not screen future placement reachability. Generated bundles insert an
explicit `simulation.move_held_object` before Place: cuRobo plans that free-space
transport, while E6, pickup and final lowering/release retain their existing
planner. The task-owned transport service snapshots live rigid objects and all
articulation links for each plan/replan. Articulated collision components become
separate link-local boxes, preserving openings; the held object is removed only
from the planner world and represented by a conservative sphere cover on a
separate tool-attached collision link. This never attaches objects in physics.
The planner and its payload attachment are released after each planning scope.
This scoped snapshot does not implement concurrent moving-obstacle prediction
or add self-collision checking to the shared cuRobo backend.
Place is planned from the acquired grasp. Composite bundles use the extended
motion sample budget rather than raising joint velocity limits. Ordinary command
planning, effect verification and E6 joint-state acceptance remain.
Execution completion does not prove an object is contained.
Evaluate actual placement/closing from the recorded physical outcome.

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
