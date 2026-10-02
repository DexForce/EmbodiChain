# Generative simulation

Scene Engine turns images into editable scenes and portable exports; the
SimReady Asset Engine ingests and generates individual assets, including
articulated assets. Neither output is by itself a complete runnable Gym task
deployment.

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
| General asset ingest | `asset_engine/pipeline/ingest.py`: `ingest_one_asset()` |
| Articulated asset generation | `asset_engine/clients/articulated_generation.py`, `asset_engine/utils/articulated_usdc_utils.py` |
| Generated task semantics | `embodichain/task_spec/` and `task_engine/task_spec.py`: TaskTemplate generation and cache adapter |
| Web app configuration | `gradio_ui/gradio_app.py`, `app_env.py` |
| Session-owned subprocesses | `gradio_ui/app_processes.py`: `SessionProcessRegistry` |

Generate: validate input → VLM understanding → Asset Engine geometry/articulation
generation → placement refinement → export. Segmentation and Asset Engine
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
TaskSpec owns the reusable normative task identity and evidence contract across
these boundaries; its schemas, predicate rules, and cache behavior are owned by
`embodichain/task_spec/` and detailed in the
[GenSim/TaskSpec design](../../../docs/architecture/gen-sim-taskspec-design.md).
Scene output is still only a SceneInstance input to that protocol, not a
certified task witness.

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
Invocation-scoped Cartesian policy also covers single-EEF-target transports;
joint targets retain their joint-space behavior. E2 alignment preserves the
acquired orientation during the staging lift, then turns the object aloft.
E2 checks its final yaw-free alignment at release height before Place preserves
that heading. Alignment staging never descends below the current height before
the planner can choose the final orientation.
Only yaw-free alignment calls enable alternative final headings; exact-pose
transport retains its orientation contract. All candidates use the same motion
and velocity checks.
E5 Robotiq grasp clearance is isolated in
`task_engine/_task_program/coordinated_grasp.py`: only the dual-grasp service
uses URDF collision surfaces and deployment open/grasp commands in TCP space.
Antipodal points seed a bounded finite-pad gap, centering and insertion search;
their separation is not assumed to be the actual contact gap. Watertight target
triangles preserve cavities that coarse convex unions can fill. Both inward pad
faces must reach contact, with the open-to-contact sweep clear of other hand
parts. Non-watertight targets fail explicitly instead of receiving guessed signs.
Near-contact inward-pad samples also require outward target normals within
60 degrees of the corresponding pad-facing direction. This excludes tangential
or back-facing proximity without treating it as force closure. Normal queries
reuse the target triangle BVH and run only on near-contact pad samples; rejected
endpoints do not enter the closing sweep. Two-sided occupancy probes orient
each queried normal using the signed-distance solid convention, not global
mesh volume: independently reversed components and nested cavities retain
their own inside/outside boundary. Ambiguous probes cannot certify contact.
Regenerate E5 bundles after changes
to this versioned geometry policy.
Single-arm generator methods and
their factories remain unchanged even in mixed E1/E5 programs; do not tune their
finger envelope or opening margin at program scope. This target-mesh screening
does not certify continuous approach clearance, contact dynamics, or attachment.
`coordinated_motion.py` composes the shared planning helpers for E5 continuation:
two verified grasps of the same object are retained through motion, and hands
open only at release. Missing or conflicting held state cannot fall back to
re-picking. Consecutive E5 relative targets accumulate for terminal checks;
release retreat exceeds the unchanged separation threshold with tracking margin.
E5 `terminal_behavior=place` with `relation=on` retains the bound rigid support
instead of converting it into an initial-pose direction. Bundle generation
reuses E1 relative-placement height geometry; the coordinated lowerer targets
the observed support position while preserving the carried object's observed
orientation. Release clearance is excluded from the resting-position target,
which is checked relative to the support again after cleanup. This uses the
existing E5 position/stability acceptance, not a new contact certificate; mesh
envelope limitations remain. Motion budgets are declared per generated segment
in `constraints.json.motion_samples`, then bound to compiled invocation IDs by
`invocation_policy.py`. The base execution policy is no longer raised by scanning
the whole program. E2/E4/E6, drawer-placement and on-support E5 recipes retain
their staged-motion budgets without changing unrelated calls. The GenSim engine
selects immutable request policies and observes the runner's initial plan;
planning, execution, recovery and state effects remain in the shared engine.
Generated PytorchSolver configurations retain GenSim's 5 mm position tolerance
explicitly, without changing the shared solver's 0.5 mm default. Existing
explicit solver tolerances take precedence; rotation tolerances are unchanged.
Larger caller budgets, control dt and
velocity limits are preserved. Missing/stale bindings require bundle regeneration.
Cartesian sampling is declared alongside budgets in `cartesian_calls`, bound
to compiled invocation IDs, and enabled only within an exception-safe planning
scope. Appending a HandOver, clearance or drawer recipe no longer switches the
whole program's generator. Hardware, initialization, remaining option selectors
and random-state scopes still require separate qualification.
Normal bundle execution writes `planning_probe.json` from the actual first
invocation via `_task_program/planning_probe.py`, without separately planning it
again. Full-robot post-resampling velocity validation still gates that plan
before dispatch; diagnostics distinguish execution evidence from the independent
`--plan-probe-only` mode. The run-local observer matches the exact invocation ID,
restores its scope on exit, and retains no executable plan. Genuine runtime
replans continue to use fresh observations; this is not a cross-run plan cache.
For non-drawer Place, the current `GenSimPlace` planning scope selects the same
Cartesian sampling policy in both checked and approach motion generators. Its
planning and recovery do not depend on unrelated E2/HandOver calls elsewhere
in the program. Outside this scope, each generator's existing policy remains
unchanged; drawer and cuRobo routes retain their own policies. Ordinary Place
paths that previously used joint interpolation may therefore change; regenerate
bundles after the motion-policy revision. For a single environment only,
exhausted Cartesian IK planning may retry through the shared motion generator
with nearby joint seeds and a joint-limit-margin preference. Successful
Cartesian plans do not enter local IK recovery;
the fallback preserves Cartesian samples, control timing and random state,
and rechecks full-robot velocity after Place resampling. This is a bounded
kinematic recovery, not collision or physical-placement qualification.
E6 recovery is bound to exact registered Slide/withdraw invocation IDs, not
all calls of those underlying skills. Failed single-environment CPU
`PytorchSolver` / `ik_interp` pull-and-release plans can screen up to four
terminal configurations, solve the contact path backwards, then connect a
free-space joint approach. Slide recovery is capped at 1.5 times the request
and 390 command frames; larger caller budgets are not shortened. Failed E6
withdrawal can try local IK without changing its targets or frame budget.
The shared skills still rebuild arm/hand commands and the shared executor
still owns execution. Both routes retain tolerances, restore RNG/planning
scope, validate final full-robot velocity, and require fresh discrete mesh
clearance for the changed free-space motion. This is not continuous collision
or zero non-target-contact certification. Successful plans, other semantic
calls, collision-planner routes and shared Lab defaults remain unchanged.
Recovery needs the GenSim `yourdfpy`/`python-fcl` dependencies; unavailable or
unsupported geometry fails closed. Regenerate E6 bundles for the recovery
fingerprint revision; non-E6 fingerprints are unchanged.
E2 release preserves the observed aligned heading instead of adding another
fixed yaw during descent; an upright instruction does not require that rotation.
Generated E1/E2 deployments, including light E5 compositions, cap only grippers
used by single-arm picks at `max_effort: 0.5` when every picked rigid body and
E5 carrier has an explicitly declared mass of at most 10 g.
The full closing range is retained; smaller existing effort limits are not
increased. Unknown/heavier loads, E5-only programs and other recipe families retain their declared
controls. Imported mimic behavior and shared robot defaults are unchanged.
Ordinary E1 placement on another object retains its position tolerance and uses
`supported_placement`: scaled local mesh vertices are cached per environment,
then transformed by observed poses for vertical gap and projected support-envelope
checks. It does not impose initial roll or either object's initial up direction.
Both objects must still remain stable for the declared window. This geometric
envelope is not contact evidence for arbitrary concave or tilted supports.
Relative side placement accepts rigid and articulation roots as spatial anchors.
Articulated anchors keep native USD paths in the planner view; geometry is read
in the runtime base-link frame with configured scale, never from a preview proxy.
Binding preserves `SceneArticulationRef` and the shared registry observes its
live root pose. GenSim stability acceptance reads the same root/displacement and
retains the position tolerance, replacing only the rigid-only duplicate validator.
This uses the authored articulation geometry envelope, not a changing-joint
collision envelope; a reference also actuated by Slide in the same program is
rejected rather than reusing stale extents. Articulated `on/above/inside` support still requires qualified
link geometry; spatial-reference support does not authorize joint manipulation.
Explicit upright, oriented hold and stack constraints retain their axis checks;
legacy presets are unchanged, so regenerate bundles to select the new policy.
Source pose matrices take precedence over Euler fields.
After UID grounding, independent E1 placement, E2 upright, and E5
lift-and-return `count`/`all` sets lower to ordered single-object steps;
line layout, E5 terminal hold, other task routes, and a downstream
`step_result` consumer still reject implicit expansion.
E6 `count`/`all` sets retain their qualified synthetic part UIDs through
binding, then expand to ordered Slide/withdraw/Park recipes with a distinct
part binding per step. Cabinet-root sets do not stand in for drawer parts.
For opening sets, measured world-space handle heights order parts bottom-to-top
within each cabinet. The existing 1 cm spatial tolerance groups same-height
parts, retaining their original lateral order, including horizontal rows and
multi-column layouts. Missing geometry leaves that cabinet's order unchanged.
Explicit separate steps and closing sets are not reordered. This is a default
sequence preference, not a collision-free planning guarantee.
E6 uses each qualified handle's position for automatic arm
preference rather than the shared cabinet center; explicit arms take precedence.
This code-owned geometry stays outside the model prompt and runtime scene config.
The final program segment rechecks the last requested position of every E6
joint, so later manipulation cannot silently invalidate an earlier drawer.
Repeated open/close commands on one joint retain only its last terminal target.
Regenerate existing E6 bundles to include these terminal checks.
Part inventory exposes a configured initial open/closed endpoint only when
the runtime default zero reset matches an explicitly authored, validated
`gen_sim:closedPosition` endpoint or its opposite endpoint. Missing endpoint
semantics, explicit `init_qpos` vectors (unqualified joint ordering), and
non-endpoint resets remain unknown. This is configuration evidence, not a
settled-state/contact observation; no asset or reset value is changed.
For an unbound existing Gym scene, Task Engine automatically renders the
current GLB assets into UID-labeled overviews, segmentation masks and crops,
then retries grounding with the same LLM transport configured for text binding
in `gen_sim/.env` or the process environment;
`--reference-image` is not this binding input. A deterministic tenth of already
bound scenes receives an audit-only visual spot check. Text and visual binding
share semantic identity rules: descriptive categories are not exact-match labels;
synonyms, translations and supported broad names may identify the same entity,
but explicit modifiers, exact UIDs and counts cannot be relaxed into substitutes.
Model responses include matching support, conflicts and nearby candidate UIDs;
reported conflicts cannot authorize a resolved binding. This evidence is not
new scene state or physical capability. Legacy four-field response replay stays
supported. Genuine missing/ambiguous results are not retried into guesses.
Run-local binding audits retain prompts, schemas, responses and model identity,
including early failures; candidates and source fingerprints survive failed
selection. Visual evidence also records scene revision and image hashes; the
VLM may select only visible inventory UIDs, while native semantic compatibility
remains authoritative. `task_engine/binding_evaluation.py` provides frozen
positive/rejection controls and per-reference/paraphrase metrics; provider calls
require `--run-model`. These text-only metrics do not certify visual or physical
success. Articulation
proxy GLBs do not represent live joint state, so E6-E9 binding does not use
this visual path. Without usable Task Engine LLM configuration, text-only
binding and its existing conflict behavior remain unchanged.
E4 may defer unnamed transfer/receiver arms as `auto`; the planner binds the
source from the held state or nearest scene side and the other arm as receiver,
while preserving explicitly named arms. GenSim declares `pick_purposes` per
generated Pick segment, bound to the compiled runtime invocation identity;
missing or stale declarations reject loading. Built-in Pick identity and
compiler-owned downstream reachability remain unchanged. Ordinary pickup tries
top-down before arm-relative diagonal approaches. Pour pickup retains its
axis-dependent ordering. E4 source lowerers reserve an object end and try both
ends top-down before either end diagonally. The wrapper marks only an owned goal
snapshot; upright and stack constraints take precedence over ordinary policy.
`GenSimPickUp` screens each
candidate strategy through the shared full pickup planner and retains the exact
successful trajectory per row; endpoint IK alone is not acceptance. Explicit
grasp poses, fixed calibrations and constrained picks retain their contracts.
Its direction prefilter removes only candidate columns whose two roll variants
are rejected in every environment, then delegates to shared `PickUp` and restores
the original candidate indices. Upright adjustment and unconstrained approaches
delegate unchanged. Keep the full environment axis; do not copy the IK loop or
change shared defaults to optimize this GenSim selection.
`GenSimHandOver` chooses the receiving end from the actual source attachment.
Vertical receiving grasps try diagonal then top-down; horizontal ones reverse
that order. Candidate, IK or path failure triggers one alternate-direction
attempt through the complete shared HandOver planner before execution. Successful
rows retain their first trajectories and attachment candidates. Exhaustion
fails normally; no source regrasp or cross-call recovery is added. Transfer,
release, retreat and state effects remain owned by shared HandOver. E5 paired
grasping and E6 handle policies remain separate. These are GenSim-local skill replacements,
not another executor or changes to shared goal/options types. Unmarked calls
delegate unchanged; drawer deployments install the same ordinary wrappers.
Constrained E2 Pick lowerers
snapshot their object/target rule selector into the goal; only that Pick uses
an immutable filtered-generator view. Unscoped sampling, including HandOver,
checks opening clearance without applying another stage's region/release rules.
Drawer transport and release remain drawer-owned. Relative placement geometry and
stability use preceding alignment operations, never future E2 state. E2 after
an explicit on/above placement retains that support and uses its measured top
for staging, release and grasp clearance. Its constrained regrasp uses a
top-down approach so uprighting yields a side grasp instead of an end-directed
rim approach. Standalone E2 retains its original table target and grasp policy.
The GenSim-owned `gen_sim.place_upright` call reuses Place while
separating the later upright release geometry from the earlier ordinary Place
route. Repeated identical selectors within either call family still reject
conflicting stage geometry instead of silently overwriting a prior route.
Terminal preserve-pose handovers verify receiver attachment and the
declared world-axis alignment; ordinary upright constraints still use world Z.
Regenerate bundles after adaptive-policy/lowerer revisions; do not bypass their
fingerprint checks or extend the shared Pick/AxisAlign configuration schema.
Multiple children
placed inside the same measured container reserve each prior landing footprint
and margin; insufficient space or an unmeasured shared floor fails preflight
instead of assigning overlapping destinations. These are planning contracts,
not physical placement acceptance.
Generated non-drawer held-object transports declare a measured attachment postcondition.
GenSim installs PickUp, HandOver, MoveHeldObject, Place and Pour wrappers for all
deployments after shared engine validation; shared Lab action classes and
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
vessel identity is bound per Pour invocation from that recipe's explicit staging
reference, not from an object-wide map. Its position is late-bound while the source's upright heading is preserved
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
layout is unchanged. Execution admits E1-E6 and standalone E9; E7/E8 remain
rejected before graph generation and bundle publication. E6 uses the explicit registered
Slide/withdraw/Park recipe in `_task_program/articulation_binding.py` and
`articulation_slide.py`. Its first supported binding is one fixed-base,
single-prismatic, self-contained metre-authored USD with uniform scale and an
unambiguous handle mesh, in one simulation environment. Asset hashes, joint
ownership, and declared limits are checked again at runtime. Public Slide owns
planning; public Task Program and Gym own execution. Joint-target retention is
checked after every recipe call; this is not in-flight contact qualification.

E9 is owned by `_task_program/press_binding.py` and `press_runtime.py`: an
explicitly bound, fixed-root prismatic button lowers to prepare/Press/Park
through the ordinary Task Program and Gym runtime. Generated deployments audit
scaled limits, pre-resize calibrated moving-link mass and an explicit release bias;
source USD and shared physics defaults remain unchanged. No synthetic latch is
inferred from a STOP label; the sensor never commands the button. Grasp TCP is offset to
the closed fingers' leading geometry. Mandatory post-policies verify released
stability, at least 0.05 mm continuous contact-backed travel and actual finger
withdrawal, again after Park. Rebound does not erase the event; full mechanical
travel, device activation and self-latching are not claimed. Only the empty symbolic effects
use projected bookkeeping; these physical post-policies cannot be removed.
One button/one environment is supported; mixed recipes and ambiguous controls
remain rejected. The old standalone press probe was removed; qualification uses
the normal Task Engine route. Regenerate E9 bundles after lowerer revisions.
E9 preserves the declared/native button, housing and finger contact envelopes;
it does not override contact or rest offsets. Normal E9 generation sets only
the selected button assembly to unit body scale, preserving XY, rotation and
the original support bottom. Other objects and source USD remain unchanged.
Table-footprint or object-AABB conflicts reject the deployment. The audit in
press_adaptation.json records original and deployed geometry; mass and release
drive retain their pre-resize calibration. This explicit deployment policy is
not gripper-driven size estimation or proof of original-scene success.
E9 physical configuration identity hashes resource contents rather than their
absolute locations, so nested transaction publication and final-bundle copies
preserve identity without allowing physical-parameter or asset-content drift.
An explicit E9 route may declare 0--10 mm extra commanded press travel;
the requested default is 10 mm. This changes only public Press's
command distance, not button limits, target state, physics or acceptance. The
extra distance is included in the route fingerprint; the bound is not a
collision-clearance or hardware-safety certificate.

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
When drawer routes exist, the adapter creates `DrawerPlacementEngine` but keeps
ordinary wrapper installation unchanged. Explicit drawer planning scopes delegate
transport and Place to the original primitives. The `gen_sim.drawer_transport`
semantic call reuses MoveHeldObject with the effectless drawer contract; ordinary
`simulation.move_held_object` calls retain attachment verification even in the
same program. Drawer motion and later Place/E6 acceptance are not a new transport
retention certificate. Adapter contract v8 requires regenerated bundles with
invocation policies and distinct drawer routing.
Inside targets share E6's asset-qualified part/link identity, never a duplicate
rigid root. Geometry supplies a floor/center hint, not an interior certificate.
The GenSim adapter refreshes the placement target from actual link/object poses
on every Place plan/replan while preserving the declared scene manifest.
No cavity-fit or separate FCL rejection gate is installed. A Pick with an explicit
downstream drawer target defers that reachability screen; a later drawer task on
the same object does not disable ordinary Pick lookahead. Generated bundles insert an
explicit `gen_sim.drawer_transport` before Place: cuRobo plans that free-space
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
| Ingest formats and metadata | `tests/gen_sim/asset_engine/` |
| UI roots/auth/session workflow | `tests/gen_sim/gradio_ui/` |
| Task intent and scene capability contracts | `tests/gen_sim/task_engine/test_agent.py`, `test_interpretation.py`, `orchestration/test_scene_adapter.py` |

Select provider-backed or simulator conversion tests only when that runtime
boundary changes; source/unit checks do not establish remote service quality.
