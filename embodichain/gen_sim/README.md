# GenSim commands

Use the unified namespace; every command also works with `python -m embodichain`.

```bash
embodichain gensim --help
embodichain gensim asset ingest --input raw_asset --category cup --output outputs/assets
embodichain gensim image clean-robot --image head.png --output outputs/clean-head
embodichain gensim scene generate --image table.png --output outputs/scene
embodichain gensim scene edit --scene outputs/scene --prompt "Move the cup to the left"
embodichain gensim scene expand --scene outputs/scene --config layouts.yaml
embodichain gensim scene preview --usd-file outputs/scene/scene_usd/scene.usda --viser
embodichain gensim task plan --image table.png --instruction "Put the cup on the tray" --output outputs/task
embodichain gensim services validate --config model_services.json --env-file services.env
embodichain gensim services doctor --config model_services.json --env-file services.env
embodichain gensim ui --host 127.0.0.1 --port 7860
```

Domain operations share their application implementation with the existing
`asset-engine`, `scene-engine`, `scene-randomization`, `preview-scene`, and
`task-engine` commands. Preview uses the Lab-owned standard USD file interface;
the old workspace/manifest preview arguments have been removed. New commands
use `--input`, `--output`, `--scene`, `--prompt`, and hyphenated option names.
The new scene operation is named `expand`; the legacy top-level command remains
`scene-randomization`.

Scene editing modifies the selected workspace in place. Task planning requires a
fresh output directory and ends with an unbound semantic graph and offline HTML
viewer. Its output alone does not construct a robot or execute a task.

Scene generation, editing and staged materialization produce an editable
`scene_export/`, a Z-up/metre/kilogram `scene_usd/scene.usda`, a relocatable
`scene_usd/scene.usdz`, and **`scene_package.zip`** for DexSim AssetRevision
`2.0-draft` interchange. The ZIP contains a canonical `asset.usda` entry with
default prim `/Asset`, its USD/texture dependency closure, `asset.json`, and
`reports/dexsim-packaging.json` / `reports/dexsim-registry.json`. DexSim's
packager validates the JSON schema, byte digests and cross-object references.
Its processed-USD packaging support is required; an export failure fails the
operation while retaining the editable workspace for diagnosis.

The AssetRevision describes packaging provenance. A scene with multiple
independent bodies is not automatically one rigidbody or articulation:
`asset_kind` remains unknown until classified. Unmeasured quality facts stay
null, the quality summary is `incomplete`, and no runtime capabilities or
certification are granted by packaging. Preview consumes the standard USD file
without interpreting GenSim UID or
initial-state tags. No scene manifest or copied native-asset preview package is
produced. The import owner resolves the USD-authored state; old custom metadata
is not restored by preview.

The UI launcher uses the active Python interpreter, preserves the selected SDK
import root, and retains the existing UI's process cleanup and authentication.
Install the `embodichain[gensim]` extra for its optional dependencies.
Model implementations and serving frameworks remain operator-selected; see the
[deployment and BYOS guide](services/DEPLOYMENT.md) and the versioned
[HTTP protocol](services/spec/v1/README.md).

### Clean a tabletop reference frame

`gensim image clean-robot` accepts a dual-arm head-camera RGB frame with the
two arms at opposite sides and a visible camera plate at the bottom center.
It uses the regular [GenSim service configuration](services/README.md) for
`image_segmentation` and an `image_generation` adapter with image-to-image
editing. For the built-in `image_http` adapter, the `generate_image_by_image_path`
option defaults to `/generate_image_by_image`. The endpoint accepts multipart
`image`, `prompt`, `width`, `height`, and `seed` fields and returns a PNG.

The output directory must be new. It contains `input.png`, the original RGB
frame, the per-part and combined masks, the editor response, and a
`preparation.json` pixel audit. The cleaned PNG keeps the input dimensions
and every decoded RGB pixel outside `robot-mask.png`. The camera support
must resolve to an unambiguous bottom-edge component. An edit that leaves all
detected robot pixels unchanged fails without publishing `input.png`.

External whole-scene generators use the independent
`embodichain.gensim_scene_providers` contract. Public operations in
`embodichain.gen_sim.workflow` publish a Task Plan and scene requirements, retain
durable provider job identities, validate a provider-neutral generated scene,
publish a structured qualification report, and prepare an execution-ready
deployment. The default remains the in-package Scene Engine; an external provider
must be selected explicitly and never silently falls back. Simulation execution
continues through the ordinary task runner.

## Generate, bind and execute

For a self-contained input fixture, see the
[scripted pick/place reproduction](../../examples/gensim/scripted_pick_place/README.md):

```bash
embodichain run-example gensim/scripted_pick_place --output outputs/gensim-scripted-a
```

It uses scripted model responses and a procedural scene, then real native
binding, Task Program preflight and verified execution. Each attempt saves the
graph frontend, reports and wall-clock timing. The observed placement failure
retains a nonzero exit status. Its timing does not qualify live image/asset
generation. For real inputs, configure the model services and use `gensim run`:

```bash
embodichain gensim run --config examples/gensim/run.yaml \
  --image /absolute/path/to/table.png --output outputs/gensim-run-001
```

The [example configuration](../../examples/gensim/README.md) selects a trusted
robot embodiment, a file-owned physics backend, and a verified execution policy.
Configuration paths resolve relative to their owning file; explicit CLI path
overrides resolve relative to the caller's working directory. Model services
retain the discovery and credential rules in [services/README.md](services/README.md).
The workflow borrows one service run across task planning, scene generation and
runtime entity grounding. It uses the existing `task_image_grounding` role to
match logical references to the generated inventory and declared robot resources.

A run performs image task planning, scene generation (or read-only reuse through
`--scene`), native entity/resource binding, strict configured deployment
composition and catalog preflight, then one batched demonstration episode through
the ordinary Gym/Task Program execution owner. It does not use the removed
action-agent pipeline. Native physical execution and its result are required;
a graph-only run cannot produce workflow success.

Automatic geometry binding covers observed rigid objects, antipodal pickup and
placement on a measured near-level support or planar container floor. Placement
reuses the shared landing-region selector, including footprint and occupied-space
checks. A run-owned Z-up OBJ copy bakes the exported Y-up scale and coordinate
conversion directly into the vertices, with isolated material/texture sidecars.
This avoids renderer-dependent GLB up-axis conversion while preserving the
original scene export.

Measured geometry also binds the registered `simulation.place_relative`,
`simulation.axis_align` and `simulation.move_held_object` calls when the run
configuration does not supply their lowerers; a configured lowerer keeps its own
targets and receives no synthesized validators. Relative placement offsets follow
the shared world axes, or the photographed view when `source_calibration` is set.
An aligned or upright-carried object's local +Z is re-framed to its length axis,
toward the narrower end, without changing world geometry. A held transport keeps
the object's initial orientation (or the upright one) and centers its contact end,
the wider end of an elongated tool, over the reference; relative to a support the
object already rests on, it lifts in place. A held transport without a reference
is validated against its world target pose. Action-option and effect-monitor
templates named after a skill (for example `axis_align`) seed the registered call
IDs that use that skill.

Registered calls require compatible configured lowerers and runtime services.
The optional `binding` mapping supplies task-owned `profile`, `runtime_services`,
`scene_binding` additions and `segments` post-policy/validator declarations
through the existing closed decoders. `binding.resource_anchors` maps declared
resources to planar `[x, y]` anchors: a single-arm call without a resource then
uses the arm already holding its object, or the arm anchored nearest to it, and
a placement on a support prefers its robot-side region. `binding.release_clearance`
(metres, default 0) releases placements above their resting pose while validators
keep the resting target; the place gate otherwise expects a released object to
separate from the open gripper before retracting. `binding.accept_settled_hold`
(default false) lets the declared effect monitors accept a held object that
shifted in the hand and came to rest: its pose error lies between the monitor's
attached and detached thresholds, it moves slower than 0.03 m/s and 0.2 rad/s,
and the grasp evidence still reports the hold. Without it such a hold stays
unresolved until the skill times out.

Further optional run fields: `task` reuses a published task-plan directory instead
of planning, after verifying its artifacts and that it planned the same instruction
and image; `planning_repairs` (default 1) bounds diagnostic re-planning;
`source_calibration` names a JSON camera/tabletop calibration of the input image,
which rescales and places each rigid object so its projection matches its image
mask (scenes with articulations, or whose table does not stay world-axis aligned
under the calibrated yaw, are rejected). The calibrated table is decomposed
with V-HACD, so objects rest at the calibrated surface height;
`room` adds a scene asset room (for example `WoodLab`) as a visual shell with its
authored floor collider; `recording` streams every camera frame and per-step state
to `recording/` during execution. Missing native parts, hidden or ambiguous
references, unsupported geometry and incompatible capabilities reject rather
than selecting a substitute or claiming execution readiness. Model grounding
never creates capabilities, rewrites control-part names or chooses physics.

### Automatic cleanup for verified head-camera inputs

For a verified RoboDojo dual-arm head frame, add its camera view and the
SHA-256 from the reference episode's `cam_head.first_png_sha256`:

```yaml
input_camera:
  view: robodojo_dual_arm_head
  sha256: <64 lowercase hex digits>
```

`image` must be that exact raw 640 × 480 frame. Other views and already cleaned
images use the ordinary configuration without `input_camera`. GenSim uses its
shared service configuration to clean the verified raw frame before Task Engine
and scene generation. Both stages consume the same cleaned PNG. If cleanup
fails, the run stops at `image_preprocessing`.

To reuse a plan on the same raw frame, keep the previous run directory with
`run_manifest.json`, `image_preprocessing/`, and `task/`. Run with the same
`input_camera` declaration and `--task PREVIOUS_RUN/task`. GenSim checks the
raw image, the Task Plan digest, and the previously grounded PNG and mask.
It copies the cleanup artifacts without requesting a new image edit. For an
independent published Task Plan, supply its exact grounded image as `image`
and omit `input_camera`.

## Outputs and verdicts

For a route comparison and the path from specification to physical evidence, see the
[Task Engine guide](../../docs/source/features/generative_sim/task_engine.md) and
[paired 20-task evaluation](../../scripts/benchmark/task_engine/README.md).
Standalone `gensim task plan` defaults to hybrid. Its published stage distinguishes
authoring only, pending visual binding and readiness for scene qualification.

The explicit template route’s v2 task-plan manifest publishes `task_spec_v1.json` and the selected
semantic graph. The v0.1 normative source remains available as `task_spec.json`.
Use `load_task_plan` to verify the publication. Scene providers receive
`scene_requirements.json`; an agent provider can consume the associated
`scene_agent_messages.json`. Expansion clients read `expansion_requirements.json`
or project a frozen document against supported dimensions in Python.

Open `task/semantic_task_graph.html` for the plan and `task_review.html` beside
the run manifest for measured call evidence. Click an action in the canvas
to inspect its step interval and recorded failure events. Existing runs can
publish a fresh offline view with
`embodichain gensim task view --plan RUN/task --run RUN --output review.html`.
Omit `--run` to review only the frozen task and graph.

To revise template graph calls against a published spec and its original grounding:

```bash
embodichain gensim task revise --parent outputs/task --proposal proposed_nodes.json \
  --expected-spec-digest "$SPEC_DIGEST" --expected-graph-digest "$GRAPH_DIGEST" \
  --expected-plan-digest "$PLAN_DIGEST" \
  --output outputs/task-revision-2
```

The proposal uses the semantic planner response fields `nodes` and
`requirements`. Each accepted revision publishes a new immutable plan
directory and records the parent manifest digest.
`load_task_plan` returns `plan_digest` (the SHA-256 of `task_plan_manifest.json`)
for the complete publication, including
semantic annotations and binding requests. Revision requests and native attempt
evidence bind this identity as well as the spec and graph digests.

Frozen clauses can carry structured `constraint` declarations for role properties,
return-to-origin and orientation restoration. Coverage checks their target, calls
and required bindings; unsupported meanings remain visibly unverified and block
revision and deployment. Both scene routes enforce explicit role properties.
An absent property measurement is unverified. Existing `generated-scene/v1`
providers may omit optional affordance metadata; those capabilities stay pending
while qualification continues to geometry checks. Explicit incompatible metadata
still rejects the scene.

A new run directory contains:

- `source_image.*`: the raw input frozen as `source_image_sha256`;
- `image_preprocessing/`: when enabled, both images, masks and the pixel audit.
  The manifest's `image_sha256` identifies the effective image used downstream;
- `task/`: source semantic graph, TaskSpec, image grounding and offline viewer;
- `scene/`: generated workspace, unless an existing workspace was selected;
- `deployment/`: robot/policy snapshots, native mesh copies, integration,
  bound graph, program and runnable `task.yaml`;
- `binding_report.json`, `preflight_report.json`, `execution_report.json`;
- `run_manifest.json` and `service_calls.json`.

Existing run directories are never overwritten. Failed runs retain completed
artifacts and stage/error information. The execution report retains the native
demo segment outcomes and measured-effect verdict; truncation and failed
execution produce a nonzero CLI exit status. Exceptions and interrupts retain
their failure status and release owned runtime resources.

The source task graph stays unbound. The bound graph and explicit deployment are
separate artifacts. The workflow records `task_spec_verdict: not_evaluated`:
a measured Task Program execution result is not a separate normative TaskSpec
certificate.

The standalone planning frontend does not automatically display native execution
results. Its elapsed-time field covers semantic planning only. The scripted
example's `process_timing.json` covers the complete measured worker lifetime;
its `comparison.json` separates matching semantic artifacts/step spans from
floating-point trajectory reproducibility.
