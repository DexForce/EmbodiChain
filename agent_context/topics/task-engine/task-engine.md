# Task Engine

`embodichain.gen_sim.task_engine` owns instruction understanding, semantic planning,
role/target binding, declarative policy selection, bundle publication and runtime
orchestration. TaskSpec owns normative identity and acceptance. Shared Lab,
Motion, Scene and GraspKit owners implement skills, algorithms and observation.
Importing the instruction/candidate front end initializes neither Lab nor Sim.

| Request | Owner under `gen_sim/task_engine/` |
|---|---|
| Instruction intent and explicit model-caller boundary | `interpretation.py` |
| Supported task intentions and relations | `ontology.py` |
| Candidates, drafts, scene requests and success declarations | `contracts.py`, `agent.py` |
| Candidate-to-normative-spec conversion | `task_spec.py` |
| Semantic graph schema, ordering and canonical call validation | `semantic_graph.py` |
| Pure program projection and explicit segment acceptance bindings | `program_declarations.py` |
| Pure bundle declarations, preflight and atomic publication | `task_program_bundle.py`, `bundle_assembly.py`, `bundle_deployment.py`, `bundle_identity.py`, `bundle_io.py` |
| Installed current-bundle loading and source revision/input inventory | `bundle_loading.py`, `bundle_inputs.py` |
| Execution and occurrence-bound evidence projection | `execution.py`, `runtime_bindings.py`, `runtime_evidence.py`, `evidence_binding.py` |
| Workflow, child lifecycle and fresh run allocation | `workflow.py`, `run_directory.py`, `orchestration/execution_resources.py` |
| Unified public CLI and worker | `cli.py`, `_bundle_runner.py`; dispatch in `embodichain/cli/main.py` |

| Source upright/carrier/handover/return graph declarations | `source_graph_declarations.py`; geometry comes from public Scene/Compute |
| Declared alignment/Pick aliases and held-resource transitions | `alignment_declarations.py`; nominal envelopes use the public hand protocol |
| Roles, binding reports and scene manifest contracts | `orchestration/contracts.py` |
| Graph/TaskSpec/grounding coverage and source lineage | `semantic_graph_validation.py` |
| Semantic acquisition requests and normalized-asset view projection | `asset_preparation.py`; consumes public Scene Engine APIs |
| Source export discovery and pure scene declarations | `orchestration/source_scene.py` |
| Explicit frozen physical-owner components and drift identity | `physical_input.py` |
| Semantic part positions/ranks from public Scene descriptions | `orchestration/part_catalog.py` |
| Normal scene-to-frozen physical declaration composition | `source_physical_input.py` |
| Semantic control selection and physical-owner receipts | `control_physical_preparation.py` |

| Ordered relation targets and occupied support reservations | `placement_declarations.py` |
| Static scene affordance/source bindings | `scene_binding_declarations.py` |
| Named Task Program stability and sensor declarations | `stability_declarations.py` |

TaskSpec identity and predicate evaluation belong to
[Task specification](../task-spec/task-spec.md). Both mappings and instances must
retain their verified semantic identity. Graph coverage checks declarations; it
is not evidence that the program ran or its physical objective succeeded.

Graph validation uses the public Task Program language decoder. Shared Atomic,
Motion, Scene and GraspKit owners remain independent of this front end; it
contains no skill, planner, engine or sensor implementation. Runtime assembly
consumes the public configured Task Program deployment and observation services.
Task Engine does not replace skill registrations, patch skill private state or
operate on planner SDK buffers. Shared owners import neither this package nor
TaskSpec. See [deployment and evidence](deployment.md) for bundle/version,
input identity, runtime and qualification contracts.

Fixed candidate/graph inputs can be checked without invoking a model. Run
`tests/gen_sim/task_engine/` for candidate/spec, coverage and import/ownership
boundaries. Public API descriptions are in the
[Task Engine reference](../../../docs/source/api_reference/embodichain/embodichain.gen_sim.task_engine.rst).

`orchestration/source_scene.py` owns supported source-export discovery, semantic
UID correspondence and immutable preparation provenance. It delegates physical
properties and pose/scale normalization to shared Scene ingress, and reads
mesh-top inputs through the public Scene geometry owner. Use `PreparedScene`'s
asset identity check before later preparation; missing or changed original input
cannot be replaced or given a new digest under its old identity. Articulation
assets use their resolved top-level source path; shape-owned asset digests stay
unchanged. Asset baking
remains Scene Engine's existing owner; this importer does not publish bundles
or connect a runtime/CLI.

`physical_input` imports an explicitly selected set of physical-owner component
files before semantic planning. Its detached `FrozenPhysicalInput` projection and
input-file inventory reject nested-data, file and newly discovered dependency
drift without re-signing an old snapshot. It reads no trajectory, receipt or
certificate and does not publish or launch a runtime bundle. Shared Scene owns
pose validation/translation and source geometry; `orchestration/part_catalog`
adds only conservative semantic part labels. The later bundle loader still owns
runtime-contract versions and complete current-owner input inventory validation.

## Read-only graph display

`semantic_graph_html.py` renders an existing graph and optional TaskSpec,
recomputed coverage and recorded execution report. `reporting.py` owns the
read-only report validator; `bundle_spec.py` recognizes explicit TaskSpec absence.
Rendering does not call a model, start a simulator or change physical verdicts.
The writer produces a standalone HTML file outside sealed bundle directories.
JSON/YAML inputs and historical certificates remain unchanged. The React/TypeScript source is under `web/task-engine/`; `web/graph-ui/` owns shared React Flow viewport controls and theme tokens, also consumed by the architecture explorer. `npm --prefix web/task-engine run build` updates the committed `_graph_react.js`, `_graph_react.css` and dependency notices in the Python package. Installed rendering embeds those resources and the licensed fonts without a Node runtime or network access.

`_semantic_graph_presentation.py` reads optional local thumbnails and run display
metadata. `_graph_localization.py` keeps display-only language labels separate
from canonical graph identity. Translation runs only through an explicit caller;
ordinary offline rendering never invokes it. Use
`tests/gen_sim/task_engine/test_semantic_graph_html.py` for HTML escaping,
identity/coverage checks, read-only input behavior and source-thumbnail selection.

Graph model projection and dependency/arm-lane layout stay in the Task Engine
frontend. Vector SVG export uses that layout directly; recorded measurements
and normative status evaluation stay with their existing Python owners.
Stage and call cards prefer the object's source render, falling back to its
cutout. These images identify the asset; they are not observations after a call.
`web/task-engine/tests/viewer.spec.ts` checks the built offline pages in Chromium,
Firefox and WebKit; `tests/gen_sim/task_engine/browser_fixture.py` creates
synthetic inputs through the public renderer without starting simulation.
See [`web/README.md`](../../../web/README.md) for both frontend build paths.


Model declaration binding uses `embodiment_declarations.py` and `robot_profiles.py`.
These consume public `embodichain_robots.task_program` descriptions and component
services; they emit declarations and provenance hashes rather than skills,
physical facts or acceptance. Normal Source physical preparation consumes this descriptor surface; bundle
publication and CLI routing retain their separate integration layer.

`resolve_source_hand_services()` includes the effective registered grasp-model
asset inventory in its returned hand-input hashes. It delegates model/side and
archive-reference selection to GraspKit's public model owner; component fallback
calibration files remain in the same hand-services provenance group. It creates
no sampler or Robot and does not add another bundle input group.

`prepare_source_physical_input` composes the explicit existing model component,
shared Scene declarations, Scene Engine normalization and optional source-control
preparation into a fresh `FrozenPhysicalInput`. Original source files and newly
derived provenance remain inventoried, with before/after drift checks. It never
reprocesses an existing frozen input or constructs Robot, simulation, IK or
skills. `control_physical_preparation` selects semantic subjects/resources and
records shared control-resize/mass/calibration receipts; those numerical and
authored-geometry algorithms stay in Scene/Compute/model owners. The selected
`dual_arm_trajectory_verified.yaml` component retains source tracking/recovery
values; Lab owns the pure source reset/cadence declaration helper.

Source effort preparation only reduces semantic acquisition/carrier/blocking
relationships into neutral robot-owner candidates. The robot facade selects
model effort thresholds, prismatic gains and initial clearance posture from
explicit requirements. Task Engine does not hold those machine calibration
values or rewrite runtime action policies.

build_program_declaration preserves each graph call and segment identity while
applying explicit owned post-policy and validator bindings. Deferred acceptance
must retain every original check at the group terminal. The same pure projection
validator is owned here for later bundle loading; it depends on no bundle reader,
simulator, robot or motion service. Complete deployment preflight remains with
the configured Task Program owner.

## Static source declarations

Placement and scene-binding resolvers consume frozen source scenes and explicit
resource mappings. Shared Scene owns mesh, articulation, collision and robot
mount adaptation; Compute owns numeric placement and reservation geometry.
Task Engine chooses semantic targets and produces existing declarative formats.
`resolve_stability_conditions` takes exact support sensor declarations (or a pure
factory), then returns detached typed conditions for the integration preset table.
It neither constructs sensors nor acquires evidence. Runtime policy evaluation
and physical observation remain Task Program and Lab responsibilities.

## Authored control declarations

Control declarations consume authored Press/Twist bindings and explicit model
calibration, then select typed options, conditions and post-policy occurrence
presets. Prismatic declarations preserve the complete Slide/withdraw/Park
recipe. Drawer declarations bind source payload/container relations and explicit
execution occurrences; drawer graph refinement owns semantic resource selection.
The control_declarations.py, prismatic_declarations.py, drawer_declarations.py and
drawer_graph_declarations.py modules construct no simulator, skill or planner.
Shared Scene inspects source geometry, Lab executes recipes and collects facts,
and TaskSpec owns normative judgments.

Source graph refinements operate on fixed input graphs and explicit authored
source views. They preserve graph target/call identity, insertion order and
semantic first-use/held-resource relationships while Scene supplies target
geometry and Compute constructs poses. Constrained aliases require explicit
geometry; ordinary E2 retains its existing downward-entry declaration. Closed
option/lowerer fragments remain declarations for later configured preflight,
not a runtime factory or a skill registration patch. Bundle/CLI integration consumes these declarations separately; the declarations
alone do not claim physical success.

## Policy and continuation declarations

semantic_program.py projects validated graphs and owner-supplied bindings into
the existing program format. action_profile_declarations.py and
policy_declarations.py select declarations for typed action, motion and recovery
options. Source verified occurrence presets explicitly select evidence for existing
release gates; terminal and attached checks retain their full clauses.
Source observed cuRobo declarations explicitly choose PRM without TimeOpt or
graph-seed fitting. These settings preserve the declared source handoff; SDK
0.8-to-0.9 optimizer differences retain their separate compatibility baseline.
They do not register skills or modify their private fields.
native_pick_declarations.py and continuation_declarations.py bind native inputs,
lookahead and released-hand selections to explicit compiled occurrences.
pour_declarations.py uses shared source target geometry and model calibration.
Its Source held-transport declarations explicitly select terminal attachment,
including measured Pour staging; this preserves terminal pose and constraint
checks without adding in-flight retention sampling.

experiment_declarations.py keeps opt-in contact, handover and reacquisition
permissions explicit. experiment_policy.py selects acceptance/reporting modes
without changing strict TaskSpec verdicts. Configuration defaults, runtime
composition, physical evidence and final bundle identity retain their own owners.

## Configuration, requests and replayable state

config.py owns the workflow, semantic planning and execution configuration plus
defaults.yaml. Explicit direct configuration and packaged workflow defaults keep
their distinct documented values. workflow_contracts.py validates existing-scene
requests and source/output separation. state_machine.py owns immutable stage
transitions and exact event replay; it starts no environment or model call.

orchestration/workflow_artifacts.py owns safe run JSON and history persistence.
It rejects writes below sealed bundle markers, including symbolic-link paths,
and preserves interrupted histories. Workflow scheduling and bundle/runtime
integration consume these contracts in their own layer.

## Semantic planning and source scene adaptation

semantic_planner.py and _semantic_recipes.py turn validated candidates and role
bindings into canonical declarations, preserving complete task recipes and
bounded control occurrence counts. They do not implement motion or atomic skills.
orchestration/scene_inventory.py and grounding.py own declared scene matching and
role/part grounding. scene_source.py fingerprints source inputs, while
scene_adapter.py composes source selection and conservative scene evidence.

The scene package owns static manifest validation, conservative relationship
graphs, the existing Scene Engine adapter and final exported-source inspection.
scene_backend.py consumes these descriptions and safe output/request contracts.
Missing geometry remains unmeasured; no source graph or proposal establishes
native physical success. Coordinator/runtime and bundle publication stay with
their integration layer.
