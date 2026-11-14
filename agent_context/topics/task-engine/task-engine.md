# Task Engine semantic contracts

`gensim task revise` validates a planner proposal against the frozen v1
document and grounding in a published plan. The caller supplies expected spec,
graph and complete publication digests plus a new output directory. The complete
identity covers deployment annotations and binding requests. The revised graph and
its coverage report share the original document and scene requirements.

`embodichain.gen_sim.task_engine` owns instruction understanding, candidate
contracts, TaskSpec generation, canonical Semantic Call graphs and declaration
coverage. Importing its front end does not initialize Lab or a simulator.

| Request | Owner under `gen_sim/task_engine/` |
|---|---|
| Image + instruction → unbound canonical-call graph | `image_planning.py`; `_image_run.py` publishes artifacts; `cli.py` through `embodichain gensim task plan` (legacy `task-engine`) |
| Instruction intent and explicit model-caller boundary | `interpretation.py` |
| Supported task intentions and relations | `ontology.py` |
| Candidates, drafts, scene requests and success declarations | `contracts.py`, `agent.py` |
| Candidate-to-normative-spec conversion | `task_spec.py` |
| Semantic graph schema, ordering and canonical call validation | `semantic_graph.py` |
| Pure program projection and explicit segment acceptance bindings | `program_declarations.py` |
| Roles, binding reports and scene manifest contracts | `orchestration/contracts.py` |
| Graph/TaskSpec/grounding coverage and source lineage | `semantic_graph_validation.py` |

Model calls use role-bound capabilities in `gen_sim/services/`; interpretation,
image grounding and display translation have separate bindings. An injected
caller takes precedence and requires no default transport credentials. TaskAgent
shares the supplied run across its joined candidate workers; provider-specific
options and protocol handling stay in adapters. See the [service guide](../../../embodichain/gen_sim/services/README.md).

TaskSpec identity and predicate evaluation belong to
[Task specification](../task-spec/task-spec.md). Both mappings and instances must
retain their verified semantic identity. Graph coverage checks declarations; it
is not evidence that the program ran or its physical objective succeeded.

Graph validation uses the public Task Program language decoder. Shared Atomic,
Motion, Scene and GraspKit owners remain independent of this front end; it
contains no skill, planner, engine or sensor implementation. Execution assembly,
physical adaptation and runtime workflow configuration are separate layers.

Fixed candidate/graph inputs can be checked without invoking a model. Run
`tests/gen_sim/task_engine/` for candidate/spec, coverage and import/ownership
boundaries. The [Task Engine guide](../../../docs/source/features/generative_sim/task_engine.md)
owns CLI configuration, artifact interpretation, offline display and runtime
handoff. Public API descriptions are in the
[Task Engine reference](../../../docs/source/api_reference/embodichain/embodichain.gen_sim.task_engine.rst).

## Image semantic planning entry

`embodichain gensim task plan --image IMAGE --instruction TEXT --output DIRECTORY`
uses the configured GenSim model to interpret the complete instruction, ground
logical image references and plan canonical calls from the data-only Task Program
catalog. It always ends at the unbound semantic graph; there are no model or
stage-selection flags. `image_planning.py` retains raw provider attempts and
allows one diagnostic repair per grounding/planning stage. It contains no
per-task graph construction templates.

`lab/task_program/language/semantic_contracts.py` owns provider-free call argument,
resource, precondition/effect and runtime-binding declarations. Configured live
lowerers consume those same argument contracts and retain their stricter scene
and route checks. The catalog must import without `lab.sim`, Gym or DexSim.
`semantic_plan_validation.py` checks task producers, source identity, topology,
held-object/arm state, structured instruction qualifiers and deferred binding needs.
Qualifier witnesses must reference their frozen target and implementing calls;
return-to-origin also requires its source-pose binding and must be the subject's
final pose-changing operation in the targeted milestone. Each clause needs its
own structured meaning; an unrelated role property cannot cover it. Unsupported meanings
remain unverified and cannot authorize a revision or deployment.
Task producer obligations derive from the existing task affordances and catalog
capabilities rather than per-E-type construction branches. Explicit part parents
retain drawer/handle identity; step-result selectors inherit earlier source IDs.
This semantic-only checker does not weaken historical grounded coverage checks.
Requested destinations and relations must be realized by each subject's producing
call, rather than an unrelated pickup annotation. Resource-state diagnostics
include proposed node/slot set/remove operations, including the original
acquisition for a named handover. Repair prompts require retaining unaffected
fields of the previous plan; suggestions do not mutate the graph or bypass the
complete strict validator. Task Engine side enums `left_arm`/`right_arm` map
explicitly to logical resources `left`/`right`; custom resource identifiers and
robot control-part names are preserved for runtime binding. Terminal holds are checked at
the source step boundary; coordinated hold is rejected until a matching hold
call is available in the catalog.

Image entities distinguish `observed`, `instruction_declared` and
`runtime_required`; only observed entities carry normalized boxes. Ambiguous references retain
observed candidate IDs rather than selecting one implicitly. Hidden or
ambiguous references retain a logical identity, description and pending runtime
binding rather than an invented visual match. Original poses are binding needs
of the source object, not fictitious observed scene objects. Runtime geometry and
policies remain outside canonical call arguments.

Fresh output directories contain source-image provenance, candidates, TaskSpec,
image evidence, the catalog snapshot, calls, source-clause traceability, pending
bindings and a standalone viewer. Failed runs save uniquely named sibling audit
JSON before temporary publication is cleaned up. `semantic_plan_validation/v1`
reports separate semantic/visual states and always sets `execution_ready: false`
and execution `not_run`; the viewer recomputes the report and displays pending
bindings. Legacy graph/report reading remains unchanged. The all-zero integration
fingerprint denotes no runtime integration, not an executable deployment.

Use `tests/gen_sim/task_engine/test_image_planning.py` for offline E1–E9 and
negative cases, and `data/image_task_expectations.json` with
`image_case_acceptance.py` in that test directory for independently authored
clauses of the pinned task1171–task1190 cohort. These checks supplement manual
semantic review and never qualify physical execution.

## Read-only graph display

Version-2 plans and revisions embed the frozen v1 specification and its pairing
report in the offline viewer. The grouped action canvas links milestones, final
goals and roles. Native run review is published separately at `task_review.html`;
`gensim task view --plan PLAN [--run RUN] --output NEW.html` also renders
existing artifacts. Only matching attempt identities, complete plan identities and
execution-report digests supply native call evidence. Per-environment details
retain row reasons, step ranges and event masks. Source images come from the
verified input reference or legacy inventory. Archived graph-only views retain
their existing renderer.

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

build_program_declaration preserves each graph call and segment identity while
applying explicit owned post-policy and validator bindings. Deferred acceptance
must retain every original check at the group terminal. The same pure projection
validator is owned here for later bundle loading; it depends on no bundle reader,
simulator, robot or motion service. Complete deployment preflight remains with
the configured Task Program owner.

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


## Configured bundle snapshots

Already-authored deployment declarations can be sealed through
`bundle_deployment.prepare_configured_bundle`, atomically published through
`bundle_io.write_bundle_atomic`, then loaded with
`bundle_deployment.load_configured_bundle`. Input hashes and implementation
revisions are explicit caller-owned contracts. `bundle_identity` binds the file
inventory and semantic identities; `bundle_inputs` validates provenance groups.
These APIs create no simulation, Robot, skill or planner. Shared Task Program
composition accepts owned inline components, and the Gym parser consumes the
sealed component snapshots when configuring an environment. Bundle changes
require preparing a new identity; historical run outputs remain separate.
