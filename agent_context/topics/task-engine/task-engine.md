# Task Engine semantic contracts

`embodichain.gen_sim.task_engine` owns instruction understanding, candidate
contracts, TaskSpec generation, canonical Semantic Call graphs and declaration
coverage. Importing its front end does not initialize Lab or a simulator.

| Request | Owner under `gen_sim/task_engine/` |
|---|---|
| Instruction intent and explicit model-caller boundary | `interpretation.py` |
| Supported task intentions and relations | `ontology.py` |
| Candidates, drafts, scene requests and success declarations | `contracts.py`, `agent.py` |
| Candidate-to-normative-spec conversion | `task_spec.py` |
| Semantic graph schema, ordering and canonical call validation | `semantic_graph.py` |
| Roles, binding reports and scene manifest contracts | `orchestration/contracts.py` |
| Graph/TaskSpec/grounding coverage and source lineage | `semantic_graph_validation.py` |

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
boundaries. Public API descriptions are in the
[Task Engine reference](../../../docs/source/api_reference/embodichain/embodichain.gen_sim.task_engine.rst).

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
