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
