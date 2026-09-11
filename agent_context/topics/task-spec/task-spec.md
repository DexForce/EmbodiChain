# TaskSpec semantic protocol

`embodichain/task_spec/` owns pure JSON task meaning, canonical task identity
and content/evidence references. It depends only on the standard library.
It does not own LLM calls, scene creation, check execution, simulation
lifecycle, retries, dataset submission or persistence.

## Owning entry points

- `validation.py` validates template fields, role references and capability
  vocabulary; `expressions.py` validates bounded predicate AST and quantities.
- `registry.py` returns a detached semantic-version/predicate-revision snapshot.
  This declares semantics, not installed checkers or Semantic Calls.
- `canonicalization.py` canonicalizes roles, inverse relations and SI units.
  Top-level self hash is excluded; unknown execution fields are rejected.
- `contracts.py` validates the five records, computes exact instance content
  identity, checks optional template/instance owners and applies certificate
  required-check acceptance. URI/digest references are not resolved here.

Template v0.1 permits six roles, 128 predicate occurrences and depth 16.
Canonical quantity values are exact decimal strings; accepted units are
m/cm/mm, s/ms and rad. Role relabeling is independent of spelling/order,
including equal role declarations. Temporal sequence order remains normative.
These are restricted equivalences, not arbitrary formula equivalence.

SceneInstance owns grounded roles, assets, scene/embodiment and observed
initial-state references. Instance roles are canonical IDs; use
`canonical_role_map` to rekey author grounding and evaluate the corresponding
`canonical_template`. This prevents ambiguous physical meaning under role
alpha-renaming. ActionWitness owns graph/program/integration/policy/
constraints and execution evidence; candidate execution is null. Existing
integration_fingerprint is reused. Certificate checks bind template, instance,
witness and one env/episode, with explicit versions, metrics and evidence.
Missing/not_run/unavailable/unsupported/failed required checks do not pass.
Hosts must verify evidence content before accepting report statuses.

## GenSim boundary

`SemanticTaskPlanner.plan(task_template=..., scene_instance=...)` consumes
paired explicit inputs and emits a candidate `semantic_task_graph/v2`.
It verifies physical-object bindings and records template_hash, instance_hash
and legacy_plan_hash. The existing recipe remains a proposed solution and
does not redefine or prove the normative goal.

Without TaskSpec inputs, TaskCandidate and graph/v1 behavior is unchanged.
`task_program_bundle.py` and `_bundle_runner.py` refuse v2 export/execution before writing artifacts until
final evaluation before data submission/reset exists. This is a planning
provenance entry, not the complete template-driven generation/runtime route.
TaskAgent migration, template-derived scene requests, actual-state capture,
shared evaluator, Workflow certificates and expansion hosting remain future
work in their current owners. See [GenSim](../gen-sim/gen-sim.md).

## Focused validation

Run `tests/task_spec/` for identity, schema, quantity and evidence boundaries.
Run `tests/gen_sim/task_engine/test_task_spec_planning.py` and existing
`test_agent.py`/`test_semantic_graph.py` for the paired planning entry,
graph versions and legacy compatibility. These CPU tests do not certify
physical rollout or installed predicate observers.
