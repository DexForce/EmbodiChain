# TaskSpec v0.1

TaskSpec owns normative task meaning and evidence references. It imports only
the Python standard library. It does not read assets, invoke checkers, plan,
step, reset, retry, write files, or commit datasets.

## Author a template

```python
from embodichain.task_spec import semantic_hash, validate_task_template

template = {
    "schema_version": "taskspec/template/v0.1",
    "semantic_version": "0.1",
    "roles": {
        "item": {"kind": "object", "capabilities": ["graspable"]},
        "reference": {"kind": "object"},
    },
    "init": [],
    "goal": [{
        "predicate": "relative_position",
        "object": "item",
        "reference": "reference",
        "relation": "left_of",
        "margin": {"value": 10, "unit": "cm"},
    }],
    "invariants": [],
    "requirements": [],
}
template["semantic_hash"] = semantic_hash(template)
template = validate_task_template(template)
```

All records are detached JSON dictionaries. Unknown fields and unsupported
versions are rejected. A present digest is excluded during hash calculation
and verified by the corresponding complete-record validator. No other field
is silently dropped: plan, trajectory, seed and robot model are invalid
template fields. The old GenSim TaskCandidate.semantic_hash keeps its existing
step-hash meaning.

Role names are binders. Alpha-renaming changes role keys and only declared
role references, never arbitrary strings. At most six roles and 128 predicate
occurrences are supported in a template. Logical depth is at most 16; the JSON
decoder also bounds nesting and total nodes. Equal role declarations are
relabelled exhaustively to choose a stable minimum, including symmetric roles.
Instances must bind canonical IDs (role_0, role_1, ...), using
`canonical_role_map(template)` to rekey author-name grounding. Checkers consume
`canonical_template(template)` against these bindings. Binding author labels
directly is rejected: equivalent templates can otherwise exchange author-role
meanings while retaining the same task and instance identities.

Canonicalization supports inverse left/right and front/behind relations, SI
unit scaling, sorting conjunction/disjunction arguments and top-level condition
lists. It does not claim general logical equivalence, simplify negations,
flatten boolean trees, or deduplicate conditions. Optional temporal defaults
to an empty list. Each temporal entry is an ordered
`{"op": "sequence", "args": [expression, expression, ...]}`, requiring observations
in that order at strictly increasing sample times. Multiple entries are
conjunctive; each entry's argument order remains normative and changes identity.

Every quantity is `{"value": ..., "unit": ...}`. Values accept integers, finite
floats, or finite decimal strings. Canonical values are exact fixed decimal
strings (with trailing fractional zeros removed); this avoids float collisions
between distinct success thresholds and dependence on process decimal context.
v0.1 accepts m/cm/mm, s/ms and rad. It rejects degrees and other units until
their conversion semantics are versioned. Decimal tokens/canonical values
are bounded to 128 characters and parsed exponent to ±256.

## Predicate vocabulary

The semantic version pins the complete registry snapshot and predicate
revisions. There is no executable evaluator registry. The snapshot describes
what observations a host would need; knowing a predicate name does not prove
that any simulator supplies them.

| Predicate | Required roles | Required quantities |
| --- | --- | --- |
| object_at_target | object, target (abstract target role) | tolerance (length) |
| relative_position | object, reference | margin (length); relation enum |
| upright | object | max_tilt (angle) |
| stack_supported | object, support | max_tilt (angle), max_gap, max_offset (length) |
| held_stable | object, holder (manipulator) | position_tolerance, angular_tolerance, duration |
| released | object, holder (manipulator) | none |
| joint_position | joint | position, tolerance (angle) |

Quantities are mandatory. Tolerances are nonnegative, held duration is
positive, and a revolute joint's normative position may be signed.
Object roles may be object/container/support; repeated roles within a binary
predicate are rejected. The frame is right-handed scene coordinates: +X right,
+Y front, +Z up; it is not Scene Engine's internal edit frame.

Conditions have bounded and/or/not forms with `args`; not has exactly one
argument. There are no callables, module paths, eval strings, liquid transfer,
force proof, visual visibility, quantifiers, or prismatic-joint predicates.
Unknown predicates are rejected when authoring a template. Evidence records
may record unsupported/unavailable predicates but cannot pass them.

## Identity and artifact ownership

| Record | Required payload besides schema_version |
| --- | --- |
| TaskTemplate | semantic_version, roles, init, goal, invariants, requirements, semantic_hash; optional temporal |
| SceneInstance | template_hash, roles, scene, embodiment, initial_state, evidence, content_hash |
| ActionWitness | template_hash, instance_hash, status, plan, execution, evidence |
| ExpansionManifest | parent, changes, operator, seed, semantic_effect, invalidates |
| ValidationCertificate | template_hash, instance_hash, witness_id, checks, policy |

A content reference is exactly `{"uri": "...", "content_hash": "<sha256>"}`.
URIs are inert identifiers; hosts must resolve them and verify referenced bytes.
Each instance role binds `{"entity_id": "...", "asset": content_reference}`.
Scene, embodiment and observed initial_state are content references; evidence
is a nonempty list of references. Use scene_instance_hash to calculate identity.
Passing template= to validate_scene_instance checks complete role membership
and semantic identity; passing instance= to validate_action_witness checks its
owner identity.

A witness plan references graph/program/integration/execution_policy/constraints
and reuses integration_fingerprint. Candidate status requires execution=null.
For succeeded/failed, execution includes env_id, episode_id, program_success,
task_success, runtime_result, task_evaluation and trajectory references.
Succeeded requires both success booleans plus execution evidence and trajectory.
This checks record consistency; it does not independently confirm report truth.

Expansion changes name a scope (language/scene/embodiment/trajectory/episode)
and child content reference. semantic_effect is preserve/mutate/unknown,
independent of scope. The manifest includes operator name/version/parameters,
nonnegative seed and invalidated check scopes. The host enforces invalidation;
the record does not execute expansion.

Each certificate check includes id, status, scope, inputs, checker, predicate,
parameters, metrics, env_id, episode_id and evidence. Inputs bind template,
instance and witness content identities. Checker/predicate records identify
name and version; predicate may be null for non-predicate checks.
A pass requires evidence and supported predicate revision where applicable.
A certificate cannot combine different env/episode rows. Policy contains id,
version and a nonempty required_checks list. certificate_passed returns false
for any absent, failed, not_run, unavailable or unsupported required check.
It never invokes a checker. The host must establish evidence authenticity,
versions and evaluation-policy applicability before accepting its result.
Cross-seed robustness aggregation remains future work, not inferred from this
single-episode policy.

## Current GenSim integration and remaining work

SemanticTaskPlanner.plan accepts optional task_template and scene_instance
together. It checks their identities and physical-object bindings, preserves
the existing E-recipes, and emits semantic_task_graph/v2 with template_hash,
instance_hash, legacy_plan_hash and status=candidate. The template is an
explicit normative input, not automatically inferred from a chosen recipe.
Only physical object/container/support roles are wired at this entry today.
Callers are responsible for supplying the actual observed instance.

Without those inputs, the existing candidate and graph/v1 behavior is unchanged.
v2 is a provenance-carrying plan candidate: it does not claim recipe satisfaction
of the template. The bundle builder and runner still refuse v2, because a full
measured SceneInstance / witness evidence qualification is not implemented.

### Opt-in bounded E2 acceptance

`task-engine prepare` and `run-all` accept `--task-template template.json`.
The template must contain one object/container role, one `upright` goal with
`0 < max_tilt < pi/2` radians, no capabilities/process constraints, and either
empty init or the exact negation of the goal as init. The selected E2 recipe
must bind one object whose upright axis is local +Z. Unsupported templates are
rejected before generation. Unsupported asset-axis bindings fail preparation.
The CLI defaults to dual_franka and rejects other profiles before generation.

The existing graph/v1 remains the executable recipe. A new strict
`gen_sim.taskspec_e2/v1` sidecar owns the explicit template and binding, bound to
the graph and referenced by exact bytes from fingerprint schema v3. Legacy
bundles without TaskSpec remain fingerprint v2. Segment upright thresholds
are derived from the template; their extra settling time remains execution
policy. No second normative success specification is authored.

After reset/settling the runner records actual poses, evaluates init, and
rejects already-satisfied goals as a **demo acceptance policy**, not task
semantics. This first host aborts the entire batch if any initial row is
invalid/unavailable/trivial, preserving per-row initial evidence; no row is
executed or saved in that case. Final goal checks are independent per-row.
Gym's `final_acceptance` hook freezes goal evidence after all segments/cleanup
and before episode metadata finalization. A successful program cannot override
a failed goal. A failed final batch is not committed as successful training data.

`initial_state.json`, `task_goal_state.json`, `initial_evaluation.json`,
`program_execution.json` and `task_evaluation.json` separate observed state,
program completion and task acceptance. Evidence references hash exact bytes.
The report deliberately records `certificate_status=unavailable`: asset-content,
full-process, actual SceneInstance identity and robustness qualification remain
unimplemented. It is not a successful certified ActionWitness. Invalid pose
rows remain unavailable, including under negation. No physical rollout or
robustness claim follows from CPU tests.

Next work belongs in the existing owners: TaskAgent seed migration and
template-derived SceneRequest, scene adapter observation/grounding evidence,
additional shared predicate measurements, step monitoring, and Workflow
certificate assembly. Full P2/P3 instance/witness qualification, E1/E4/E5
template execution, Gradio migration, decoder extension consolidation and motion
expansion hosting remain subsequent increments. Public registered protection
and empty-plan recovery are implemented in their existing execution owners.
