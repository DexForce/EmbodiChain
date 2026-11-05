# Task specification

`embodichain.gen_sim.task_spec` owns a simulator-independent normative task
contract, evidence evaluation and a content-addressed cache. Core protocol/cache
and evidence cold imports use only the standard library. Physical proof rechecks
lazily load shared compute algorithms. Task generation, scene grounding, physical
observation and execution belong to callers; importing the protocol loads none
of those owners.

## Entry points and boundaries

| Request | Owner |
|---|---|
| Task roles and required properties | `contracts.py`: `RoleSpec` |
| Normative identity and ordered milestones | `spec.py`: `TaskSpec`, `MilestoneSpec` |
| Predicate vocabulary and temporal windows | `expressions.py` |
| Structure validation and evidence evaluation | `validation.py` |
| Placement binding, physical proof checks and certificate assembly | `evidence.py` |
| Frozen graph/role/lineage projection | `embodichain/gen_sim/task_engine/evidence.py` |
| Cache reads, identity checks and atomic writes | `registry.py`: `TaskSpecCache` |
| JSON snapshots and semantic hashing | `canonicalization.py` |

Paths above are relative to `embodichain/gen_sim/task_spec/`.

Decode through `TaskSpec.from_dict`, then validate observation declarations when
required and evaluate supplied snapshots or trajectories. Both mappings and
existing instances undergo semantic-hash validation and yield independently
owned nested values. Frozen dataclasses still contain JSON containers: changing
one in place invalidates its original identity. Construct a new TaskSpec for an
intentional semantic change. Instructions, task identifiers and provenance
metadata do not participate in normative identity.

`RoleSpec.is_single_member` is a read-only query for `one` or explicit
`count=1`. It does not rewrite the authored selector, serialized fields or
semantic identity. Larger counts and `all` require complete separate binding
and evidence; they cannot be reduced to an available subset.

Milestone dependencies must be acyclic. All predicates for one milestone must
hold at the same snapshot strictly after its predecessors. Missing evidence and
incomplete temporal windows cannot certify success. Only an entirely satisfied
evaluation sets `certificate_ready`; the evaluator does not establish the
physical authenticity of caller-provided measurements.

The cache reuses a validated semantic identity across metadata differences.
It rejects corrupted or mismatched documents and publishes writes atomically.

## Physical evidence boundary

Placement binding version 2 seals the explicit `raw_snapshot` or
`historical_record` input contract alongside program/spec identities and exact
release, support and boundary occurrences. A raw input requires a complete
COM-patch acquisition channel and cannot switch validators by adding a record;
a single report cannot mix raw and historical contracts. Older binding versions
must be rebuilt from declarations rather than re-signed automatically. The frontend supplies resolved descriptors and
existing named conditions; it reads no pose buffer and computes no acceptance.
New placement bundles project the configured owner condition directly through
`project_configured_placement_evidence_binding`, without a legacy constraint
file or digest. The B0 fixture adapter separately checks its frozen source identity.
TaskSpec verifies receipts, acquisition masks, episode IDs, sampled support windows
and recorded execution prefixes. Lab owns the physical collector;
[episode recording](../env-framework/env-framework.md#expert-trajectory-contract)
supplies historical values with validity and episode/lifetime checks; the
observation manager owns their semantic schema.
Env dispatches capture/reset boundaries;
no Lab module imports TaskSpec or Task Engine.

Current goals cannot borrow recorded milestone success. The legacy instantaneous
final-support scope requires its exact measured release and actual final runtime
boundary, but not an unexecuted later wait/park policy. Failed post-policy execution
therefore does not erase an available final physical negative. Recorded milestones
still require their complete group prefix and original support receipt. Boundary snapshots do
not establish a complete control-step trajectory. The explicit legacy contact
scope and version-2 sampled-support scope retain their original meanings;
revision-1/full-footprint region declarations require their own verifier and are
rejected by this initial rigid-on slice. Raw Lab snapshots carry no accepted or
released claim. Historical claims are checked when present.

## Focused validation

Run `python -m pytest -q --noconftest tests/gen_sim/task_spec` for identity,
serialization, cache, missing-evidence, temporal-window and milestone contracts.
The [API supplement](../../../docs/source/api_reference/public_api.rst) documents
the public imports. Provider calls and simulation runs are unnecessary for this
protocol and snapshot-validation package.
