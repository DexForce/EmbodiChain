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
| Orientation/joint state and milestone qualification | `condition_evidence.py` |
| Common per-environment physical reports | `evidence_report.py` |
| Explicit held-object acceptance over normalized facts | `manipulation_evidence.py`; `handover_evidence.py`, `coordinated_evidence.py` |
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
Lab validates raw receipts, acquisition masks, actor/episode identities, sampled
support windows and recorded execution prefixes before normalization. TaskSpec
checks normalized facts against the sealed task criteria. Lab owns the collector;
[episode recording](../env-framework/env-framework.md#expert-trajectory-contract)
supplies historical values with validity and episode/lifetime checks; the
observation manager owns their semantic schema.
Env dispatches capture/reset boundaries;
no Lab module imports TaskSpec or Task Engine.

HandOver and coordinated acquisition/holding use
`manipulation_evidence.py`: `ManipulationEvidenceBinding` seals the predicate,
final/recorded scope and explicit `HoldAcceptanceCriteria`. The evaluator reads
normalized `PhysicalEvidenceFrame`/`Batch` JSON, retaining validity, missing
reasons, interval coverage and per-environment episode identities. It never
selects control parts, interprets skill phases or adopts monitor thresholds.
Instance and mapping inputs both verify their original content identities.

Raw actor/query, receipt, release, contact timeline and geometry normalization
belong to `lab/task_program/integrations/evidence/`; see
[Task Program ownership](../task-programs/task-programs.md). The legacy placement,
condition, control and Pour binding APIs retain their existing declarations.
Placement and Pour receive normalized release facts; linked placement receives
normalized operation prerequisites and geometric measurements. Normative math,
role cardinality, milestone ordering and certificate assembly remain here.
`evidence_report.evaluate_physical_evidence_report` dispatches normalized
manipulation and the existing normative binding families without importing Lab
or Task Engine. Raw integration tests and frozen protocol fixtures live with Lab;
TaskSpec tests exercise task criteria, identity and normalized facts.

HandOver and coordinated holding separate protocol validation from task acceptance.
Lab's `task_program/integrations/evidence/` owns raw receipt/actor/query/clock
validation and observation requests. Its offline normalizers emit source-bound
physical frame/batch JSON without importing TaskSpec or Task Engine.
`ManipulationEvidenceBinding` seals explicit `HoldAcceptanceCriteria`, predicate,
program, episode and final/recorded scope. TaskSpec compares attachment drift,
grips, target position, orientation and observed duration against those criteria;
it never derives acceptance limits from a runtime monitor's resolved parameters.
Valid measured negatives fail; missing or contradictory acquisition stays
unavailable. Historical boundary overrides must match the verified record. Raw held poses
retain the canonical history row/episode/step identities and original native
acquisition times; final observations match source time, while recorded reads
may be collected later without equating native time to episode step count.
Existing placement/state/control/Pour APIs keep their current contracts.

Current goals cannot borrow recorded milestone success. The legacy instantaneous
final-support scope requires its exact measured release and actual final runtime
boundary, but not an unexecuted later wait/park policy. Failed post-policy execution
therefore does not erase an available final physical negative. Recorded milestones
still require their complete group prefix and original support receipt. Boundary snapshots do
not establish a complete control-step trajectory. The explicit legacy contact
scope and version-2 sampled-support scope retain their original meanings.
Explicit revision-1 support also checks release-phase windows, actor/episode
identity, stable policy clocks and raw contact patches. Declared full-footprint
regions retain the original planar `on`/`inside` checks; they do not establish
general volume containment. Transfer/coordinated-transport tasks still require
their separate operation proofs. Raw Lab snapshots carry no accepted/released
claim; historical claims are checked when present. For a recorded region, Lab
reprojects the exact raw history poses against owned geometry and the original
support snapshot, retaining availability/errors and making no live acquisition.

## Focused validation

Run `python -m pytest -q --noconftest tests/gen_sim/task_spec` for identity,
serialization, cache, missing-evidence, temporal-window and milestone contracts.
The [API supplement](../../../docs/source/api_reference/public_api.rst) documents
the public imports. Provider calls and simulation runs are unnecessary for this
protocol and snapshot-validation package.

Orientation and articulation state bindings seal criteria, native identities,
episode, format and exact execution occurrences. Current state is independent
of program success. Recorded milestones retain their complete successful prefix
and original policy/validator receipts. E6 rechecks the authored joint target,
tolerance and one-second stable window; invalid or missing evidence remains
unavailable per environment. Lab state collectors use the same canonical
ObservationManager history as support collection, preserving raw validity,
acquisition errors and temporal identities without a second acquisition path.
The Task Engine projection supplies declarations only and selects no thresholds.

`pour_evidence.py` qualifies a recorded rigid tilt/restore occurrence using sealed
source/receiver, episode, program, release and cadence identities. The original
30-degree tilt, 15-degree restore and receiver-projection bounds are unchanged;
this does not qualify fluid transfer. Missing acquisition remains unavailable.
Lab reads recorded values through the canonical ObservationManager schema/raw
history, never live poses or a second trajectory buffer. Historical control
receipts are normalized by the Lab control-evidence adapter before normative evaluation.

`control_evidence.py` seals the original v1/v2 source declarations and evaluates
normalized control-effect observations. It checks source/measurement identities,
row/episode association and channel validity without interpreting skill phases,
contact windows or raw receipts. Valid negative facts fail; invalid acquisition
remains unavailable. Raw and historical sources first pass through Lab's
`normalize_control_evidence`; the declaration and program identities remain
unchanged, while normalization has its own measurement identity.
