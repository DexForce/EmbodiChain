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
| Placement bindings and physical proof checks | `evidence.py` |
| Complete finite-count placement groups | `placement_group_evidence.py` |
| Orientation/joint state and milestone qualification | `condition_evidence.py` |
| Complete finite orientation/part groups | `condition_group_evidence.py`; common member aggregation in `_group_evidence.py` |
| Operation-bound historical Press/Twist qualification | `control_evidence.py` |
| Rigid Pour motion qualification and exact occurrence selection | `pour_evidence.py` |
| Transfer/receiver hold and coordinated single-object proof | `handover_evidence.py`, `coordinated_evidence.py`; shared raw/receipt checks in private proof helpers |
| Data-only held acquisition requests from validated receipts | `observation_requests.py` |
| Common per-environment physical reports | `evidence_report.py` |
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

Milestone dependencies must be acyclic. All predicates for one milestone must
hold at the same snapshot strictly after its predecessors. Missing evidence and
incomplete temporal windows cannot certify success. Only an entirely satisfied
evaluation sets `certificate_ready`; the evaluator does not establish the
physical authenticity of caller-provided measurements.

The cache reuses a validated semantic identity across metadata differences.
It rejects corrupted or mismatched documents and publishes writes atomically.

`transport_evidence.py` qualifies a past, exact coordinated-acquisition
occurrence from its recorded prefix and native contact identity. It retains the
original B=1/env0 scope and does not require an attachment to survive a later
release. Missing/invalid history remains unavailable; a past event cannot certify
a final held state or placement. Shared held proof primitives stay in their
existing owner, and Lab remains responsible for acquiring facts.

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
and recorded execution prefixes. Lab owns the physical collector; the injected recording reader owns historical
pose access and episode/lifetime checks. Env only assembles recording components;
no Lab module imports TaskSpec or Task Engine.

Current goals cannot borrow recorded milestone success. The legacy instantaneous
final-support scope requires its exact measured release and actual final runtime
boundary, but not an unexecuted later wait/park policy. Failed post-policy execution
therefore does not erase an available final physical negative. Recorded milestones
still require their complete group prefix and original support receipt. Boundary snapshots do
not establish a complete control-step trajectory. The explicit legacy contact
scope and version-2 sampled-support scope retain their original meanings.
The explicit revision-1 scope additionally verifies release-phase windows,
actor/episode identity, stable policy clocks and raw contact patches. Declared
full-footprint regions preserve the original planar `on`/`inside` checks; they
are not general volume containment. Task families requiring handover or
coordinated transport requires its explicit operation linkage: transfer precedes
receiver release, or both acquisitions and releases precede sampled support.
Raw sampled timelines require explicit native/program acquisition identities;
planar and directional scopes declare their exact native geometry mode/owners.
Complete finite-count placement groups require every member and occurrence,
without copying missing members or accepting a partial group. Raw Lab snapshots
carry no accepted/released claim; historical claims are checked when present.
Recorded region geometry is recomputed from seeded poses and the original
native load, then compared with the original policy snapshot, without sampling
current contacts or substituting live poses.

The Task Engine `project_condition_evidence_binding` adapter only projects
frozen role/lineage and explicit compiled occurrence descriptors. It receives
already resolved owner criteria and native IDs; it does not select thresholds
or acquire physical state.

Orientation and articulation endpoint bindings seal their native entity,
criteria, episode and input-format identities. Current state is evaluated
independently of program success. Recorded milestones require the complete
successful occurrence prefix and original policy/validator receipts. E6 retains
the exact semantic slide target, joint-coordinate tolerance and one-second
stability contract; missing or failed milestone evidence remains unavailable.
Raw joint receipts additionally bind their actual environment-step acquisition
clock and boundary. A common report owner combines supported physical families
without filling absent predicates or historical boundaries.

Finite condition groups validate the original full TaskSpec/program and exact
count-to-execution/member bijection before reusing per-member state proofs.
Logical/native articulation membership includes part/joint, so one articulation
can supply distinct members. Missing any required member remains unavailable;
complete valid final negatives remain failed. `one` and explicit `count=1` use
the same single-member gate without changing role serialization or semantic
hashes. No unobserved `all` universe is inferred. E1 final upright/vertical facts
remain independent of placement; recorded E1 permits only the original upright
fact. Original E6 normal multi-part milestones retain their unavailable boundary.

Control qualification consumes a plain sealed source route and exact execution
occurrences, not a live Press/Twist factory. It retains source sampling-window,
cleanup and clearance requirements and the explicitly declared coarse/calibrated
criterion. Only explicit historical receipt replay is currently supported; raw
control acquisition is rejected before observation until the complete window
contract is implemented. A final control goal remains an operation-event proof,
not a replacement for a current joint-position measurement. Failed or missing
operation receipts remain unavailable.

Rigid Pour proof preserves the explicit 30-degree tilt/15-degree restore proxy,
receiver projection and measured release. Its source/receiver and compiled
occurrence descriptors are plain declarations; it does not consume Task Engine
graph objects or live Lab factories. New raw observations require their geometry
channels, episode, cadence, compiled program fingerprint and exact
window/boundary identities. The event proof
cannot establish ``poured`` fluid transfer. Recorded milestones additionally
require the successful operation prefix; valid measured negatives remain failed
and missing channels remain unavailable per environment.

HandOver and coordinated-hold proofs retain the original single-member scope
(`one` or finite `count=1`); larger counts and unobserved universes remain
unavailable. They seal exact resources/occurrences and recheck attachment, hold,
raw pose/grip or historical receipt facts independently. Fresh grip requests
carry original endpoint/query ordinals, not an inferred hand or cached trace.
`held_observation_request` returns only plain acquisition declarations from
validated receipts; Lab performs the reads. These predicates cannot certify
transported cargo, placement or fluid transfer. The report dispatch adds only
these supported proof families; the TaskSpec does not import live factories.

## Focused validation

Run `python -m pytest -q --noconftest tests/gen_sim/task_spec` for identity,
serialization, cache, missing-evidence, temporal-window and milestone contracts.
The [API supplement](../../../docs/source/api_reference/public_api.rst) documents
the public imports. Provider calls and simulation runs are unnecessary for this
protocol and snapshot-validation package.
