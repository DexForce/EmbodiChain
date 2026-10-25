# TaskSpec Acceptance Evidence

## Semantic Boundaries

TaskSpec declares the requested result; recorded execution supplies evidence.
Neither command completion nor video appearance alone changes the contract.
Missing evidence is unavailable, not an invented success or a measured failure.

- Events, including action-only pouring, retain their exact historical window.
  Returning the vessel or moving the receiver afterwards does not erase the event.
- Group terminal stage states use the compiled group's seeded end-step slot.
  An explicit E5 coordinated `on` placement also records its separately
  measured two-hand acquisition before terminal release/support; missing or
  reordered native boundaries remain unavailable under TaskSpec's `after` rule.
- Final states are observed before reset, not copied from an earlier milestone.
- New E3 instruction interpretation requires explicit `pour_goal_mode`. `motion`
  verifies alignment, tilt, restoration and release, never liquid transfer.
  An omitted mode enters bounded interpretation repair. Legacy authored candidates
  retain their original optional-field semantics; migration creates a new spec.

## Observation Owners

`task_engine/_task_program/` owns these application-level observations:

| Concern | Owner |
|---|---|
| Native tilt/restore occurrence | `pour_evidence.py` |
| Placement/release geometry | `placement_evidence.py`, `task_spec_runtime.py` |
| Bounded contact observations on Gym's clock | `acceptance_observation.py` |
| Coordinated acquisition and dual release | `coordinated_placement_evidence.py` |
| Stable native reference support | `stable_support_evidence.py` |
| Same-episode comparison and terminal snapshots | `acceptance_audit.py` |

These observers do not step physics, send commands or replace Task Program.
Runtime profile, assets and execution safety policies remain separate owners.

## Relation Evidence

TaskSpec planar placement uses the union of real native triangle projections.
A convex hull can invent occupied material across payload concavities. Actual
reference holes remain excluded; no blanket geometric buffer fills them.
The execution-side planar gate retains its previous default; exact silhouette
selection is local to the semantic observer.

A rejected floor projection does not prove violation of a sloping container
cavity. When released payload geometry overlaps a raised reference and only
the floor footprint fails, containment remains unavailable until a qualified
three-dimensional cavity observer exists. Clear misses, penetration and excessive
support gaps remain failures. Diagnostics retain the native projected areas.
Current planar containment records carry a per-environment `available` mask;
matchers recompute it from the measured geometry and release flags. An ambiguous
row remains unavailable without erasing other rows' final or stage observations.

An E5 composition may collect a native contact timeline through a sensor
registered in both SimulationManager and Gym's sensor dictionary. The existing
physics-substep lifecycle supplies samples; all native substeps in each control interval
are retained with source/topology revisions and explicit capacity/loss reporting.
Single-ended filtering preserves other counterparts rather than hiding them.

Stable support is distinct from COM projection inside an instantaneous contact
hull. Its v2 observer requires a complete post-release window and maximum relative
pose drift within the actual decoded placement preset's bounds. An upward load
path may traverse known dynamic rigid cargo, but must exist within one native
substep, not a union of contacts at different times. Each control interval needs
a measured root path; every substep excludes external positive-impulse contacts
of the entire coupled cargo component, including frictional side contacts.
The reference itself may rest on a table. The legacy COM classifier is
retained for audit. This is sampled native support evidence, not a continuous
friction-cone or hardware force-closure certificate.
Zero active contacts cannot distinguish native sleep from absent support and
remain unavailable unless an independent observation proves a violation.
Support by a named carrier, whole-footprint containment in a region, and an
explicit anchor over a region are different contracts, never automatic fallbacks.
Installing an E5 contact timeline does not replace an E1 `supported_region`'s
full footprint, contact-gap and exact release-window requirements with support alone.

Coordinated transport observation currently qualifies one environment and an
explicit configured `on` placement. It binds two distinct resource slots and
disjoint pad actors, actual two-hand contact in the lift window, an exact
successful dual release receipt and measured independent open-hand joints.
Standalone coordinated-hold observation keeps its original contract.
For an explicit E5 `on` relation, the formal placement evidence is native stable
support by the named reference. Planar footprint and gap are diagnostics only;
full coverage cannot certify a released payload hovering above the reference.

## Audit And Qualification

The runner evaluates legacy and current observers before reset in one episode.
The legacy result is audit-only and cannot override the current selected verdict.
State guards cover the control clock, step counts, robot qpos, rigid poses and
articulation poses/qpos. Runtime/program/spec digests bind both reports.

Native meshes and exact terminal poses are saved separately. Existing trajectory
exports can omit the seeded final slot; never substitute the previous slot.
Preserve historical specs, videos, receipts and failures. Explicit semantic
migrations record parent/new hashes and cannot masquerade as same-spec repairs.

Calibration separates human episode-level observations from independently labelled
per-predicate evidence. Synthetic negatives remain a separate test stratum; they
do not establish a real-world false-positive rate. Freeze source and input hashes
before full native regression and report program completion and strict acceptance
separately.
