# Trajectory expansion

Return to [motion planning](motion-planning.md) for facade and timing ownership.
The implementation lives in `embodichain/lab/sim/motion/expansion/`.

## Templates, sessions and hosts

Expansion Profiles are callable-free task-local resources, decoded separately
from environment and Task Program deployments. `SourceAdapter` converts
handwritten, planner or grounded Atomic sources to immutable templates;
`CandidateCoordinator` generates, deduplicates and queues candidates.
`lab.expansion.ExpansionSession` is the sole owner of identities, RNG, budgets,
reservations and receipts; the motion session module re-exports it. Motion keeps
trajectory coverage adapters. Hosts own physical slots, stepping, reset,
observations, acceptance, media and persistence; expansion imports no Gym.

`motion/execution.py` provides injected restore/executor/sink ports and drives
session transitions for a B=1 host. Episode-level `expansion.combined` enumerates
reference-family, affordance and trajectory recipes, selects visual profiles and
records `schedule_digest`; it does not acquire physical authority. Configured
spatial deduplication tolerances must be smaller than their perturbation scale
and are forwarded by the Task Program integration.

## Operators and permissions

`variants.py` combines joint-space, task-space and manipulability factors through
the same host-supplied kinematics callable. Joint residual, via-point,
null-space and timing operators preserve phase endpoints with zero endpoint
residual derivatives and never edit contact/hold phases. Limit rejection covers
moved joints, so an untouched measured reference slightly beyond a limit does
not invalidate every candidate.

Only one joint-path operator runs per proposal; a list of methods produces
separate variants. Task-space projection precedes joint edits and is repeated
after them; retiming runs last. Null-space edits use the projected path's
Jacobian and hold declared task rows to first order only. Callers own Jacobian
frames/columns; a fully constrained task rejects null-space expansion.

`task_space.py` applies bounded root-frame pose residuals to explicitly
permitted free phases and projects editable interiors through fixed-step DLS.
Global and phase permissions must both allow edits. Conversion grants new
permissions only for caller-named `editable_phase_ids`; extraction intersects
existing permissions. Locked samples and endpoints retain reference qpos.
Results include pose residuals and success evidence; hosts reject failed rows
and independently validate collision, dynamics and tasks.

`stitching.py` admits replacement segments only when phase semantics, motion
kind, joint layout, controlled joints and tensors agree. It retains source
lineage and unique phase identities, sharing one index at continuous boundaries.
Discontinuities require an injected bridge planner. The minimum-jerk fallback
has sampled velocity/acceleration limits, without a collision certificate.
Stitched templates enter the ordinary source-adapter and host paths.

## Sampling, rejection and coverage

`expand_trajectory_variants` deduplicates measured geometry/timing for one scene.
Only `ProposalRejected` counts as rejection; malformed inputs propagate.
`max_attempts` counts all numerical draws, including guidance's inner proposals
and valid unselected draws. Omitted config selects supported default factors;
explicit config is never overwritten and unsupported enabled factors raise.
Approach/grasp-pose factors produce Cartesian targets that require replanning.

Atomic preparation preserves the original batched grasp RNG scope. Finite
position search reuses its pool; approach search draws one pool per direction.
Each successful row freezes independently. Per-row resets or filtering before
ranking/NMS change populations. Freeze grasp, direction and position before
comparing planners and review population fingerprints after selection changes.
`utils.nms.pose_nms_scope` supplies a nested, task-local pairwise tile budget;
explicit call arguments override it. It preserves threshold arithmetic and
selection order without changing an imported backend function. Copied planning
contexts inherit the budget while independent threads keep their own default.
Explicit goal-conditioned endpoint searches also change the population and
retain rejected goals; they establish neither paths nor physical success.

`manipulability.py` consumes supplied Jacobians and compute Yoshikawa scores.
Band normalization uses a positive reference per initial state. Optional bounded
guidance ranks local proposals; coverage-only mode retains the original draw.
Per-band quotas and admission use measured observations, not planned scores.
The coordinator and session must share normalization. Guidance metadata remains
separate from path, dynamic and task validation.

## Consumers

- `examples/expansion/trajectory/` exercises production operators and supplies
  the CPU browser visualizer and analytic stitching/IK examples.
- The Atomic example host compiles permitted free segments through the shared
  plan adapter and owns row-local replay/provenance. Lift retiming preserves
  callback alignment; contact, hand and protected sample boundaries stay exact.
- Task Program collection retains its own lifecycle; see
  [Atomic execution](../atomic-actions/execution.md),
  [Real2Sim](../data-pipeline/real2sim.md) and
  [Expansion Engine](../expansion-engine/expansion-engine.md).

Use `tests/sim/motion/expansion/` for algorithm/contracts coverage and host tests
for stepping, acceptance and persistence. Examples are not production test APIs.
