# PR 681 Generation Foundation Hardening Design

## Objective

Harden PR #681's source-neutral trajectory-generation foundation so its public
contracts are internally consistent, candidate generation remains novel across
queue refills, failures cannot strand collection capacity, and physical host
orchestration lives outside the pure expansion package.

This change fixes the confirmed review findings without implementing the later
T6-T10 roadmap from issue #670. In particular, multi-slot execution, cache and
writer pipelines, coverage-per-cost scheduling, and observation fan-out remain
future integration work.

## Chosen Scope

The change follows a foundation-hardening approach:

- preserve `CandidateSpec`, `SourceAdapter`, `CandidateCoordinator`, and
  `GenerationSession` as the shared source-neutral contracts;
- make unsupported future configuration fail during decoding rather than
  silently selecting another behavior;
- preserve the B=1 synchronous integration slice while moving its orchestration
  to the existing motion execution owner;
- keep Task Program syntax and environment-owned backend, `num_envs`, control
  period, robot, sensor, and recorder configuration unchanged; and
- add regression tests for every corrected failure mode.

The change does not merge or copy implementation from PR #591, #594, or #653.
It leaves explicit ports for those integrations to implement later.

## Ownership Boundaries

`embodichain.lab.sim.motion.expansion` continues to own only:

- source-neutral templates and candidate values;
- deterministic trajectory expansion;
- logical candidate coordination and compatibility metadata;
- generation budgets, state transitions, measured coverage, and receipts.

`embodichain.lab.sim.motion.execution` owns the B=1 host integration:

- initial-state restoration and verification;
- physical execution dispatch;
- measured episode admission;
- persistence submission and receipt application; and
- integration-level failure cleanup.

`SingleSlotRunner`, its host ports, `FixedSceneInitialStatePort`, and
`SingleSlotOutcome` move from `motion.expansion.single_slot` to
`motion.execution`. Because PR #681 is not merged, the incorrect expansion
import path is removed instead of becoming a compatibility surface.

`GenerationSession` remains value-only and never calls a simulator or sink.
The host runner invokes its transitions but does not duplicate session state.

## Atomic Plan Integrity

The call-scoped transform boundary must preserve all `ActionPlan` invariants:

1. The engine validates the original action-produced plan.
2. The transform receives an independently owned snapshot, so it cannot mutate
   the original plan through tensor storage.
3. The returned value is snapshotted again. Reconstructing the snapshot runs
   the full `ActionPlan` constructor validation before engine-specific checks.
4. The public `plan()` and `plan_request()` entry points expose the same
   optional call-scoped transform. Execution recovery continues to omit it
   unless an owning caller explicitly supplies one.

`rebuild_plan_from_trajectory()` validates that the original plan belongs to
the supplied request and context before copying semantic metadata. Its initial
same-grid mode requires identical batch size, full joint width, environment
IDs, device, frame count, and arrival-interval tensor. Timing changes and phase
index remapping remain unsupported in this helper.

The rebuilt plan continues to flow through `AtomicAction.build_plan()`, which
recreates commands, velocity targets, tracking, named segments, dependency
bounds, effects, guards, and recovery-owned policy from typed values.

## PlanResult Source Normalization

`PlanResultSourceAdapter` converts every physically representable successful
single-row result into the stricter `TrajectoryTemplate` timing contract.

- A positive first arrival interval is preserved by prepending a duplicate of
  the first position at time zero.
- A later zero interval is removed only when its position is identical to the
  preceding retained position. A zero-duration position change is rejected as
  physically unrepresentable.
- Explicit phases are remapped over inserted or removed samples. A conversion
  that would erase a named phase is rejected.
- The template-wide `allowed_operators` value is the explicit adapter value
  when supplied, otherwise the stable union of per-phase permissions.
- Controlled-joint indices continue to cover the declared full joint order.

The normalization preserves total duration and does not silently retime onto a
different control period. Fixed-grid retiming remains owned by the existing
trajectory utilities and the caller that selects an execution grid.

## Session-Owned Generation Streams and Identity

`GenerationSession` gains a deterministic generation-stream allocator separate
from physical candidate admission. The stream key includes scene case, initial
state, source ID and revision, source unit, and operation identity. Each call
returns a private CPU generator and advances a session-owned ordinal without
consuming a physical candidate budget.

`CandidateCoordinator` uses this generator for expansion instead of allowing
`variants.py` to restart from local ordinal zero on every enqueue. It retains a
bounded fingerprint set for templates it has already enqueued, so a repeated
nominal result is removed before candidate identity allocation or physical
execution.

Accepted expanded rows are admitted through an atomic `GenerationSession`
batch-proposal operation. The operation validates the complete request and
available proposal capacity before changing ordinals, attempts, or counters.
The existing single-item `propose()` API delegates to this operation.

The coordinator no longer copies provisional geometry-family IDs from the
standalone expansion helper. Session-created physical identities receive unique
families; `CoverageIndex` remains responsible for aliasing families whose
measured geometry is actually equivalent. Consequently, different Affordance
branches cannot collide merely because they share a template ID and local
variant ordinal, while timing-only variants still converge to one measured
geometry family.

## Scheduling Configuration

Only FIFO scheduling is implemented by this foundation. Configuration decoding
rejects `coverage_per_cost` with an explicit not-implemented error. The schema
may retain cost, exploration, and reference-family fields needed by the future
T9 implementation, but no accepted value may imply behavior the coordinator
does not provide.

When T9 is implemented, the decoder can enable `coverage_per_cost` together
with confirmed-coverage scoring and an exploration floor.

## Host Lifecycle and Persistence Failures

The host runner treats each transition as a transaction around the
`GenerationSession` state:

- planning or admission failure releases a still-uncommitted candidate;
- restore or physical execution failure releases assigned/running capacity;
- structural measured-episode rejection releases a running reservation before
  returning or re-raising; and
- a final `CommitReceipt` is always applied before a terminal outcome is
  returned.

Persistence ambiguity stays fail-closed. A sink exception is not blindly
converted to a failed receipt because the write may already be durable. The
runner returns a `write_pending` outcome containing episode, commit, and
submission IDs while the session retains its reservation. The host can later
apply a reconciled idempotent receipt. A sink that knows a write definitively
failed returns an ordinary unconfirmed `CommitReceipt`, enabling the existing
retry path.

The executor contract documents that `mark_rollout_started()` must correspond
to the first actual command. The B=1 port receives a one-shot callback and calls
it at that boundary; calling it twice is rejected by the session state machine.

## Affordance Provenance

The shared provenance helper validates that selected poses, success flags,
environment IDs, and optional reference poses have the same row count and
device before serializing them. Specialized samplers call this validation even
when random sampling is disabled, preventing nominal paths from emitting
misaligned lineage.

## Showcase and Public Documentation

The Atomic tutorial recording helper accepts an optional explicit video path.
The task-environment showcase enables auto-play and routes the replay MP4 to its
`--output-dir` while preserving existing tutorial defaults for direct callers.
No tutorial-focused production test is added; existing syntax/import checks and
small helper-level tests cover the forwarding contract.

The Atomic Actions API page restores its package `currentmodule` after the
adapter-specific autosummary so engine and execution classes resolve from their
real public module. Expansion API documentation removes the relocated host
runner and motion execution documentation owns it.

## Error Handling

Validation occurs at the earliest owning boundary and must not leave partial
state:

- malformed transforms and changed control grids fail before execution;
- unrepresentable `PlanResult` timing fails before candidate generation;
- proposal-budget exhaustion fails before any candidate is enqueued;
- measured evidence errors release their reservation; and
- ambiguous persistence remains explicitly pending rather than being reported
  as a confirmed failure.

Errors retain candidate or commit identity wherever a caller needs to reconcile
or audit the operation.

## Validation Strategy

Regression tests are written before each production fix and demonstrate the old
failure. Coverage includes:

- in-place transform mutation and complete reconstruction validation;
- changed `dt`, mismatched request/plan, and valid same-grid rebuilding;
- positive first intervals, removable trailing padding, zero-duration motion,
  phase remapping, and operator permissions;
- repeated queue refill, distinct Affordance branches, measured geometry-family
  separation, atomic proposal-budget failure, and FIFO-only decoding;
- planning, restore, execution, measured admission, final write failure, and
  ambiguous write cleanup paths;
- Affordance provenance row alignment;
- public imports and Sphinx module scope; and
- showcase argument forwarding and static import/syntax behavior.

The final proportional checks are:

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion \
  tests/sim/atomic_actions
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
/root/miniconda3/envs/py311/bin/python -m black .
git diff --check
python .agents/skills/project-dev-context/scripts/context.py affected \
  --base origin/main --explain
```

Live GPU/simulator execution is reported separately if unavailable; passing CPU
tests does not claim physical validation of PR #591 or #594 integrations.

## Completion Criteria

The hardening is complete when all confirmed review reproductions pass, no
unsupported scheduling mode is silently accepted, the expansion package is
host-side-effect free, public docs resolve the relocated APIs, project context
remains accurate, and the focused/pre-commit validation selected by the project
skills passes without modifying unrelated user work.
