# ODS PickPlace redesign review

Date: 2026-10-17. Reviewed head: `d9912e5e4692f5b7109edeaab297dbdc3ad5d13d`;
showcase base: `6bd83208`. This is a review of implementation and experiment
design, followed by the proposed [redesign](REDESIGN.md). Runtime fixes and new
training experiments have not been performed as part of this review.

Implementation follow-up: both confirmed findings below are now fixed in the
worktree with regression coverage. This document preserves the review of the
pinned head; [implementation evidence](IMPLEMENTATION.md) describes current
behavior and the remaining learning/repeatability failure.

## Confirmed implementation findings

### [P2] Include the evaluated initial checkpoint in best-checkpoint selection

[Runner, line 670](/root/sources/EmbodiChain-ods-pick-place-showcase/examples/data_pipeline/ods_pick_place_showcase.py:670)
evaluates `initial_report` but initializes `best_key` to negative sentinels.
The first trained round therefore always replaces the initial checkpoint,
even when every measured metric becomes worse. This is reachable when resuming
a competent policy and undermines best-checkpoint reporting. Initialize the key
from the initial report using the same ordering as subsequent evaluations, and
test a sequence in which all new checkpoints regress. A synthetic comparison
confirmed that `(0, .1, -.3)` replaces the initial candidate even when its actual
report is `(.9, 1, -.01)`. This does not explain the poor from-scratch runs.

### [P2] Preserve invalid language indices in padded history frames

[History sampler, line 1362](/root/sources/EmbodiChain-ods-pick-place-showcase/embodichain/data_pipeline/engine/data.py:1362)
zeros every padded tensor, including `task_index` and `subtask_index`. Zero is
a valid registry entry. Passing this padded batch to `resolve_language()` thus
returns task-zero text for padding, contrary to its documented rejection of
invalid/padding indices. The language resolver checks index range but not `valid`.
Use the invalid sentinel for language IDs, or introduce explicit masked decoding
with a documented contract. Extend padding coverage to language resolution.
The lead's CPU reproduction with the existing fake engine, seed 1 and a 16-by-3
history request produced 372 padded frames, all with task ID 0; they decoded as
`unknown_task`. The PickPlace learner ignores language, so this is also separate
from its learning failure.

| Priority | Location | Defect | Impact | Recommended action |
| --- | --- | --- | --- | --- |
| P2 | [Runner:670](/root/sources/EmbodiChain-ods-pick-place-showcase/examples/data_pipeline/ods_pick_place_showcase.py:670) | Baseline metrics excluded from selection | Resume can select a worse model | Seed selection from initial report; regression test |
| P2 | [Engine:1362](/root/sources/EmbodiChain-ods-pick-place-showcase/embodichain/data_pipeline/engine/data.py:1362) | Padding converted to valid language IDs | Padding silently acquires real task text | Preserve invalid IDs or explicitly mask decoding; regression test |

## What the experiments establish

The [retained experiment report](EXPERIMENTS.md) records the following results.
Each run used 5,000 updates; selected checkpoints can precede the last update.

| Policy / data | Complete test success | Test lift |
| --- | --- | --- |
| Absolute joints / online | 0/100 | 0/100 |
| Absolute joints / frozen | 0/100 | 1/100 |
| Residual joints / online | 0/100 | 25/100 |
| Residual joints / frozen | 2/100 | 20/100 |

The physical experts qualified at 40/40 in each run. New ODS generations reached
supervised updates, model parameters changed, and lifecycle checks passed.
Consequently, failure cannot be explained simply as stale-only sampling or no
optimization. Those engineering checks do not establish a learnable control
representation, robustness after small errors, or complete-task competence.

Inspected residual rollouts include sustained grasps followed by failed transport
and release; approach failures also exist. These traces do not isolate one cause
across the population. No teacher takeover, phase input, action-order mismatch,
coordinate-frame mismatch, or teacher-only object attachment was found.

## Design gaps and hypotheses, not confirmed causal defects

1. **No physical small-data overfit gate.** Synthetic loss reduction and reliable
   experts do not show that the student can reproduce even one complete episode.
   Establish that capability before broadening scenes or streaming generations.
2. **The current objective measures the wrong scale for the residual head.**
   A residual output maps to at most 0.25 rad, but its error is divided by the
   entire joint range. For a small equal head-output error, away from clamping,
   the coefficient ratio of the gripper loss to one arm coordinate is
   `35 * (joint_span / 0.5)^2` because arm coordinates are averaged and the
   gripper weight is 5. The saved limits imply roughly 1,170–5,094. This objective
   property is verified; its causal contribution to rollout failure is not.
   First compare residual-space loss with the existing formulation on identical
   data and architecture. Audit whether expert residuals exceed the output bound.
3. **Geometry is implicit.** Current inputs contain joint state, cube pose and
   goal, requiring the network to learn forward kinematics and relative geometry.
   Add measured TCP and relative vectors. Compare against the same architecture
   before attributing gains to a new model.
4. **Timing and recovery remain unresolved.** The expert includes long close and
   settle dwells. Nearly identical current states can correspond to different
   commands around transitions. An RNN can represent elapsed history in principle;
   this is not proof that the task requires a large recurrent network. Inspect
   conflicting state/action neighborhoods and compare current state, short
   measured history, and recurrent memory. Nominal successful trajectories also
   provide little evidence of corrective behavior after student deviations.
5. **Action conversion needs physical qualification.** A Cartesian controller is
   promising but changes IK, joint trajectories and contacts. Static FK found the
   reset TCP z-axis 45.607 degrees from downward vertical, and its full rotation
   177.655 degrees from the expert grasp orientation. Position-only control that
   holds current orientation cannot reproduce the present approach. Use pose
   control, or explicitly change and qualify the reset/orientation controller.
6. **Online/frozen conflates several effects.** Replacing 32 episodes increases
   total generated diversity while discarding old data. Equal update counts do
   not isolate coverage, forgetting, or production/training overlap. Full causal
   prefixes also produced 44.8 million repeated input frames for 7.68 million
   supervised targets. Report unique data, target exposure, compute and elapsed
   time separately. Neither single-seed result proves an online advantage.

## Review summary

| Review item | Result |
| --- | --- |
| Review target | `6bd83208...d9912e5e4692f5b7109edeaab297dbdc3ad5d13d` |
| Scope | 19 changed files in the showcase range; focused runtime, task, ODS, tests and experiment-contract review; control/recording neighbors inspected |
| Execution | Multi-agent, three independent specialists plus lead verification |
| Findings | P0: 0; P1: 0; P2: 2; P3: 0 |
| Merge assessment | Changes requested; learning acceptance also remains unmet |
| Validation | Static source and trace review; architecture reviewer: 10 focused CPU sampling tests passed; lead: padding reproduction and checkpoint-selection comparison; robotics reviewer: CPU FK; official source inspection |
| Residual risks | No new live simulation or training; proposed Cartesian teacher and learner unqualified; effectiveness of each representation change unknown |

| Role | Selection reason / scope | Executor / mode | Status / evidence | Candidate disposition |
| --- | --- | --- | --- | --- |
| robotics | Frames, actions, contact, expert/student contract | `/root/ods_robotics_review`, independent | Complete; task/controller source, retained traces, cached Panda URDF FK | No confirmed code defect; G1–G4 retained as design hypotheses; position-only reset assumption rejected |
| ai, verification lens | BC loss/history, split/selection, experiments | `/root/ods_ai_review`, independent | Complete; runner, tests, saved results and official BC sources | `ai-1` accepted as P2; H1–H4 kept as hypotheses, not proven causes |
| architecture, engine/reliability lenses | Buffer publication, provenance, lifecycle, ownership | `/root/ods_architecture_review`, independent | Complete; sampling/publication code, CPU tests and padding reproduction | `architecture-1` accepted as P2; ODS/replay/coverage distinctions retained |

API/HPC considerations were inspected within these scopes; they were not
separate independent passes. No new security or agentic execution surface
warranted another specialist. No remote review was published. Context decision:
no update; this review adds a proposal, not changed routing or runtime contracts.
