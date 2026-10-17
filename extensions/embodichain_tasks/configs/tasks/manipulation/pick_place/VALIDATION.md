# PickPlace showcase validation

Validation date: 2026-10-17 (Asia/Shanghai).

This report preserves the original fixed-window learner's results. See
[revised experiments](EXPERIMENTS.md) for the subsequent causal-history,
observation-processing and refill changes and their measured outcomes.

The engineering data/training/evaluation loop passes its checks. The reference
student does **not** meet the 80% task-learning target. These are separate results.

## Physical expert and engineering checks

- The final physical expert completed **40/40** qualification episodes using
  seeds 10000–10039. No grasp attachment, cube teleport during execution, expert
  takeover during student evaluation, or relaxed success predicate was used.
- A live forkserver probe verified robot-state width 9, action width 8, complete
  publication of two generations, and matching sampled generation IDs.
- The latest engineering smoke performed ten parameter updates, reported
  `parameters_updated=true`, and trained on generations 1 and 2.
- Frozen-data smoke consumed only generation 1. Its initial neural-network
  parameters matched the online smoke, with the same seed and update budget.
- Resume smoke continued from step 10 to step 20 under a new ODS run identity.
  This restores the learner, not the previous shared-memory buffer.
- Viser and TensorBoard returned HTTP 200 during the viewer smoke. Fixed-seed
  videos retained failed student episodes. Workers and viewer processes exited
  after validation; no stale worker or recorder queue was left running.

## Default-budget learning experiment

One online run used seed 42, 32 episode slots, five rounds of 1,000 updates,
64 sequences per batch, sequence length 32 and burn-in length 8. It used the
specified GRU-128, normalized Huber losses, gripper weight 5, AdamW at 3e-4 and
norm clipping at 1. There was no pretrained checkpoint or offline input dataset.

| Measurement | Observed result |
| --- | --- |
| Parameter updates | 5,000; checkpoint parameters changed |
| Completed ODS generations | 5 |
| Accepted expert episodes | 160 |
| Rejected expert attempts in this run | 0 |
| Generated valid frames | 53107 |
| Cumulative fill time | 840.74 seconds |
| Fill throughput, excluding initialization and idle time | 63.17 frames/second |
| Sampled sequence chunks | 320000 |
| Initial student: held-out complete success | 0/100 |
| Selected trained student: held-out complete success | 0/100 |
| Selected trained student: held-out measured lift | 6/100 |
| Learning target | **Not met** |

All five development checkpoints had zero complete successes on the same
20 development scenes. Lift rates were 0%, 15%, 15%, 5% and 5%. Selection by
complete-success rate retained the earliest tied checkpoint, step 1000. The
held-out score above therefore belongs to the selected checkpoint, not a
separate test of the step-5000 checkpoint. The final 100 seeds, 30000–30099,
were used only after training and checkpoint selection.

Training consumed frames from all five generations. Their counts were
G1: 4093952, G2: 2048000, G3: 2048000, G4: 2048000, G5: 2048.
These include recurrent burn-in frames. The latest generation was consumed
by only the final minibatch: production was slower than the intervening
updates. The report does not imply equal training exposure across generations.

Training losses decreased, but autonomous task success did not improve. A full
frozen-data learning experiment was not run, so no online-versus-frozen learning
benefit is claimed. The run uses one training seed and is not a general
algorithm benchmark.

The default-budget experiment started before subsequent diagnostic,
visualization-marker and observation-space validation refinements. Its saved
checkpoint paths and raw reports identify the actual experiment. The final
implementation separately passed the latest engineering smoke and live schema
probe.

## Software checks and environment

- Related data-pipeline/Gym/context regression selection: 430 tests passed.
- Final focused selection, including the subsequent fixes: 150 tests passed.
- Public API documentation: 2602/2602 exports covered.
- Project-context validation and diff whitespace checks passed.
- Black 26.3.1 was run across the checkout and checked before commits.
- The Sphinx dummy build succeeded with 822 warnings; this was not a
  warnings-as-errors build.

Validation used Python 3.11.14, PyTorch 2.7.0+cu128, and an NVIDIA RTX PRO 5000
72GB Blackwell. The installed editable DexSim build was
0.5.1+401968ebfd112a3c11ab9deea338e2418542fe6e. Its Newton kernel facade had
outdated exports, so validation used an isolated temporary Python overlay
against the existing native build. No fallback simulator or physics shortcut
was added to this implementation. A compatible DexSim installation is required
for ordinary execution.

## Retained artifacts

The implementation host retained the default run in
`/tmp/ods-showcase-default-budget`, including the expert qualification,
per-seed development and test reports, five checkpoints, TensorBoard events,
data manifest and summary. The latest engineering, frozen, resume and viewer
smoke runs were saved separately. The validation learning-curve figure is
`/tmp/ods-showcase-learning.png`.

`comparison_videos/` contains initial and selected-checkpoint recordings for
the same first three test seeds, 30000–30002, including all failures. Their
separate JSON reports identify the checkpoint and evaluation seeds.

See the adjacent README for reproduction commands. Failure records and the
unmet learning target are retained rather than replaced with a successful-only
recording or a weakened acceptance criterion.
