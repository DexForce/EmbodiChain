# Experiment evidence and review scope

The source tree contains the runnable experiment, qualification tools and
report generators. Generated logs, plots, checkpoints, source snapshots and
machine-specific manifests belong in run output directories rather than the
code change. Local `results/` and `outputs/` directories are ignored.

The complete pre-cleanup tree is preserved at commit
[`3829593c07392431282187ac0d810a8a4a0c5373`](https://github.com/DexForce/EmbodiChain/tree/3829593c07392431282187ac0d810a8a4a0c5373/examples/ods/data_stream_demo/results),
also named by branch `codex/ods-stream-design-backup-20261024`. The links below
use that immutable commit. Removing generated evidence from the PR does not
change experiment qualification or turn historical failures into successes.

| Study | Scope and retained result |
| --- | --- |
| [Delivery diagnostic](https://github.com/DexForce/EmbodiChain/blob/3829593c07392431282187ac0d810a8a4a0c5373/examples/ods/data_stream_demo/results/2026-10-21/RESULTS.md) | 27 primary runs and nine diagnostics; measures delivery/decoding rather than training efficiency |
| [Copy optimization](https://github.com/DexForce/EmbodiChain/blob/3829593c07392431282187ac0d810a8a4a0c5373/examples/ods/data_stream_demo/results/2026-10-21-copy-optimization/README.md) | CPU copy timings and independent-storage checks; not an end-to-end speedup |
| [Visual training](https://github.com/DexForce/EmbodiChain/blob/3829593c07392431282187ac0d810a8a4a0c5373/examples/ods/data_stream_demo/results/2026-10-21-visual-training/README.md) | 30 attempts, 29 qualified; one audit-process shutdown failure remains failed |
| [Budget-aware prefetch](https://github.com/DexForce/EmbodiChain/blob/3829593c07392431282187ac0d810a8a4a0c5373/examples/ods/data_stream_demo/results/2026-10-22-budget-aware-prefetch/README.md) | Six capacity preflights and eight pilot runs; six primary outcomes/four pairs, with eager controls separate |
| [Publication handoff](https://github.com/DexForce/EmbodiChain/blob/3829593c07392431282187ac0d810a8a4a0c5373/examples/ods/data_stream_demo/results/2026-10-23-publication-handoff/README.md) | Ten qualified pilot/diagnostic runs; earlier six-run smoke is functional qualification only |

## Measured limits

The formal visual study uses U128/B32/H2/F32: 4,096 trained chunks and
131,072 valid action targets per attempt. Its complete-time medians are
382.8/478.3 seconds for ODS overlap and 332.7/413.1 seconds for Offline-32
(single/dual; dual Offline-32 has only two qualified runs). ODS is slower in
that workload. All six ODS policies score 0/20 successes; among qualified
formal evaluations, only Offline-128/dual/seed19001 scores 1/20. These are
systems measurements, not evidence of converged PickPlace learning. The failed
attempt is Offline-32/dual/seed19003; its original evaluation was not run.
Later shutdown repair and supplemental validation do not replace that outcome.

The later publication pilot keeps the same work budget, uses one seed and
requests no policy evaluation. Both camera modes complete 48 updates before
their generation fully publishes. Post-initial publication-to-first-completed-
update p50/p95 changes from 9.662/19.650 to 0.271/0.310 seconds single, and
24.030/47.079 to 7.698/22.576 seconds dual, relative to generation handoff.
Complete times are 204.842/334.537 seconds for publication overlap,
209.339/335.005 for generation handoff, 200.088/334.321 for serial, and
216.944/350.906 for Offline-32. Serial remains slightly faster than overlap;
state coverage remains similar. This supports earlier usable data, not a
positive isolated overlap speedup or a diversity advantage.

Serial matches numeric trajectories, initialization, normalization and ordered
source/anchor work. Raw rendered RGB equality is not established, and final
model weights may differ. AV1 is lossy. Shared-host observations and one-seed
pilots do not establish a hardware-independent speedup.

## Reproduction and artifacts

Use the [README](README.md) to generate fresh data, run calibration/training,
audit saved standard datasets and render reports. The historical tree retains
exact protocol files, executed source snapshots, qualification outputs, raw
metadata archives and their hash inventories. Large datasets/checkpoints remain
at the explicitly inventoried external local paths; they were never uploaded
to Git and are not portable public artifacts. A rerun must create a fresh output
directory and record its own source, package, hardware and native-library
versions. Compare within that frozen recipe rather than mixing historical runs.

## Scope retained in PR #828

Keep the ODS flow controller, engine health/status and copy fixes with their
production tests; the simple BC caller migration; the lazy shared ResNet10
encoder and tests; and the runnable data-stream/visual-training examples with
their audit/report tools. These form active dependency chains. Keep API and
operational documentation beside the public contracts.

The largest removable group is the 1,040 generated evidence files: 174,638,781
logical bytes and approximately 4.59 million added text lines. Their backup
is linked above. If a second review split is wanted, separate the core ODS
flow/copy changes and simple BC migration from the visual encoder and showcase
as dependent changes. The shared recorder/acquisition/report helpers serve
both delivery diagnostics and visual training; deleting one entry point alone
does not remove that shared implementation. Future VLA/WAM adapters and richer
task distributions require their own implementation changes.
