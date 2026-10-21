# ODS visual BC: fixed training work, data coverage and complete pipeline cost

Under this frozen workload, ODS starts parameter updates earlier than a
collect-then-train 32-episode corpus, but **does not finish the complete
pipeline sooner at comparable observed 32-source coverage**. The paired serial
control records a small complete-time reduction of about 3%, with numerical
replay limitations described below. Larger offline
corpora cover more states and cost more time. These results describe a measured
time–coverage tradeoff, not a universal ODS speedup or a successful learned
PickPlace policy.

All **30 attempts are terminal: 29 complete and one failed**. There are 23/24
qualified main runs, 6/6 qualified serial controls and 23/24 eligible paired
comparisons. Every training process, including the failed attempt, completed
the declared 128 optimizer updates. The failed attempt remains excluded; no
seed was substituted or rerun to manufacture a complete matrix.

## Frozen protocol and what is held fixed

The implementation is `9b616757900944cd1935c43171b69d569125ed66` and the protocol
is `ods-visual-fixed-budget-protocol-v2`. Its exact 29 source-file hashes and
source snapshot are retained. The formal root is
`/tmp/ods-visual-training-formal-20261021-v3`; formal v1/v2 are separate earlier
attempts, not alternative cells in this cohort.

Each attempt completes 128 real optimizer updates of 32 chunks, with two
causal history images/states and 32 predicted action targets: **4,096 trained
chunks and 131,072 valid future action targets**. FP32 AdamW, the policy,
initial model, initial-cache numerical normalization and hardware assignments
match within each vision/seed group. Data quantity, later samples and coverage
are outcomes rather than equal-data restrictions. Fixed update counts are not
asserted to be equal data epochs across differently sized corpora.

The policy uses a trainable-from-scratch ResNet10/GroupNorm/spatial-embedding
encoder adapted from the pinned RLInf reference. Both history frames and views
share the encoder. Robot state is allowed; cube pose, task phase and expert
progress are not policy inputs. The `single` mode uses a fixed third-person
camera; `dual` adds a wrist camera. The offline paths use standard LeRobot SDK
access, real CPU TorchCodec decoding of AV1 and four bounded reader workers;
they do not train from an entirely resident decoded corpus.

The main arms are Offline-8, Offline-32, Offline-128 and ODS Overlap, each with
both visions and seeds 19001–19003: 24 attempts. Six additional ODS Serial
controls replay their matching Overlap's accepted numeric trajectories and
ordered source/anchor/generation training manifest. All paths save standard
LeRobot data and the same checkpoint formats.

Within each seed/vision group, the suite alternates whether Overlap runs before
or after the offline series. Offline-8 → Offline-32 → Offline-128 remains in
ascending order and Serial always follows its matching Overlap trace. This
is not randomization or full order counterbalancing. Every path permits
bounded recording to overlap generation; Serial excludes generation/model
update overlap without artificially serializing generation against recording.

All arms in a group retain the same physical simulation/training GPUs **0/2**
and eight CPU cores **0–7**. The explicitly disclosed passive graphics driver
footprint on GPU7 is bounded pure-G memory, not additional training compute.
GPU admission and placement records are retained. This is a shared host, not
an exclusive device reservation.

## Three clocks and the complete cost boundary

- **First update:** launch to the first synchronized completed optimizer update.
- **Last update:** launch to completion of the fixed optimizer budget.
- **Complete pipeline:** launch through accepted-data recording, finalization,
  checkpoint writing and normal worker shutdown. An already requested
  generation is completed and saved after training ends. The post-update
  interval contains remaining generation, recording, checkpoint and teardown;
  it is not merely video encoder drain.

Numeric/H2 SDK validation, the full-video integrity audit and the independent
20-state policy evaluation run **after** this timed pipeline. They remain
visible as excluded costs; they are not part of a quoted pipeline-time ratio.
Audits take approximately 10.7–42.0 seconds and completed evaluations take
258.9–309.3 seconds per run. Complete study wall time also includes these
checks and earlier validation, so it is longer than the sum of pipeline times.

The table gives qualified median [minimum, maximum] complete time; the other
clock columns are qualified medians, all in seconds. Their medians need not
add together. Individual seed values and the retained failed attempt are in
`comparison.csv` and the original summaries.

| Vision | Path | n | Complete pipeline s | First update s | Last update s | Post-update s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single | Offline-8 | 3 | 136.1 [136.1, 139.5] | 120.3 | 131.5 | 4.7 |
| single | Offline-32 | 3 | 332.7 [326.3, 350.6] | 317.5 | 328.0 | 4.6 |
| single | Offline-128 | 3 | 1,116.6 [1,115.1, 1,131.9] | 1100.9 | 1111.9 | 5.2 |
| single | ODS Overlap | 3 | 382.8 [381.8, 393.8] | 79.1 | 277.4 | 108.8 |
| single | ODS Serial | 3 | 395.5 [394.8, 397.1] | 80.0 | 283.6 | 111.1 |
| dual | Offline-8 | 3 | 179.5 [177.7, 182.8] | 159.8 | 174.9 | 4.9 |
| dual | Offline-32 | 2 | 413.1 [410.1, 416.1] | 392.9 | 407.9 | 5.2 |
| dual | Offline-128 | 3 | 1,361.4 [1,346.7, 1,367.7] | 1341.2 | 1356.0 | 5.1 |
| dual | ODS Overlap | 3 | 478.3 [469.1, 481.2] | 91.5 | 326.4 | 151.7 |
| dual | ODS Serial | 3 | 491.3 [486.5, 491.7] | 87.7 | 335.6 | 154.7 |

For the matched Offline-32 comparison, the median paired first-update lead is
232.1 seconds for single and 300.9 seconds for dual; the last-update lead is
48.6 and 81.0 seconds. Complete pipeline time nevertheless increases by a
median paired **17.3%** and **16.1%**, respectively. The dual comparison has
only two qualified seeds because the third Offline-32 attempt failed.

ODS saves 40 trajectories but trains on 32; its eight untrained tail
trajectories contribute to generation, persistence and drain cost, not training
coverage. The median post-update interval is 108.8 seconds for single and
151.7 seconds for dual, compared with roughly five seconds for offline.
Earlier model access therefore does not imply earlier normal-exit completion.

## Actual trained diversity and reuse

Only data that participated in a completed optimizer update contributes to
training diversity. Initial XY uses the fixed 8×8 grid of 1 cm cells;
anchor-state XYZ uses 2 cm voxels. Unique action-target frames are deduplicated
by immutable source/frame identity, not the number of overlapping 32-step
prediction windows. Source IDs alone do not establish a new physical state.

The following coverage columns are qualified medians. Effective source counts
match the trained counts because source usage is balanced. The half-integer
dual Offline-32 voxel median comes from its two qualified runs.

| Vision | Path | n | Accepted / trained / untrained | XY cells / 64 | Anchor voxels | Unique target frames | Repeated target occurrences |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| single | Offline-8 | 3 | 8 / 8 / 0 | 7 | 123 | 3,914 | 97.01% |
| single | Offline-32 | 3 | 32 / 32 / 0 | 25 | 188 | 15,712 | 88.01% |
| single | Offline-128 | 3 | 128 / 128 / 0 | 55 | 220 | 55,433 | 57.71% |
| single | ODS Overlap | 3 | 40 / 32 / 8 | 25 | 185 | 15,714 | 88.01% |
| single | ODS Serial | 3 | 40 / 32 / 8 | 25 | 185 | 15,714 | 88.01% |
| dual | Offline-8 | 3 | 8 / 8 / 0 | 7 | 123 | 3,914 | 97.01% |
| dual | Offline-32 | 2 | 32 / 32 / 0 | 25 | 198.5 | 15,870 | 87.89% |
| dual | Offline-128 | 3 | 128 / 128 / 0 | 55 | 220 | 55,433 | 57.71% |
| dual | ODS Overlap | 3 | 40 / 32 / 8 | 25 | 185 | 15,714 | 88.01% |
| dual | ODS Serial | 3 | 40 / 32 / 8 | 25 | 185 | 15,714 | 88.01% |

ODS has C=8 shared-cache episodes, P=2 producer environments and K=128 lifetime
delivered chunks per trajectory. At the required 4,096 chunks, this cap imposes
a minimum of `ceil(4096/128) = 32` trained sources. All ODS sources respect the
cap and no untrained tail inflates coverage. K is an observable engineering
protection, not a proven learning-optimal or universally safe reuse setting.
The offline corpora may be reused to finish the identical training budget.

ODS clearly uses more initial states than Offline-8, but Offline-8 completes
sooner. ODS and Offline-32 have the same trained-source count and paired
initial XY coverage; their trajectory-voxel counts are close, not identical.
ODS does not dominate that comparable-coverage result. Offline-128 has more
sources, initial cells and unique targets, at a substantially higher cost.
No single sampled configuration is claimed to dominate all time–coverage
metrics.

## Paired time ratios and serial attribution

Ratios below are **baseline complete time / matching ODS Overlap complete
time**, summarized across qualified same-seed pairs. A ratio above 1 means
Overlap finishes sooner. These are medians of paired ratios, not ratios of
unpaired medians, and each ratio must be read beside its coverage difference.

| Vision | Baseline | Qualified pairs | Baseline / Overlap median [min, max] |
| --- | --- | ---: | ---: |
| single | Offline-8 | 3 | 0.355 [0.346, 0.365] |
| single | Offline-32 | 3 | 0.852 [0.845, 0.918] |
| single | Offline-128 | 3 | 2.917 [2.874, 2.921] |
| single | ODS Serial | 3 | 1.031 [1.004, 1.040] |
| dual | Offline-8 | 3 | 0.379 [0.373, 0.382] |
| dual | Offline-32 | 2 | 0.861 [0.852, 0.870] |
| dual | Offline-128 | 3 | 2.843 [2.816, 2.902] |
| dual | ODS Serial | 3 | 1.028 [1.021, 1.037] |

The 2.92×/2.84× ratios against Offline-128 do not establish equivalent data
coverage: that baseline trains on 128 sources and ODS on 32. The serial replay
controls hold numeric trajectories, sampled indices and completed work fixed,
providing a closer scheduling comparison without exact RGB/numerical replay.
Their median paired complete
time reduction is **3.03% single** and **2.73% dual**, with median paired
absolute savings of 12.0 and 13.4 seconds. These are small descriptive results
on a shared host, not evidence of a general speed factor. Unresolved visual
and numerical differences also prevent attributing every saved second solely
to overlap.

Actual synchronized GPU update service totals only about 5–6 seconds for
single and about nine seconds for dual across the entire 128-update run.
Generation, data handling, reuse-cap waits and final drain dominate elapsed
time. This small model/work budget provides little model compute to hide
behind acquisition. Overlapping stage services are not added to manufacture
an estimate of wall-clock savings.

## Quality and equivalence boundaries

Every qualified checkpoint was evaluated on the same 20 development scenes,
outside pipeline timing, using independent policy execution and the task's
physical success criterion. All six ODS Overlap checkpoints and all six Serial
checkpoints have **0/20 PickPlace successes**. Offline-128/dual/seed19001 has
1/20; the other 28 qualified checkpoints have 0/20. Partial lifts are not
completed successes. These 29 evaluations reuse the same fixed scene set;
they are not 580 independent draws for an aggregate statistical claim.

This budget is therefore a systems/accounting pilot, not a converged visual
BC showcase or evidence of greater learning efficiency. All successes,
failures and model versions remain recorded. The protocol was not changed to
improve quality or remove inconvenient outcomes.

All six Overlap/Serial controls pass the mandatory accepted numeric trajectory
count/digest, exact ordered training manifest, recipe, initial model and
normalization checks. Their retained raw-reference array hashes also match.
Final model state is bitwise equal for single/19002 and dual/19001, and differs
for the other four pairs. These hashes cover model weights, not checkpoint
metadata. Final-weight equality remains a separately reported diagnostic;
manifest equality does not establish exact optimizer numerical replay or full
raw training RGB identity.

The separate post-terminal diagnostic measures all six pairs rather than
extrapolating a near-identical early loss trace from one seed:

| Vision | Seed | Model bitwise equal | Maximum absolute model-tensor difference | Differing decoded YUV frame hashes |
| --- | ---: | --- | ---: | ---: |
| single | 19001 | no | 0.0006812271 | 6 |
| single | 19002 | yes | 0 | 2 |
| single | 19003 | no | 0.0293007 | 2 |
| dual | 19001 | yes | 0 | 12 |
| dual | 19002 | no | 0.02937917 | 8 |
| dual | 19003 | no | 0.01464683 | 10 |

The actual per-frame YUV checksum columns differ for **40 of 180,174** compared
camera frames. No non-finite model elements were found. These are retained
encoded-video decode and model-state observations, not a full raw-RGB
comparison. Rendering, codec and numerical contributions have not been
separated or causally assigned. The Serial arm is a numeric/index/work control,
not a claim of exact raw-image or numerical optimizer replay. See
[the detailed CPU diagnostic](terminal-diagnostic/ods-v3-terminal-diagnostic.json)
for the separate diagnostic and its scope.

Native online RGB and offline AV1-decoded RGB are not asserted to be
pixel-identical. AV1 is lossy. All six control pairs have different encoded
video-file bytes, which by itself proves neither different decoded pixels nor
different raw training pixels. A framehash text-file SHA can differ because
its headers differ; inspect actual frame-hash columns before a pixel claim.
Only the retained raw-reference arrays are compared exactly. Fixed-reference
PSNR/MAE, numeric manifests and compressed-video hashes have distinct scopes.

## The retained failed attempt

`offline32/dual/seed19003` completed training with exit 0, 32 saved/trained
sources, 128 updates, 4,096 chunks and 131,072 targets. Its complete pipeline
measurement is **405.514 seconds**. The separately persisted numeric/H2 and
full-video receipts report passed content checks. However, post-audit process
group closure failed with `Child left an owned process group alive`, so the
supervisor marked the run failed before evaluation. Its evaluation remains
pending/not executed, not passed.

Independent archival verification reconstructed its trained-source ledger and
verified the saved final model/optimizer at step 128 and full-video receipt.
Those checks confirm retained content; they do not overturn the original
process-lifecycle failure, certify an unrun evaluation, or qualify its timing
comparison. It remains excluded from aggregates and the dual Offline-32 paired
ratio. No replacement seed or later repair is counted as a successful v3 run.

## Evidence and limits

The independent audit reconstructs completed-work source counts, source
entropy, anchors/history/target-frame unions, XY/voxel coverage and the
untrained-tail exclusion from chronological events. It checks every initial/
final checkpoint on CPU, all source ledgers and all six ordered serial traces.
Raw archive members were decompressed and SHA-verified. Re-rendering from the
archived hierarchy preserves all fairness gates and reproduces aggregate
metrics and qualification counts.

The original SDK numeric/H2 audit checks each saved source and fixed RGB
references. A separate `full-video-integrity/v1` receipt sequentially
software-decodes every saved AV1 frame to YUV420p, checks per-file/per-camera
counts and binds the compressed video and metadata hashes. All 30 attempts'
saved full-video receipts were checked, including the failed run's content
receipt and all untrained tail data. Sequential bitstream completeness is not
lossless RGB fidelity or a guarantee for every possible SDK temporal seek.

SDK writer spans include create/write/encode/bookkeeping/finalize service;
reader-worker spans include SDK initialization, read/decode/conversion/
validation. Owner packing, waits and quota waits are separate lanes. Prefetch
can place worker activity before the parent requests a batch. Neither lane
is pure codec CPU occupancy or a GPU-kernel timeline.

Three seeds are a small descriptive sample. Report qualified n, seed values
and median/range rather than broad statistical significance. Partial order
balancing does not remove cache, clock/thermal, OS scheduling or foreign-work
variation. GPU utilization is whole-board NVML and includes foreign activity;
continuing-PID CPU samples can undercount newly created or disappearing
processes. Sampled memory maxima are observations, not guaranteed allocation
peaks. State grids are task-specific coverage proxies, not semantic diversity
or proof of improved policy quality.

This compares the named collect-then-train SDK paths with one bounded live ODS
recipe. It does not claim the LeRobot ecosystem cannot support streaming or
that ODS always outperforms file-backed training. Longer model workloads or
other production/reuse ratios require their own frozen experiments.

## Artifacts and reproduction

The bundle retains all 30 summaries, launch/process/initial contracts,
chronological logs, standard-dataset/checkpoint inventory, full-video and
original SDK receipts, and 29 completed evaluation reports. The 12 core figure
files are the four measured figures in PNG/SVG/PDF; original report bytes are
retained. See [the report](report/report.md), [comparison CSV](comparison.csv),
[comparison JSON](comparison.json), [independent audit](independent-audit.json),
[reproduction check](reproduction-check.json), [protocol](protocol.json),
[raw-file index](raw-log-files.json) and [artifact inventory](external-artifacts.json).

The formal inventory contains **2,270,204,459 bytes** of standard dataset
files and **3,394,539,030 bytes** of initial/final checkpoints. Including all
retained earlier diagnostics, the external inventory covers 1,180 files and
9,563,614,561 bytes, each with exact original absolute path, logical byte size
and SHA256. Large binaries remain at those recorded local paths; availability
of those external paths is required for full future model/video re-audit.
Measured dataset/checkpoint logical sizes exclude temporary disk peaks, logs
and post-timing audit/evaluation artifacts.

Formal v1 failed hardware admission; v2 was superseded after a control
recording-submission scheduling confound was found. Both remain separate
archived attempts. Development, smoke, wait, calibration and final-validation
logs are diagnostics, never replacement formal cells. The exact 29 frozen
source files are retained under `frozen-source/`; a later shutdown fix must
not be described as the code that produced v3 timings.

## Post-study shutdown repair

After the frozen study and archive verification, the current supervisor adds
a two-second normal-exit grace and records remaining PID/PPID/state identities.
A nonempty group at the deadline still fails, including zombies. This addresses
the unobserved exit-race risk without assigning an unsupported cause to the
historical failure. Its wait belongs to the corresponding stage clock.

[Separate validation](qualification/normal-exit-grace/README.md) passes three
controlled process cases and a new dual-camera Offline-4 pipeline, including
actual four-worker SDK training, full-video audit and normal closure. The
original failed checkpoint also completes a separate 20-state evaluation:
0/20 successes, 10/20 lifts. Original records and qualification remain unchanged;
these later checks are not substituted into formal aggregates or ratios.

Use [PACKAGING.md](PACKAGING.md) for archive layout and scope. Re-render figures
from the retained measurements without starting simulation or training:

```bash
python -m examples.ods.data_stream_demo.training_report \
  examples/ods/data_stream_demo/results/2026-10-21-visual-training/main \
  --output /tmp/ods-visual-report-reproduction
```

For complete source-fingerprint re-audit, use a full checkout of frozen commit
`9b6167579`; the `frozen-source/` files are byte evidence, not a complete runnable
checkout. The top-level manifest also covers the separate repair validation.
No original measurement or failed outcome was rewritten.
