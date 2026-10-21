# ODS visual training: fixed-work time and diversity

Qualified runs: **29/30**; qualified formal pairs: **23**. Failed, unsupported and unqualified runs remain in report.json and are excluded from aggregate values and paired time ratios.

Each result completes the declared optimizer-update, batch and valid-target budget. Data amount and final coverage may differ. Ratios compare total pipeline time, including acquisition, recording, decode, actual training, checkpoint and normal teardown; they do not establish equal policy quality or a universal ODS speedup.

## single

Values are median [minimum, maximum] across qualified seeds.

| Path | n | Total s | Trained sources | Effective sources | XY cells / 64 | Unique targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Offline-8 | 3 | 136 [136, 140] | 8 [8, 8] | 8 [8, 8] | 7 [7, 8] | 3.91e+03 [3.9e+03, 3.98e+03] |
| Offline-32 | 3 | 333 [326, 351] | 32 [32, 32] | 32 [32, 32] | 25 [23, 27] | 1.57e+04 [1.56e+04, 1.6e+04] |
| Offline-128 | 3 | 1.12e+03 [1.12e+03, 1.13e+03] | 128 [128, 128] | 128 [128, 128] | 55 [53, 56] | 5.54e+04 [5.54e+04, 5.56e+04] |
| ODS overlap | 3 | 383 [382, 394] | 32 [32, 32] | 32 [32, 32] | 25 [23, 27] | 1.57e+04 [1.55e+04, 1.6e+04] |
| ODS serial replay | 3 | 396 [395, 397] | 32 [32, 32] | 32 [32, 32] | 25 [23, 27] | 1.57e+04 [1.55e+04, 1.6e+04] |

| Baseline / ODS overlap | Paired seeds | Total-time ratio median [min, max] |
| --- | ---: | ---: |
| Offline-8 | 3 | 0.355 [0.346, 0.365] |
| Offline-32 | 3 | 0.852 [0.845, 0.918] |
| Offline-128 | 3 | 2.92 [2.87, 2.92] |
| ODS serial replay | 3 | 1.03 [1, 1.04] |

| Path | CPU mean / assigned capacity % | Simulation GPU board mean % | Training GPU board mean % |
| --- | ---: | ---: | ---: |
| Offline-8 | 46.1 [46, 46.6] | 19.3 [19.2, 20.5] | 2.52 [2.16, 3.13] |
| Offline-32 | 54.2 [54.1, 54.5] | 32.5 [30.8, 33] | 1.31 [0.937, 1.39] |
| Offline-128 | 58.6 [58.6, 58.7] | 37.9 [37.8, 38.3] | 0.307 [0.135, 0.383] |
| ODS overlap | 56.2 [55.9, 56.4] | 35.2 [34.6, 35.6] | 1.17 [0.906, 1.59] |
| ODS serial replay | 55.6 [55.6, 55.7] | 34.1 [33.5, 34.5] | 1.16 [0.966, 1.58] |

CPU values use continuing-PID sampled interval deltas normalized by the assigned cores; new or disappeared processes may be undercounted. GPU values are whole-board NVML samples and include foreign work; they do not attribute kernel utilization to this run. Missing observations remain unknown (—), not zero. See raw resources for sample counts and process-coverage gaps.

| Path | Sampled PSS max GiB | Sampled owned GPU max GiB | Dataset MiB | Checkpoints MiB |
| --- | ---: | ---: | ---: | ---: |
| Offline-8 | 8.19 [7.93, 8.26] | 5.9 [5.9, 5.9] | 9.83 [9.79, 10] | 106 [106, 106] |
| Offline-32 | 8.32 [8.02, 8.33] | 5.9 [5.9, 5.9] | 37.8 [37.5, 38.6] | 106 [106, 106] |
| Offline-128 | 8.37 [8.32, 8.5] | 5.9 [5.9, 5.9] | 150 [149, 151] | 106 [106, 106] |
| ODS overlap | 9.38 [9.03, 9.39] | 7.84 [7.84, 7.84] | 47.3 [47.1, 47.9] | 106 [106, 106] |
| ODS serial replay | 9.37 [9.03, 9.39] | 7.84 [7.84, 7.84] | 47.3 [47.1, 47.9] | 106 [106, 106] |

Memory maxima are sampled observations, not guaranteed allocator/process peaks. Disk sizes are logical finalized dataset and initial/final checkpoint bytes; they exclude temporary disk peaks, logs and post-timing audit/evaluation artifacts.

## dual

Values are median [minimum, maximum] across qualified seeds.

| Path | n | Total s | Trained sources | Effective sources | XY cells / 64 | Unique targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Offline-8 | 3 | 179 [178, 183] | 8 [8, 8] | 8 [8, 8] | 7 [7, 8] | 3.91e+03 [3.9e+03, 3.98e+03] |
| Offline-32 | 2 | 413 [410, 416] | 32 [32, 32] | 32 [32, 32] | 25 [23, 27] | 1.59e+04 [1.57e+04, 1.6e+04] |
| Offline-128 | 3 | 1.36e+03 [1.35e+03, 1.37e+03] | 128 [128, 128] | 128 [128, 128] | 55 [53, 56] | 5.54e+04 [5.54e+04, 5.56e+04] |
| ODS overlap | 3 | 478 [469, 481] | 32 [32, 32] | 32 [32, 32] | 25 [23, 27] | 1.57e+04 [1.55e+04, 1.6e+04] |
| ODS serial replay | 3 | 491 [487, 492] | 32 [32, 32] | 32 [32, 32] | 25 [23, 27] | 1.57e+04 [1.55e+04, 1.6e+04] |

| Baseline / ODS overlap | Paired seeds | Total-time ratio median [min, max] |
| --- | ---: | ---: |
| Offline-8 | 3 | 0.379 [0.373, 0.382] |
| Offline-32 | 2 | 0.861 [0.852, 0.87] |
| Offline-128 | 3 | 2.84 [2.82, 2.9] |
| ODS serial replay | 3 | 1.03 [1.02, 1.04] |

| Path | CPU mean / assigned capacity % | Simulation GPU board mean % | Training GPU board mean % |
| --- | ---: | ---: | ---: |
| Offline-8 | 47.8 [47.7, 47.8] | 17.1 [16.7, 18.1] | 4.79 [4.2, 5.39] |
| Offline-32 | 57.4 [57.4, 57.5] | 30.6 [30.5, 30.7] | 1.88 [1.85, 1.91] |
| Offline-128 | 62.5 [62.5, 62.6] | 36.8 [36.7, 36.8] | 0.643 [0.627, 0.691] |
| ODS overlap | 59.8 [59.7, 59.9] | 33.1 [32, 33.2] | 1.9 [0.981, 2.32] |
| ODS serial replay | 58.9 [58.8, 59.1] | 32.1 [32.1, 32.3] | 1.39 [1.14, 1.76] |

CPU values use continuing-PID sampled interval deltas normalized by the assigned cores; new or disappeared processes may be undercounted. GPU values are whole-board NVML samples and include foreign work; they do not attribute kernel utilization to this run. Missing observations remain unknown (—), not zero. See raw resources for sample counts and process-coverage gaps.

| Path | Sampled PSS max GiB | Sampled owned GPU max GiB | Dataset MiB | Checkpoints MiB |
| --- | ---: | ---: | ---: | ---: |
| Offline-8 | 10.7 [10.5, 10.8] | 5.91 [5.91, 5.91] | 14.6 [14.6, 14.9] | 110 [110, 110] |
| Offline-32 | 10.8 [10.6, 11] | 5.91 [5.91, 5.91] | 56 [55.4, 56.5] | 110 [110, 110] |
| Offline-128 | 11 [10.8, 11.2] | 5.91 [5.91, 5.91] | 219 [218, 221] | 110 [110, 110] |
| ODS overlap | 12 [11.9, 12] | 10.1 [10.1, 10.1] | 69.3 [69, 70] | 110 [110, 110] |
| ODS serial replay | 12 [12, 12.1] | 10.1 [10.1, 10.1] | 69.3 [69, 70] | 110 [110, 110] |

Memory maxima are sampled observations, not guaranteed allocator/process peaks. Disk sizes are logical finalized dataset and initial/final checkpoint bytes; they exclude temporary disk peaks, logs and post-timing audit/evaluation artifacts.

## Figures and interpretation

![End-to-end time and actual trained-state coverage](01_time_diversity.png)

![Coverage after completed optimizer updates](02_coverage_growth.png)

![Measured pipeline stages](03_pipeline_timeline.png)

![Lifetime source reuse, including retired sources](04_lifetime_reuse.png)

Coverage uses completed training only. Initial XY bins are fixed 1 cm cells over the task's 8 cm × 8 cm reset area; anchor-state voxels are 2 cm. Source identity growth alone is not physical diversity. Overlapping prediction windows share target frames and are deduplicated by immutable frame identity. Published but untrained data does not increase these metrics.

AV1 decoding is lossy. Compare source quality audits and held-out evaluation alongside timing. Differences between online raw pixels and decoded pixels prevent a claim of identical optimization trajectories. Serial trace replay isolates overlap only when its source/anchor manifest audit passes. A faster small offline run or broader large offline corpus is a valid result.

SDK recording spans cover real create/write/encode/bookkeeping/finalize service; SDK temporal fetch spans include worker initialization, dataset reads, decode, conversion and validation. Neither lane claims pure encoder/decoder CPU work or GPU kernel activity. Owner packing/copy and waiting remain separate. Intervals may overlap and are not summed into total time. Timeline rows use the first observed qualified seed per vision, without selecting the fastest run. PNG, SVG and PDF versions share measured inputs.

Formal ratios require matching initial-model/normalization SHA256, initial-C numeric prefix, camera order, declared/completed training budget, ordered primary GPU UUIDs and the same eight CPU cores. Retained supervisor launch masks/thread limits, placement guards, process exit and clocks must agree. New formal inputs additionally require chronological sdk-worker-service/v1 intervals and a revisioned full-video-integrity/v1 receipt for every saved frame; the original numeric/H2 audit remains required. Full-video YUV sequential integrity is distinct from lossless RGB fidelity or every SDK temporal seek. Missing evidence blocks claims; smoke/calibration or legacy CPU rendering fixtures are not formal comparisons. Post-timing corpus audit and evaluation durations are excluded from pipeline time.

## Excluded paired claims

- dual / offline32 / seed 19003: requires exactly one qualified run per arm and seed.

## Retained unqualified runs

- `/tmp/ods-visual-training-formal-20261021-v3/offline32/dual/seed19003/summary.json`: status=failed, no explicit qualification pass.
