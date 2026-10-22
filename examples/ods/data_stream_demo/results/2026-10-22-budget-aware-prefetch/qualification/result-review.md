# Final budget-prefetch systems pilot: measured results and limits

All eight attempts are terminal and engineering-qualified: six strict primary results (four paired comparisons) and two separate eager diagnostics. This is seed19001 with common P4, not a multi-seed formal study. All runs requested zero policy-evaluation scenes.

Each run completed **128 real updates ×32 chunks ×32 future targets =131,072 valid action targets**. H=2, C=8, P=4, K=128; model initialization, normalization, GPU pair, eight-core allocation and declared work pass the primary pairing gates. Data amount is an outcome, not the primary matching restriction.

## Complete timing

| Vision/path | Total s | First update s | Last update s | Post-update s | Accepted / trained / untrained |
| --- | ---: | ---: | ---: | ---: | --- |
| single/Budget | 232.636 | 57.416 | 175.626 | 57.009 | 32 / 32 / 0 |
| single/Offline32 | 243.006 | 227.768 | 238.237 | 4.769 | 32 / 32 / 0 |
| single/Serial | 228.039 | 55.278 | 178.002 | 50.037 | 32 / 32 / 0 |
| single/Eager control | 277.191 | 54.932 | 172.660 | 104.531 | 40 / 32 / 8 |
| dual/Budget | 364.626 | 64.303 | 269.476 | 95.149 | 32 / 32 / 0 |
| dual/Offline32 | 385.823 | 365.213 | 380.575 | 5.248 | 32 / 32 / 0 |
| dual/Serial | 361.901 | 63.433 | 267.218 | 94.683 | 32 / 32 / 0 |
| dual/Eager control | 451.649 | 65.132 | 271.235 | 180.414 | 40 / 32 / 8 |

Total runs from pipeline launch through all accepted-data recording, checkpoint and normal worker exit; each value agrees with successful process.json elapsed. Post-timing SDK audits are excluded. Post-update time is all remaining work, not merely codec drain. Do not substitute last-update time or total experiment wall time.

| Vision | Baseline | Budget total reduction s | Budget time saving |
| --- | --- | ---: | ---: |
| single | Offline32 | 10.370 | +4.27% |
| single | Serial | -4.597 | -2.02% |
| single | Eager control | 44.555 | +16.07% |
| dual | Offline32 | 21.197 | +5.49% |
| dual | Serial | -2.724 | -0.75% |
| dual | Eager control | 87.024 | +19.27% |

Positive saving means budget finishes sooner. Thus budget beats the measured Offline32 path by **4.27% single / 5.49% dual**, and eager by **16.07% / 19.27%**. It is **2.02% / 0.75% slower than Serial**. These are exact descriptive ratios for this seed, not medians or confidence estimates.

Budget starts updates **170.35s / 300.91s** earlier than Offline32 and finishes its update budget **62.61s / 111.10s** earlier. Its remaining recording/cleanup still costs **57.01s / 95.15s**. Eager saves the same 32 trained sources but also persists eight untrained sources, with **104.53s / 180.41s** post-update tails.

## Actual trained coverage

All paths train 32 sources with effective source count approximately 32, 26/64 initial XY cells and 4,096 unique anchors. Budget, Serial and Eager each cover 192 anchor-state voxels and 16,100 unique target frames; Offline32 covers 188 voxels and 16,157 targets. Target-repeat fractions are 87.72% versus87.67%, owing to overlapping 32-step windows. Therefore this pilot demonstrates **comparable measured coverage, not a meaningful diversity increase over Offline32**. The future target counts are equal; unique target-frame counts need not be.

The first 32 accepted numeric trajectories match across all compared arms. Serial and Eager also match budget’s exact ordered episode-ordinal/anchor/generation update manifest; Offline32 has a different sample order. These are observed control checks, not proof of full raw-image identity. Native online and lossy AV1-decoded offline images remain distinct input paths. Reported model hashes match budget/Serial/Eager for single and differ for dual; they are diagnostics, not independent evidence of universal numerical replay.

## Service and critical-path observations

| Vision/path | Generation busy s | Writer episode busy s | Prep+transfer+train s | Quota / data wait s | Staging wait s | SDK busy after last update s |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| single/Budget | 155.71 | 171.50 | 10.83 | 100.09 / 0.00 | 42.76 | 52.39 |
| single/Offline32 | 156.51 | 168.09 | 12.96 | 0.00 / 10.91 | 141.18 | 0.00 |
| single/Serial | 156.70 | 168.97 | 9.94 | 0.00 / 27.09 | 114.71 | 45.55 |
| single/Eager control | 196.99 | 217.74 | 10.14 | 104.21 / 0.00 | 84.33 | 99.37 |
| dual/Budget | 183.48 | 296.60 | 19.37 | 114.76 / 0.00 | 131.91 | 90.24 |
| dual/Offline32 | 185.96 | 296.49 | 17.26 | 0.00 / 0.16 | 264.35 | 0.00 |
| dual/Serial | 180.19 | 294.47 | 18.59 | 0.00 / 1.23 | 244.49 | 89.90 |
| dual/Eager control | 230.77 | 382.79 | 19.55 | 114.74 / 0.00 | 214.65 | 175.55 |

Each service is measured by its interval union. These columns overlap and **must not be summed into wall time**. Writer episode busy excludes create/finalize; SDK busy after the last update includes them. Ready service is the parent preparation/transfer/synchronized-update path, not pure kernel occupancy; Offline batch service can include reader waits. Host overlap does not prove simultaneous GPU kernels. Real model GPU service is only about 5s single and 9s dual across the entire run.

Serial being slightly faster does not negate observed asynchronous execution. Budget generation/ready-learner service overlaps in the trace; Serial’s equivalent intervals do not. Serial forbids generation/model-update overlap while still allowing recording to overlap generation and model work; it is not a fully serialized recording pipeline. Recording/finalization can therefore determine normal-exit completion after useful model work has stopped. The small Serial-versus-budget timing differences do not isolate their cause.

- **single:** Budget’s last update is 2.375s earlier than Serial; its post-update interval is 6.972s longer. The final SDK service ends 4.460s later and subsequent exit differs by 0.137s, yielding budget’s 4.597s later completion. This locates observed clock differences; it does not identify an unmeasured contention, scheduler, codec or caching cause.
- **dual:** Budget’s last update is 2.258s later than Serial; its post-update interval is 0.466s longer. The final SDK service ends 2.604s later and subsequent exit differs by 0.120s, yielding budget’s 2.724s later completion. This locates observed clock differences; it does not identify an unmeasured contention, scheduler, codec or caching cause.

Budget eliminates the observed fifth-generation untrained tail, but recording 32 accepted sources still remains on the completion path. Its lower total versus Offline32 cannot be attributed solely to concurrent generation/training: Serial is also faster than Offline32 here. The offline path waits for dataset finalization before SDK reading/training, whereas both streaming paths can execute model updates before recording finishes. That is an observed sequencing difference, not an isolated quantitative attribution to decode speed or overlap. Treat the main comparison as an observed whole-pipeline difference and the negative serial control as an explicit limit on a parallelism claim.

## Configuration guidance and conclusion boundary

The short P2/P4/P8 calibration selects **single P4, dual P2** by the smallest measured P at ≥90% of best effective service `min(post-initial generator rate, writer episode busy rate)`. Common P4 is the declared pilot recipe, including all same-mode controls. Dual P4 reaches about 98.9% of that short calibration’s best measured rate, so it is near the plateau but is not the method’s smallest qualifying choice. Do not report this common-P4 pilot as a measurement of a dual-P2 pipeline.

Each calibration observes only one post-initial generation. Its rates are short service-capacity estimates, not long-run throughput guarantees; the hot learner demand remains much faster than protected data supply. K128 is a reuse cap, not a target to increase to manufacture utilization. No GPU job or code change was performed for this review.

This pilot supports stopping untrainable acquisition and correctly reporting earlier access, complete cost and actual coverage. It does **not** establish robust speedup, a positive overlap-only gain, policy quality or learning efficiency. One seed and two camera costs do not provide a multi-seed statistical result. Preserve the complete negative control and all raw results rather than selecting the favorable offline comparison.
