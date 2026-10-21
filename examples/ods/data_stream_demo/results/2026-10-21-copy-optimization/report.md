# ODS CPU sampling-copy benchmark

Synchronous production methods; no simulation, LeRobot reader, learner or GPU. Returned output release, correctness and memory observation are excluded from latency. Repeated samples support median and observed range; no percentile-tail claim is made.

## Time & Memory

| workload | implementation | cost_time_ms | min–max ms | cpu_delta_mb | sampled_peak_delta_mb | gpu_delta_mb | peak_gpu_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| camera=0 B=1 chunk | baseline | 1.002 | 0.940–1.062 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=1 context | baseline | 0.634 | 0.616–0.662 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=1 source | baseline | 0.784 | 0.774–0.810 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=32 chunk | baseline | 1.494 | 1.459–1.575 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=32 context | baseline | 1.039 | 1.019–1.061 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=32 source | baseline | 1.184 | 1.180–1.211 | 0.0 | 0.0 | N/A | N/A |
| camera=1 B=1 chunk | baseline | 8.351 | 5.013–9.137 | 8.1 | 8.1 | N/A | N/A |
| camera=1 B=1 context | baseline | 9.235 | 9.051–9.634 | 16.4 | 16.4 | N/A | N/A |
| camera=1 B=1 source | baseline | 14.341 | 13.814–17.364 | 0.0 | 0.0 | N/A | N/A |
| camera=1 B=32 chunk | baseline | 255.414 | 252.036–260.332 | 264.0 | 527.6 | N/A | N/A |
| camera=1 B=32 context | baseline | 406.041 | 402.308–423.986 | 264.0 | 526.1 | N/A | N/A |
| camera=1 B=32 source | baseline | 556.343 | 554.100–577.127 | 264.0 | 527.0 | N/A | N/A |
| camera=2 B=1 chunk | baseline | 15.356 | 14.990–16.005 | 32.7 | 32.7 | N/A | N/A |
| camera=2 B=1 context | baseline | 24.510 | 24.138–24.744 | 32.7 | 32.7 | N/A | N/A |
| camera=2 B=1 source | baseline | 33.343 | 33.300–34.480 | 32.7 | 32.7 | N/A | N/A |
| camera=2 B=32 chunk | baseline | 507.132 | 506.097–508.127 | 528.0 | 1053.7 | N/A | N/A |
| camera=2 B=32 context | baseline | 812.297 | 802.499–818.335 | 528.0 | 1055.8 | N/A | N/A |
| camera=2 B=32 source | baseline | 1113.004 | 1101.229–1149.415 | 528.0 | 1053.0 | N/A | N/A |
| camera=0 B=1 chunk | optimized | 0.707 | 0.681–0.745 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=1 context | optimized | 0.368 | 0.356–0.393 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=1 source | optimized | 0.415 | 0.403–0.455 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=32 chunk | optimized | 1.095 | 1.079–1.119 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=32 context | optimized | 0.928 | 0.915–0.974 | 0.0 | 0.0 | N/A | N/A |
| camera=0 B=32 source | optimized | 1.011 | 0.993–1.051 | 0.0 | 0.0 | N/A | N/A |
| camera=1 B=1 chunk | optimized | 4.542 | 4.400–6.751 | 0.0 | 0.0 | N/A | N/A |
| camera=1 B=1 context | optimized | 8.695 | 8.652–9.013 | 0.0 | 0.0 | N/A | N/A |
| camera=1 B=1 source | optimized | 8.920 | 8.733–8.983 | 0.0 | 0.0 | N/A | N/A |
| camera=1 B=32 chunk | optimized | 169.096 | 167.666–172.444 | 264.0 | 264.0 | N/A | N/A |
| camera=1 B=32 context | optimized | 318.266 | 315.115–319.680 | 264.0 | 264.0 | N/A | N/A |
| camera=1 B=32 source | optimized | 318.719 | 316.182–322.302 | 264.0 | 264.0 | N/A | N/A |
| camera=2 B=1 chunk | optimized | 11.313 | 11.139–11.783 | 16.1 | 16.1 | N/A | N/A |
| camera=2 B=1 context | optimized | 19.816 | 17.033–20.139 | 16.2 | 16.2 | N/A | N/A |
| camera=2 B=1 source | optimized | 19.897 | 19.789–20.475 | 16.2 | 16.2 | N/A | N/A |
| camera=2 B=32 chunk | optimized | 346.618 | 345.215–349.341 | 527.9 | 527.9 | N/A | N/A |
| camera=2 B=32 context | optimized | 645.841 | 637.809–657.122 | 528.0 | 528.0 | N/A | N/A |
| camera=2 B=32 source | optimized | 641.989 | 638.478–654.979 | 528.0 | 528.0 | N/A | N/A |

## Success & Other Metrics

| workload | implementation | success_rate | all-leaf and RNG equivalence | padding frames | padding correct | owned after replacement |
| --- | --- | --- | --- | --- | --- | --- |
| camera=0 B=1 chunk | baseline | 100% | True | 0 | True | True |
| camera=0 B=1 context | baseline | 100% | True | 3 | True | True |
| camera=0 B=1 source | baseline | 100% | True | 3 | True | True |
| camera=0 B=32 chunk | baseline | 100% | True | 0 | True | True |
| camera=0 B=32 context | baseline | 100% | True | 60 | True | True |
| camera=0 B=32 source | baseline | 100% | True | 60 | True | True |
| camera=1 B=1 chunk | baseline | 100% | True | 0 | True | True |
| camera=1 B=1 context | baseline | 100% | True | 3 | True | True |
| camera=1 B=1 source | baseline | 100% | True | 3 | True | True |
| camera=1 B=32 chunk | baseline | 100% | True | 0 | True | True |
| camera=1 B=32 context | baseline | 100% | True | 60 | True | True |
| camera=1 B=32 source | baseline | 100% | True | 60 | True | True |
| camera=2 B=1 chunk | baseline | 100% | True | 0 | True | True |
| camera=2 B=1 context | baseline | 100% | True | 3 | True | True |
| camera=2 B=1 source | baseline | 100% | True | 3 | True | True |
| camera=2 B=32 chunk | baseline | 100% | True | 0 | True | True |
| camera=2 B=32 context | baseline | 100% | True | 60 | True | True |
| camera=2 B=32 source | baseline | 100% | True | 60 | True | True |
| camera=0 B=1 chunk | optimized | 100% | True | 0 | True | True |
| camera=0 B=1 context | optimized | 100% | True | 3 | True | True |
| camera=0 B=1 source | optimized | 100% | True | 3 | True | True |
| camera=0 B=32 chunk | optimized | 100% | True | 0 | True | True |
| camera=0 B=32 context | optimized | 100% | True | 60 | True | True |
| camera=0 B=32 source | optimized | 100% | True | 60 | True | True |
| camera=1 B=1 chunk | optimized | 100% | True | 0 | True | True |
| camera=1 B=1 context | optimized | 100% | True | 3 | True | True |
| camera=1 B=1 source | optimized | 100% | True | 3 | True | True |
| camera=1 B=32 chunk | optimized | 100% | True | 0 | True | True |
| camera=1 B=32 context | optimized | 100% | True | 60 | True | True |
| camera=1 B=32 source | optimized | 100% | True | 60 | True | True |
| camera=2 B=1 chunk | optimized | 100% | True | 0 | True | True |
| camera=2 B=1 context | optimized | 100% | True | 3 | True | True |
| camera=2 B=1 source | optimized | 100% | True | 3 | True | True |
| camera=2 B=32 chunk | optimized | 100% | True | 0 | True | True |
| camera=2 B=32 context | optimized | 100% | True | 60 | True | True |
| camera=2 B=32 source | optimized | 100% | True | 60 | True | True |

## Leaderboard

| rank | algorithm | overall_success_rate | cases |
| --- | --- | --- | --- |
| 1 | baseline | 100% | 18 |
| 2 | optimized | 100% | 18 |

Memory is a separate warmed pass: CPU retained RSS delta and maximum observed increment from 1 ms process-RSS polling, not an allocator peak. Persistent cache bytes and software/hardware/affinity are recorded in the JSON. Shared-memory fixture construction and dirty-tail/continuity setup are excluded. No worker writes during timing, so producer contention and training overlap remain unmeasured. Implementations use the same script and seeds; checkout order alternates by camera count. First-implementation hashes are the comparison reference, not independent semantic proof.
