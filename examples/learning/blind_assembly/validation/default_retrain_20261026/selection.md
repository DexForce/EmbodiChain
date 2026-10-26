# Balanced checkpoint validation

Balanced development validation of every saved update; not independent-test or paper-reproduction evidence.

## Time & Memory

| jobs | elapsed_wall_seconds | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |
|---:|---:|---|---|---|
| 4 | 906.797 | unmeasured | unmeasured | unmeasured |

## Success & Other Metrics

| update | success_rate | status | convergence_budget_exhausted |
|---:|---:|---|---|
| 50 | 0.591796875 | completed | True |
| 100 | 0.681640625 | completed | True |
| 150 | 0.77734375 | completed | True |
| 200 | 0.736328125 | completed | True |
| 250 | 0.806640625 | completed | True |
| 300 | 0.830078125 | completed | True |
| 350 | 0.80078125 | completed | True |
| 400 | 0.814453125 | completed | True |
| 450 | 0.86328125 | completed | True |
| 500 | 0.84765625 | completed | True |
| 550 | 0.861328125 | completed | True |
| 600 | 0.841796875 | completed | True |
| 650 | 0.830078125 | completed | True |
| 700 | 0.8671875 | completed | True |
| 750 | 0.861328125 | completed | True |
| 800 | 0.87109375 | completed | True |
| 850 | 0.90234375 | completed | True |
| 900 | 0.875 | completed | True |
| 950 | 0.873046875 | completed | True |
| 1000 | 0.88671875 | completed | True |

## Leaderboard

| rank | update | validation_success_rate |
|---:|---:|---:|
| 1 | 850 | 0.90234375 |
| 2 | 1000 | 0.88671875 |
| 3 | 900 | 0.875 |
| 4 | 950 | 0.873046875 |
| 5 | 800 | 0.87109375 |
| 6 | 700 | 0.8671875 |
| 7 | 450 | 0.86328125 |
| 8 | 550 | 0.861328125 |
| 9 | 750 | 0.861328125 |
| 10 | 500 | 0.84765625 |
| 11 | 600 | 0.841796875 |
| 12 | 300 | 0.830078125 |
| 13 | 650 | 0.830078125 |
| 14 | 400 | 0.814453125 |
| 15 | 250 | 0.806640625 |
| 16 | 350 | 0.80078125 |
| 17 | 150 | 0.77734375 |
| 18 | 200 | 0.736328125 |
| 19 | 100 | 0.681640625 |
| 20 | 50 | 0.591796875 |

Elapsed time includes child startup, completed native evaluations and teardown. No independent-test episodes participate in selection. Candidate processes use the same device; concurrency is a resource setting, not a measured speedup claim. Raw native health reports distinguish convergence budgets from capacity loss.
