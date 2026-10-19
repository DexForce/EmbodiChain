# Exact usage-accounting optimization: final review handoff

**Correctness passed; the strict performance gate remains incomplete.** The final selected-backend C32 p50 overhead is +5.127%, narrowly exceeding the unchanged 5% limit. It is not rounded to a pass. No additional experiments or performance retries follow this final run.

## Final API and integration

`UsageTracker(*, episode_capacity_hint: int | None = None)` chooses its exact counter backend once. The default and hints below 64 use the original plain-dictionary counters; hints of 64 or more use compact blocked counters with duplicate-aware integer scatter. `counter_backend` is a read-only property returning `dict` or `blocked`. Publications do not trigger migration, and the hint changes no sampler, metric, budget or numerical result.

The CPU crossover study tested 16/32/64/128/256 sources. Block/dict median accounting ratios were 0.880/0.875/0.687/0.534/0.418; the smallest case showed startup variability. The conservative cutoff 64 retains the original small-capacity path and selects the first measured case showing over 25% CPU benefit. This is a workload-specific engineering choice based on CPU evidence, not a universal optimum or a GPU gate-selected threshold.

Root integration after the final profile binds live ODS to `buffer_size`, resident offline data to its eligible valid row count, and frozen/offline-32 imports to eligible generation 1 rows. Replacing the tracker also rebinds its CompactFrameSet view. Both usage.json write paths include the selected backend. No running pilot source was changed.

## All retained complete-consumer results

| Variant | Resident episodes | p50 overhead | p95 overhead | Mean overhead | Both p50/p95 ≤5% |
|---|---:|---:|---:|---:|---|
| Original | 32 | -0.569% | -0.986% | +0.285% | True |
| Original | 256 | +12.993% | +13.554% | +12.732% | False |
| Candidate 1: dense block accumulation | 32 | +4.401% | +2.295% | +4.250% | True |
| Candidate 1: dense block accumulation | 256 | +6.546% | +8.886% | +6.579% | False |
| Candidate 2: duplicate-aware integer scatter | 32 | +4.161% | +6.428% | +4.021% | False |
| Candidate 2: duplicate-aware integer scatter | 256 | +3.320% | +1.591% | +3.215% | True |
| Fixed capacity hint: dict below 64, blocked otherwise | 32 | +5.127% | +4.685% | +5.075% | False |
| Fixed capacity hint: dict below 64, blocked otherwise | 256 | +3.577% | +3.924% | +3.539% | True |

The C256 cardinality penalty is substantially reduced. The final C32 result uses the original dictionary path and still narrowly fails the median gate. Earlier failures are retained, including the original C256 penalty, candidate1 C256 failure and candidate2 C32 tail failure. No result is replaced by a favorable retry. These component measurements do not establish an end-to-end ODS/offline speed advantage or combined telemetry cost. ResourceSampler remains excluded.

## Implementation and validation

CPU profiling attributed 0.351s of 0.445s across 240 original sample/complete pairs to repeated per-source loops. Compact blocks keep at most 256 nearby source identifiers, allocate only observed rows, grow row capacity geometrically and frame width in 64-element increments. Sparse or maximum-int64 source IDs do not allocate a maximum-ID-sized array. Exact per-source dictionary views, all four frame/anchor counters, age histograms, first-use metrics, provenance, eviction behavior and denominators remain intact. This uses some spare row/frame capacity and a 256-entry lookup per touched block to avoid repeated Python loops.

- 84 focused tests passed across usage, added accounting tests and benchmark helpers.
- 960 complete-summary/raw-counter comparisons passed: 40 randomized seeds exercised both final backends against the frozen original implementation.
- Coverage includes multiple runs, sparse/max-int64 IDs, duplicates, padding and target masks, sample-only batches, delayed lifecycle publication and completion after eviction.
- All measured ABBA arms match input tensors/masks/target IDs, initial optimizer/RNG and final model/optimizer/RNG. Final selected-backend hashes also match the original revision.
- Source and SDK hashes were unchanged during every profile. Root integration edits occurred after the final profile completed. GPU5 was released after each run.

## Measurement scope

Every complete-consumer case used the same GPU5 UUID, NUMA-local CPU 128–143, three idle/VRAM checks, source SDK dataset, G1 normalization, model initialization, batch 256, history 2 and prediction horizon 64. Each ABBA block includes 20warmup and 500measured updates, totaling 2,000 measured updates per capacity. Both arms use unrestricted sampling and identical per-update CUDA completion. OFF retains compact exact target counting; ON adds complete UsageTracker accounting using the selected fixed backend. SDK loading, simulation, production, evaluation, resource sampling and final reporting are outside the measured update. Boundary GPU inventories are not continuous host-isolation guarantees.

## Evidence

- `RESULTS.json`: combined machine-readable outcome and unchanged gate.
- `backend-crossover.json`: CPU backend choice evidence.
- `oracle-selected-backends.json`: 960 old-reference comparisons.
- `consumer-selected-backends/`: final raw update latencies, hashes, source diff and harness.
- `consumer-candidate/`, `consumer-candidate2/`: all earlier measurements and failed gates.
- `REPORT-candidate2.md`, `RESULTS-candidate2.json`: preserved pre-final handoff.
- `baseline-hotpaths.txt`, `candidate1-hotpaths.txt`, `sparse-counter-micro.json`: optimization evidence.
- `reference_usage.py`, `candidate1_usage.py`, `candidate2_usage.py`: preserved source references.
- Original harness failure and baseline telemetry measurements remain under `/tmp/ods-offline-pilot/overhead-3cd6379a*`.

Implementation changes owned by this agent are `_ods_pick_place_usage.py` and `tests/data_pipeline/test_usage_accounting.py` in `/root/sources/EmbodiChain-ods-offline-accounting`. Root owns the subsequent integration edits. No commit was made by this agent.

## Runner integration and interpretation

The online tracker receives the configured live buffer capacity. A resident
offline corpus receives its number of eligible nonempty rows; frozen/offline-32
filters this hint to generation 1. Construction selects the backend once. Both
normal and cleanup `usage.json` reports record `counter_backend`; sampler
selection, quota enforcement and lifecycle denominators remain unchanged.

The original C32 path measured close to noise in the first profile and +5.13%
in the final profile. This variation reinforces why identical telemetry flags
alone cannot establish timing fairness. The strict gate remains incomplete.
These profiles isolate resident consumption and exclude ResourceSampler,
production, SDK loading, simulation and deferred evaluation. They cannot be
subtracted from the four-arm wall times or establish an ODS speed advantage.

Validation after integration: 133 focused tests passed, Black 26.3.1 passed,
and public API coverage remained 2614/2614. No public API changed. The context
review selected no update: the owning online-sampling guidance remains accurate
and these private counter storage choices do not alter its contracts.

Two additional real-runtime single-update checks on GPU5 passed 19 audit
conditions. Loading the complete 256-trajectory SDK corpus selects `blocked`;
loading that same corpus in G1-only mode selects `dict` and consumes only G1.
Both retain 256 source records, complete exactly one update/256 chunks, keep
the frozen source unchanged, use only the selected GPU and exit all observed
children. They validate runner integration, not learning quality or throughput.
Raw evidence: `/tmp/ods-offline-accounting-audit/integration-smokes-v2/AUDIT.json`.
The first harness attempt failed before child launch because an unused packet
metadata file was missing; its artifacts remain in `integration-smokes/`.
