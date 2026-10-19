# Exact usage-accounting validation

Correctness passed; the strict performance gate remains incomplete. The selected
C32 median overhead is +5.127%, above the unchanged 5% limit. This result limits
timing claims in the [offline comparison](OFFLINE_PILOT_RESULTS.md).
[Structured evidence](ODS_ACCOUNTING_AUDIT.json) retains the original baseline,
selected implementation, parity checks and runtime integration audit.

## Selected implementation

`UsageTracker(episode_capacity_hint=None)` chooses its counter backend once.
The default and hints below 64 preserve the original dictionary path; hints of
64 or more use compact blocks with duplicate-aware integer scatter. A CPU
16/32/64/128/256-source comparison measured block/dict median ratios of
0.880/0.875/0.687/0.534/0.418. The cutoff 64 preserves the small-cache path and
selects the first tested capacity with over 25% CPU benefit.

Blocks allocate observed source rows and frame extents, including sparse or
maximum-int64 identities. They preserve per-source dictionary views, sampled
and completed anchors, supervised/history frames, ages, provenance and eviction
metrics. The tradeoff is spare row/frame capacity and a bounded block lookup.

The runner supplies live cache capacity or the resident corpus's eligible row
count. G1-only sampling uses eligible G1 rows. Tracker replacement rebinds its
`CompactFrameSet` view; both `usage.json` output paths report `counter_backend`.
This choice changes accounting cost, not sampling or numerical results.

## Baseline and selected results

| Implementation | Sources | Median overhead | p95 overhead | Both ≤5% |
| --- | ---: | ---: | ---: | --- |
| Original dictionary | 32 | -0.569% | -0.986% | Yes |
| Original dictionary | 256 | +12.993% | +13.554% | No |
| Selected, dictionary | 32 | +5.127% | +4.685% | No |
| Selected, blocked | 256 | +3.577% | +3.924% | Yes |

The unchanged small-cache path varied from measurement noise to +5.127%.
The selected C256 implementation reduces the cardinality penalty, but the
combined gate remains incomplete. The four-arm pilot used the original
implementation and its times have not been algebraically corrected.

## Validation and measurement scope

- 960 original-reference summary/counter comparisons cover both backends over
  40 randomized seeds, sparse identities, duplicates, masks, delayed publication
  and completion after eviction. Integration-focused CPU validation passed 133 tests.
- Paired GPU runs preserve input tensors, masks, target identities, and final
  model/optimizer/RNG hashes exactly. Source and SDK hashes stayed unchanged.
- Two real single-update checks passed 19 audit conditions: the full corpus
  selects blocked counters; importing the same corpus for G1-only sampling
  selects dictionaries. Both update successfully and exit all observed children.
- ABBA blocks used 20 warmup and 500 measured updates on GPU5/CPU128–143,
  batch 256, history 2 and horizon 64, with matched CUDA completion boundaries.
- The measured path includes resident sampling, preparation/H2D and training.
  SDK loading, production, simulation, evaluation and ResourceSampler are excluded.
  The protected-offline quota-snapshot caller still needs full-path qualification.

These scoped results establish counter correctness and an implementation cost
tradeoff. They do not qualify complete-pipeline instrumentation overhead or an
ODS speed advantage. Intermediate implementation attempts and handoff notes are
removed from the current documentation; the original baseline and failed final
gate remain visible.
