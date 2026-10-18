# Initial ODS tuning findings

Status: development evidence, not a final recommended configuration. The live nine-cell screen and three-seed confirmation are separate gates. See [the protocol](TUNING.md).

## Controlled initial reuse and replacement

Two learner seeds (12001/12002), one fixed two-generation corpus, 20 development scenes per policy. Each fork restores the same model, optimizer and CPU/CUDA RNG state. All evaluations use a fresh B=1 world per checkpoint. Formal comparison panels remain untouched.

| Initial G1 reuse | Continuation reuse | Before continuation, seeds 1 / 2 | Continue G1, seeds 1 / 2 | Replace with G2, seeds 1 / 2 |
| ---: | ---: | --- | --- | --- |
| 8 | 32 | 17/20 / 19/20 | 16/20 / 20/20 | 20/20 / 20/20 |
| 32 | 32 | 18/20 / 20/20 | 19/20 / 19/20 | 19/20 / 20/20 |
| 128 | 32 | 19/20 / 20/20 | 20/20 / 17/20 | 17/20 / 20/20 |
| 128 | 128 | 19/20 / 20/20 | 18/20 / 17/20 | 20/20 / 20/20 |

The continuation128 extension was declared separately after the initial diagnostic; its starting model and normalizers match the initial128 checkpoint bitwise. It uses the same development scenes and is exploratory, not an independent confirmation.

High initial reuse did not itself lower the pre-refresh score here. After a short continuation on G2, seed12001 dropped from19/20 to17/20; after longer G2 training both seeds reached20/20. In contrast, extended training on G1 ended at18/20 and17/20. These results do not support a blanket claim that more updates or any replacement inevitably harms learning. They suggest measuring the amount of training each newly published source actually receives, alongside total reuse. Two seeds and one acquired corpus are insufficient for a universal setting.

The fixed-training-context audit also shows that old-source fitting error can increase without a corresponding decrease in physical success. Teacher-state error is a diagnostic, not a replacement for autonomous task evaluation. Raw records: [TUNING_CAUSAL_RESULTS.json](TUNING_CAUSAL_RESULTS.json); local fixed-context audits and plot are under `/tmp/ods-usage-study/`.

## Scheduling and accounting checks

The bounded asynchronous P2/C4/r1 smoke generated8 accepted episodes and2662 eligible anchors, completed41 B64 updates (2624 chunks), consumed both generations and stopped with38 unusable residual credits. The producer and evaluator closed cleanly. No policy-quality claim is made from this smoke.

The same run exposes unequal episode exposure: the first four episodes received29,34,612,605 completed chunk selections. An aggregate r budget is not a per-episode cap; sequential replacement changes residence time. Configuration guidance must therefore include distributions, coverage and generation-specific use, not just a global mean.

## Production setting for the next screen

These are one-seed, two-generation engineering pilots, with only one development
scene and the same pre-optimization instrumented source across P. They establish
throughput, not learning quality. P8 is best within this tested grid; larger P
has not been ruled out. Per-P qualification scenes follow independent seeded
vector streams and are not identical physical initial states across P.

| P | Concurrent G2 trajectories/s | Time to first update s | Active update capacity/s |
| ---: | ---: | ---: | ---: |
| 1 | 0.122 | 287.0 | 61.8 |
| 2 | 0.229 | 167.4 | 60.5 |
| 4 | 0.439 | 104.5 | 63.3 |
| 8 | 0.813 | 81.1 | 63.8 |

The corrected-source P8 reference measured269.3 new anchors/s and64.52 active
updates/s. With B256 and r32, the sustainable estimate is33.66updates/s, not the
observed two-generation loop average59.35updates/s: that short run spends its
initial cache credits. The longer C/r screen is required to measure steady use.
G1/G2 consumed334,849/346,111 chunks in this reference. Its one-scene success is
an engineering check, not quality confirmation. Raw accounting and profile data
are in [TUNING_SYSTEM_RESULTS.json](TUNING_SYSTEM_RESULTS.json).

Reuse must be specified with batch size and chunk horizon: lowering B at fixed
chunk count creates more optimizer updates. If the winner lies on a search
boundary, describe it as best within the tested range rather than a global
optimum. Full three-seed stability confirmation remains required.

## Readiness and remaining gates

- P1/P2/P4/P8 each passed40/40 physical expert qualification. Production throughput repetitions and actual-learner pilots are recorded separately.
- GPU accounting before selected-ID reuse exceeded the5% instrumentation gate. The landed implementation passes a real shared-memory consumer ABBA check: mean block medians13.024→13.495ms (+3.62%), mean block p95 values13.699→14.256ms (+4.07%). Inputs, supervised identities and final weights match bitwise. This includes lock/copy, raw observation conversion, H2D and training; the producer is idle after filling, and both arms synchronize every update. It does not measure active-writer contention or changing the legacy50-update synchronization cadence.
- Timing pilots use one development scene and a reused evaluator; they establish plumbing/throughput only. Their post-shutdown valid-row metric and throttle bookkeeping limitations are explicitly retained in the pilot report.
- Full parameter screening and independent three-seed stability confirmation remain required before publishing a recommended P/C/B/r recipe.
