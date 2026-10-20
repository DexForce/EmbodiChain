# Evaluate ODS workloads

ODS evaluation should establish three separate results: whether sampling obeys
its contracts, whether the policy learns, and what resources the complete
experiment consumes. Rate matching or bounded reuse alone establishes neither
learning quality nor an efficiency advantage. Start with the
[ODS guide](../online_data.md) and [configuration guide](configuration.md).

## What the public interfaces measure

The core package supplies counters and accounting primitives; the training
application supplies optimizer, evaluation and resource instrumentation.

| Interface | Implemented information and limits |
| --- | --- |
| `OnlineDataEngine.get_stats()` | Successful/failed episode counts, generated frames, cumulative production seconds, generations and sampled chunks. Readable-frame counts exclude the active write range. Sampled chunks are deliveries, not completed optimizer work. |
| `get_sampling_budget_snapshot()` | With quotas enabled, retained source identities, assigned/delivered/remaining allowances and locked state; lifetime delivered chunks include replaced sources. The unlocked total is only a coarse eligibility bound: the requested context or sampling mode can restrict it further. |
| `drain_lifecycle_events()` | Optional publication/eviction records, source identities, eligible anchors and host monotonic timestamps. Enable `lifecycle_event_capacity` and drain regularly, including during waits; a full log backpressures the producer. |
| `include_provenance=True` | Copied source/frame identities tied to sampled payloads. Engine-local identities support exposure accounting; they do not establish portable content equality or restore a previous cache. |
| `measure_ods_throughput()` | Calculates active and wall rates from caller-supplied counts and durations in the same absolute monotonic window. It does not collect intervals or establish simultaneous saturation. |
| `configure_ods()` | Plans settings from measured rates and an explicit qualified or provisional reuse profile. It changes no runtime state; the consumer must enforce pacing and apply the returned quota settings. |

`TensorDataSource.configure_sampling_budget()` supplies optional resident-source
protection before sampling begins. Its ledger belongs to the creating process
and thread and has no refill operation. Unrestricted resident sampling remains
a valid offline baseline.

Lifetime per-source exposure distributions, retired unused allowances, optimizer
completion, latency histograms and native resource peaks need application
instrumentation. The BC example's `usage.json`, `flow_measurements.json` and
TensorBoard tags are example outputs, not general ODS APIs or automatic engine
reports.

## Measure the deployed workload

Use the actual observation/action schema, history, prediction horizon, learner,
producer concurrency, batch size and device placement. Record startup
separately, then observe several complete refills with production and training
active. Retain raw counts and interval boundaries, not just calculated rates.

| Quantity | Recommended definition |
| --- | --- |
| Accepted production rate | Successfully published episodes divided by productive refill wall time, including failed attempts and retries. Also report whole-window supply including intentional producer idle time. |
| Anchor supply | Newly published eligible sampling anchors divided by the same production interval. Measure directly when episode lengths or eligibility vary. |
| Active learner rate | Completed chunks or optimizer updates divided by active consumer service time: selection, locked copy, feature/target conversion, transfer, forward/backward and optimizer completion. Exclude deliberate pacing and evaluation. |
| Realized learner rate | Completed work divided by the complete steady observation window, including pacing and data/quota waits. Report acquisition-inclusive rates separately. |
| Publication latency | Request-to-first-publication and full-refill distributions, with observation count, median and p95. A few observations do not establish a reliable tail estimate. |
| Exposure | Per-source selections, per-anchor draws, valid supervised-frame appearances, source age, first-use delay, generation shares and unused data at eviction. |

Record a common monotonic origin so producer and learner intervals can be
intersected. Matching windows do not prove both components were saturated:
label productive-refill rates measured under learner pacing with that schedule.
An intentionally idle producer's whole-run rate is not production capacity;
a throttled learner's wall rate is not active learner capacity.

Lifecycle `monotonic_ns` values use a host monotonic clock, while sampled
`ods_accepted_at_ns` values use Unix nanoseconds. Compute ages within one clock
domain or record an explicit alignment; do not subtract one domain from the
other. Preserve units and histogram/bin boundaries when reporting quantiles.

Measure jointly after changing concurrency, batch size, profile, device
placement or pacing. Shared CPU/GPU resources can change both rates. A synthetic
resident-data run can diagnose contention but does not replace measurement of
the complete consumer. Preserve the measurement packet, scope, raw intervals
and configurator output, including provisional-profile qualifications.

## Audit selections, supervision and retirement

One returned batch row costs one chunk selection from its source episode,
regardless of context length. Repeated selections of the same source within a
batch each count. This is distinct from unique anchors, whole-episode passes
and supervised-frame appearances: overlapping contexts can expose a target
many times, and padding contributes no valid target. Count supervision through
the sampling contract's `loss_mask`; use `valid` for real context frames.

Quota debit follows a successful copied batch under the sampling lock.
Insufficient quota cannot partially spend a batch. A later consumer failure,
dropped prefetch or unsuccessful optimizer step does not refund delivery.
Record delivered chunks and chunks used in completed optimizer updates
separately; investigate any difference before equating either with training
work.

For every protected source, audit delivered selections against its assigned
budget and hard limit. Report minimum/mean/p50/p95/maximum selections, sources
reaching the cap, unused allowance retired on replacement and allowance remaining
at shutdown.
Retain identities across slot reuse in the experiment recorder: the engine
snapshot contains current rows, not an unbounded history of retired sources.
Cross-check the sum of lifetime source deliveries against the engine lifetime
total. Report observed cap violations explicitly. An unrestricted baseline
exceeding the protected arm's budget is expected reuse, not a violation.

For `N` accepted sources, budget `K` selections per source and batch size `B`,
`floor(N*K/B)` is only an upper bound on complete updates. Retirement, eligibility
and remaining partial batches can reduce it. The initial `C`-source cache can
fund at most `floor(C*K/B)` updates before further publications are needed.
Do not raise a source cap after production ends to repair a learning-budget
shortfall discovered during the run.

## Compare online and offline systems fairly

Predeclare the question and retain a strong small-data baseline. A useful
comparison includes these arms, with task-specific sizes chosen in advance:

| Arm | Question |
| --- | --- |
| Protected online supply | What happens when training overlaps acquisition under bounded source reuse? |
| Small fixed offline corpus, unrestricted reuse | Can a simple, inexpensive corpus already solve the task? |
| Larger offline corpus, unrestricted reuse | How competitive is conventional training after larger acquisition? |
| Larger offline corpus, matched source quota | How do complete systems compare under the same exposure ceiling? |

Match the learner, initialization procedure, normalization procedure, optimizer,
batch/update budget, supervision masks and evaluation scenes. Freeze
normalization after the predeclared fitting subset. Keep accepted-data budgets
and concurrency explicit; parallel generation can benefit either system.
Equal-quota arms still differ in residence, retirement, sample order and data
availability. They measure complete systems, not isolated transport overhead.

Report cold-start and data-ready comparisons separately. A cold offline arm
pays collection, failed attempts, persistence, finalization and loading before
training; cold online latency includes producer startup and initial fill.
An existing-dataset comparison answers a different question and must identify
the acquisition costs it excludes. State whether offline data is loaded once
into resident tensors: such a run does not measure per-minibatch disk reads.
Validate persistence readback for schema, ordered actions, valid lengths,
continuity, masks and causal observation/action pairing.

Equal seeds do not establish equal numeric data in vector simulation. Record
accepted/rejected sources and compare ordered payload hashes, schemas, masks,
initial-model hashes and frozen-normalizer hashes. Claim exact-data pairing
only when these match. Otherwise disclose differences and describe the result
as a comparison of matched procedures.

Fix development scenes and learner/producer/sampling seeds before execution.
Use several independent seed blocks for variability and balance arm order to
reduce cache/order effects. Keep held-out scenes sealed during tuning; a single
development block is descriptive evidence, not held-out generalization.

## Compare checkpoints and elapsed time

Save a common, predeclared update-based checkpoint panel, including
initialization and the last completed policy if training ends early. Include
early enough checkpoints for the latency question. Evaluate identical immutable
policies on the same development scenes and retain failed or missing panels.
Report actual updates, completed chunks, valid supervised targets, final/best
quality and checkpoint-to-checkpoint regressions, alongside task success counts
and task-relevant errors.

Record process launch, application origin, first completed update, checkpoint
save completion, training completion, evaluation completion and cleanup.
Align origins explicitly; a reconstructed process-origin offset is not a direct
measurement of import latency. Keep active training, intentional pacing,
quota/data waiting, acquisition/persistence and evaluation intervals separate.
Overlapping intervals cannot be added to reconstruct total wall time.

Deferred evaluation can retrospectively identify when an evaluated qualifying
checkpoint became available. It cannot locate threshold crossing between saved
checkpoints or measure when an online evaluator certified the result. Report
the checkpoint's save-completion time and evaluation-result time separately;
keep missing timestamps unknown. A faster first update alone establishes
neither an earlier useful policy nor a shorter total experiment.

## Account for resource and instrumentation cost

Record full process wall time, allocated GPU-hours, process-tree PSS, CPU time,
native per-process GPU memory where available, and dataset/checkpoint/log bytes.
Include startup and waits; track descendants and verify shutdown. Label sampled
peaks and incomplete counters. Torch allocator memory omits simulator/native
allocations, and a bounded cache does not bound total process memory. Cache
storage allowances from configuration cover episode payloads only.

Measure accounting overhead at both online-cache and offline-corpus sizes.
Include quota checks, exposure tracking and resource sampling in a complete
consumer measurement; identical logging flags may cost more at larger source
cardinality. A counter microbenchmark cannot qualify end-to-end overhead, and
its percentage must not be subtracted from observed wall time. Retain failures,
partial artifacts, cleanup cost and separately recorded retries.

## Diagnose before retuning

| Observation | Check first |
| --- | --- |
| Learner waits without quota | Intentional pacing and source-budget policy, then measured additional supply. |
| Quota remains but sampling fails | Unlocked rows, valid windows, continuity and context/mode eligibility; distinguish quota exhaustion from terminal worker failure. |
| New sources retire unused | Residence time, cache/concurrency ratio, refill timing and completed consumption. More production can worsen this. |
| Old sources dominate | Refill gaps, accumulated allowance and training after production stops. |
| Healthy throughput but worse policy | Exposure distributions, optimization budget and fixed development rollouts. |
| More producer environments reduce progress | Joint CPU/NUMA/GPU contention and device placement. |

The [BC configuration study](https://github.com/DexForce/EmbodiChain/blob/main/examples/ods/simple_bc_demo/docs/CONFIGURATION.md)
and [offline pilot protocol](https://github.com/DexForce/EmbodiChain/blob/main/examples/ods/simple_bc_demo/docs/OFFLINE_PILOT.md)
are worked evidence with task, revision and measurement limits. Their settings
are not default reuse recommendations or proof that online training must win.
