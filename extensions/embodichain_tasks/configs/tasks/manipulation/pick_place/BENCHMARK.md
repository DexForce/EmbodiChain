# ODS versus offline data: training-efficiency evaluation

Status: proposed experiment protocol, 2026-10-17 (Asia/Shanghai). No benchmark
results are claimed here. This extends [the learner redesign](REDESIGN.md).
Core data-delivery and reporting modes are now implemented; see
[implementation evidence](IMPLEMENTATION.md). Exact MLP manifest replay is
verified, recurrent manifest replay is not implemented, and serial refresh is
currently an explicitly unmatched diagnostic. Age, eviction, device-wide peak,
kernel-overlap and attributed-I/O counters remain unmeasured/null. Full learning
comparisons are gated on repeatable student execution.
Run learning comparisons only after its small-data and frozen-generalization
gates pass. The objective is to measure whether ODS delivers a competent policy
sooner and with fewer resources, and identify which mechanism produces any gain.

## Questions and claims

Separate three questions:

1. From no task data, does generation overlapping training reduce time to a
   usable policy relative to collect-save-load-train?
2. For the same supervision, does shared-memory delivery reduce loading cost,
   learner stalls or storage relative to a competent offline loader?
3. Does refreshing demonstrations improve success per unit of time or resources
   relative to a frozen initial dataset?

ODS can improve one without improving the others. More optimizer steps per second
does not imply faster learning. New generations do not imply better sample
efficiency. Data freshness is diagnostic for this stationary task, not an outcome
to maximize. Report negative and neutral results as well as gains.

## Experiment arms

Use the same qualified teacher, task, feature/action schema, model, optimizer,
batch size, normalization, seed pairing, success predicate and evaluation schedule.
Default data budget is 160 accepted episodes in five groups of 32; count all
failed attempts and their costs. Default training budget is 5,000 updates of
256 valid targets, or 1,280,000 supervised targets. Budget changes must be frozen
before comparison runs and applied consistently.

| Arm | Production and training | Purpose |
| --- | --- | --- |
| ODS-overlap | Live producer, 32 shared slots, G1–G5, current bounded overlap/refill schedule | Proposed end-to-end system |
| Offline-full | Generate and persist all 160 episodes, validate/load them, then sample all 160 throughout training | Strong practical offline baseline; full-data mixing is allowed |
| Offline-matched | Same saved 160 episodes, but train with the same batch identities/order and generation exposure as ODS | Isolate delivery/storage under matched supervision |
| ODS-serial | Same live generation and batch schedule as ODS-overlap, with production and optimization serialized | Isolate production/training overlap without requiring disk storage |
| Offline-32 | Generate/persist G1 only, close producer, then train on those 32 episodes (`--mode offline-32`) | Strong small offline baseline and refresh ablation |
| Frozen-32 | Keep live G1 without refill (`--mode frozen`) | Shared-memory frozen-data diagnostic |

Primary comparisons include ODS-overlap versus Offline-full and Offline-32;
include the small offline baseline to avoid attributing a larger precollection
delay alone to better learning. Mechanism comparisons:
ODS-overlap versus Offline-matched and ODS-serial. Frozen-32 alone is insufficient
evidence of superiority to offline datasets because it has fewer unique episodes.

Offline-full may retain all low-dimensional tensors in RAM after one load;
also measure a memory-mapped/batched loader if RAM limits require it. Use contiguous
numeric storage with correct schema metadata and an optimized loader as the
reference, not per-frame Python file I/O. A repository LeRobot adapter can be an
additional integration comparison, with its own format/version recorded. Do not
charge camera/video decoding to a state-only offline policy. G1-only normalization
is shared in controlled runs; a separately labeled practical offline variant may
use all training-data statistics if that is its normal recipe.

### Content and schedule matching

Specify training reset seeds, attempt seeds, teacher/controller version and
episode order before running. Same seeds alone do not prove identical data.
Compare accepted episode lengths and hashes of policy observations/actions.
Use a content identity independent of each engine's run ID to pair datasets.
Retry budgets and acceptance rules are identical; acceptance failures cannot be
silently replaced with easier episodes.

For a strict delivery diagnostic, capture one ODS-produced corpus and its sampled
frame manifest, persist that same content, and replay the same sequence of
minibatches through offline delivery. This is an explicitly instrumented diagnostic,
not an uninstrumented ODS throughput measurement. Account for snapshot memory and
writes. Repeat the main end-to-end arms with minimal logging. If independently
regenerated content differs, report paired-reset comparisons rather than claiming
bitwise-identical supervision.

For overlap isolation, freeze a manifest from a qualification run before timed
trials. At every update the same frame must be available in both schedules.
Use only bounded snapshots of frames that would otherwise be overwritten, or
version a deterministic publication/consumption schedule applied to both arms.
Report these diagnostic changes and memory. If exact matching cannot be achieved,
omit the causal speedup claim and report descriptive timing only.

## Time and resource boundaries

Report both acquisition-inclusive and data-ready scenarios:

- **Cold end-to-end:** `t0` is launch after common software/assets are installed,
  before task-data generation, worker startup, task-specific warm-up, saving or
  loading. Include failed generation, initialization, transfers, normalization,
  training, checkpointing, scheduled evaluation and verification. Offline-full
  starts learning only after its full dataset is ready; ODS can learn after G1.
- **Data-ready:** offline data already exists. Start the timer before loading and
  learner startup. Label this separately; do not charge acquisition again. A
  frozen in-memory ODS sampler can be compared here for delivery overhead, but
  that is not a measurement of live production. Also report the actual ODS run's
  production cost rather than hiding it behind a data-ready label.

Use one physical host and a fixed maximum GPU/CPU/RAM allocation per paired run.
Begin with the intended single-GPU deployment; shared producer/learner contention
is part of the result. Any two-GPU variant has a corresponding two-GPU allowance
for offline runs and is reported separately. Record GPU model, CPU affinity,
threads, memory cap, storage, simulator/library versions and background load.
Disable Viser/video for timed measurements in every arm; run the final visual
showcase separately. Evaluation cadence and its resource use are common.

Predeclare a wall-time budget H using training-side pilot timings that allow
Offline-full meaningful optimization after collection, plus the update/data caps.
Run until the first cap, and always report which cap ended the run. If a run
finishes early, hold its best available policy for the remaining time when
computing fixed-horizon quality metrics. Do not choose H after inspecting results.

Use monotonic timestamps. CPU wall timings around asynchronous CUDA launches do
not measure completed training work: use CUDA events for device compute and
completed-update windows for throughput, with periodic rather than per-step
global synchronization. Record profiler/logging overhead. Record cache state;
do not purge shared host caches or assume that a repeated file read is cold.

## Primary metrics

Let `s_j` be complete physical success on the fixed 20-state development panel
for checkpoint j, `t_j` its evaluation-result publication time relative to t0,
and `S_best(t)` the success of the best evaluated checkpoint available by t.
Before any evaluated policy is available, set `S_best(t)=0`. All rates are fractions.

| Metric / log key | Definition | Interpretation |
| --- | --- | --- |
| Development time to 80%, `efficiency/t80_dev_seconds` | First `t_j` at which both checkpoint j and the preceding scheduled checkpoint achieve at least 16/20; includes the second evaluation's elapsed time | Main acquisition-inclusive time-to-quality endpoint; lower is better |
| Target attainment, `efficiency/reached_80` | Whether the above event occurs before the preregistered caps | Report with T80 so failed runs are not silently removed |
| Fixed-budget final success, `quality/test_success_at_budget` | Success of the dev-selected checkpoint on the reserved 100 test states, after the run ends | Quality guardrail at equal time/update/resource budget |
| Time-integrated useful quality, `efficiency/auc_success_time` | `(1/H) * integral_0^H S_best(t) dt`, using a left-constant step curve updated only when results arrive | Rewards earlier competent policies; no look-ahead interpolation |
| GPU allocation to target, `cost/gpu_hours_to_80` | Integral of the number of allocated physical GPUs from t0 to T80, divided by 3600 | Detects speed gains purchased with more hardware |
| Peak training-data footprint, `cost/data_peak_bytes` | Peak resident training-data memory and peak persistent training-data bytes, reported separately | Exposes bounded-cache/storage tradeoffs; also report total process memory |

T80 is explicitly a **development** endpoint; 20 states do not certify population
success of 80%. Two consecutive checkpoints are an operational stability check,
not independent statistical evidence. Freeze final checkpoint choices before
opening the reserved test panel. At final evaluation, test both the first
T80-confirming checkpoint (when present) and the budget-selected checkpoint;
neither test result may select a replacement checkpoint. Call a time-to-quality
gain validated only if its T80 checkpoint also meets the unchanged 80/100 test
target. Final verification time is reported separately and added when presenting
total delivery cost; never call T80 a repeatedly measured test-set endpoint.

If a run does not attain the target, record `reached_80=false` and the observed
limit/time as a right-censored result, not T80=0 or a fabricated finite T80.
Report attainment counts and individual results before medians. Compute paired
speedup `T80_offline / T80_ODS` only for pairs attaining the target; if some do not,
do not present the surviving-pair average as the overall speedup. A lower bound
requires a baseline observed for the entire stated horizon and is labeled as such.

## Diagnostic metrics

| Metric / log key | Definition / measurement |
| --- | --- |
| First update latency, `pipeline/first_update_seconds` | Launch to completion of the first optimizer update, including initialization and data readiness |
| Supervised throughput, `pipeline/targets_per_second` | Valid targets whose optimizer updates completed / elapsed training-loop wall time; include data waits, exclude evaluation; also report acquisition-inclusive targets/second |
| Exposed data-wait fraction, `pipeline/data_wait_fraction` | Union of learner intervals blocked solely on no eligible data, generation barriers or offline read/prefetch completion / training-loop wall time; exclude ordinary sampling/copy work and evaluation |
| Batch service latency, `pipeline/batch_latency_ms_p50_p95` | Sampling/read request to model-ready tensors, including indexing, cloning, transfer and exposed waits; separate component timers where available |
| Generation throughput, `producer/accepted_frames_per_second` | Accepted valid frames / producer elapsed production time, including failed attempts; also episodes/second and rejection rate |
| Actual overlap, `pipeline/producer_learner_overlap_fraction` | Time intersection of active production and completed learner-work intervals / time union; use device traces when host intervals only show queued work |
| Update/sample efficiency, `learning/targets_to_80` | Total valid supervised targets, with repetitions, through the T80 checkpoint; separate from unique data |
| Unique data to target, `learning/unique_frames_to_80` | Distinct supervised content-frame identities consumed through that checkpoint; also generated/accepted episode and frame counts |
| Accepted-data utilization, `data/consumed_unique_fraction` | Distinct accepted frames supervised at least once / all accepted frames produced by that endpoint; pending live-cache frames reported separately |
| Waste at overwrite, `data/evicted_unused_fraction` | Accepted frames evicted before any supervised use / all accepted frames evicted; undefined until an eviction occurs |
| Reuse, `data/targets_per_unique_frame` | Total valid supervised targets / distinct supervised frames; high reuse is neither automatically good nor bad |
| Generation exposure and age, `data/generation_targets`, `data/age_seconds_p50_p95` | Targets by generation; age from episode acceptance/publication to batch use, with exact timestamp convention recorded; apply the same definition to offline data |
| CPU work, `cost/cpu_core_seconds` | Sum of user+system CPU time over all owned processes; separate from CPU cores reserved |
| Memory, `cost/host_pss_peak_bytes`, `cost/gpu_peak_bytes` | Peak host proportional-set-size across processes to avoid double-counting shared pages; GPU memory per physical device, with allocator/driver definitions recorded |
| Storage/I/O, `cost/data_disk_peak_bytes`, `cost/data_write_read_bytes` | Peak retained dataset bytes, cumulative logical dataset reads/writes, and physical device I/O where attributable; checkpoints, logs and videos listed separately |

Do not sum child RSS as unique RAM or add producer and learner GPU-hours when
they share the same physical GPU interval. GPU utilization is supplementary;
simulator work can raise utilization while slowing useful learning. Optional
energy measurements integrate device power with a stated scope and sampler;
GPU energy alone is not whole-machine energy.

ODS already exposes generation, success/failure counts, generated frames,
production seconds and sampled chunks. Current runner wait timing includes
sampling/copy time, so it cannot directly populate exposed data-wait fraction.
Add reason-specific wait timers, within-episode frame identity, publication/use
timestamps and process/resource accounting. The existing counter is not a count
of unique supervised frames. Keep model-specific accounting in the runner;
generic provenance and publication facts belong to ODS. No metrics above should
be labeled implemented until their measurement path has been validated.

## Reuse, durability and reporting

Offline datasets can be reused across model seeds and experiments. Report both
single-run cold costs and K=1, 5, 10 training-job workloads:

```text
offline total(K) = collect + persist + sum(load_i + train_i + evaluate_i)
ODS total(K)     = sum(live_produce_and_train_i + evaluate_i)
```

For a parallel workload use its actual makespan and summed allocation costs;
the displayed sum above describes sequential jobs. Offline acquisition is charged
once to a reused corpus, not once per seed. Fresh ODS production is charged each
time it occurs. If ODS also archives/reuses data, label that variant and count its
storage/loading costs. Show any amortization crossover instead of claiming an
unconditional advantage. ODS without persistence provides a different durability
service; compare optional archiving separately when durable reuse is required.
Within each job, use the union of elapsed intervals: overlapped evaluation,
production, loading or optimization time is counted once in wall time. Component
work totals remain separate and must not be summed into an inflated makespan.

Low-dimensional PickPlace data is small. Disk I/O may be negligible once cached;
the strongest plausible benefit may be earlier training or bounded storage,
not higher raw SGD throughput. Measure this rather than projecting gains from
image-heavy workloads. Images remain display-only in this milestone.

Run three paired training seeds initially and expand to five for a headline
efficiency claim. Show individual runs, attainment fractions, median/range and
paired ratios; give test-success uncertainty and avoid strong significance claims
from three seeds. For multi-seed cold trials use a paired independent corpus per
trial; the shared-corpus amortization workload is a separate experiment.
Common qualification/design costs are disclosed separately, not charged only
to one arm. If required during each invocation, include them for both.

Publish these four figures and a numeric result table:

1. Best available development success versus acquisition-inclusive wall time,
   including the offline collection interval and the 80% threshold.
2. Success versus allocated GPU-hours, with fixed-budget final test markers.
3. Producer/loader/learner/evaluator timeline explaining overlap and exposed waits.
4. Peak training-data memory/storage plus amortized cost versus K.

The headline table contains arm, attained/total seeds, T80_dev, T80-checkpoint test
success, final test success, AUC, GPU-hours, wait fraction and memory/disk peaks.
Report same-budget quality differences in percentage points. Optional engineering
targets (for example, at least 20% lower cold T80) must be preregistered and are
aspirations, not assumed results. Qualify any efficiency statement by its measured
scenario: cold start, data-ready, fixed resources, or repeated data reuse.

## Implementation tasks

1. Freeze the viable learner/controller before changing data delivery.
2. Add offline export/load and content-hash validation for the identical qualified
   expert policy-request schema; retain frame/episode boundaries and provenance.
3. Add arm configuration, resource/timing events and censored-target reporting.
4. Validate accounting on a tiny known manifest: repeated samples, evictions,
   padded frames, overlapping process intervals and timeout/failure paths.
5. Pilot timing only on training/development data, freeze budgets, then run the
   matched experiment matrix and reserved final evaluation.

This authorizes a locally generated offline comparison dataset, not external
pretraining or a change to the student task. No DAgger, feedback collection,
unbounded ODS replay or universal offline-data framework is required.
