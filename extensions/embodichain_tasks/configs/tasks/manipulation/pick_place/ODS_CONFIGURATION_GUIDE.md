# Configure ODS from measured supply and training demand

Status: configurator, strict CPU shared sampling quotas, measurement helpers and
PickPlace integration are implemented; profile qualification and cross-rate
validation remain separate work. The two explicit
user inputs are data-generation throughput and model-training throughput.
The system reads batch size, sampling units, trajectory metadata, resource limits
and a versioned reuse-protection profile from the measurement JSON. Missing
latency metadata must be measured before configuration; the first implementation
does not silently invent it or launch an automatic saturation sweep. It derives pipeline settings within that protection profile.
Existing measurements are examples, not defaults for other tasks or hardware.
See [quota and learning validation](ODS_QUOTA_TESTS.md) and the
[completed offline comparison](OFFLINE_PILOT_RESULTS.md).

## 1. Measure the actual concurrent workload

Use the deployed observation/action representation, chunk horizon, learner,
producer and device placement. Record startup separately, then observe several
complete refills with producer and learner both active. For each candidate
producer concurrency P and batch B, record interval boundaries and raw counts.

| Quantity | Definition and purpose |
| --- | --- |
| Accepted episodes/s | Successful published episodes divided by measured production wall time, including failures/retries. Attempted trajectories are not useful supply. |
| Published anchors/s, lambda | Newly published eligible chunk-start frames divided by the same time. Measure this directly; episode length can vary. |
| Active learner capacity, u_cap | Completed optimizer updates divided by active training time, including sample selection, locked copy, feature/target conversion, H2D, forward/backward and optimizer completion. Exclude deliberate pacing waits and evaluation. Measure while production is active. |
| Realized updates/s, u_wall | Completed updates divided by the complete steady observation window, including intentional waits and refill stalls. Report acquisition-inclusive rate separately. |
| Publication latency | Request-to-first-publication and full-refill latency distributions; report median/p95 and observation count. Few measurements do not establish a reliable p95. |
| Exposure and coverage | Per-episode anchor draws and supervised-frame appearances; p50/p95/max, unused frames at eviction, age, first-use delay and generation shares. |
| Resource cost | Process-tree sampled PSS, native GPU memory if measured, CPU usage, GPU allocation time and topology/affinity. Missing measurements remain unknown. |

Distinguish production capacity while a refill is active from effective supply
over the whole window, which also includes intentionally idle producer time.
Report both. Do not diagnose a generation bottleneck using an idle producer's
whole-run rate, or derive capacity from a learner that is already throttled.
Synthetic resident training is a contention diagnostic; it does not substitute
for the real consumer pipeline. Repeat the joint measurement after changing P,
B, device placement or the pacing schedule: sharing a GPU changes both rates.

Capture a shared monotonic time origin for producer events and learner intervals
so concurrent windows can be intersected. Productive-refill rates measured under
intentional learner throttling describe that schedule. Record the contention
and pacing state rather than treating these rates as saturation capacity.

## 2. Automatic configuration with protected trajectory budgets

Normalize the two explicit inputs to g = accepted published trajectories/s and
t = completed training chunks/s under concurrent load. If the training probe
reports updates/s, read its batch size B and convert t=B*u_cap. Do not use a
rate already reduced by credit waiting as learner capacity. Observation frames/s
cannot be treated as chunks/s without the actual sampling/target-count metadata.

One trajectory selection means one sampled chunk anchored in that trajectory.
Selecting a trajectory ten times in one batch counts ten selections, not one.
This is not ten complete trajectory passes. Frame supervision appearances and
normalized anchor reuse remain separate diagnostics, especially when lengths
or prediction horizons differ.

The task/model/sampler profile supplies three integers:

- K_min and K_max: the recommended budget range, qualified for that profile.
- K_hard: an absolute per-trajectory cap, with 1 <= K_min <= K_max <= K_hard.

The profile is a protection policy, not something inferred from throughput.
Record its version, task, model, history/horizon and trajectory-length range.
Without a matching qualified profile, a provisional profile must be explicitly
labeled and validated; the configurator must not invent a scientifically safe
range. Changing hardware changes throughput, not this hard cap. Material changes
to trajectory lengths or learner semantics require profile revalidation.

```text
balance budget       K_balance = t / g                       [chunks/trajectory]
recommended budget   K = clamp(floor(K_balance), K_min, K_max)
training target      t_target = min(t, g * K)                [chunks/s]
update target        u_target = t_target / B                 [updates/s]
producer demand      g_target = t_target / K                 [trajectories/s]
training duty estimate        = t_target / t
producer duty estimate        = g_target / g
```

Invalid, zero or stale rates require an explicit diagnosis/calibration state;
do not divide by zero or silently remove protection. Rates are local estimates
under the measured schedule, not guaranteed constants. Re-measure after pacing
changes because shared CPU/GPU contention can change both capacities.

When training is much faster than supply, cap K and pace the learner. When supply
is much faster, preserve K_min as the planned budget and pace production through
refill requests/backpressure. K_min is not a guarantee that every trajectory is
used that often: replacement, failures or early termination can leave it unused,
which must be reported. Do not generate an unbounded backlog to meet the plan.

Illustrative arithmetic only: g=1 trajectory/s, t=1000 chunks/s, profile
K_min=200, K_max=600, K_hard=1000 gives K=600, training600 chunks/s and estimated
60% learner duty. The system must not raise K to1000 just to keep the GPU busy.
These profile values are hypothetical, not validated Franka defaults.

Automatic outputs include K, its profile/cap, pacing targets, initial fill/cache
size, refill trigger, initial training budget and end-of-supply stopping rule.
Batch size and producer concurrency remain at the measured settings unless the
system runs an additional calibration curve: two scalar throughputs cannot
predict unmeasured B/P behavior. Optimizer/model hyperparameters are outside this
pipeline configurator's scope.

The long-run g*K supply estimate assumes assigned budgets are actually usable.
Full-cache replacement may retire unused allowances. Measure this utilization
and lower the realized supply estimate when necessary; never transfer discarded
allowances to an already exhausted trajectory to recover utilization.

## 3. Enforce the bound at sampling time

The enforced invariant for every immutable (run_id, episode_id) is:

```text
delivered_chunks + reserved_chunks <= assigned_K <= K_hard
```

Assign K when the trajectory is published. Automatic adaptation can change the
budget of future trajectories, but cannot replenish existing budgets to avoid
waiting. Counters belong to trajectory identity, never a reusable cache slot.
Different generations cannot transfer their unused quota to an exhausted
trajectory. Track delivered chunks and optimizer-completed chunks separately.

Filter exhausted trajectories and reserve the whole batch's per-trajectory
multiplicities atomically with selection/copy under the existing shared lock.
Concurrent consumers must share the same counters. A batch must not exceed a
quota merely because it contains repeated selections of the same trajectory.
Convert reservations to delivered counts before exposing data. Return a failed
copy's reservation only when no data was delivered; training failure after
delivery does not refund sampling quota. Cache replacement and counter updates
must use the same identity/lock boundary.

If there is insufficient unlocked quota for a full batch, report budget
exhaustion separately from a worker failure or invalid sampling windows. The
showcase may wait with a bound, request refill, or stop after supply ends; it
must not silently oversample. Preserve the current immediate-error sampling
contract for existing callers and make quota protection explicitly configurable.
Resume with a new ODS run creates new source identities; replaying old source
identities must restore their usage state rather than reset their allowance.

For a finite run, completed updates cannot exceed total delivered chunks/B,
and total delivered chunks cannot exceed the sum of assigned trajectory
budgets. Initial credit permits a burst above the long-run rate; it must not be
reported as sustainable throughput. After production ends, spend only remaining
trajectory budgets and stop when no full batch can be funded.

## 4. Derive the remaining pipeline controls

The two throughput inputs are sufficient for the balance calculation, but
cache/refill settings also require latency, memory and writer-batch metadata.
Read these from the environment or a short automatic probe rather than adding
manual tuning inputs or guessing them from the two rates.

For a latency margin s from the system profile and measured p95 first-publication
latency L, a starting refill threshold is Q_lead=ceil(t*L*s) remaining usable
chunk selections. For fixed K and writer batch P, a provisional cache size is
the next multiple of P at or above max(2P, P+ceil(Q_lead/K)). This reserves a
writer batch plus quota covering latency. Limit it by measured memory capacity;
if it does not fit, lower active training demand or report infeasibility, never
raise the sampling cap. This is a sizing heuristic to validate, not a guarantee
against arbitrary production stalls. Subsequent publication spacing and minimum
trajectory diversity also constrain the profile.

Initial training budget is floor(sum of initial assigned_K/B). Runtime refill
decisions use remaining unlocked quota and observed latency, not a fixed number
of updates. If demand reduction is needed for latency protection, pacing must
bound short-window consumption too; an average sleep after an unlimited burst
does not protect the available quota during refill.
The implemented consumer spaces full-batch service starts and permits one
immediate initial batch; short-window accounting therefore includes a one-batch
burst allowance. Per-trajectory count caps remain strict without this allowance.

| Control | How to choose and verify it |
| --- | --- |
| Producer environments P | Increase only while joint accepted-anchor throughput improves enough to justify memory/CPU/GPU cost. Verify expert success and learner capacity again; producer-only scaling is insufficient. |
| Batch B | Measure active updates/s and chunks/s together. Holding r and data volume fixed, reducing B increases the number of optimizer updates. Recheck learning behavior and latency; B is not merely a throughput knob. |
| Cache episodes C | This study uses C >= 2P so a writing batch leaves another batch available. This is a starting constraint, not a universal sufficient size. Check unlocked eligible anchors, memory, unused eviction and data age. Larger C increases initial-fill cost and may lengthen residence; it does not create new supply. |
| Trajectory sampling K | Derive from the throughput ratio within the profile's recommended range; enforce each assigned budget and the hard cap. Audit normalized anchor/frame exposure as well. |
| Refill lead time | Estimate remaining funded training time as credits/(B*u_cap). Request before it falls below a measured publication-latency margin. Distinguish first publication from full refill; the first batch may not sustain demand until the next batch. |
| Initial fill and end of supply | Bound updates on the initial cache. When production ends, make remaining credit and the stopping rule explicit; continued updates on the last cache are a separate fixed-data phase. |

Compatibility mode: the example also retains aggregate publication credits via
`--usage-schedule anchor_budget --anchor-reuse R --usage-statistics`, together
with `--producer-envs P`, `--buffer-size C`, `--batch-size B` and an explicit
`--max-generated-episodes D`. Every new eligible anchor grants r credits; each
completed optimizer batch spends B. Credit capacity is max(B,r*A0), overflow is
reported, and a remainder smaller than B cannot fund another update. Very small
initial budgets therefore require checking actual grants and batch rounding.

Current refill lead time is1.25 times measured p95 first-publication latency,
with a30-second cold prior. These are implementation heuristics, not validated
universal constants or independent CLI settings. An excessive cold prior can
replace initial rows too early. Record the prior and actual latency separately.
Credits are transferable across generations; this scheduler does not enforce
per-episode reuse limits. Cache replacement is sequential, and writing rows are
unavailable to consumers. Short windows and first/last generations are not
steady-state estimates.

## 5. Diagnose deviations before changing parameters

| Observed behavior | Next check and action |
| --- | --- |
| Learner waits with zero credits | Check intentional budget throttling first. If additional supply is needed, measure whether increasing P improves joint throughput. Do not raise r solely to keep the GPU busy. |
| Learner has credits but cannot sample | Check unlocked rows and valid context windows. Consider C relative to P, continuity and chunk settings; this is distinct from exhausted credits and terminal worker failure. |
| New trajectories are evicted unused | Inspect per-row residence, C/P, refill timing and actual consumption. Increasing production can worsen this condition. |
| Old data dominates or age rises | Inspect refill gaps, accumulated credits and production stopping. Separate insufficient supply from intentional reuse. |
| Early or late generation receives disproportionate updates | Audit startup allowance, cold latency prior, row replacement order and post-production training. Global r alone does not constrain these tails. |
| Learning regresses while throughput is healthy | Examine exposure and fixed development rollouts, then adjust reuse or optimization budget. Throughput formulas cannot diagnose all policy failures. |
| Larger P lowers total performance | Measure CPU/NUMA, GPU contention and learner capacity; reduce concurrency or revise placement based on the paired measurements. |

## 6. Remaining deliverables and validation

1. Use the implemented common-clock flow reports to validate joint supply/capacity,
   wall rates, waits, latency distributions, exposure and measurement scope.
2. Qualify task/model reuse profiles for the implemented two-throughput configurator: read the versioned K profile and
   runtime metadata, derive budgets/pacing/cache/refill settings, and explain
   saturation, uncertainty and infeasible constraints in its report.
3. Maintain per-trajectory atomic sampler protection with focused tests for
   within-batch repeats, concurrent consumers, replacement, exhausted quotas,
   failure/refund boundaries and end-of-supply stopping. Global credits and accounting alone are insufficient; strict mode now uses
   the shared sampler ledger.
4. Validate rate predictions in representative supply-limited, learner-limited
   and variable-latency conditions. Compare predicted and observed rates/exposure,
   document estimator error and revise assumptions where they fail. Use existing
   runs first; add targeted experiments only for missing conditions.
5. Complete the B64/256/1024 sensitivity to explain the fixed-supervision versus
   optimizer-update tradeoff. Preserve all results without naming a winner.
6. Publish worked examples, CLI mappings, startup/steady/end-of-supply guidance,
   failure diagnosis and a short recalibration checklist for new tasks/hardware.

Acceptance is a reproducible procedure that explains the measured operating
regime and yields testable settings. Autonomous policy quality remains a guard
against a fast but ineffective training schedule. Exhaustive grid completion
and a top-two multi-seed tournament are not prerequisites for this manual.

## 7. Run the implemented path

The input JSON contains `accepted_episodes_per_second`,
`training_chunks_per_second`, `profile`, `runtime` and `binding`. Profile fields
are `profile_id`, `scope`, explicit `qualified`, `min_chunks_per_episode`,
`max_chunks_per_episode`, `hard_max_chunks_per_episode`, and optional
`latency_safety_factor`. Runtime fields are `batch_size`, `producer_envs`,
`first_publication_p95_seconds`, optional paired `cache_memory_bytes` and
`bytes_per_episode`, `min_cache_episodes` and `measurement_scope`. Memory is the
episode-payload allowance, not total process/GPU memory.

Binding fields must match the run: task=`PickPlace-Franka-v1`,
policy=`chunk-pose`, `prediction_horizon`, `action_horizon`, and `pose_ik`. The
runner rejects conflicting explicit batch/cache/concurrency values. Bounds from
a provisional profile remain labeled provisional in every output.

```bash
python -m scripts.benchmark.data_pipeline.configure_ods \
  --input /path/to/measured-input.json --output /path/to/configuration-report.json
python examples/data_pipeline/ods_pick_place_showcase.py \
  --policy chunk-pose --auto-config-input /path/to/measured-input.json \
  --output /path/to/new-run --sim-device cuda:0 --train-device cuda:0 \
  --no-viewer --no-video
```

The calculator accepts optional `--generation-throughput` and
`--training-throughput` overrides and records them explicitly. Pass the same
effective input to the showcase; the report wrapper is not itself a run input.
The auto path enables exact usage statistics and deferred development evaluation
to avoid evaluator contention and always preserve the final quota-limited
checkpoint. It defaults production to cache-size times rounds; an explicit
accepted-episode cap must be divisible by the derived cache size. Existing
update/wall-time budgets can stop earlier.

Outputs include `auto_configuration.json` with input hash and resolved settings,
`sampling_budget.json` with retained-source and lifetime delivery counts,
`usage.json` with all observed episode exposures, and `flow_measurements.json`
with active/wall learner rates, productive warm-refill rates, waits and excluded
intervals. `timeline.json` records the common monotonic origin; source hashes
record the configurator and reporter too. These are observed paced-workload
rates, not a claim of simultaneous saturation. Recalibrate after changing the
profile or runtime placement. No built-in profile is claimed universally safe.

## 8. Implementation validation

At implementation revision `8748aa22`, 429 focused runtime tests and 49 context/
documentation tests passed. API coverage is2614/2614. A real P2/C4/B64 run with
provisional budget/hard limit64 produced8 accepted trajectories across2 generations
and performed6 parameter updates. All384 delivered chunks were consumed; every
trajectory stayed at or below64 selections. The last four trajectories consumed
their full budgets;128 unused initial-generation credits were retired on
replacement. The final step6 checkpoint was saved and evaluated despite a1000-step
periodic evaluation interval. Budget depletion and normal cleanup both passed.

The measured service-start spacing respected the configured12.047 chunks/s cap
with the one-batch burst allowance. The tiny budget intentionally tests plumbing
and termination, not learning success. CPU sampler measurements at C32 and B64/
256/1024 showed p50 quota overhead of4.29%/0.92%/1.17%; these include selection,
lock and clone, but are not end-to-end training overhead claims. Full Sphinx
validation remains blocked by the environment's missing `open3d.cpu` dependency.
Detailed evidence and limitations: [validation record](ODS_CONFIGURATION_VALIDATION.json).

## 9. Subsequent regime and learning tests

See [the protected-quota test report](ODS_QUOTA_TESTS.md) for the deterministic
regime matrix, two-seed learning screen and same-seed replication. Rate matching
and strict caps are engineering properties; sufficient learning remains a
separate profile qualification. In particular, `floor(N*K/B)` is an upper bound
on updates from N trajectories, and retirement reduces it. The two throughput
inputs cannot determine the total work required for convergence.

The learning screen fixes256 accepted trajectories and a generous update cap.
The generic CLI defaults to cache-size times rounds for production; it is not
a validated learning budget for every profile. Set the task's explicit production
budget (divisible by derived cache size), then inspect actual delivered chunks
and optimizer updates. Never raise source caps merely to compensate for a
shortfall discovered after data generation stops.

## 10. Interpret efficiency against an offline baseline

Rate matching and bounded reuse do not establish a learning-efficiency advantage.
Compare the protected online run with both a small fixed corpus and a larger
acquisition-inclusive corpus. Give the offline learner the same initial model,
normalization procedure, update budget and evaluation scenes. Preserve an
unrestricted offline baseline as well as an equal-quota control: they answer
different questions, and the quota may never bind in a large resident corpus.
Report per-source minimum/mean/maximum selections and the number reaching the cap.

For C initial episodes, K selections per episode and batch B, the initial cache
can fund at most `floor(C*K/B)` updates. If a scheduled checkpoint exceeds that
budget, online training must await new sources even when a fixed offline corpus
could reuse its data immediately. This is the cost of the chosen protection
policy, not evidence that the learner is malfunctioning. A faster first update
does not imply an earlier useful checkpoint or shorter total experiment time.

Predeclare sufficiently early checkpoint panels for the desired latency question.
Deferred evaluation identifies the availability of the first **evaluated**
qualifying checkpoint retrospectively; it does not measure the true threshold
crossing between saved checkpoints or the time an online evaluator certifies
success. Include failed checkpoints and report final quality, regressions, active
training, intentional throttling, acquisition/persistence, evaluation and cleanup.

Measure accounting overhead at both online-cache and offline-resident sizes.
Identical logging flags can have different costs when source cardinality differs.
Include quota checks and resource sampling in a complete-consumer measurement;
a faster counter microbenchmark alone cannot qualify the whole training path.
Do not subtract an isolated percentage from measured wall time. Likewise, a
bounded source cache does not guarantee lower total PSS or native GPU memory:
simulation and concurrently live workers can dominate a small state-only dataset.

The [four-arm pilot protocol](OFFLINE_PILOT.md) applies these controls.
The [accounting audit](ODS_ACCOUNTING_AUDIT.md) retains both correctness evidence
and the incomplete performance gate. Use these as measurement guidance, not a
universal K recommendation or an assumption that online training must win.
