# Configure ODS from measured throughput

Use `embodichain.data_pipeline.configuration` to plan bounded data consumption
from two rates: accepted trajectory production and active learner demand.
The result is a static configuration plan. A consumer must apply its cache and
sampling settings and implement the returned pacing/refill policy.

This method applies across tasks and learners. The [Franka integration](https://github.com/DexForce/EmbodiChain/blob/main/examples/ods/simple_bc_demo/docs/CONFIGURATION.md)
provides a worked binding; its measurements and reuse bounds are provisional
case evidence. See [ODS](../online_data.md) for engine and source ownership and
[evaluation](evaluation.md) for measuring the resulting behavior.

## Inputs and units

| Input | Meaning and ownership |
| --- | --- |
| `accepted_episodes_per_second`, g | Successfully accepted/published trajectories per active production second, including failed attempts and retry cost. Measure complete production intervals. |
| `training_chunks_per_second`, t | Completed training selections per active learner second, including sampling, copies, feature conversion, transfer and optimizer completion. Exclude deliberate pacing and evaluation waits. |
| `ODSReuseProfileCfg` | A versioned task/model/sampler protection policy with recommended minimum/maximum selections and a hard ceiling. Supplied by the caller. |
| `ODSRuntimeCfg` | Measured batch B, producer concurrency P, first-publication latency, minimum cache diversity, measurement scope and optional storage allowance. |

One selection is one returned chunk anchored in a source trajectory. Ten draws
from a trajectory in a batch consume ten selections. This is neither ten full
trajectory passes nor a count of supervised frames. History/horizon, trajectory
length and sampling distribution affect exposure; record them in the profile.
If the probe reports completed updates/s, multiply by its batch size B to obtain
t. Observation frames/s require an explicit conversion and cannot be substituted.

Measure production and training under the intended concurrent placement. A
producer-only benchmark does not establish its capacity while sharing a GPU
with the learner. A paced wall rate is not the learner's active capacity.
Repeat measurements after changing concurrency, batch, representation, device
placement, display or the pacing schedule.

`measure_ods_throughput(production, training)` accepts two
`ODSRateMeasurementCfg` records with the same absolute monotonic window,
explicit count units and active durations. It returns active and wall rates.
It calculates rates from supplied observations; it does not run a probe or
verify simultaneous saturation. `get_stats()` is also an observation interface,
not an automatic capacity estimator. Counters and production duration advance
at different publication/refill boundaries, so a mid-refill counter ratio can
misstate throughput. Prefer completed refill intervals or lifecycle timestamps.

## Reuse profile

The profile supplies integer bounds `K_min <= K_max <= K_hard`, an explicit
`qualified` flag, an identity/scope and a latency safety factor s >= 1.
Throughput cannot establish learning-safe bounds. A new profile may be marked
`qualified=False`; the plan retains that qualification and a diagnostic reason.

Changing hardware changes measured rates. Material changes to the model,
trajectory distribution, horizon or sampler require profile revalidation.
The public calculator validates fields and ranges, not the scientific truth
of a caller's qualification or scope. It also cannot detect stale measurements
from two scalar rates; freshness and workload compatibility belong to the host.

## Balance, cache and rate limits

The calculator keeps B and P at the measured values. Two rates do not predict
the scaling curve of other batch sizes or producer counts, nor determine the
optimizer/model settings or total updates needed for convergence.

```text
K_balance = t / g
K = clamp(floor(K_balance), K_min, K_max)

L_safe = first_publication_p95_seconds * s
C_base = round_up_to_P(max(2*P, min_cache_episodes, P + ceil(B/K)))
C_desired = round_up_to_P(max(C_base, P + ceil(t*L_safe/K)))
```

For memory allowance M and measured episode-storage bytes E:

```text
C_memory = P * floor(M / E / P)
C = min(C_desired, C_memory)
```

Both memory fields must be supplied together. The calculator rejects capacity
below `C_base`. Without a memory allowance it uses `C_desired`. The allowance
covers episode storage, not model, native simulator, SDK cache or allocator
memory; reserve these separately.

The actual rates also include the active limit imposed by usable cache quota:

```text
unlocked_budget = (C - P) * K
t_active_cap = min(t, unlocked_budget / L_safe)
t_target = min(t_active_cap, g*K)
u_target = t_target / B
g_target = t_target / K
refill_remaining_chunks = max(B, ceil(t_active_cap * L_safe))
initial_max_updates = floor(C*K / B)
learner_duty_estimate = t_target / t
producer_duty_estimate = g_target / g
```

The implementation conservatively rounds published floating-point limits down
where required. Use returned fields rather than reproducing floating arithmetic
in another scheduler. Cache/refill sizing is a latency heuristic: first
publication does not guarantee sustained subsequent supply or protection from
arbitrary stalls. Validate later publication spacing and actual unlocked windows.

When demand exceeds bounded supply, pace the learner. When supply exceeds
demand, request less production while retaining the profile's planned minimum
budget. K_min is not a guarantee that every trajectory receives that exposure:
replacement, failures and early stopping may leave allowances unused.

## Calculate a plan

The following complete CPU-only example uses illustrative rates and bounds,
not a qualified task profile:

```python
from dataclasses import asdict
from embodichain.data_pipeline.configuration import (
    ODSReuseProfileCfg,
    ODSRuntimeCfg,
    configure_ods,
)

profile = ODSReuseProfileCfg(
    profile_id="illustrative-v1",
    scope="Illustrative numeric chunk workload; requires learning validation",
    qualified=False,
    min_chunks_per_episode=200,
    max_chunks_per_episode=600,
    hard_max_chunks_per_episode=1000,
)
runtime = ODSRuntimeCfg(
    batch_size=64,
    producer_envs=4,
    first_publication_p95_seconds=2.0,
    min_cache_episodes=8,
    measurement_scope="Illustrative concurrent rates, not a hardware result",
)
plan = configure_ods(1.0, 1000.0, profile=profile, runtime=runtime)
assert plan.episode_sample_budget == 600
print(asdict(plan))
```

`ODSConfiguration` includes assigned budget and hard limit, cache size/storage,
initial credits/update ceiling, active/average rate targets, refill threshold,
duty estimates, profile qualification, diagnostic reasons and stopping rule.
Invalid/non-positive rates, invalid profiles and infeasible memory raise an
error; they do not silently remove protection.

## Apply the plan

| Plan field | Runtime owner |
| --- | --- |
| `buffer_size`, `batch_size`, `producer_envs` | Host constructs the cache/consumer at these measured settings. |
| `episode_sample_budget`, `episode_sample_hard_limit` | Pass to `OnlineDataEngineCfg`; engine samplers enforce per-source delivery under the producer lock. Resident sources have their own optional ledger. |
| `active_chunks_per_second_cap` | Consumer limits short-window service starts. The engine does not throttle optimizer calls. |
| `training_chunks_per_second_target` | Consumer controls average delivery/completed-training rate and reports actual waits. |
| `refill_remaining_chunks` | Host compares remaining usable quota and schedules manual refill. |
| `publication_episodes_per_second_target` | Host paces production/refill requests. The engine has no finite accepted-episode production budget. |

For example, after supplying the task's validated `gym_config` and state width:

```python
from embodichain.data_pipeline.engine import OnlineDataEngineCfg

engine_cfg = OnlineDataEngineCfg(
    gym_config={**your_gym_config, "num_envs": plan.producer_envs},
    state_dim=your_state_width,
    action_dim=your_action_width,
    buffer_size=plan.buffer_size,
    buffer_device="cpu",
    refill_policy="manual",
    episode_sample_budget=plan.episode_sample_budget,
    episode_sample_hard_limit=plan.episode_sample_hard_limit,
    include_provenance=True,
)
```

The planner's remaining-quota threshold is **not** engine `refill_threshold`.
The engine's default `sample_threshold` policy counts delivered samples against
`refill_threshold * buffer_size` after successful calls. Quota exhaustion does
not itself request refill. Use manual scheduling for the plan above.

`request_refill()` returns the target generation and coalesces an outstanding
request. `wait_for_generation()` waits for the complete requested generation,
not first publication. Continue useful training while supply is active when
appropriate, observe health, drain enabled lifecycle events and use bounded
waits. Source errors remain failures, not permission to continue on stale data.

## Strict budget and consumer accounting

For an immutable `(run_id, episode_id)`, each requested batch's selection,
copy and debit share the producer lock. Within-batch duplicate draws count
separately. The assigned K is a delivery allowance; the hard limit bounds its
configuration. Current budgets are fixed for an engine/source lifetime: the
planner does not implement online adaptation or replenish exhausted identities.

Failed optimization, transforms or prefetched data do not refund delivered
chunks. Track optimizer-completed chunks separately. Live replacement retires
unused credits; it cannot move them to another identity. A new live run creates
new identities. Replaying existing identities requires restored usage accounting;
the demo rejects protected offline resume without that state.

The initial update count is a ceiling, not a convergence recommendation.
After production stops, the host may consume only remaining quotas and must
stop when a requested full batch cannot be funded. No generic background
controller derives a new safe K or training budget automatically.

## Recalibration and diagnosis

| Observation | Check before changing configuration |
| --- | --- |
| Learner waits at zero quota | Separate intentional protection from missing production. Measure joint supply before changing P; keep profile bounds. |
| Quota remains but sampling fails | Check write locks, continuity, sequence length and segment/generation filters. Positive credits alone do not prove an eligible context exists. |
| New trajectories retire unused | Inspect residence, cache replacement and refill timing; more production may increase waste. |
| Old/first-generation data dominates | Inspect startup allowance, gaps, source age and post-production training. The average target alone does not constrain every tail. |
| Quality regresses at healthy throughput | Audit exposure and development rollouts; rates cannot determine convergence or diagnose all policy failures. |
| More producers reduce throughput | Re-measure learner capacity and CPU/NUMA/GPU contention under joint load. |

Record the resolved plan and inputs, actual accepted/delivered/completed counts,
generation shares, retired allowances, latency observations, waits and resource
scope. Compare predictions with measured rates after changing the schedule.
See [evaluation and comparison](evaluation.md) for the metric definitions and
the distinction between engineering constraints and learning quality.
