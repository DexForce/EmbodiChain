# Configure protected ODS supply from measured throughput

The two explicit rate inputs are successful published trajectories per second
`g` and active learner chunks per second `t`. A chunk is one returned sampling
row, not a complete trajectory pass. Repeated selections within a batch count
separately. The input JSON also owns the measured publication latency, batch B,
producer concurrency P, cache-memory limits, and versioned reuse-protection
profile. Throughput alone cannot establish a learning-quality bound.

## Measure the deployed workload

Use the actual policy, observation/action schema, horizon, batch size, producer
concurrency and device placement. Exclude startup, evaluation and intentional
pacing from active capacity. Include sampling, locked copies, feature/target
conversion, H2D, forward/backward and optimizer completion. Measure while the
producer is active, because sharing a GPU changes both rates.

| Quantity | Record |
| --- | --- |
| Supply | Accepted trajectories and eligible anchors divided by productive refill wall time, including retries |
| Learner capacity | Completed optimizer chunks divided by active consumer service time |
| Realized throughput | Completed chunks divided by the complete training-loop window, including waits |
| Latency | Request-to-first-publication and full-refill distributions, with observation counts |
| Exposure | Per-source chunk selections, supervised-frame appearances, coverage and generation shares |
| Resources | Placement, native GPU memory, process-tree sampled PSS, CPU usage and disk footprint |

Keep a common monotonic origin. Distinguish productive capacity from whole-run
supply when the producer is deliberately idle. A paced probe measures the actual
scheduler regime; it does not establish simultaneous saturation capacity.
Re-measure after changing P, B, devices, representation or pacing.

`python -m examples.ods.simple_bc_demo.measure --help` describes the bounded
qualification/production/consumer probes. Save the raw counts, intervals and
hardware metadata, then construct a deployment-specific input.

## Protection profile and derived settings

The profile supplies a recommended minimum and maximum selections per source,
and a strict hard maximum. Bounds belong to a task/model/sampler/horizon version.
Qualify their learning stability separately with fixed development scenes and
multiple learner/producer seeds. `qualified: false` remains visible in outputs.

The raw rate-balance requirement is `t/g` chunks per new trajectory. The generic
`embodichain.data_pipeline.configuration.configure_ods()` clamps this requirement
inside the profile bounds, derives a per-source budget K, and calculates the
paced training/publication targets. It does not raise the hard limit to keep a
fast learner busy. Cache memory and first-publication latency additionally bound
short-window consumption and refill lead time.

The calculator returns a reviewable report including K/hard limit, cache size,
initial credits/update ceiling, refill remaining-credit threshold, active-rate
cap, wall-rate target and expected throttling. The demo validates measured
shared-buffer payload size against the configured memory allowance.

```bash
python -m examples.ods.simple_bc_demo.configure \
  --input examples/ods/simple_bc_demo/configs/throughput.example.json \
  --output /tmp/simple-bc-plan.json
```

The sample input is the prior P8/C32/B256 engineering measurement:
`g=0.93264`, `t=14218.41`, provisional fixed K/hard limit2660. These are an
illustration for that deployment, not defaults for other hardware. Change the
rates and latency/memory metadata after a new probe. Optional
`--generation-throughput` and `--training-throughput` overrides are recorded in
the calculator report; use the same effective input when training.

```bash
python -m examples.ods.simple_bc_demo.train \
  --auto-config-input /path/to/measured-input.json \
  --max-generated-episodes 256 --rounds 1 --updates-per-round 2500 \
  --eval-every 500 --deferred-evaluation-schedule updates \
  --dev-scenes 20 --final-test-scenes 0 --no-viewer --no-video \
  --output /tmp/simple-bc-configured
```

Binding must match `SimpleBCPickPlace-Franka-v1`, `chunk-pose`, its horizons and
IK method. Explicit B/P/C flags may only confirm the calculated values. Choose
an accepted-episode budget divisible by C. `floor(N*K/B)` is an upper update
bound from N sources; replacement retires unused allowances and reduces it.
Two rate inputs cannot determine the optimizer work needed to converge.

## Runtime protection and diagnosis

The CPU shared sampler debits per-source allowance while copying under its
publication lock. Duplicate draws count separately. Locked/incomplete rows are
excluded. Exhaustion requests fresh data while production remains available;
when no complete batch can be funded after supply ends, training stops and saves
the last policy. Eligibility failures wait at most 30 seconds; worker failures
propagate instead of becoming empty-data retries.

| Observation | Interpretation / action |
| --- | --- |
| Most time intentionally throttled | Supply cannot sustain active learner demand at K; this is expected protection, not high GPU occupancy |
| Data waits despite adequate average supply | Inspect publication latency/bursts, refill lead, locked rows and memory limits |
| Assigned credits remain unused | Inspect row replacement and final partial-batch remainder; allowances are not guaranteed consumption |
| First cache receives too much training | Inspect per-source maxima and actual debits, rather than only generation-average counts |
| Quality drops despite bounded counts | Requalify the profile/model/data recipe; throughput balance is not a convergence guarantee |

Inspect `auto_configuration.json`, `flow_measurements.json`, `usage.json`,
`sampling_budget.json` and `timeline.json`. Compare input rates with the measured
productive and active rates, preserving excluded intervals and sample counts.
Resume creates a new ODS run and fresh cache, not exact replay recovery.

## Compare with a strong offline control

Retain both small fixed-data and larger cold offline baselines with equal initial
model, normalization procedure, update budget and scenes. Keep unrestricted
reuse alongside equal-quota sampling. For C initial trajectories, K selections
and batch B, G1 funds at most `floor(C*K/B)` updates. A checkpoint beyond that
point must await new online sources, while an unrestricted offline corpus can
reuse the same data immediately.

Report the number of sources that reach K: a large offline dataset's cap may
never bind. Faster first update, earlier useful checkpoint, active learner
throughput and total delivery cost are separate metrics. Deferred evaluation
identifies the availability of the first evaluated qualifying checkpoint
retrospectively. It does not measure true threshold crossing between saved
checkpoints or online certification.

Measure full-consumer instrumentation at both cache and offline-resident sizes,
including quota checks and resource sampling. Equal logging flags do not imply
equal overhead. Do not subtract an isolated microbenchmark percentage from wall
times. A bounded source cache also does not guarantee lower total simulator or
process memory. See [the retained pilot](OFFLINE_PILOT_RESULTS.md) and
[accounting audit](ODS_ACCOUNTING_AUDIT.md); the strict accounting overhead gate
was incomplete in that evidence.
