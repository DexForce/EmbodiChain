# Configure the Franka ChunkBC demo

The reusable method now lives in the ODS documentation:

- [ODS overview and source contracts](../../../../docs/source/features/online_data.md)
- [Throughput, reuse budget, cache/refill and runtime checks](../../../../docs/source/features/ods/configuration.md)

Those guides define rate units, exact planner limits, sampler protection and
consumer responsibilities for any task. This page describes the demo's binding,
CLI and artifact names. [Run the demo](../README.md).

## Input packet and binding

`configs/throughput.example.json` is a complete historical packet. Its rates
and K2660 profile are provisional evidence for the recorded workload. Measure
your concurrent producer/learner placement and validate its profile before
using it as a recommendation for another setup.

The demo adapter accepts exactly five top-level keys:

| Key | Demo use |
| --- | --- |
| `accepted_episodes_per_second` | Accepted production rate passed to the public calculator. |
| `training_chunks_per_second` | Active learner rate passed to the public calculator. |
| `profile` | Fields of `ODSReuseProfileCfg`, including explicit qualification and selection bounds. |
| `runtime` | Fields of `ODSRuntimeCfg`, including measured B/P, latency and optional storage allowance. |
| `binding` | Exact local task, policy, horizon and controller match described below. |

The binding must match:

```json
{
  "task": "examples.ods.simple_bc_demo.environment:SimpleBCPickPlace-Franka-v1",
  "policy": "chunk-pose",
  "prediction_horizon": 64,
  "action_horizon": 16,
  "pose_ik": "dls"
}
```

`binding` and `--auto-config-input` belong to this adapter, not the public
`configure_ods()` function. The runner rejects conflicting explicit batch,
cache or producer-concurrency values. It preserves the packet hash, the public
plan's diagnostic reasons and the profile's provisional qualification.

## Calculate and run

From the repository root with the native runtime set up as in the README:

```bash
python -m examples.ods.simple_bc_demo.configure \
  --input examples/ods/simple_bc_demo/configs/throughput.example.json \
  --output /tmp/simple-bc-configuration.json

python -m examples.ods.simple_bc_demo.train \
  --auto-config-input examples/ods/simple_bc_demo/configs/throughput.example.json \
  --max-generated-episodes 256 --rounds 1 --updates-per-round 2500 \
  --eval-every 500 --deferred-evaluation-schedule updates \
  --dev-scenes 20 --dev-seed-base 810000 --final-test-scenes 0 \
  --no-viewer --no-video --sim-device cuda:0 --train-device cuda:0 \
  --output /tmp/simple-bc-ods
```

Pass the original input JSON to training; the calculator's report is a reviewable
output, not another input format. The calculator accepts
`--generation-throughput` / `--training-throughput` overrides and records them;
use the same effective inputs for training.

This auto path supports only live online ChunkBC in the showcase stage. It
cannot combine with offline input or smoke overrides. It configures strict
source budgets, remaining-quota refill scheduling and service-start pacing.
It also enables usage accounting and deferred evaluation. Without it, the
example retains its fixed-round scheduling path.

The explicit accepted-episode cap must be divisible by the derived cache size.
Without a cap, this adapter uses cache-size times rounds. Requested updates,
wall time or exhausted usable quotas can stop training earlier. The resulting
finite-data update ceiling is not a convergence guarantee; cache replacement
can retire unused allowances.

## Inspect the result

| Artifact | Meaning in this example |
| --- | --- |
| `auto_configuration.json` | Input hash, resolved public plan, qualifications and finite-run budget ceiling. |
| `sampling_budget.json` | Retained-source allowances and lifetime delivered-chunk count. |
| `usage.json` | Consumer-maintained per-source completed exposure and generation shares. |
| `flow_measurements.json` | Observed productive refill/learner rates, waits and excluded intervals under this scheduler. |
| `timeline.json` | Recorded intervals and their common monotonic origin. |
| `environment.json`, `cleanup.json` | Runtime/source identity and process shutdown evidence. |

These filenames and the detailed flow/exposure/resource instrumentation belong
to the example. The shared engine exposes counters, budget snapshots and optional
lifecycle events; it does not produce a generic TensorBoard dashboard or these
complete learner reports. Recorded rates under deliberate pacing do not establish
simultaneous saturation capacities.

Compare planned versus observed rates, inspect initial/last-generation exposure,
retired credit, maximum per-source deliveries and the actual stopping reason.
Keep model quality separate from rate matching. The maintained small offline
baseline may solve this task sooner than collecting a larger corpus.

The [offline comparison protocol](OFFLINE_PILOT.md) owns the four experiment
arms, matched learner/data settings, checkpoint panels, timing and resource
accounting. It distinguishes acquisition-inclusive runs from data-ready runs
and checkpoint availability from evaluation completion. A new study must
predeclare its quality threshold, seed blocks and held-out scenes; the recorded
single-seed pilot remains descriptive evidence.

[Historical quota trials](ODS_QUOTA_TESTS.md), [configuration validation](ODS_CONFIGURATION_VALIDATION.json),
[the offline pilot](OFFLINE_PILOT_RESULTS.md), and [the accounting audit](ODS_ACCOUNTING_AUDIT.md)
retain their original revisions, failures and limits. They are worked evidence
for the [general ODS method](../../../../docs/source/features/ods/configuration.md),
not qualified defaults for every task or hardware setup.
