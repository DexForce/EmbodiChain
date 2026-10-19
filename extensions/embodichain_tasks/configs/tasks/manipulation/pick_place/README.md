# Franka ODS PickPlace showcase

Train a small state-based ChunkBC policy from scratch using a live expert
producer or an offline LeRobot dataset, then evaluate it in an independent
simulator. The entry point is `examples/data_pipeline/ods_pick_place_showcase.py`.

| Guide or evidence | Purpose |
| --- | --- |
| [Throughput configuration](ODS_CONFIGURATION_GUIDE.md) | Derive pacing, cache/refill settings and bounded per-trajectory reuse. |
| [Working policy validation](WORKING_SHOWCASE.md) | Physical qualification, successful checkpoint and retained refresh regression. |
| [Quota validation](ODS_QUOTA_TESTS.md) | Enforcement, rate regimes and limits of reuse recommendations. |
| [Comparison protocol](OFFLINE_PILOT.md) and [results](OFFLINE_PILOT_RESULTS.md) | Completed four-arm experiment, including the strong offline32 baseline. |
| [Accounting audit](ODS_ACCOUNTING_AUDIT.md) | Exact-counter validation and the incomplete performance gate. |

## Run

From the repository root, with compatible DexSim runtime, robot assets and
PyTorch installed:

```bash
python examples/data_pipeline/ods_pick_place_showcase.py \
  --sim-device cuda:0 --train-device cuda:0 --output /tmp/pick-place
```

The default recipe uses a 32-episode CPU ODS cache, two rounds of 5,000 updates,
batch size 256 and DLS action processing. Open Viser at
`http://127.0.0.1:8080` and TensorBoard at `http://127.0.0.1:6006` during the run.
Use `--no-viewer --no-video` for headless runs; ports are configurable.

The first run qualifies the expert, collects initial data, trains, evaluates
checkpoints and verifies the development-selected policy. A matching
`--qualification-report` can reuse physical qualification. Load the model named
by `summary.json.best_checkpoint`: historical verification achieved 99/100 at
step 5,000 and regressed to 65/100 at step 10,000. Those exposed test scenes are
not fresh held-out evidence for a new study.

For a short process/data/control check:

```bash
python examples/data_pipeline/ods_pick_place_showcase.py --smoke \
  --no-viewer --no-video --output /tmp/pick-place-smoke
```

Smoke is an engineering check, separate from learning qualification. Generation
and physical evaluation dominate this small learner's runtime.

## Physical task and policy

The Franka must physically grasp a 5 cm cube, lift it at least 5 cm, settle it
within 3 cm of the fixed goal at table height, open the fingers, withdraw the
TCP by 10 cm, and remain stable for 0.3 seconds. Initial cube XY varies by ±4 cm;
physics/control frequencies are 240/30 Hz with a 600-step horizon. There is no
object attachment, teleport or expert takeover during student execution.

ChunkBC uses two measured 51-feature observations: robot qpos/qvel, TCP/cube
geometry, goal, relative vectors and cube velocities. A two-layer MLP-512
predicts 64 absolute TCP/gripper targets and executes 16 before replanning.
The DLS controller converts target error to bounded relative requests against
live measurements. Targets are reconstructed from causal pre-action state and
recorded commands; no phase, clock, success label or expert plan enters inference.

Observation and target normalization are frozen from initial training data.
Training uses AdamW at 3e-4, weight decay 1e-6, gradient norm limit 1 and 0.01
normalized observation noise. `--prediction-horizon`, `--action-horizon` and
`--observation-noise` are explicit settings; changing them requires a matching
reuse profile and new throughput measurement.

## Protected online supply

Prepare measured inputs using the [configuration guide](ODS_CONFIGURATION_GUIDE.md):

```bash
python -m scripts.benchmark.data_pipeline.configure_ods \
  --input /path/to/measured-input.json --output /tmp/ods-configuration.json
python examples/data_pipeline/ods_pick_place_showcase.py \
  --auto-config-input /path/to/measured-input.json \
  --max-generated-episodes 256 --rounds 1 --updates-per-round 2500 \
  --eval-every 500 --deferred-evaluation-schedule updates \
  --dev-scenes 20 --dev-seed-base 810000 --final-test-scenes 0 \
  --sim-device cuda:0 --train-device cuda:0 --no-viewer --no-video \
  --output /tmp/pick-place-protected
```

The configurator combines generation/training throughput with measured runtime
metadata and a versioned protection profile. It enables usage statistics and
deferred evaluation. Per-source quotas are enforced during locked sampling;
unusable allowances can be retired on replacement. Throughput matching does
not determine how much work is needed for convergence.

## Offline and frozen controls

Cold offline collection writes accepted episodes directly through a bounded
LeRobot SDK writer, waits for finalization, then performs one schema/hash-audited
SDK load into resident CPU tensors. Install the optional LeRobot SDK; version
0.4.4 was validated. Numeric dataset collection uses LeRobot exclusively.

```bash
python examples/data_pipeline/ods_pick_place_showcase.py \
  --mode offline-full --offline-episodes 256 \
  --buffer-size 32 --producer-envs 8 --batch-size 256 \
  --rounds 1 --updates-per-round 2500 --eval-every 500 \
  --seed 19001 --producer-seed 1019001 --sampling-seed 29001 \
  --defer-evaluation --deferred-evaluation-schedule updates \
  --dev-scenes 20 --dev-seed-base 810000 --final-test-scenes 0 \
  --sim-device cuda:0 --train-device cuda:0 --no-viewer --no-video \
  --usage-statistics --output /tmp/pick-place-cold-lerobot256
```

The explicit episode count is independent of training rounds and must be a
cache-size multiple. The output contains `lerobot_dataset/`, the run-local SDK
cache, checkpoints and reports. Subsequent minibatches use resident context
sampling; this does not measure LeRobot per-minibatch disk I/O.

`--offline-memory-limit-mib` bounds resident numeric dataset tensors (default
512 MiB). The public loader checks this before allocating the padded corpus;
SDK caches and bounded staging are additional costs. Feature/action schema and
fixed-rate timestamps are validated rather than inferred from vector width.

- `--mode offline-32 --offline-episodes 32`: collect the initial cache, close
  the producer and train on that fixed corpus for the same update budget.
- `--mode frozen`: retain the initial live ODS cache without refill.
- `--mode offline-full --offline-data /path/to/lerobot_dataset`: train with an
  existing dataset. Report this data-ready scenario separately from cold costs.
- `--offline-sample-budget K --offline-sample-hard-limit LIMIT`: optional
  per-source protection for resident data; conventional baselines reuse freely.

The pilot used provisional K=2660, not a universal recommendation. Its offline32
baseline reached a qualifying checkpoint earlier than ODS, while the offline256
cap never bound. Keep the same learner, update budget, panels and hardware when
comparing. The [protocol](OFFLINE_PILOT.md) defines resource, timing and failure
accounting. Generate a descriptive report with:

```bash
python examples/data_pipeline/ods_pick_place_compare.py \
  /tmp/pick-place-protected /tmp/pick-place-cold-lerobot256 \
  --output /tmp/pick-place-comparison
```

## Online/offline alignment

The paired gate uses the real ODS sampler with a deterministic shared CPU
buffer, official LeRobot SDK serialization/readback and the current learner:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/data_pipeline/test_ods_pick_place_alignment.py
```

Install the optional LeRobot SDK to run the roundtrip checks. Identical source
data and RNG states must give exactly equal observations, actions, masks,
provenance and optimizer updates. Tests cover starts/ends, validity gaps, dirty
tails, horizon 1/4/64 and binding source quotas. Canonical feature padding is
applied after geometry conversion; unsupported continuity is rejected during
offline snapshot creation. This validates data semantics, not an ODS speedup.

The [alignment evidence](ODS_ALIGNMENT_VALIDATION.json) also records a real
producer check: two generations/four accepted episodes, 1,330 valid frames,
32 paired chunks and four matched optimizer updates. Every training-batch field,
sampling RNG, loss, gradient and model/optimizer tensor matched exactly on CPU;
all eight observed processes exited. No held-out final tests were opened.

The [public loading design](../../../../../../docs/architecture/dataset-loading-design.md)
is implemented for numeric data: `SequenceSpecCfg`, `BatchSource`,
`OnlineDataSource`, `TensorDataSource`, `LeRobotDataSource` and `OfflineDataset`
are exported from `embodichain.data_pipeline.datasets`. `OnlineDataset` keeps
its chunk API and adds opt-in context iteration. The showcase consumes the
common source interface; its private code retains writing and sidecar/hash
auditing, with the duplicate resident sampler and SDK loading loop removed.
Standard LeRobot numeric datasets can use the public reader without this
showcase's sidecar, through an explicit schema and field mapping.

[Public loading validation](DATASET_LOADING_VALIDATION.json) records the API
regressions, pre-extraction golden outputs, bounded loading, two real-process
integrations and scoped sampler overhead. It does not qualify the unresolved
full-path telemetry gate or establish an ODS learning/throughput advantage.

## Artifacts and recovery

Use fresh output directories. Runs retain configuration/source hashes,
immutable checkpoints, source provenance, losses, physical evaluations, resource
samples and cleanup status. Deferred evaluation reports checkpoint availability
retrospectively; it cannot certify online time-to-target. Unsupported resource
peaks remain unknown, and failed policies remain in reports and recordings.

`--resume CHECKPOINT` restores ChunkBC, optimizer, normalization and RNG state.
Live resume creates and fills a fresh ODS source; shared-cache replay is not
restored. Protected offline resume is rejected until its quota ledger can be
restored. Only ChunkBC checkpoint schemas are supported.

`--qualification-only` validates the physical expert on 40 independent scenes.
Optional `--stage overfit-one|overfit-four` checks repeatability on training
resets; their failures are not held-out learning results. Cameras are for
presentation only. Feedback-driven collection, DAgger, aggregate replay and
RLinf integration are outside this showcase.
