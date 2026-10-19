# Simple BC demo with ODS and LeRobot

Generate physical Franka pick-and-place demonstrations, train a state-based
ChunkBC policy from scratch, and evaluate the student in an independent
simulator. The same learner consumes live ODS supply or a fixed LeRobot corpus.
Cameras provide optional display and video; policy inputs come from measured
robot and object state.

Run the commands below from the repository root. This directory owns the local
task, configuration, model, trainer, checkpoint tools and evaluation. The shared
ODS engine, throughput configurator and numeric source APIs remain in
`embodichain.data_pipeline`.

## Layout

| File | Responsibility |
| --- | --- |
| `environment.py`, `config.py`, `configs/` | Physical task, reset, expert, measured success and local configuration |
| `models.py`, `observations.py` | ChunkBC, causal features, masked loss and frozen normalization |
| `train.py`, `trainer.py` | CLI and shared training/experiment loop |
| `checkpoint.py`, `evaluation.py` | Immutable checkpoints, recovery and independent student execution |
| `data.py`, `collection.py`, `lerobot.py` | Corpus identity, accounting and bounded direct LeRobot persistence |
| `autoconfiguration.py`, `configure.py` | Bind measured rates and a reuse profile to this learner |
| `usage.py`, `flow.py`, `compare.py` | Source exposure, timing and existing-run comparison |
| `docs/` | Configuration method and retained experimental evidence |

The task ID is
`examples.ods.simple_bc_demo.environment:SimpleBCPickPlace-Franka-v1`.
`config.load_task_config(device, pose_ik="dls")` resolves the local YAML files;
the demo does not require registration in the official task catalog.

## Requirements

Use an EmbodiChain source installation with its matching DexSim runtime, Franka
assets and a CUDA-capable simulation device. Install the presentation tools if
needed, and the optional LeRobot SDK for offline collection/loading:

```bash
python -m pip install -e .
python -m pip install tensorboard viser imageio imageio-ffmpeg
python -m pip install 'lerobot==0.4.4'
```

LeRobot 0.4.4 was used for the retained experiments. The task declares
`physics: default`; its Python facade and native DexSim extension must match.
Select the physical GPU before importing Python libraries; it becomes local
`cuda:0` in these commands. Use a fresh output directory for each run.

On the validation host, activate `py311` and select its existing compatible
DexSim runtime from the repository root before running native checks:

```bash
conda activate py311
export PYTHONPATH="$PWD:$PWD/extensions:$PWD/extensions/embodichain_tasks:/tmp/ods-dexsim-runtime"
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The unmodified installed facade on this host fails to import
`reconstruct_cable_rings_kernel` from its Newton module. The temporary overlay
above is host-specific and is not bundled with this example; other installations
need a matched DexSim runtime. The physical task uses the default backend.

## 1. Check the engineering loop

```bash
python -m examples.ods.simple_bc_demo.train \
  --smoke --no-viewer --no-video --output /tmp/simple-bc-smoke
```

This checks expert generation, ODS sampling, parameter updates, checkpoint
loading, evaluation and shutdown with tiny budgets. Smoke success does not
qualify policy learning. Direct file entry also works:
`python examples/ods/simple_bc_demo/train.py --help`.

## 2. Run the small-data BC baseline

```bash
python -m examples.ods.simple_bc_demo.train \
  --mode offline-32 --offline-episodes 32 \
  --buffer-size 32 --producer-envs 8 --batch-size 256 \
  --rounds 1 --updates-per-round 2500 --eval-every 500 \
  --seed 19001 --producer-seed 1019001 --sampling-seed 29001 \
  --defer-evaluation --deferred-evaluation-schedule updates \
  --dev-scenes 20 --dev-seed-base 810000 --final-test-scenes 0 \
  --usage-statistics --no-viewer --no-video \
  --sim-device cuda:0 --train-device cuda:0 \
  --output /tmp/simple-bc-offline32
```

The run qualifies the expert, collects 32 accepted episodes directly into
`lerobot_dataset/`, finalizes the bounded SDK writer, and performs one audited
numeric load into resident CPU tensors. It saves steps
0/500/1000/1500/2000/2500 and evaluates them after training and producer shutdown.
The default expert qualification uses 40 separate scenes and requires at least
38 successes. Only physically accepted demonstrations enter the training data.

The retained pre-reorganization pilot finished at 20/20 development successes
with this small-data recipe. This is one-seed development evidence; see the
[full comparison and its limits](docs/OFFLINE_PILOT_RESULTS.md).

## 3. Run protected online ODS

The configuration input contains generation and training rates, runtime
metadata, resource limits and a versioned per-trajectory reuse profile. See the
[configuration method](docs/CONFIGURATION.md).
`configs/throughput.example.json` contains historical provisional measurements;
replace its rates and latency/memory metadata after measuring your deployment.

```bash
python -m examples.ods.simple_bc_demo.configure \
  --input examples/ods/simple_bc_demo/configs/throughput.example.json \
  --output /tmp/simple-bc-configuration.json

python -m examples.ods.simple_bc_demo.train \
  --auto-config-input examples/ods/simple_bc_demo/configs/throughput.example.json \
  --max-generated-episodes 256 --rounds 1 --updates-per-round 2500 \
  --eval-every 500 --deferred-evaluation-schedule updates \
  --seed 19001 --producer-seed 1019001 --sampling-seed 29001 \
  --dev-scenes 20 --dev-seed-base 810000 --final-test-scenes 0 \
  --no-viewer --no-video --sim-device cuda:0 --train-device cuda:0 \
  --output /tmp/simple-bc-ods
```

Pass the original measurement input to training; the calculator's report is a
reviewable output. Auto-configuration enables exact usage accounting and
deferred evaluation. Per-source quotas are enforced during locked sampling;
replacement can retire unused allowances. The historical K=2660 is provisional.
Throughput matching determines pace, while the learning protocol determines
the total production and update budgets.

Measure the actual learner, representation, horizons, batch size and producer
concurrency together. The public `measure_ods_throughput` helper preserves rate
units and observation windows. The generic CPU sampler-cost benchmark remains
`scripts/benchmark/data_pipeline/benchmark_ods_sampling_budget.py`; it does not
measure concurrent simulation/training capacity.

## 4. Evaluate a checkpoint independently

```bash
python -m examples.ods.simple_bc_demo.evaluation \
  --checkpoint /tmp/simple-bc-offline32/step_2500.pt \
  --episodes 20 --seed-base 810000 --sim-device cuda:0 \
  --output /tmp/simple-bc-evaluation
```

This starts a fresh evaluator, loads the frozen checkpoint, executes network
targets, and writes `evaluation.json` and `cleanup.json`. It creates no learner
or ODS producer. Add `--viewer` for Viser or `--video` to record the first three
scenes, including failures. `--timeout` bounds each worker response. Load only
trusted checkpoints written by this example.

Success is measured by `environment.py`: lift the 5 cm cube at least 5 cm, place
it within 3 cm of the goal at table height, open the gripper, withdraw the TCP
by 10 cm, and remain stable for 0.3 seconds. Expert acceptance and student
evaluation share this measurement. The cube's initial XY varies by ±4 cm;
physics/control frequencies are 240/30 Hz with a 600-step horizon.

ChunkBC is the sole model. Two measured 51-feature observations feed a
102→512→512 network predicting 64 absolute TCP/gripper targets; the student
executes 16 before replanning through bounded relative-pose DLS control.
Observation and target normalization are frozen from the first training cache.
The student receives no expert phase, clock, success label or plan at inference.
No object attachment, teleport or expert takeover assists student execution.

## 5. Compare online and offline runs

Keep learner/data/sampling seeds, normalization procedure, model, batch/update
budgets, hardware and evaluation panels equal. For a cold 256-episode offline
arm, use `--mode offline-full --offline-episodes 256`. An equal-quota control
adds `--offline-sample-budget 2660 --offline-sample-hard-limit 2660`.
Retain the unrestricted 32-episode baseline.

```bash
python -m examples.ods.simple_bc_demo.compare \
  /tmp/simple-bc-ods /tmp/simple-bc-offline32 --output /tmp/simple-bc-comparison
```

`--offline-data /path/to/lerobot_dataset` loads existing data and measures a
data-ready scenario. Cold collection pays its own acquisition, persistence and
loading costs. Subsequent minibatches use resident numeric tensors; they do not
measure per-minibatch disk reads. `--offline-memory-limit-mib` bounds the padded
numeric corpus allocation (default 512 MiB); SDK caches and bounded staging
require additional memory.

`--mode frozen` retains the initial live cache without refill. Protected offline
resume is rejected until quota-ledger restoration exists. Deferred results
describe checkpoint availability retrospectively and cannot certify online
time-to-target. The [pilot protocol](docs/OFFLINE_PILOT.md) defines timing,
resource and failure accounting. The retained pilot established neither a
general ODS efficiency advantage nor a qualified full-path instrumentation cost.

## Outputs and recovery

Runs retain resolved configuration and source hashes, expert qualification,
immutable checkpoints, physical panels, losses, resource samples and cleanup
status. Usage-enabled runs also write `usage.json`, `sampling_budget.json` and
`flow_measurements.json`. Inspect the stopping reason and `cleanup.json` after
budget depletion. Unsupported resource peaks remain unknown.

`--resume CHECKPOINT` restores ChunkBC, optimizer, normalization and RNG state
in a fresh output directory. Live resume fills a new ODS cache under a new run
identity. It does not restore the previous shared cache. Training budgets apply
to additional updates; saved checkpoint steps remain cumulative. Only ChunkBC
checkpoint schemas are supported.

A matching `--qualification-report` can reuse expert qualification after
source/configuration checks. The move to this local task requires fresh
qualification. `--qualification-only` runs that check separately;
`--stage overfit-one|overfit-four` provides training-reset diagnostics.

Omit `--no-viewer` to expose Viser at port 8080 and TensorBoard at 6006. Omit
`--no-video` to record comparison videos. Re-measure throughput after changing
display, video or device placement.

## Validation and retained evidence

Unit tests cover reusable production contracts rather than the demo:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/data_pipeline/test_sources.py \
  tests/data_pipeline/test_lerobot_source.py
```

Install the optional LeRobot SDK for reader checks. These tests cover source
parity, schema mapping, timestamps, masks, quota/error boundaries and bounded
memory. The local model, persistence and orchestration are validated through
runnable experiments without dedicated example unit tests.

The [public loading design](../../../docs/architecture/dataset-loading-design.md)
describes `BatchSource`, `SequenceDataset` and the online/resident/LeRobot
adapters consumed by this demo. `OfflineDataset` remains a compatibility alias;
`OnlineDataset` retains its chunk API. The demo owns writing, content auditing
and feature conversion; it contains no duplicate resident sampler or SDK loader.

| Evidence | Scope |
| --- | --- |
| [Evidence index](docs/README.md) | Historical source revisions, original artifact paths and limitations |
| [Working policy](docs/WORKING_SHOWCASE.md) | Historical 99/100 selected checkpoint and retained 65/100 refresh regression |
| [Quota validation](docs/ODS_QUOTA_TESTS.md) | Strict caps, rate regimes and provisional profile limits |
| [Four-arm pilot](docs/OFFLINE_PILOT_RESULTS.md) | Offline32, offline256 and protected ODS comparison |
| [Accounting audit](docs/ODS_ACCOUNTING_AUDIT.md) | Exact counters; strict performance gate remains incomplete |
| [Alignment](docs/ODS_ALIGNMENT_VALIDATION.json) and [public loaders](docs/DATASET_LOADING_VALIDATION.json) | Historical exact batch/optimizer evidence, pre-extraction golden outputs, public API and physical integration records |
| [Local-package runtime validation](docs/PY311_VALIDATION.json) | Candidate `d54cfc8a`; 482 production/API/context checks and 31 runtime audits passed in conda `py311` |

Structured historical records retain their executed paths, hashes and test
counts verbatim. Earlier learner/optimizer checks are historical evidence and
are not current CI coverage. Exposed development and reserved scenes must not
be treated as fresh held-out data in a later study.

The local-package rerun used Python 3.11.14, Torch 2.7.0/CUDA 12.8 and LeRobot
0.4.4 on 2026-10-19 with the matching native runtime documented above. Online
training completed 10 updates/40 chunks across two generations; cold LeRobot
collection wrote/read four episodes (1,330 frames), consumed four chunks and
stopped within its one-selection-per-source budget. Resume restored the model,
normalization and sampler RNG from step 10 and completed 15 additional updates
to step 25 under a new data-source run ID. Independent loading of the retained
trained policy succeeded on two reused development scenes. All observed child
processes exited normally; source files stayed unchanged during validation.

These are engineering checks, not a rerun of the complete learning studies.
Training smokes used engineering scenes 920000–920001 and 921000–921001; the
independent checkpoint regression reused development scenes 810000–810001.
The original installed DexSim import failure and the earlier unit-test setup
attempts remain in the validation record. Successful native checks require the
matching runtime and CUDA visibility documented above.
