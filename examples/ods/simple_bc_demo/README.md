# Simple BC demo with ODS and LeRobot

A self-contained Franka pick-and-place example: generate physical expert
trajectories, train a small behavior-cloning policy, and evaluate the student in
an independent simulator. It supports live ODS supply and fixed offline data.
The student uses measured simulation state. Cameras are optional display/video
outputs; no pretrained policy or external demonstrations are required.

Run the commands below from the repository root. This directory owns the task
implementation and configuration, models, trainer, checkpoint tools, evaluation,
and offline comparison. The shared ODS engine and throughput configurator remain
in `embodichain.data_pipeline`.

## Layout

| File | Responsibility |
| --- | --- |
| `environment.py`, `configs/` | Physical scene, reset, handwritten expert, and measured success |
| `models.py`, `observations.py` | BC policies, causal inputs, losses, frozen normalization |
| `train.py`, `trainer.py` | CLI and shared training/experiment loop |
| `checkpoint.py` | Immutable checkpoint saving/loading |
| `evaluation.py` | Independent student evaluator and standalone checkpoint CLI |
| `data.py`, `collection.py`, `lerobot.py` | Resident sampling and bounded direct LeRobot persistence |
| `autoconfiguration.py`, `configure.py`, `measure.py` | Throughput inputs, configuration calculation, measurement tools |
| `usage.py`, `flow.py`, `compare.py` | Per-source exposure, timing, and existing-run comparison |
| `tests/`, `docs/` | Focused checks, configuration method, and retained experiment evidence |

The default `chunk-pose` model uses two 51-dimensional observations, a
102→512→512 MLP, and 64 predicted absolute TCP/gripper targets. It executes 16
predictions before replanning through the relative-pose controller. MLP and
geometry-GRU policies remain available as ablations. All selectable policies use
one trainer. The superseded joint-GRU model, its separate training loop, and its
CLI have been removed.

## Requirements

Use a working EmbodiChain source installation with its matching native DexSim
runtime, Franka assets, and a CUDA-capable GPU. Python 3.11, PyTorch 2.7/CUDA 12.8,
and LeRobot 0.4.4 were used for the retained experiments.

```bash
python -m pip install -e .
python -m pip install tensorboard viser imageio imageio-ffmpeg
# Required only for direct LeRobot collection/loading:
python -m pip install 'lerobot==0.4.4'
```

DexSim must already be installed/built using the project's installation process.
Its Python facade and native extension must match. An import failure mentioning
Newton render kernels or `open3d.cpu` is an installation failure that occurs
before the demo creates a world. The task itself uses `physics: default`.

Select an available physical GPU before importing Python libraries; it becomes
local `cuda:0` in the processes below. Use a fresh output directory for every run.

```bash
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## 1. Check the engineering loop

```bash
python -m examples.ods.simple_bc_demo.train \
  --smoke --no-viewer --no-video --output /tmp/simple-bc-smoke
```

This uses tiny data and update budgets. It checks expert generation, ODS
sampling, parameter updates, checkpoint reload, evaluation and shutdown. It is
not a learning-quality test. The smoke's evaluation seeds are separate from the
retained experiment's development scenes.

Direct file entry also works:
`python examples/ods/simple_bc_demo/train.py --help`.

## 2. Reproduce the working small-data BC baseline

```bash
python -m examples.ods.simple_bc_demo.train \
  --mode offline-32 --offline-storage lerobot --offline-episodes 32 \
  --buffer-size 32 --producer-envs 8 --batch-size 256 \
  --rounds 1 --updates-per-round 2500 --eval-every 500 \
  --seed 19001 --producer-seed 1019001 --sampling-seed 29001 \
  --defer-evaluation --deferred-evaluation-schedule updates \
  --dev-scenes 20 --dev-seed-base 810000 --final-test-scenes 0 \
  --usage-statistics --no-viewer --no-video \
  --sim-device cuda:0 --train-device cuda:0 \
  --output /tmp/simple-bc-offline32
```

The run qualifies the expert, collects 32 accepted trajectories directly into
`lerobot_dataset/`, finalizes the bounded SDK writer, loads and checks the
numeric data once, then trains from resident CPU tensors. It saves checkpoints
at steps 0/500/1000/1500/2000/2500 and evaluates them after training.
The qualification uses 40 separate scenes and requires at least 38 successes.
Only demonstrations accepted by the task enter the training data.

The retained pre-reorganization pilot achieved 20/20 final development successes
with this small-data recipe. It is one-seed development evidence, not a promise
for every installation or held-out performance. See
[the full comparison](docs/OFFLINE_PILOT_RESULTS.md).

## 3. Run protected online ODS

The configuration input combines two measured rates with sampling metadata,
resource limits, and a versioned per-trajectory reuse profile. See
[the configuration method](docs/CONFIGURATION.md). `configs/throughput.example.json`
contains the previous hardware's provisional measurements; replace its rates
and latency/memory metadata after measuring your deployment.

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

The original input JSON is the training input; the calculator's report is a
reviewable output. Auto-configuration enables usage accounting and deferred
assessment. It derives cache/pacing/refill settings within the profile's bound.
The example K=2660 is provisional, not a universally optimal or safe setting.

To inspect generation throughput, start with
`python -m examples.ods.simple_bc_demo.measure --help`. Re-measure with the actual
producer concurrency, batch size, representation and model placement. Active
learner capacity and wall throughput under deliberate pacing are different.

## 4. Evaluate an existing checkpoint independently

```bash
python -m examples.ods.simple_bc_demo.evaluation \
  --checkpoint /tmp/simple-bc-offline32/step_2500.pt \
  --episodes 20 --seed-base 810000 --sim-device cuda:0 \
  --output /tmp/simple-bc-evaluation
```

This starts a fresh evaluation process, loads the model, executes student
network actions, and writes `evaluation.json` and `cleanup.json`. It does not
create an ODS producer or optimizer. Use trusted checkpoint files written by
this example. Add `--viewer` for Viser, or `--video` to record the first three
scenes, including failures. `--timeout` bounds each worker response.

Success is owned by `environment.py`: the cube must have been lifted, placed
within 3 cm, released with the gripper open and TCP withdrawn, and remain low
speed for 0.3 seconds. Expert acceptance and student evaluation share this
measurement. Planning completion alone is insufficient.

## 5. Compare online and offline runs

Keep seeds, model, normalization procedure, batch/update budgets, hardware and
evaluation panels equal. For a 256-trajectory cold offline arm, change the
baseline command to `--mode offline-full --offline-episodes 256`. For an
equal-quota offline control add
`--offline-sample-budget 2660 --offline-sample-hard-limit 2660`.
Retain the unrestricted 32-trajectory baseline: a small fixed corpus may already
solve this task faster.

```bash
python -m examples.ods.simple_bc_demo.compare \
  /tmp/simple-bc-ods /tmp/simple-bc-offline32 --output /tmp/simple-bc-comparison
```

`--offline-data PATH` loads an existing `.pt` corpus or this example's LeRobot
directory; it measures a data-ready scenario. Cold collection includes its own
acquisition, persistence and loading costs. The old `.pt` corpus format remains
available through `--offline-storage pt`.

Fit normalization only from the first training cache; freeze it thereafter.
Development scenes select checkpoints. Enable a disjoint final panel explicitly
only after freezing the experiment protocol. The retained pilot did not open
final tests. Deferred results describe checkpoint availability retrospectively;
they do not measure online threshold certification.

## Outputs, recovery and display

A run contains `config.json`, resolved `task_config.json`, runtime/source hashes,
expert qualification, checkpoints, per-panel JSON, TensorBoard `events/`, and
`summary.json`. Usage-enabled runs additionally record `usage.json`,
`sampling_budget.json`, timing/resource reports and flow measurements. Inspect
`cleanup.json` and the reported stopping reason after early budget exhaustion.

To reuse a matching expert qualification pass `--qualification-report
PREVIOUS_RUN/expert_qualification.json`; controller/task source and physical
configuration must match. Migration to this local task requires a fresh
qualification. Resume uses `--resume CHECKPOINT` with a fresh output directory
and the same model/sampling settings. ODS is recreated and filled under a new
run ID; the shared cache is not restored exactly. Protected offline resume is
rejected because its per-source quota state is not restored.

Omit `--no-viewer` in training to expose Viser on port 8080 and TensorBoard on
6006. Omit `--no-video` to record the configured comparison videos. Display and
rendering change resource contention; repeat throughput measurements when those
settings change.

## Validation and evidence

```bash
# CPU model/data/CLI contracts, without importing the native simulator:
python -m pytest -q -c /dev/null --noconftest \
  examples/ods/simple_bc_demo/tests \
  --ignore=examples/ods/simple_bc_demo/tests/test_environment.py

# Environment predicates/reset tests, with the native dependencies installed:
python -m pytest -q -c /dev/null --noconftest \
  examples/ods/simple_bc_demo/tests/test_environment.py
```

`docs/` retains the final offline pilot and its accounting audit. Those results
were measured at source revision `3cd6379a`, before this directory reorganization;
paths in the evidence JSON identify that historical execution. Interim designs,
failed-model diaries, and one-off tuning campaigns remain accessible in Git
history. The retained pilot did not establish a general ODS speed advantage;
instrumentation overhead qualification and multi-seed testing remain necessary.

Reorganization checks: 281 CPU model/data/CLI checks and 7 environment checks
passed. A real online smoke consumed two generations and completed 10 updates.
A real cold LeRobot smoke wrote/read 4 episodes (1,330 frames), exhausted its
per-source quota after one update, saved the final checkpoint, and shut down.
The independent evaluator loaded the retained trained model and succeeded on
2/2 development scenes. Those small checks verify integration; the full learning
curves above are the earlier pilot. This host's native DexSim facade is currently
inconsistent; real checks used the same existing compatible runtime overlay as
the earlier experiments, without changing the global installation.
