# Visual PushCube imitation learning with official LeRobot ACT

This example uses EmbodiChain to collect Franka demonstrations and evaluate a
policy in GPU simulation. LeRobot 0.6.1 owns the ACT network, dataset loader,
normalization, loss, temporal ensemble, optimizer updates and native checkpoints.
The example invokes `lerobot.scripts.lerobot_train` directly. The shared
EmbodiChain SL trainer remains available for the dynamics model workflows.

## Environment and control contract

Use Python 3.12 with `lerobot[training]==0.6.1`, DexSim 0.5.1rc1 and the native
camera renderer. The current Python 3.12 dependency profile uses torch 2.11.0+cu128,
torchvision 0.26.0+cu128, TorchCodec 0.11.1 and TensorDict 0.15.3. Install the
main package through the normal resolver using the project's private DexSim index.
For an existing Python 3.12 checkout environment, select matching CUDA wheels:

```bash
uv venv --python 3.12 /tmp/embodichain-act
source /tmp/embodichain-act/bin/activate
uv pip install --torch-backend cu128 -e . \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' \
  'torchcodec==0.11.1' 'tensordict==0.15.3' \
  --index-strategy unsafe-best-match \
  --extra-index-url http://pyp.open3dv.site:2345/simple/
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

Use `cuda:0` after selecting a physical GPU. Each training or evaluation process
uses one GPU. The first full training downloads the pinned torchvision
ResNet-18 ImageNet weights into the Torch cache. Videos require imageio and
imageio-ffmpeg. Reports identify the actual GPU and software; lower-memory
hardware needs its own qualification.

The existing task uses a 4 cm cube, randomized XY within ±10 cm, and a green
10 cm radius target 20 cm in +X from the initial cube. Success requires the cube
to remain on the table and enter that target. Physics/control run at 240/30 Hz.
The policy sees a current 224×224 RGB frame and nine measured joint positions;
its eight commands are seven absolute arm joint targets plus a normalized
parallel-gripper command. Geometry and goal poses are expert/evaluator inputs.

## 1. Qualify and collect

Use fresh output directories. Qualification must succeed on at least 95 of
100 independent attempts and match the current task identity. Collection saves
only successful full episodes with raw policy-request action semantics, with
at most four attempts per requested episode. It stores RGB as image features.

```bash
python -m examples.learning.act.collect qualify \
  --episodes 100 --seed-base 500000 --output /tmp/act-qualification
python -m examples.learning.act.collect collect \
  --episodes 100 --seed-base 100000 \
  --qualification /tmp/act-qualification/report.json --output /tmp/act-train
python -m examples.learning.act.collect collect \
  --episodes 20 --seed-base 200000 \
  --qualification /tmp/act-qualification/report.json --output /tmp/act-validation
```

Keep each `report.json` with its finalized dataset. Reports record successful
and failed attempts, seeds, schema, task identity and runtime. For an engineering
check, `collect --smoke --episodes 4` marks an unqualified corpus; full training
rejects smoke reports.

## 2. Train with the upstream workflow

```bash
python -m examples.learning.act.train \
  --train-report /tmp/act-train/report.json \
  --validation-report /tmp/act-validation/report.json \
  --seed 19001 --output /tmp/act-seed19001
```

Repeat for seeds 19002 and 19003 with fresh outputs. Defaults are 30,000 updates,
batch 32, and a checkpoint every 2,000 updates. The model uses the official
ResNet-18/Transformer/CVAE architecture, a 30-step chunk, and `10 × KL` plus the
SDK's masked L1. AdamW uses learning rates 1e-4 and 1e-5 for the main network
and backbone, weight decay 1e-4, and the upstream preset's gradient clipping.
Training is FP32 with four DataLoader workers; `--num-workers` changes that count.

Only the training dataset is passed to LeRobot. Numeric statistics come from
its training-only metadata; RGB uses ImageNet statistics. Validation is a separate
dataset and never contributes training statistics. After training, the wrapper
records deterministic prior-prediction L1 for each checkpoint in
`offline_validation.json`, using frozen training processors and up to 128 anchors.
Physical development scenes select the final policy.

The wrapper saves `recipe.json`, the exact `lerobot_config.json`, dataset content
fingerprints and `training_report.json`. Native artifacts live under
`lerobot/checkpoints/002000/pretrained_model/` and corresponding update folders.
They contain `config.json`, `model.safetensors`, both processor JSON files and
normalization safetensors, plus an `embodichain_act.json` task/action manifest.
The sibling `training_state/` is owned by LeRobot and supports native recovery.

For a short engineering run, use `--smoke --updates 20 --batch-size 2
--eval-every 10`. This selects a small uninitialized model and zero workers.
Resume from the latest checkpoint with `--resume /path/to/checkpoint/pretrained_model`, the same output and
recipe arguments. The wrapper checks source contents, checkpoint ownership and native configuration, and permits a larger update
budget; the native trainer restores its optimizer, scheduler, RNG and sampler.

The default is four DataLoader workers. Choose a worker count appropriate for
the host CPU and memory; it is recorded in the recipe and must match on resume.

For a learning diagnostic, collect four qualified scenes, train the full model
for 4,000 updates, and evaluate those same four scenes using `--panel overfit`.
At least 3/4 should succeed before spending the full three-seed training budget.

## 3. Select and evaluate in GPU simulation

```bash
python -m examples.learning.act.evaluate --run /tmp/act-seed19001 \
  --panel development --episodes 50 --seed-base 300000 --output /tmp/act-dev19001
python -m examples.learning.act.evaluate --checkpoint /tmp/act-dev19001/best \
  --panel test --episodes 100 --seed-base 400000 --video --output /tmp/act-test19001
```

Development evaluates all native checkpoints on the same 50 scenes, ranking by
success rate, then mean final goal distance, then earlier update. Its `best/`
directory is a portable native policy with processors and provenance. `--policy-dir`
is an alias for `--checkpoint`; paths now refer to directories, not `.pt` files.
`play --checkpoint /path/to/best --seed 300000 --output /tmp/act-play` renders one
MP4 and JSON diagnostic without claiming learning acceptance.

The loader calls LeRobot's native model/processor APIs. RGB is scaled to [0,1]
once; the official preprocessors normalize it and state, `select_action` applies
the official temporal ensemble, and the official postprocessor unnormalizes
actions on the simulation GPU. Every reset clears policy history. Environment
action bounds are applied before stepping, and live schema/task identities must
match the recorded contract. The prior latent is zero during evaluation.

Run the same held-out test panel for all three seeds and both baselines:

```bash
python -m examples.learning.act.evaluate --baseline fixed --panel test \
  --episodes 100 --seed-base 400000 --output /tmp/act-fixed
python -m examples.learning.act.evaluate --baseline untrained \
  --reference-checkpoint /tmp/act-dev19001/best --panel test \
  --episodes 100 --seed-base 400000 --output /tmp/act-untrained
```

The untrained baseline reinitializes ACT with its original backbone initializer
and the same training statistics; it does not load trained weights. The target
is ≥90% mean test success across three training seeds, ≥80% for every seed, and
at least 30 percentage points above the better baseline. Test scenes never
select checkpoints or hyperparameters. Task identity and scene-overlap checks
remain active; `overfit` is the explicit training-scene diagnostic.

Reports contain per-scene outcomes, Wilson intervals, distances and mean/median/
P95 policy latency. Videos retain up to three successes and failures each.
The strict 3 cm metric is measured when the normal 10 cm termination fires.
A development-only `--shuffle-images-report` retains the existing visual ablation.

## Compatibility

The previous EmbodiChain ACT network, SL recipe and custom temporal ensemble
have been replaced. Historical `.pt` checkpoints require retraining; use Git
history to reproduce those models. Keep generated datasets, checkpoints,
evaluation reports and videos in a local output directory outside the checkout.

Core trainer/dataset/robot support on Python 3.11 remains independent of
this Python 3.12 ACT workflow. The focused production tests exercise real SDK
configuration, native training/resume, portable policy reload, normalization,
reset and content checks without starting a simulator. GPU collection and
learning acceptance require actual rendered rollouts.

See the [LeRobot integration guide](../../../docs/source/guides/lerobot.md) for dependency profiles and the [framework comparison](../../../docs/source/guides/training_frameworks.md) for native training and RLinf alternatives.
