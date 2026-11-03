# Visual PushCube imitation learning with ACT

This example collects scripted Franka demonstrations, trains a visual Action
Chunking with Transformers (ACT) policy offline, and evaluates the policy in a
fresh simulator. It uses the project task, action manager, LeRobot recorder,
causal data sources, and supervised trainer. The policy sees one RGB camera and
robot joint positions; object/goal geometry is available only to the expert and
the evaluator.

The physical task is inspired by ManiSkill's PushCube and the model follows
[ACT](https://tonyzhaozh.github.io/aloha/) and the
[ManiSkill baseline](https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/act).
This is a separate EmbodiChain task, not a numerical reproduction of ManiSkill.

The default task uses a 4 cm cube, a green 10 cm radius target and randomized
cube XY within ±10 cm. The target is 20 cm in +X from the initial cube. An
episode succeeds when the cube is on the table and its XY distance to the
target is below 10 cm. Simulation/control run at 240/30 Hz; actions are seven
absolute arm joint targets and a normalized parallel-gripper command.

The RL and visual deployments share `extensions.success_threshold` and the
`distance_to_goal` metric. Expert geometry and segment lengths are declared in
`env.visual.yaml`; the task does not supply a second set of expert defaults.

## Requirements

Run from a source checkout with EmbodiChain's matching DexSim runtime, Franka
assets, PyTorch/torchvision, CUDA, the Python-selected LeRobot SDK (0.4.4 on
Python 3.10/3.11, 0.6.1 on Python 3.12), and a native camera renderer.
This example still uses the EmbodiChain ACT implementation; changing the SDK
dependency does not replace its model or qualify its learning results on a new
environment. See the root README for the dependency validation matrix.
The default recipe uses one GPU with 24 GB or more. Select a physical GPU
before starting Python, for example `export CUDA_VISIBLE_DEVICES=0`; use
`cuda:0` in the commands below. The first full training run downloads the
torchvision ImageNet ResNet-18 weights into the Torch cache. `imageio` and
`imageio-ffmpeg` are needed for videos.

Commands use separate fresh output directories. Recorder datasets live inside
their collection output, and `report.json` records the actual dataset path,
ordered schema, attempts, seeds and task identity. Keep reports with datasets.

## 1. Qualify and collect the expert

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

The qualification report must match the current task sources and show at
least 95% success across at least 100 independent attempts. Collection rejects
qualification seed overlap, records only successful full episodes, and stops
after at most four attempts per requested episode. RGB is stored as PNG.

For an engineering-only check use `collect --smoke --episodes 4` with a fresh
output. A smoke run bypasses expert qualification and must not be used as a
learning acceptance result.

## 2. Train

```bash
python -m examples.learning.act.train \
  --train-report /tmp/act-train/report.json \
  --validation-report /tmp/act-validation/report.json \
  --seed 19001 --output /tmp/act-seed19001
```

Repeat with seeds 19002 and 19003 and fresh output directories. Defaults are
30,000 updates, batch size 32, and a saved checkpoint and offline validation
every 2,000 updates. The model predicts 30 actions with ResNet-18, Transformer
and a CVAE; training minimizes masked action L1 plus `10 × KL`. Only training
frames determine state/action normalization. The checkpoint captures model,
optimizer, RNG, action schema, task identity and dataset identity.

`--smoke --updates 20 --batch-size 2 --eval-every 10` uses an uninitialized
backbone to check the engineering loop without downloading weights. Resume is
available with `--resume CHECKPOINT` and the original arguments/output recipe.
The SDK dataset and image files must remain unchanged for exact resume.

## 3. Select and evaluate

Development selection evaluates every periodic checkpoint on the same 50
scenes. It chooses highest success rate, then lowest mean final goal distance,
then the earlier checkpoint. Only development scenes select a model.

```bash
python -m examples.learning.act.evaluate --run /tmp/act-seed19001 \
  --panel development --episodes 50 --seed-base 300000 --output /tmp/act-dev19001

python -m examples.learning.act.evaluate --checkpoint /tmp/act-dev19001/best.pt \
  --panel test --episodes 100 --seed-base 400000 --video --output /tmp/act-test19001
```

Repeat for the other seeds using the **same** development and test scenes.
Selection output contains `best.pt`; evaluation reports each scene, success
rate, Wilson interval, goal distance and policy inference time. Videos include
up to three successes and three failures when available. The reported strict
3 cm result is measured at the ordinary 10 cm termination point; this is a
diagnostic of entry accuracy, not a separately extended rollout.

Compare the same held-out panel with two baselines:

```bash
python -m examples.learning.act.evaluate --baseline fixed --panel test \
  --episodes 100 --seed-base 400000 --output /tmp/act-fixed
python -m examples.learning.act.evaluate --baseline untrained --panel test \
  --episodes 100 --seed-base 400000 --output /tmp/act-untrained
```

The learning acceptance target is ≥90% mean test success across the three
training seeds, ≥80% for every seed, and at least 30 percentage points above the
better baseline. A four-scene training-reset diagnostic should reach at least
3/4 before launching full training. `--panel overfit --episodes 4 --seed-base
<training seed>` runs that diagnostic. Test results must not be used for
checkpoint selection or tuning.

The retained [qualification reports](results/README.md) show the measured
results and their scope for one completed run.

For a development-only visual diagnostic, add `--shuffle-images-report
/tmp/act-validation/report.json` to a single-checkpoint development evaluation.
It replaces each live frame with the corresponding-time frame from a different
recorded validation episode while keeping the live joint state. Compare against
the same checkpoint and development seeds without the flag; this is an ablation,
not a held-out test result.

Evaluation checks the task identity and rejects train/validation or recorded
development seed overlap. Load only trusted checkpoints, since PyTorch training
checkpoints include optimizer and RNG state.

To render a single rollout for inspection (including a training scene), use
`python -m examples.learning.act.play --checkpoint /tmp/act-dev19001/best.pt
--seed 300000 --output /tmp/act-play`. The output contains an MP4 and a JSON
record; single-rollout playback is not an acceptance evaluation.

## Data and resource boundaries

Each sampled training item uses the RGB and joint state at time `t` and up to
30 actions starting at `t`. Episode-end padding is excluded from the posterior
attention and loss. Numeric data is resident in bounded CPU memory; the image
reader decodes only anchor frames and has a 512 MiB CPU cache by default.
Inference uses the CVAE prior mean (`z=0`) and temporally ensembles overlapping
chunks, clearing its history on every reset.

Report peak GPU memory and wall time on the target machine. Acceptance is based
on physical rollouts rather than loss alone. On a physical 24 GB GPU, run the
full recipe there to qualify that hardware independently.
