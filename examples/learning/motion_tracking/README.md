# G1 motion tracking with an online world model

This example collects physical G1 transitions, fits a world model with supervised
learning, and updates an APG Actor-Critic policy through imagined trajectories.
The official `UnitreeG1MotionRL-v1` task provides physical collection and independent
policy evaluation. The published model supports 29-joint G1 motion tracking at
50 Hz.

The commands below use the installed `embodichain run-example` entry. Data and
outputs can live outside the source checkout.

## Install the runtime

Use the candidate source revision and the matching runtime wheels supplied with
the review build. From the repository root, place those wheels in
`runtime-wheels/`, then install EmbodiChain and its bundled tasks and models:

```bash
RUNTIME_WHEELS="$PWD/runtime-wheels"
uv venv --python 3.11 .venv-g1
uv pip install --python .venv-g1/bin/python --torch-backend cu130 \
  --find-links "$RUNTIME_WHEELS" .
uv pip check --python .venv-g1/bin/python
source .venv-g1/bin/activate
```

Use an isolated environment for each candidate runtime build. Record the source
revision and wheel filenames with the run results. The runtime build must provide
native libraries compatible with the host's Python, CUDA driver and Linux ABI.

The declared runtime includes `dexsim_engine==0.5.1`, PyTorch 2.10, TorchVision
0.25 and TorchCodec 0.10. The CUDA 13.0 selection above matches the NVIDIA RTX PRO
5000 training environment; select a PyTorch backend supported by the GPU and
driver in use. The complete training recipe was run on a single 72 GiB GPU.
The functional recipe uses smaller networks, batches and replay storage.

TorchCodec requires FFmpeg shared libraries, and the simulator requires EGL/OpenGL
runtime libraries. On Ubuntu, install them with
`sudo apt install ffmpeg libegl1 libgl1 libglu1-mesa libopengl0 libsm6 libvulkan1`. Conda also provides FFmpeg through
`conda install -c conda-forge ffmpeg`. FastRT replay uses an NVIDIA GPU.

Create a workspace and select its asset cache:

```bash
mkdir -p g1-workspace
cd g1-workspace
export EMBODICHAIN_DATA_ROOT="$PWD/data"
```

Keep this environment variable set for data preparation, evaluation, training and
replay. Without it, assets use `~/.cache/embodichain_data`.

## Prepare the motion data

```bash
embodichain run-example learning/motion_tracking/prepare_lafan1.py
```

The entry downloads the fixed public source revision and prepares
`$EMBODICHAIN_DATA_ROOT/LAFAN1G1/`:

| Output | Contents |
|---|---|
| `motion.npz` | 40 clips, 441,140 frames, 29 joints, 50 Hz, XYZW root quaternions and world-frame velocities |
| `robot/` | G1 URDF, meshes and robot-local notices |
| `preparation.json` | Source revision, clip/frame/joint counts and preparation parameters |
| `SOURCES.md`, `sources/retargeting/` | Source URLs, revision and upstream notices |

The data sources are:

- Original motion capture: [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), under [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt).
- G1 retargeting project: [Unitree Robotics LAFAN1 Retargeting Dataset](https://www.modelscope.cn/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset).
- Download mirror: [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74), fixed at revision `ce1572906efe6157840e8474d5a0d7aa87481e74`.

Preparation resamples the supplied 30 Hz joint trajectories to 50 Hz on the local
machine and retains their source notices. The robot description retains its
upstream BSD 3-Clause notice. `--source` accepts an existing source directory;
`--cache-dir` selects a Hugging Face download cache.

## Download and evaluate the pretrained model

Download the published seed-0 inference bundle:

```bash
embodichain run-example learning/motion_tracking/download_pretrained.py \
  --revision e08d6548315be2b24a401e93dcb6434a614ea49f \
  --output models/g1-motion
MODEL=$(python -c 'import json; print(json.load(open("models/g1-motion/download.json"))["checkpoint"])')
```

The [model bundle](https://huggingface.co/DexForceAI/embodichain_model/tree/main/neural_dynamics/motion_tracking/g1/v1)
contains actor, critic and world-model weights, normalization statistics, portable
configuration and evaluation results. The download record stores its immutable
revision and checkpoint path. Checkpoint loading uses `weights_only=True`.

Evaluate ten seconds of walking and save the states needed for replay:

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/evaluate_online.py "$MODEL" \
  --clips walk1_subject1 --max-steps 500 \
  --output outputs/pretrained-walk/evaluation.json \
  --trajectories outputs/pretrained-walk/trajectory.npz
```

For the complete 40-clip evaluation, omit `--clips` and `--max-steps`:

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/evaluate_online.py "$MODEL" \
  --output outputs/pretrained-full40/evaluation.json \
  --trajectories outputs/pretrained-full40/trajectory.npz
```

The published seed-0 model completed 40/40 full clips with a macro normalized
return of 0.94054 on this motion library. The model bundle's `evaluation.json`
records per-clip results and the measurement environment.

Ordinary evaluation uses `--mode quality`. Each clip starts at its first frame
with nominal physical parameters and deterministic FP32 policy actions, and runs
to reference completion or physical failure. Normalized return divides cumulative
reward by the full reference length times the maximum step reward; the macro score
weights clips equally. Six tracking RMSE values cover usable successor frames,
including the terminal frame. Body pose errors align reference heading and
horizontal root position; root pose and body velocity errors use world coordinates.
`measured_steps` gives each error's denominator. Short evaluations record
`full_horizon: false`. `--mode physics-regression` enables deterministic Warp
execution for physical regression checks.

## Render the recorded policy with FastRT

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/replay.py \
  outputs/pretrained-walk/trajectory.npz \
  --clips walk1_subject1 --seconds 10 \
  --output outputs/pretrained-walk/replay.mp4
```

Replay displays the recorded policy states with a scene robot, a green reference
robot marker with scene lighting, red policy tracking points and green reference
tracking points. The trajectory archive stores both states and their matched
reference, joint names and tracked-link names.
Playback writes the recorded root and joint states through the robot API and
synchronizes rendering without advancing physics. The opaque reference robot uses
scene lighting; tracking points and floor traces use the overlay layer. On Linux
without a display, the entry selects SDL offscreen
initialization; an explicitly configured SDL video driver is preserved.

Create a labelled multi-clip video from the full evaluation archive:

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/replay.py \
  outputs/pretrained-full40/trajectory.npz \
  --clips walk1_subject1,dance1_subject1 --seconds 10 \
  --output outputs/pretrained-full40/playlist.mp4
```

`--resolution 1920 1080`, `--spp 8` and `--denoising dlss` control rendering.
The playlist concatenates independently evaluated clips; each segment shows its
clip name, reference time and the next clip countdown. The upper-left World XY
panel uses fixed bounds for each recorded clip and occupies 300 × 350 pixels at
1080p. The panel and ground traces show three seconds of executed motion and two
seconds of future reference in world coordinates. The green robot and reference
tracking points align heading and horizontal root position for body-pose comparison.
The panel reports world root XY error and aligned tracked-link RMS position error
in centimetres.

## Check world-model predictions

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/evaluate_world_model.py "$MODEL" \
  --control policy --num-envs 80 --collect-steps 128 --batch-size 128 \
  --horizons 1 8 16 32 --seed 10042 \
  --output outputs/pretrained-world-model
```

The evaluator collects fresh physical episodes, samples windows within each
episode, and supplies measured actions to open-loop model rollouts. It reports
state errors, reward errors and termination disagreement at each horizon, alongside
a state-hold predictor using the same windows. `--control reference` samples
reference-PD transitions. The output records the selected motion data, hardware,
software, sample counts and checks that model weights and normalization statistics
remain unchanged. These measurements cover fresh trajectories from the training
motion library.

## Train and resume

The online recipe uses the shared `trainer / policy / algorithm` configuration
blocks. `trainer.learning_env` selects the differentiable model environment;
`policy` selects the goal-conditioned actor and critic; `algorithm` configures APG
Actor-Critic and its two optimizers. `workflow.collection_env` and
`workflow.evaluation_env` separately select physical task configurations. The
workflow builds one WM and Replay, then injects that WM, motion library, task and
Replay start provider into the registered learning environment. The model is
frozen throughout each policy update.

`workflow` also owns measured batch size, replay, WM fitting, update ratio and run
budget. `trainer.num_envs` is the imagined rollout batch size. The saved
`config.yaml` contains the complete resolved recipe; `run-manifest.json` records
the execution environment. Existing checkpoints with flat recipe fields are
normalized when loading their learning state.

Configuration paths below resolve through the installed task package:

```bash
CONFIGS=extensions/embodichain_tasks/configs/tasks/locomotion/tracking/g1_motion/agents
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/train.py \
  --config "$CONFIGS/model_based.yaml" --output outputs/functional
```

The functional recipe runs two clips with 128 physical environments. It completes
200 collection iterations, 210 world-model updates and 80 actor updates, and writes
an independent evaluation. Inspect `invalid_transitions`, `apg/skipped_update` and
`apg/critic_skipped_updates` alongside the optimizer counters in `result.json`.

Continue the same run to a total of 220 collection iterations:

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/train.py \
  --config "$CONFIGS/model_based.yaml" --output outputs/functional \
  --iterations 220 --resume outputs/functional/latest.pt
```

`latest.pt` restores networks, normalizers, optimizers, schedule counters, random
state and adaptive start-sampling statistics. Recovery constructs a physical scene
and refills replay before continuing. The published inference bundle contains
weights and configuration; optimizer recovery uses a training run's `latest.pt`.

| Configuration | Purpose | Clips | Environments | Collection iterations | WM updates per actor update |
|---|---|---:|---:|---:|---:|
| `model_based.yaml` | Functional run | 2 | 128 | 200 | 2 |
| `model_based_development.yaml` | Development training | 4 | 512 | 2,000 | 10 |
| `model_based_full.yaml` | Complete training | 40 | 4,096 | 60,000 | 10 |

Each iteration collects one vector step. `utd: 0.4` schedules two actor updates per
five collection steps. All recipes prefill replay for 64 steps; functional and
development recipes then perform 50 and 500 model-only warmup updates. The full
recipe starts joint online updates after prefill and completes 240,000 WM,
24,000 actor and 480,000 critic updates. Each actor update uses a 32-step imagined
trajectory. The full recipe compiles the actor, WM objective and imagined transition;
first calls include compilation and CUDA graph preparation.

Run complete training with a new output directory and explicit seed:

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/train.py \
  --config "$CONFIGS/model_based_full.yaml" --seed 0 --output outputs/full-seed0
```

To train a new policy through the published, fixed WM:

```bash
CUDA_VISIBLE_DEVICES=0 embodichain run-example learning/motion_tracking/train.py \
  --config "$CONFIGS/model_based_fixed_wm.yaml" --wm-checkpoint "$MODEL" \
  --output outputs/fixed-world-model
```

This configuration updates the actor and critic while holding the world-model
weights and statistics fixed. The model architecture and robot must match the
selected checkpoint.

## Run records

| Output | Contents |
|---|---|
| `config.yaml` | Resolved recipe and seed |
| `metrics.jsonl` | Phase, update counts, WM/APG metrics, physical reward and failures, timing and resource samples |
| `latest.pt` | Complete learning state; replay is reconstructed on resume |
| `policy-NNNNNN.pt` | Evaluation weights and task configuration |
| `evaluation-*.json` | Per-clip return, failures, tracking errors and macro score |
| `environment-*.json` | Dataset selection, hardware, imported package versions, precision and source checkout |
| `result.json` | Actual optimizer counters, phase times, replay storage, Torch peaks, native GPU samples and CPU peak RSS |

Timing records identify their process invocation. `apg` includes actor rollout and
critic work; its subtimings are components of that total. Torch allocator peaks and
sampled native process GPU allocation describe separate memory measurements.
`optimizer_steps` records minimum and maximum Adam step counters across parameters.

## Components and gradients

| Component | Responsibility |
|---|---|
| `MotionLibrary` | Reference queries, clip boundaries and adaptive start sampling |
| `UnitreeG1MotionRL-v1` | Robot/control mapping, physical reset and measured transitions |
| Shared tracking task | Observations, future goals, rewards and termination from tensor states |
| `KinematicWorldModel` | Kinematic features, normalized state increments and integration |
| `RobotWorldModelObjective` and `SupervisedLearner` | Eight-step measured windows and WM optimization |
| Model environment | Replay starts, history, reference progress and differentiable stepping |
| Motion-tracking policy | Goal encoder, action decoder and scalar critic |
| APG Actor-Critic | Differentiable rollout, actor objective, lambda-return targets and critic regression |
| Online schedule and `train.py` | Collection/update order, normalization cadence, records and recovery |

The full model has 9,166,918 WM parameters, 9,214,877 actor parameters and 291,585
critic parameters. The actor encodes 730 goal features into a 128-dimensional latent,
then combines it with 930 history features to produce 29 actions. Its encoder and
decoder use hidden widths 2048/1024/512/256. The critic maps 1,009 privileged features
through 256/128 hidden units to one value.

Replay stores detached physical transitions. WM fitting differentiates through
eight predicted steps. During the actor's 32-step rollout, WM and critic parameters
and normalization statistics are fixed, while derivatives through actions, states,
rewards, continuation and terminal value remain connected. Actor backward finishes
before critic normalization changes. Critic inputs and lambda-return targets are
detached for value regression. This recipe uses APG pathwise gradients; `ppo.yaml`
provides the separate PPO task recipe.

Control uses unclamped PD position targets with physical joint and effort limits.
The legs use absolute setpoints; the upper body uses reference residuals. The
policy receives robot-relative root position and orientation goals, and adaptive
sampling caps individual difficult-bin probabilities.
