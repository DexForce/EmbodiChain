# Sharpa in-hand rotation with APG

This example trains a Sharpa Wave right hand to rotate cylinders in hand. Physics
runs natively through EmbodiChain `SimulationManager` on Newton/MuJoCo-Warp.
The pinned upstream HORA teacher, world model and replay from
[Mingrui-Yu/sharpa_rl_unilab](https://github.com/Mingrui-Yu/sharpa_rl_unilab)
are reused, with actor updates through EmbodiChain APG and a graph-connected
terminal value bootstrap.

The workflow has three steps:

1. `reference.py` trains the pinned upstream reference teacher. Its saved
   training config parameterizes the native run.
2. `native_full.py` trains a fresh actor with eight cylinder sizes and full
   domain randomization in native physics.
3. `native_full_evaluate.py` evaluates a saved checkpoint in a fresh scene.

## Requirements

- A clean checkout of the upstream reference at
  `9816885a3b1fbdb947db71e72c27ed75effcbf9f`, synchronized with its locked
  environment:

  ```bash
  git clone https://github.com/Mingrui-Yu/sharpa_rl_unilab /path/to/sharpa-reference
  git -C /path/to/sharpa-reference checkout 9816885a3b1fbdb947db71e72c27ed75effcbf9f
  uv --directory /path/to/sharpa-reference sync --frozen
  ```

- For steps 2 and 3, a Python environment with simulation-capable EmbodiChain
  and DexSim plus the upstream learning dependencies, with the upstream `src/`
  on `PYTHONPATH`.

## 1. Reference teacher

```bash
python examples/learning/sharpa_inhand/reference.py \
  --upstream /path/to/sharpa-reference \
  --output outputs/sharpa_reference_teacher_seed1
```

The runner requires clean pinned sources and the upstream locked interpreter.
It records source and asset hashes, installed packages, the effective config,
upstream test logs, checkpoints and the original evaluator's result. It trains
the original SuperMimic teacher on MJWarp with 2048 environments, seed 1 and
3000 iterations, keeping upstream defaults for physics, randomization, rewards,
models and learning parameters. `--verify-only` runs the checks without
training. `training/config.yaml` in the output is the `--reference-config` for
the native steps.

## 2. Native eight-scale training

```bash
python examples/learning/sharpa_inhand/native_full.py \
  --upstream /path/to/sharpa-reference \
  --reference-config outputs/sharpa_reference_teacher_seed1/training/config.yaml \
  --output outputs/sharpa_native/full_seed1 \
  --num-envs 256 --iterations 3000 --seed 1 --resource-profile reference_host
```

The scene contains all eight fixed cylinder sizes in one manager and samples
their raw grasp caches. PD gains, friction, mass, COM, gravity direction and
persistent object forces are randomized. The reference MJCF is compiled only
to translate the hand into a native articulation source with source joint
gains, armature and passive friction. Zero world gravity plus per-step cylinder weight
reproduces the source's hand gravity compensation. Control runs at 20 Hz with
integrated targets and the source 0.9 actuator-range scaling. Observations use
the source HORA frames; tactile channels use native geometry-pair contact net
forces on the elastomer pads. Resets write gains once per batch and preserve
cached joint-limit excursions exactly (`clamp_qpos=False`).

This is a native variant of the upstream task, not strict physics parity. Hand
self-contacts use `condim=6`, metal torsional/rolling friction `0.1/0.02` and
impedance `0.95/0.99`, whereas upstream hand geoms use `condim=3`, metal
friction `0.005/0.0001` and impedance `0.9/0.95`. Native summed joint drive
torques are capped at 1000 Nm. Dependency versions and learner resource sizes
also differ, so a result here is not an algorithm comparison.

Resource profiles:

- `reference_host` (default) keeps the source 2048-wide world model,
  2048-sample world-model batches, 1024-sample policy batches and 2,048,000
  replay transitions. Replay lives on CPU and world-model activations are
  recomputed during backward, trading host RAM and transfers for lower VRAM.
- `reference_eager` keeps the same sizes with replay on the GPU.
- `shared` uses a 1024-wide world model, 256/128-sample batches and 262,144
  replay transitions for a shared GPU.

`--min-free-gpu-gib` sets a free-memory wait before training and control steps.
It does not reserve memory against other processes. For a wiring check, add
`--smoke --iterations 4 --num-envs 16`; this checks collection, fitting and an
independent APG actor-gradient reduction with a small model.

Host-mode checkpoints also save `native_replay_latest.pt`, paired with the
model by SHA256. To resume into a new output directory with the saved
environment count and profile:

```bash
python examples/learning/sharpa_inhand/native_full.py \
  --upstream /path/to/sharpa-reference \
  --reference-config outputs/sharpa_reference_teacher_seed1/training/config.yaml \
  --resume-native RUN/native_teacher_03000.pt \
  --resume-replay RUN/native_replay_latest.pt \
  --output outputs/sharpa_native/continued \
  --num-envs 256 --iterations 5000 --resource-profile reference_host
```

The iteration count is cumulative, and the saved learning configuration and
seed are reused. Without `--resume-replay`, models and optimizers recover but
replay is refilled. The physical state and its local RNG restart, so this is not
an exact trajectory resume. Changed physical sources or adapter files are
rejected.

## 3. Evaluation

```bash
python examples/learning/sharpa_inhand/native_full_evaluate.py \
  --checkpoint outputs/sharpa_native/full_seed1/native_teacher_03000.pt \
  --upstream /path/to/sharpa-reference \
  --output outputs/sharpa_native/full_seed1_eval
```

The evaluator checks the checkpoint's physical source hashes and native
adapter snapshots before building a fresh scene. Each episode lasts 20 seconds.
A pass needs at least 90% stable scenes and at most 5% drops, both overall and
for every size. A stable scene survives, turns at least 2π and keeps rotating in
the final 5 seconds. Each reset seeds the Torch/NumPy observation noise and the
task's physical generator, so use disjoint seeds for validation and holdout.
`--allow-reference-diagnostic` and `--allow-implementation-transfer` are
transfer diagnostics only; they do not qualify unchanged training and
evaluation physics.

## Measured result

One seed-2 training run, started from a randomly initialized actor, passed the
rotation gates at iteration 4500. On an unseen test of 1536 episodes (192 per
size, 20 s each) it reached 98.57% stable rotation and 1.43% drops, with a mean
net world-Z rotation of 12.24 rad. All eight sizes passed separately.

The initial states come from the bundled grasp caches. This shows in-hand
manipulation from cached grasps, not grasp acquisition from a table. The teacher
uses privileged state; student distillation and hardware deployment are out of
scope. The run needed more than the source 3000-update budget, and it is one
seed in a native physics variant, not a controlled comparison of algorithm
quality or sample efficiency.

## Resources

- Hand: the Sharpa Wave MJCF bundled with the pinned upstream reference
  (`src/sharpa_rl_unilab/assets/robots/sharpa_wave`), compiled with MuJoCo only
  for asset translation.
- Grasp caches: bundled with the pinned upstream reference. The upstream package
  attributes them to `unilabsim/unilab-caches`; using them does not relicense
  that data.
- Cache poses use XYZ+WXYZ and are converted once to EmbodiChain XYZ+XYZW at
  reset. Joints are mapped by name, never by native traversal order.
