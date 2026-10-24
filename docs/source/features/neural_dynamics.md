# Neural Dynamics Models

A neural dynamics model predicts how a robot's physical state changes under
applied joint efforts and contact. EmbodiChain supports these models as named
Newton solver extensions, so the simulation world, scene objects, state access,
and environment lifecycle remain shared with analytical physics.

[NeRD (Neural Robot Dynamics)](https://neural-robot-dynamics.github.io/) is the
first implementation. Its causal Transformer consumes a short history of state,
efforts and contact features, then predicts the next state. The current reference
systems are contact-free Cartpole and free-base Ant on a static plane.

## Comparison videos

These recordings compare the learned NeRD transition with its matching Newton
Featherstone solver. Each video shows passive motion, shared random efforts, and
a PD controller. Passive and random-effort comparisons use the same initial
state and applied effort sequence. In the PD segment, each simulator computes
feedback from its own state, so the comparison shows closed-loop behavior.

```{raw} html
<figure>
  <video controls playsinline preload="metadata" style="width: 100%; border-radius: 12px;"
         poster="https://huggingface.co/DexForceAI/embodichain_model/resolve/main/neural_dynamics/nerd/media/cartpole-comparison.webp">
    <source src="https://huggingface.co/DexForceAI/embodichain_model/resolve/main/neural_dynamics/nerd/media/cartpole-comparison.mp4" type="video/mp4">
    Your browser does not support embedded MP4 video.
  </video>
  <figcaption>Cartpole: analytical and learned dynamics.</figcaption>
</figure>
<figure>
  <video controls playsinline preload="metadata" style="width: 100%; border-radius: 12px;"
         poster="https://huggingface.co/DexForceAI/embodichain_model/resolve/main/neural_dynamics/nerd/media/ant-comparison.webp">
    <source src="https://huggingface.co/DexForceAI/embodichain_model/resolve/main/neural_dynamics/nerd/media/ant-comparison.mp4" type="video/mp4">
    Your browser does not support embedded MP4 video.
  </video>
  <figcaption>Ant: analytical and learned contact dynamics.</figcaption>
</figure>
```

Download the [Cartpole video](https://huggingface.co/DexForceAI/embodichain_model/resolve/main/neural_dynamics/nerd/media/cartpole-comparison.mp4)
or [Ant video](https://huggingface.co/DexForceAI/embodichain_model/resolve/main/neural_dynamics/nerd/media/ant-comparison.mp4).
The recordings visualize the released reference models; they are not a claim of
paper-level accuracy or equivalent behavior in arbitrary robot scenes.

## Published datasets and models

Research datasets live in
[DexForceAI/embodichain_datasets](https://huggingface.co/datasets/DexForceAI/embodichain_datasets).
Deployable models live in
[DexForceAI/embodichain_model](https://huggingface.co/DexForceAI/embodichain_model).
Both repositories organize assets by model family, implementation and robot:

| Reference system | Path in each repository | Deployment profile |
| --- | --- | --- |
| Cartpole | `neural_dynamics/nerd/cartpole/v1` | Contact-free, `warp_reference`, 1/60 s |
| Ant | `neural_dynamics/nerd/ant/v1` | Static plane, `warp_reference`, 1/60 s |

Dataset directories contain a manifest and HDF5 shards with explicit
train/validation/test trajectory splits. Each trajectory stores T+1 states and T
efforts/contact inputs. Windows cannot cross trajectory or split boundaries.
Model directories contain `weights.pt` and `manifest.json`, including network
configuration, normalization buffers, codec, timestep and physical identity.
These are inference exports; resumable training checkpoints are local training
outputs, not part of the published model package.

The dataset and model cards describe the generating solver, supported dependency
versions, data size and evaluation limits. In particular, long-horizon Ant
predictions can drift. Assess the model against its matching analytical solver
for the intended motion and prediction horizon before using it downstream.

## Download and evaluate

Run from the repository root. The Hugging Face client is an EmbodiChain
dependency. Download one matching dataset/model pair:

```python
from huggingface_hub import snapshot_download

robot = "cartpole"  # or "ant"
prefix = f"neural_dynamics/nerd/{robot}/v1"
snapshot_download(
    repo_id="DexForceAI/embodichain_datasets",
    repo_type="dataset",
    allow_patterns=[f"{prefix}/*"],
    local_dir="outputs/datasets",
)
snapshot_download(
    repo_id="DexForceAI/embodichain_model",
    allow_patterns=[f"{prefix}/*"],
    local_dir="outputs/models",
)
```

Evaluate the saved test transitions without starting a simulator:

```bash
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.eval.pretrained.yaml
```

The pretrained configuration reads the downloaded paths above. Replace
`cartpole` with `ant` for Ant. Use `--device cpu` or `--device cuda:0` to choose
the tensor device. Offline evaluation measures held-out transition prediction;
it does not measure accumulated rollout error.

For a live rollout, install a DexSim build exposing named Newton solver
extensions and the Scene snapshot/functional APIs. Use the matching reference
scene and physics profile:

```bash
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.rollout.pretrained.yaml
```

This command advances both the released model and a matching Newton teacher at
the same macro timestep. The preset evaluates 16 trajectories at horizons 1, 10
and 100. It saves the report, per-trajectory metrics and raw state evidence.
Select a fresh output path with `--output` when repeating an evaluation.

The released models were trained for `physics_profile: warp_reference`, which
reconstructs a specific reference asset and material configuration using current
Newton. The analytical teacher uses five substeps for Cartpole or sixteen for
Ant, while NeRD predicts the entire 1/60-second interval in one neural substep.
The default `native`/MuJoCo workflow configurations describe a different physical
system and do not match these released exports.

## Generate data, train and export

The workflow configurations are under
`extensions/embodichain_tasks/configs/tasks/classic_control/<robot>/`.
They configure `embodichain dynamics` commands and are separate from runnable
Gym deployment configs.

For a short local workflow, use the smoke presets:

```bash
embodichain dynamics generate --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.generate.smoke.yaml
embodichain dynamics train --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.train.smoke.yaml
embodichain dynamics export --checkpoint outputs/dynamics/cartpole/smoke/train/best.pt --output outputs/dynamics/cartpole/smoke/export
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.eval.smoke.yaml
```

Smoke presets use a small network and short budget to exercise the workflow.
Full presets use the reference network scale and a substantially longer budget.
Neither preset name is an accuracy qualification.

To train on a downloaded dataset, copy `dynamics.train.full.yaml`, then set
`dataset` to the downloaded robot's `v1` directory and `output_dir` to a new
training directory. Preserve its model feature dimensions and history length.
Run the copied configuration with `embodichain dynamics train --config PATH`,
then export its validation-selected `best.pt` with the `export` command.

Training checkpoints retain model, optimizer, normalization, RNG and committed
sampling progress. Continue the same run with `train --resume PATH/last.pt`.
Initialize a new run from an inference export with `train --fine-tune PATH`.
Export destinations must be new directories. Paths written in configuration
files resolve relative to the file; explicit CLI path overrides resolve relative
to the working directory.

## Deploy NeRD in a matching scene

An installed solver entry point makes `nerd` available through the same Newton
configuration used for analytical solvers:

```yaml
physics: newton
physics_config:
  physics_dt: 0.016666666666666666
  num_substeps: 1
  solver_cfg:
    solver_type: nerd
    options:
      checkpoint: outputs/models/neural_dynamics/nerd/cartpole/v1
```

The scene must match the export's ordered joints, gravity, contact schema,
physical signature and timestep. The pretrained rollout presets construct that
reference scene. A model cannot be attached to a different importer/material
profile by changing its manifest or scaling its predicted state delta.

NeRD consumes generalized efforts. It rejects unsupported free-base and external
forces and does not provide physical contact impulse or friction-force reports.
The current provider runs eager FP32 inference and does not support CUDA Graph
capture. Compare performance with the analytical solver's supported Graph mode
enabled, keeping latency and throughput scopes explicit.

## Functional dynamics and gradients

{class}`~embodichain.lab.gym.envs.differentiable_dynamics_env.DifferentiableDynamicsEnv`
and {class}`~embodichain.lab.sim.diff.dynamics.FunctionalDynamics` expose
graph-connected physical state, per-world history and valid history lengths.
The ordinary native step and functional step share the same learned transition;
publishing a functional result does not advance the world again.

The Torch/Warp FK boundary supplies first-order derivatives. Shared tensor
storage alone does not establish a gradient path. Continuous state and contact
geometry have local derivatives, while contact selection, angle wrapping and
quaternion signs contain discrete branches. Physical parameters and contact
topology are not differentiable through this provider.

Reset clears the selected worlds' histories while preserving other rows and
already-computed terminal rewards. Call `detach_state()` at an intentional
rollout boundary, and update trainable model parameters only after backward.

## Add another neural solver

Implement the DexSim solver-provider contract and expose it through an installed
entry point:

```toml
[project.entry-points."dexsim.newton_solvers"]
my_solver = "my_package.solver:MySolverProvider"
```

The provider declares builder attributes and capabilities, then creates its
solver after the scene model is finalized. Mutable histories belong to each
World. The configured `solver_type` is the entry-point name; auto-selection does
not choose a learned model implicitly. Duplicate or unknown names fail.

Keep state transforms in {mod}`embodichain.compute.dynamics`, model/checkpoint
code in {mod}`embodichain.learning.models.nerd`, offline training in
{mod}`embodichain.learning.dynamics.workflow`, and scene adaptation in
{mod}`embodichain.lab.sim.physics.solvers`. Dataset storage belongs to
{mod}`embodichain.data_pipeline.dynamics.shards`.
