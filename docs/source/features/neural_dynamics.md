# Neural Dynamics Models

A neural dynamics model predicts how a robot's physical state changes under
applied joint efforts and contact. EmbodiChain provides a simulation-independent NeRD model, trajectory
storage, training, offline evaluation and portable export workflow. Data generation
uses Newton's built-in analytical solvers directly and does not require a custom
DexSim build.

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

The released datasets were collected through the matching Newton/Scene
integration. The independent collector below builds a new Newton model directly;
its manifests use a separate `nerd-direct-newton-builder-v1` physical identity.
Both datasets share the same state/contact schema and can be trained and evaluated
here. Do not treat different model-building paths or solver versions as identical
physical systems, or use offline transition error as a live rollout measurement.

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

## Use the exported model in PyTorch

Load tensor weights and their recorded normalization buffers:

```python
from embodichain.learning.models.nerd import load_model

model, manifest = load_model(
    "outputs/models/neural_dynamics/nerd/cartpole/v1", device="cpu"
)
```

`NeRDStateCodec` and `encode_batch` construct features from the dataset's state,
effort and contact fields. The model accepts feature windows and valid history
lengths; the training and evaluation workflows use this same preprocessing.

Live Scene stepping, `solver_type: nerd`, and differentiable Gym integration are
provided separately by the [neural dynamics solver PR](https://github.com/DexForce/EmbodiChain/pull/822).
The comparison videos above were produced using that integration; they are
reference recordings, not a command available in this offline workflow.

Keep state transforms in {mod}`embodichain.compute.dynamics`, model/checkpoint
code in {mod}`embodichain.learning.models.nerd`, training in
{mod}`embodichain.learning.dynamics.workflow`, and trajectory storage/generation in
{mod}`embodichain.data_pipeline.dynamics`.
