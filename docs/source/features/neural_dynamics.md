# Neural Dynamics Models

A neural dynamics model predicts physical state changes under joint efforts and
contact. EmbodiChain bundles the simulation-independent NeRD network, codec,
trajectory reader and SL recipe in `embodichain_models.dynamics.nerd`. Shared
supervised updates and recovery remain in `embodichain.learning.sl`; general
MLPs remain in `embodichain.learning.models`.

[NeRD (Neural Robot Dynamics)](https://neural-robot-dynamics.github.io/) uses a
causal Transformer over state, effort and contact histories. The current
reference systems are contact-free Cartpole and free-base Ant on a static plane.
The `learning/nerd` showcase owns preparation, collection, training commands and
reference research. Its default training command uses the common SL recipe.

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

## Pinned resources

The showcase prepares these resources at fixed revisions:

| Resource | Repository | Revision |
| --- | --- | --- |
| Dataset | [DexForceAI/embodichain_datasets](https://huggingface.co/datasets/DexForceAI/embodichain_datasets) | `b9ec58f36da99cc99a93f97adfe74ebc9fd66586` |
| Portable model | [DexForceAI/embodichain_model](https://huggingface.co/DexForceAI/embodichain_model) | `78e581526570069410bf269b38c0c52c800b24a7` |
| Raw teacher assets | [DexForceAI/embodichain_data](https://huggingface.co/DexForceAI/embodichain_data/tree/97e202e41fe7fb504656a6b163291305e400780d/demo/nerd/reference_assets/v1) | `97e202e41fe7fb504656a6b163291305e400780d` |

Datasets and models use `neural_dynamics/nerd/<robot>/v1`, where `<robot>` is
`cartpole` or `ant`. Raw assets use `demo/nerd/reference_assets/v1`; the verified
archive includes the URDF/XML, attribution and license. These downloaded
payloads are stored outside Git and the wheel. The model implementation and its
third-party license ship in the main EmbodiChain wheel with the same version.

Dataset manifests publish trajectory-level train/validation/test splits and
physical provenance. Trajectories have T+1 states and T effort/contact inputs;
windows cannot cross trajectories or splits. Portable models contain
`weights.pt` and `manifest.json`, with the network, normalization, codec,
timestep and physical identity. They are inference exports, not resumable
training checkpoints. Dataset/model cards describe dependency and evaluation
limits; long-horizon Ant predictions can drift.

## Prepare and evaluate

These commands also work from an installed wheel outside the checkout:

```bash
embodichain run-example learning/nerd prepare --robot cartpole --resource-root ./outputs/nerd-resources
embodichain run-example learning/nerd eval --robot cartpole --resource-root ./outputs/nerd-resources --device cpu
```

Use `--robot ant` for Ant. `prepare` verifies pinned manifests, shards, weights
and reference asset hashes. Add `--assets-only` to prepare only teacher assets,
or `--local-files-only` to require an existing verified cache without network
access. Without `--resource-root`, the showcase uses
`EMBODICHAIN_NERD_RESOURCE_ROOT`, otherwise the cache under `XDG_CACHE_HOME` or
`~/.cache/embodichain/examples/nerd`. Evaluation writes
`outputs/nerd/<robot>/offline.json` unless overridden with `--output`.

Offline evaluation measures held-out transitions and does not start a simulator.
It does not establish accumulated rollout accuracy. Published Scene-generated
trajectories retain their original physical identity; the independent direct
Newton collector uses `nerd-direct-newton-builder-v1`. A shared schema or profile
name does not make different solvers, versions or construction paths identical.

## Train, recover and export

After preparation, train through the default common SL recipe:

```bash
embodichain run-example learning/nerd train --robot cartpole --resource-root ./outputs/nerd-resources --device cuda:0 --output-dir ./outputs/nerd/cartpole/train
embodichain run-example learning/nerd export --checkpoint ./outputs/nerd/cartpole/train/last.pt --output ./outputs/nerd/cartpole/export
embodichain run-example learning/nerd eval --robot cartpole --resource-root ./outputs/nerd-resources --device cpu --model ./outputs/nerd/cartpole/export --output ./outputs/nerd/cartpole/trained-eval.json
```

`--device cpu` is also supported for training. The shipped `sl.train.yaml`
configures 1,000 updates; this bounded example does not claim full convergence.
The default output is `outputs/nerd/<robot>/train`. Resume with the same recipe
and resources using `train --resume PATH/last.pt`; choose an artifact-free output
for a new run. Recovery checks data, recipe, feature order and run identity.
Different or newer metric history is preserved and requires a fresh destination.
The final SL checkpoint is not a claim of validation-selected accuracy.

The generic entry point accepts the same installable factory without example
bootstrap. Copy `examples/learning/nerd/configs/<robot>/sl.train.yaml`, set
`data.path` to the prepared dataset and `trainer.output_dir` to a fresh output,
then run:

```bash
embodichain train-sl --config /path/to/recipe.yaml
embodichain train-sl --config /path/to/recipe.yaml --resume /path/to/train/last.pt
```

Relative paths in a common SL recipe resolve from the working directory. Its
factory is `embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer`.
Export accepts both SL checkpoints and historical NeRD checkpoints and preserves
floating dtype, normalization, ordered features and attention policy. Export
destinations must be new directories.

## Teacher collection and historical experiments

Prepare reference assets before invoking the independent collector:

```bash
embodichain run-example learning/nerd prepare --robot cartpole --resource-root ./outputs/nerd-resources --assets-only
embodichain run-example learning/nerd generate --robot cartpole --resource-root ./outputs/nerd-resources --device cpu --output-dir ./outputs/nerd/cartpole/generated
```

The default collection preset is a bounded smoke workflow. Generation uses
Newton's built-in analytical solvers without DexSim's custom solver/Scene
extension. Its manifest commits the candidate cursor and rejection evidence
with shards. Rejection-only progress preserves data identity; legacy sidecars
remain readable, but resuming requires matching collector source and physical
identity.

Explicit `dynamics.train.*.yaml` configurations in the showcase select the
historical trainer. It retains its NeRD checkpoint format and comparison
sampling/scheduling, including historical fine-tuning and shared sample-order
options. Its `best.pt` is a model-only selected snapshot for export; `last.pt`
is resumable and retains the selected best weights. These options are separate
from default SL training. See the showcase's [research protocol](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/nerd/research/acceptance.md)
for frozen source revisions, paired-seed gates and historical evidence limits.
Moving code or rerunning a smoke does not rerun those frozen experiments.

## Use the model in PyTorch

```python
from embodichain_models.dynamics.nerd import load_model

model, manifest = load_model(
    "outputs/nerd-resources/models/neural_dynamics/nerd/cartpole/v1",
    device="cpu",
)
```

The codec and `integration.encode_batch` construct named feature windows from
state, effort and contact records. `model(inputs)` predicts every token;
`model.evaluate(inputs, valid_lengths)` selects each row's last valid history
position. Both preserve input gradients. The [official model API](../api_reference/model_catalogs.rst)
describes the runtime model, data and objective components.

Live Scene stepping and differentiable Gym integration are separate consumers,
tracked in the [neural dynamics solver PR](https://github.com/DexForce/EmbodiChain/pull/822).
The comparison videos above remain reference recordings from that integration;
they do not claim that this offline showcase provides a native simulation solver
or reproduces paper-scale training.
