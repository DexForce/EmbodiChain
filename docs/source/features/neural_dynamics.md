# Neural Dynamics Models

Neural dynamics models learn how physical systems evolve from observations,
controls and physical context. EmbodiChain includes three model families:
**NeRD** for robot dynamics with state and effort histories, **RWM** for
action-conditioned robot state prediction, and **RigidFormer** for the motion
of interacting rigid geometry.

The models ship as `embodichain_models.dynamics.<family>` in the main
`embodichain` wheel. Import the family you need explicitly. Model imports do
not start a simulator; datasets, reference assets and pretrained weights are
prepared separately by each example.

## Choose a model

| Capability | [NeRD](neural_dynamics/nerd.md) | [Robot world model (RWM)](neural_dynamics/rwm.md) | [RigidFormer](neural_dynamics/rigidformer.md) |
| --- | --- | --- | --- |
| Prediction subject | Articulated robot state | Robot state at a fixed control period | Interacting rigid objects |
| Architecture | Causal Transformer over history | MLP with frame-local state integration | Geometry encoder and attention with anchor prediction |
| Inputs | State, effort and optional contact histories | Current state, action and optional physical context | Geometry history, reference geometry, anchors and physical properties |
| Outputs | Relative state changes decoded by a physical codec | Next robot state | Future object geometry |
| Action conditioning | Generalized efforts | Explicitly configured control vector | No action input |
| Gradients | Network and codec retain input gradients | State/action gradients through multi-step prediction | `forward()` trains parameters; public `rollout()` disables autograd |
| Runtime | PyTorch on CPU or CUDA | PyTorch on CPU or CUDA | CUDA for geometry prediction; `embodichain[rigidformer]` dependencies |
| Included workflows | Cartpole/Ant resources, offline evaluation, training and export | Controlled fitting, differentiable prediction and transition replay | MOVi resources, engine demo, training, export and evaluation |

Use **NeRD** when your transition model needs a history of joint efforts and
contact observations. Its shipped reference systems are Cartpole and Ant.
Use **RWM** when you have measured state/action transitions and need a
graph-connected predictor with an explicit control period and joint order.
Use **RigidFormer** when your input is a sequence of rigid-object geometries
and you want to predict their subsequent motion from observed warmup frames.

Each guide explains the model's working principles, tensor layout, runnable
workflow and available validation:

```{toctree}
:maxdepth: 1

neural_dynamics/nerd
neural_dynamics/rwm
neural_dynamics/rigidformer
```

## Shared supervised training

All three families provide installable factories for the common
`embodichain.learning.sl` trainer:

| Model | Recipe factory |
| --- | --- |
| NeRD | `embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer` |
| RWM | `embodichain_models.dynamics.rwm.sl_recipe:build_rwm_trainer` |
| RigidFormer | `embodichain_models.dynamics.rigidformer.sl_recipe:build_rigidformer_trainer` |

After following a model guide to prepare its data and recipe, use the same
entry point for training and recovery:

```bash
embodichain train-sl --config /path/to/recipe.yaml
embodichain train-sl --config /path/to/recipe.yaml --resume /path/to/train/last.pt
```

Relative paths in SL recipes resolve from the working directory. Use a fresh
`trainer.output_dir` for a new run. Recovery uses the same recipe, data and
model identity; changing the output destination does not make an incompatible
checkpoint resumable. The [SL API and launch guide](../api_reference/embodichain/embodichain.learning.sl.rst)
describes checkpoints, distributed launches and per-rank batch sizes.

The shared trainer owns optimizer updates and recovery. Each family owns its
representation, data adapter and objective, so inference inputs and export
formats remain model-specific. A training checkpoint is distinct from an
inference export, and the final checkpoint does not imply validation-based
model selection.

## Original papers

| Model | Paper |
| --- | --- |
| NeRD | Xu et al. (2025). [Neural Robot Dynamics](https://arxiv.org/abs/2508.15755). |
| RWM (methodological inspiration) | Liao et al. (2026). [SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480). |
| RigidFormer | Dou et al. (2026). [RigidFormer: Learning Rigid Dynamics using Transformers](https://arxiv.org/abs/2605.09196). |

Each model guide provides the full author list and BibTeX. The RWM guide
distinguishes the SuperMimic paper from its independent implementation reference.

## Results and further reading

The guides distinguish offline transition error, accumulated rollout error,
engine comparisons and controlled fitting. Their reported results apply to the
recorded data, physical setup and evaluation protocol. Use each model's
reference workflow for published-weight or retraining comparisons.

- [Official model API](../api_reference/model_catalogs.rst): NeRD network,
  codec, data and SL components, with a link to the RWM API.
- [RWM API](../api_reference/rwm.rst): state integration, prediction, replay
  and supervised training.
- [RigidFormer API](../api_reference/model_rigidformer.rst): geometry inputs,
  checkpoint loading, MOVi source, objective and trainer factory.
- [Model catalog](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_models/README.md):
  model families and public import conventions.
