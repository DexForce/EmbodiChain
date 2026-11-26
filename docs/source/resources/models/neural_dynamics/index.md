# Neural Dynamics Models

Neural dynamics models learn how physical systems evolve from observations,
controls and physical context. **NeRD** models robot dynamics with state and
effort histories. **RWM** predicts robot states from actions, **LeWM** predicts
visual latents from images and actions, and **RigidFormer** predicts how
interacting rigid geometry moves. **WorldParticle** provides frozen particle
prediction and optional native forward-inference deployment.

The models ship as `embodichain.extensions.models.dynamics.<family>` in the main
`embodichain` wheel. Import the family you need explicitly. Model imports do
not start a simulator; datasets, reference assets and pretrained weights are
prepared separately by each example.

## Choose a model

| Capability | [NeRD](nerd.md) | [Robot world model (RWM)](rwm.md) | [LeWM](lewm.md) | [RigidFormer](rigidformer.md) | [WorldParticle](worldparticle.md) |
| --- | --- | --- | --- | --- | --- |
| Prediction subject | Articulated robot state | Robot state at a fixed control period | Visual environment dynamics | Interacting rigid objects | Particles with static authored boundaries |
| Architecture | Causal Transformer over history | MLP with frame-local state integration | ViT image encoder and action-conditioned causal predictor | Geometry encoder and attention with anchor prediction | Source particle network with local convolution and attention |
| Inputs | State, effort and optional contact histories | Current state, action and optional physical context | RGB observation history and actions | Geometry history, reference geometry, anchors and physical properties | Current state, total acceleration and fixed features/boundaries/topology |
| Outputs | Relative state changes decoded by a physical codec | Next robot state | Predicted future embeddings | Future object geometry | Next world-space particle position/velocity |
| Action conditioning | Generalized efforts | Explicitly configured control vector | Raw actions grouped into model steps | No action input | Total acceleration; native force/mass plus gravity |
| Gradients | Network and codec retain input gradients | State/action gradients through multi-step prediction | Differentiable latent rollout with gradient-free CEM | `forward()` trains parameters; public `rollout()` disables autograd | Frozen inference; no qualified functional-gradient/APG contract |
| Runtime | PyTorch on CPU or CUDA; reference online FK uses Newton/Warp | PyTorch on CPU or CUDA | PyTorch on CPU or CUDA for the model, with DexUni and native RGB for the Cube example | CUDA for geometry prediction; `embodichain[rigidformer]` dependencies | Checked Python 3.12 / Torch 2.11 CUDA / SM120 operator build; manual provisioning |
| Included workflows | Cartpole/Ant SL, learning-env policy training and optional native solver deployment | Controlled fitting, differentiable prediction and transition replay | OGBench Cube RGB collection, training, CEM evaluation and video export | MOVi resources, engine demo, training, export and evaluation | Asset preparation, standalone inference and four native demos |

Use **NeRD** when your transition model needs a history of joint efforts and
contact observations. Its shipped reference systems are Cartpole and Ant.
Use **RWM** when you have measured state/action transitions and need a
graph-connected predictor with an explicit control period and joint order.
Use **LeWM** when observations and goals are images and planning uses a
learned visual representation.
Use **RigidFormer** when your input is a sequence of rigid-object geometries
and you want to predict their subsequent motion from observed warmup frames.

Use **WorldParticle** for the supported frozen particle demonstrations; it
currently provides inference/reset rather than a shared training recipe.

Each guide explains the model's working principles, tensor layout, runnable
workflow and available validation:

```{toctree}
:maxdepth: 1

nerd
rwm
lewm
rigidformer
worldparticle
```

## Shared supervised training

NeRD, RWM, LeWM and RigidFormer provide installable factories for the common
`embodichain.learning.sl` trainer:

| Model | Recipe factory |
| --- | --- |
| NeRD | `embodichain.extensions.models.dynamics.nerd.sl_recipe:build_supervised_trainer` |
| RWM | `embodichain.extensions.models.dynamics.rwm.sl_recipe:build_rwm_trainer` |
| LeWM | `embodichain.extensions.models.dynamics.lewm.sl_recipe:build_lewm_trainer` |
| RigidFormer | `embodichain.extensions.models.dynamics.rigidformer.sl_recipe:build_rigidformer_trainer` |

WorldParticle currently has no shared-SL recipe, learning-env or APG entry point.

After following a model guide to prepare its data and recipe, use the same
entry point for training and recovery:

```bash
embodichain train-sl --config /path/to/recipe.yaml
embodichain train-sl --config /path/to/recipe.yaml --resume /path/to/train/last.pt
```

Relative paths in SL recipes resolve from the working directory. Use a fresh
`trainer.output_dir` for a new run. Recovery uses the same recipe, data and
model identity; changing the output destination does not make an incompatible
checkpoint resumable. The [SL API and launch guide](../../../api_reference/embodichain/embodichain.learning.sl.rst)
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
| LeWM | Maes et al. (2026). [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312). |
| RigidFormer | Dou et al. (2026). [RigidFormer: Learning Rigid Dynamics using Transformers](https://arxiv.org/abs/2605.09196). |

The NeRD, RWM and RigidFormer guides provide author lists and BibTeX.
WorldParticle links its upstream source and retained license/provenance record. The RWM guide
distinguishes the SuperMimic paper from its independent implementation reference.

## Results and further reading

The guides distinguish offline transition error, accumulated rollout error,
engine comparisons and controlled fitting. Their reported results apply to the
recorded data, physical setup and evaluation protocol. Use each model's
reference workflow for published-weight or retraining comparisons.

- [Official model API](../../../api_reference/model_catalogs.rst): NeRD network,
  transition/geometry, optional Providers and WorldParticle inference, with a
  link to the RWM API.
- [RWM API](../../../api_reference/rwm.rst): state integration, prediction, replay
  and supervised training.
- [LeWM API](../../../api_reference/lewm.rst) covers the model, SL recipe and CEM planner.
- [RigidFormer API](../../../api_reference/model_rigidformer.rst): geometry inputs,
  checkpoint loading, MOVi source, objective and trainer factory.
- [Model catalog](https://github.com/DexForce/EmbodiChain/blob/main/embodichain/extensions/models/README.md):
  model families and public import conventions.
