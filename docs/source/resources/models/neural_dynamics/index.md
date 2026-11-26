# Neural Dynamics Models: Predicting the Next State

Choose a dynamics model by the quantity you observe and the quantity you want
to predict. Each family ships as `embodichain.extensions.models.dynamics.<family>`
in the main `embodichain` wheel. You can import a model independently of a
simulator. Its example handles the required data, assets and weights.

## Choose a model

| Model | Inputs | Prediction | Example |
| --- | --- | --- | --- |
| [NeRD](nerd.md) | History of robot states, joint efforts and contacts when present | State change decoded into the next articulated state | [Cartpole and Ant](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/nerd) |
| [RWM](rwm.md) | Current robot state, control and optional physical context | Next robot state at a fixed control period | [Controlled fitting and replay](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/rwm) |
| [LeWM](lewm.md) | RGB history and actions grouped into model steps | Future visual embeddings for image-goal planning | [OGBench Cube](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/lewm/cube) |
| [RigidFormer](rigidformer.md) | Point-cloud history, reference geometry, anchors, physical properties and step size | Future geometry of interacting rigid objects in free rollout | [MOVi rigid bodies](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/rigidformer) |
| [WorldParticle](worldparticle.md) | Particle state, total acceleration, material features, static boundaries and optional topology | Next world-space particle positions and velocities | [Fluid, cloth and elastic solids](https://github.com/DexForce/EmbodiChain/tree/main/examples/learning/worldparticle) |

```{toctree}
:hidden:

nerd
rwm
lewm
rigidformer
worldparticle
```

## Architecture and runtime

| Model | Architecture and gradients | Runtime |
| --- | --- | --- |
| NeRD | Causal Transformer over history. The network and physical codec retain input gradients. | PyTorch on CPU or CUDA. Reference online forward kinematics uses Newton/Warp. |
| RWM | MLP with frame-local state integration. State and action gradients propagate through multiple steps. | PyTorch on CPU or CUDA. |
| LeWM | Vision Transformer and action-conditioned latent predictor. The rollout is differentiable and the CEM planner samples actions. | PyTorch on CPU or CUDA. The Cube example uses DexUni and native RGB. |
| RigidFormer | Object-level attention and anchor acceleration prediction. `forward()` trains parameters and `rollout()` runs inference without autograd. | Geometry prediction needs CUDA and the `rigidformer` extra. |
| WorldParticle | Local particle convolutions, token merging and cross-attention. The integration supports frozen forward inference and reset. | The tested configuration uses Python 3.12, Torch 2.11 CUDA and separately provisioned SM120 operators. |

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
| WorldParticle | Wang et al. (2026). [WorldParticle: Unified World Simulation of Lagrangian Particle Dynamics via Transformer](https://arxiv.org/abs/2605.15305). |

Each model guide includes the project site and a BibTeX entry. For RWM's
implementation lineage, see the [RWM references](rwm.md#references).

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
