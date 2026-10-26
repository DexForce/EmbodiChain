# Official models

`embodichain_models` contains official model implementations grouped by model
family, including `embodichain_models.dynamics.nerd`. It ships in the same
`embodichain` wheel and uses that distribution's version and dependencies. There
is no separate model distribution, version file or model registry.

Import a named model package explicitly. The top-level package and family
packages do not eagerly import individual models or initialize a simulator.
Shared optimizer/trainer contracts remain in `embodichain.learning.sl`; shared
MLP and normalization building blocks remain in `embodichain.learning.models`.
The core does not import official model implementations.

NeRD datasets, teacher assets and learned weights are prepared by
`examples/learning/nerd/_resources.py` from pinned Hugging Face revisions.
They remain outside Git and the wheel; the showcase owns their download and
verification lifecycle. Model-specific supervised factories live in the
installable model package so `embodichain train-sl` can resolve them from any
working directory. Example helpers remain owned by their example workflows.

## Domains and public imports

Organize implementations as `embodichain_models.<domain>.<model_family>`.
The first level describes the model's prediction contract:

| Domain | Prediction contract |
| --- | --- |
| `policy` | Observations/history to actions, action distributions or action chunks. |
| `dynamics` | State/history, with optional controls and physical context, to future robot/environment states, geometry, observations or latent transitions. |

Use the following canonical packages for the four current model directions:

| Model family | Public package | Landing status |
| --- | --- | --- |
| Ordinary state/visual BC | `embodichain_models.policy.bc` | Target for the BC follow-up; not created yet. |
| NeRD | `embodichain_models.dynamics.nerd` | Implemented in this change. |
| Robot world model (RWM) | `embodichain_models.dynamics.rwm` | Implemented. |
| RigidFormer | `embodichain_models.dynamics.rigidformer` | Target for the RigidFormer follow-up; not created yet. |

RWM predicts action-conditioned robot state transitions, so it belongs to
`dynamics`. World models also describe future-state/observation prediction;
`world_models` is not a parallel top-level domain or an alternative import path.
Keep each implementation in one canonical package. Robots/tasks, state/RGB/point
modalities, MLP/Transformer backbones and SL/RL training methods do not determine
the first-level directory. Add another domain only for a concrete responsibility
that neither existing prediction contract covers.

The directory does not imply capabilities such as action conditioning,
state/action gradients or solver deployment. Those remain explicit model
contracts: the current RigidFormer adapter predicts rigid geometry without an
action input and does not qualify action gradients.

## Ownership inside a model family

A family owns its model/configuration, portable-checkpoint support where
available, codecs, model-format data adapters and model-specific objectives.
Its `sl_recipe.py` owns the installable trainer factory. Shared training,
checkpoint/recovery and distributed orchestration belong to `learning.sl`;
model packages do not introduce parallel trainer frameworks.

Collection, task/controller integration, reference research and downloadable
resources remain in the corresponding example. Shared MLPs and normalizers
remain in core. All four directions use the same SL contracts regardless of
their domain. Create a family only when its implementation lands, without empty
placeholder directories, eager cross-family imports or a new registry.
