# Official models

`embodichain.extensions.models` contains official model implementations grouped by model
family, including `embodichain.extensions.models.dynamics.nerd`. It ships in the same
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

Organize implementations as `embodichain.extensions.models.<domain>.<model_family>`.
The first level describes the model's prediction contract:

| Domain | Prediction contract |
| --- | --- |
| `policy` | Observations/history to actions, action distributions or action chunks. |
| `dynamics` | State/history, with optional controls and physical context, to future robot/environment states, geometry, observations or latent transitions. |

Use the following canonical packages for the current model directions:

| Model family | Public package | Landing status |
| --- | --- | --- |
| Ordinary state/visual BC | `embodichain.extensions.models.policy.bc` | Target for the BC follow-up; not created yet. |
| Visual ACT | `embodichain.extensions.models.policy.act` | Official LeRobot ACT configuration, provenance and simulator inference. |
| NeRD | `embodichain.extensions.models.dynamics.nerd` | Network/codec, shared SL, independent transition/geometry and learning-env consumers. |
| Robot world model (RWM) | `embodichain.extensions.models.dynamics.rwm` | Implemented. |
| LeWM visual latent dynamics | `embodichain.extensions.models.dynamics.lewm` | Image-conditioned dynamics, shared SL and CEM planning. |
| RigidFormer | `embodichain.extensions.models.dynamics.rigidformer` | Native author architecture, official weights and shared-SL recipe. |
| WorldParticle | `embodichain.extensions.models.dynamics.worldparticle` | Frozen particle transition; optional native deployment and operator qualification are separate. |

WorldParticle's [showcase](../../../examples/learning/worldparticle/README.md),
[input contract](../../../examples/learning/worldparticle/INPUTS.md),
[native quickstart](../../../examples/learning/worldparticle/DEPLOYMENT.md) and
[runtime requirements](../../../examples/learning/worldparticle/README.md#runtime-requirements)
cover the frozen inference path. A [verification summary](../../../examples/learning/worldparticle/README.md#verification)
states the checked headless/offscreen scope. It has no shared-SL
recipe, learning-env or qualified functional-gradient/APG contract in this stage.

Visual ACT's configuration and inference adapters are documented in the
[API reference](../../../docs/source/api_reference/visual_act.rst); its runnable
workflow lives in [examples/learning/act](../../../examples/learning/act/README.md).

NeRD's [showcase](../../../examples/learning/nerd/README.md) separates offline SL,
learning-env policy optimization and the optional
[native solver quickstart](../../../examples/learning/nerd/DEPLOYMENT.md).
Its [execution record](../../../examples/learning/nerd/VALIDATION.md) states the
tested runtime/budgets. Native deployment adapters have their own
[solver extension guide](../solvers/README.md).

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
Shared-SL families use `sl_recipe.py` for their installable trainer factory. The
ACT family delegates training to the upstream LeRobot workflow. Shared training,
checkpoint/recovery and distributed orchestration belong to `learning.sl`;
model packages do not introduce parallel trainer frameworks.

Collection, task/controller integration, reference research and downloadable
resources remain in the corresponding example. Shared MLPs and normalizers
remain in core. Shared-SL families use the same SL contracts regardless of
their domain. Create a family only when its implementation lands, without empty
placeholder directories, eager cross-family imports or a new registry.

For third-party training ownership and dependency profiles, see the [LeRobot guide](../../../docs/source/guides/lerobot.md) and [training framework comparison](../../../docs/source/guides/training_frameworks.md).
