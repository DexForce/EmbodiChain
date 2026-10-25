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

Organize implementations by domain and model type. The current package is
`dynamics/nerd`; future contributions can use policy families for BC,
`world_models` for RWM, or `dynamics` for RigidFormer-style models as their
contracts settle. All use the shared SL core. Add a family when its
implementation lands, without empty placeholder directories or a new registry.
