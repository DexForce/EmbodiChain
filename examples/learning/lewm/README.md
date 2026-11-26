# LeWM visual world model

LeWorldModel learns action-conditioned dynamics in a visual embedding space.
EmbodiChain bundles the model, supervised training recipe, and CEM planner in
`embodichain.extensions.models.dynamics.lewm`. See the
[model guide](../../../docs/source/resources/models/neural_dynamics/lewm.md) and
[Maes et al. (2026)](https://arxiv.org/abs/2603.19312).

## OGBench single Cube

Use the [Cube reproduction guide](cube/README.md) to prepare the pinned inputs,
adapt the published visual encoder to the EmbodiChain camera, and evaluate on
the fixed 50 queries in DexUni. See the [runtime setup](cube/DEPLOYMENT.md) and
[Cube task contract](OFFICIAL_CUBE.md) for dependencies and settings.
