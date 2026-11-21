# LeWM visual world model

LeWM learns action-conditioned latent dynamics from images. The model, common
supervised-learning recipe and CEM planner are bundled in
`embodichain.extensions.models.dynamics.lewm`. The model package also supports
offline training and planning from outside the simulator.

## OGBench single Cube

The [Cube example](cube/README.md) prepares pinned inputs, renders native RGB,
trains a validation-selected checkpoint and runs the official 50-query
benchmark. It constructs the Cube environment on DexUni CPU
rigid-body physics with an EmbodiChain camera. Use the
[deployment guide](cube/DEPLOYMENT.md) to set up the SDK and Python environment.

```bash
embodichain run-example learning/lewm/cube --help
```

The [validation record](cube/VALIDATION.md) gives the measured results and
runtime profile. The [Cube contract](OFFICIAL_CUBE.md) describes the control,
physics and planner settings. The [research notes](research/README.md) cover
the historical UR push-cube example and earlier solver experiments.
