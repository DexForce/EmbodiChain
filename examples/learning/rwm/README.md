# Robot world model supervised acceptance

See the [RWM guide](../../../docs/source/features/neural_dynamics/rwm.md) for
working principles, state layout, prediction and replay examples.

Run with the `py311` environment from a source checkout or an installed wheel:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 embodichain run-example learning/rwm \
  --output outputs/rwm-controlled-seed0 --seed 0
```

The runner writes a fixed training corpus, independent validation corpus, resolved
SL recipe, checkpoints, training metrics and `summary.json`. Its one-joint model
must reduce four-step held-out normalized state error below 15% of its initial
value within 300 updates. A separate 29-joint, three-context fit checks the model
layout and graph-connected initial-action gradients. Repeat seeds 0, 1 and 2.

These controlled accelerations verify supervised fitting and interfaces. They
do not model G1 physics, reproduce SuperMimic, or establish policy performance.
No ODS producer, native simulator or APG learner is constructed.

The emitted `recipe.json` also works with `embodichain train-sl --config FILE`.
Its installable factory is
`embodichain_models.dynamics.rwm.sl_recipe:build_rwm_trainer`; the same symbol is
exported from `embodichain_models.dynamics.rwm`.
Use a fresh output directory for a new run. `--resume FILE` restores the saved
learning boundary through the shared trainer; GPU replay snapshots are optional,
and a workflow may explicitly rebuild/refill replay instead.

Canonical robot states have position (3), a row-major body-to-world rotation
matrix (9), world linear/angular velocity (3 each), ordered joint positions and
ordered joint velocities. Controls, their units, the joint order and fixed
control period belong to the dataset/model contract. Replay callers must supply
the physical next state before any automatic reset.

Reports bind the exact source and tensor-corpus hashes, Python/Torch versions,
process peak RSS and optional Torch CUDA allocator peaks. The measured fit loop
includes its periodic validation and checkpoint I/O; interpreter/import startup
and final held-out validation are outside that interval. CUDA allocator peaks
do not include driver/native allocations.

The model package owns frame-local state math, the autoregressive objective,
graph-connected predictor and bounded transition replay. The core SL package
owns optimizer updates, run artifacts and recovery. This example owns controlled
corpus generation and fitting qualification; its private helpers are not an
installed model API. The model requires PyTorch and the shared core utilities;
importing it does not load a simulator SDK.
