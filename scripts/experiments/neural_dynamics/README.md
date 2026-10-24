# NeRD reference experiments

These command-line tools compare an upstream NeRD implementation with the
EmbodiChain implementation. The normal dataset, training, export
workflow is documented in
[Neural Dynamics Models](../../../docs/source/features/neural_dynamics.md).
Downloadable datasets and inference models are linked there. Training logs,
checkpoints used for investigation, per-run reports and machine-specific plans
belong in an external output directory rather than this source tree.

## Environments

Run the unchanged upstream source in its supported Python/Torch environment.
Use the Warp release required by the selected upstream model. Run current
Newton collection and EmbodiChain training in a separate EmbodiChain environment.
The original-runtime helpers avoid importing the installed simulation SDK.
Use explicit interpreter and source paths; do not resolve a virtualenv executable
symlink to its base interpreter.

External source/checkpoint execution requires the tools' explicit trust option.
Pin the source revision and keep any portability adapter separate from the
upstream numerical implementation. Use each tool's `--help` for its arguments.

## Tool ownership

`_portable_reference.py` and `_historical_physics.py` are private support modules
for the isolated original runtime.

## Comparison contracts

- Separate model/trainer changes, runtime kernels, coordinate transport and
  analytical-solver versions instead of assigning every difference to migration.
- Preserve original sampling, input/output normalization, loss and model
  selection when evaluating training-port parity. Equal random seeds alone do
  not prove the same consumed sample sequence.
- Replay exactly the same realized open-loop efforts for predictive accuracy.
  With feedback control, label comparisons that use each model's own state as
  closed-loop behavior rather than same-action transition accuracy.
- Keep current Newton deployments and historical Warp models in their recorded
  physical domains. Do not replace physics signatures or normalizers to make an
  incompatible export load.
- Retain all requested trajectories and report nonfinite outputs and task
  failures. Separate endpoint error from trajectory-time average error.
- Use the matching analytical solver as each model's accuracy reference. A
  distance to another neural model is model similarity, not ground-truth error.
- State dataset size, update budget, evaluation population and horizon
  independently. A README example or a complete optimizer budget does not prove
  reproduction of a paper's training-data scale or final accuracy.

These tools produce their outputs under caller-supplied directories. They do not
change the production numerical policy or provide a blanket accuracy acceptance.
