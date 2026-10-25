# Supervised learning (SL)

`embodichain.learning.sl` owns common supervised optimizer updates and offline
training orchestration. Shared MLPs and feature normalization remain in
`learning.models`; concrete model families and their model-specific adapters live
in the bundled `embodichain_models` extension. Generic numerical state and geometry
operations remain in `compute`; source storage belongs to data adapters.

## Entry points

- `SupervisedBatch` and `LossResult` in `learning/sl/types.py` preserve structured
  inputs, targets, masks and objective-specific reductions.
- `SupervisedLearner.update()` in `learning/sl/learner.py` owns a finite optimizer
  update. Its optional numerical context covers forward, backward and clipping.
- `TensorBatchSource` snapshots fixed CPU tensor trees and returns owned samples
  under the caller's explicit CPU generator.
- `SupervisedTrainer` in `learning/sl/trainer.py` composes a source and learner,
  saving optimizer, RNG, sampling progress, source identity and recipe semantics.
- `embodichain train-sl --config FILE [--resume FILE]` lazily resolves a
  `factory: module:function`. The factory receives the full model/data/objective/
  optimizer/trainer/evaluation configuration and returns a `SupervisedTrainer`.
- The [NeRD recipe](../neural-dynamics/neural-dynamics.md) is an installable factory
  at `embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer`.
  The core does not import the model extension or example helpers. The NeRD
  showcase's default `train` command uses this same common recipe.

Objectives own loss masks, normalization, integration and multi-step graph
construction. The optimizer does not infer action semantics or detach histories.
Schema, action/frame/timestep and numerical-policy identifiers belong in each
recipe's checkpoint signature. Tensor sources additionally bind content identity.
`learning/sl/_artifacts.py` owns run identities and output-history admission for
both SL and historical NeRD trainers. A fresh run claims `run.json`; resuming an
occupied directory requires the checkpoint's matching run identity. Checkpoints
without an identity require an artifact-free destination. Newer or incomplete
metric history is retained and requires a fresh resume destination.
Validation callbacks preserve RNG, sampling state and model buffers, including
checkpoints saved inside a callback. Only development data may select checkpoints.
Acquisition and producer lifecycles remain application-owned.

## Validation

`tests/learning/sl/` checks independent masked updates, graph propagation,
nonfinite update boundaries, owned samples, simulator-free imports and exact
CPU recovery with dropout, optimizer and sampling state. Model-specific
qualification belongs to each model's acceptance recipe.


## Behavior cloning

`embodichain_models.policy.bc` owns state/visual chunk models, their frozen
normalization and encoder, the masked BC objective, bounded HDF5 reader,
source/prefetch adapters and per-environment target queues. Its installable
`sl_recipe:build_supervised_trainer` composes stored physical targets and frozen
statistics with the common SL trainer. It does not import simulation or infer
controller semantics. `tests/models/policy/bc/` covers this family boundary.
The reader defaults to complete future windows with its historical source
identity. Generic recipes may explicitly enable `include_partial_targets` for
complete-history tail anchors, finite right padding and `target_valid` masks;
that view has a distinct recovery identity. The frozen acceptance driver keeps
the default. Encoders see only real targets, grouped by valid length.

`examples/learning/bc` lazily exposes collection, training, evaluation and
reference parity through `run-example learning/bc`. Private `_support` modules
own task target construction, controller decoding and the frozen PickPlace
acceptance protocol. Historical source/data/checkpoint identities and negative
results remain sealed research evidence; namespace parity does not qualify
physical task success. The generic model recipe accepts already stored targets
so standalone `train-sl` does not require example bootstrap.
