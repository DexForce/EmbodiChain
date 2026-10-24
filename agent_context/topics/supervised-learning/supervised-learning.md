# Supervised learning (SL)

`embodichain.learning.sl` owns common supervised optimizer updates and offline
training orchestration. Models remain in `learning.models`; numerical state and
geometry operations remain in `compute`; source storage belongs to data adapters.

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

Objectives own loss masks, normalization, integration and multi-step graph
construction. The optimizer does not infer action semantics or detach histories.
Schema, action/frame/timestep and numerical-policy identifiers belong in each
recipe's checkpoint signature. Tensor sources additionally bind content identity.
Fresh runs require artifact-free output directories. Restoring an older
checkpoint into a directory with later metric history requires a fresh output
directory so prior results are retained.
Validation callbacks use frozen models and restored RNG; only development data
may select checkpoints. Acquisition and producer lifecycles remain application-owned.

## Validation

`tests/learning/sl/` checks independent masked updates, graph propagation,
nonfinite update boundaries, owned samples, simulator-free imports and exact
CPU recovery with dropout, optimizer and sampling state. Model-specific
qualification belongs to each model's acceptance recipe.

## Robot world models

`learning/models/rwm.py` owns configurable frame-local state-increment models.
`compute/dynamics/robot_state.py` owns canonical state transforms and SO(3)
integration. `learning/sl/rwm.py` owns autoregressive supervision and the
`train-sl` recipe factory. `RWMPredictor` owns graph-connected prediction state;
selected reset replaces rows without modifying outstanding graph tensors.

`data_pipeline/datasets/transition_replay.py` owns fixed-byte-budget device
replay. Append copies detached pre-reset transitions; sampled windows and
histories respect episode boundaries and ring overwrite. Physical acquisition
and APG update scheduling remain caller-owned. Freeze model parameters and
normalization during action/state differentiation and finish dependent backward
work before changing model weights.

Focused contracts are covered by `tests/learning/sl/test_rwm.py`, model/state
tests and `tests/data_pipeline/test_transition_replay.py`. Controlled experiment
recipes under `scripts/experiments/rwm/` preserve measured fitting evidence.
