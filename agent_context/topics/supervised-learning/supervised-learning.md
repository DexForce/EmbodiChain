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

## Behavior cloning

`learning/models/chunk_bc.py` owns configurable state/visual chunk prediction
and per-environment causal history/target queues. `learning/sl/bc.py` owns the
masked normalized-target objective and offline source adaptation. Optional
vision imports are deferred until a visual policy is constructed.

`data_pipeline/datasets/bc_hdf5.py` reads fixed per-episode windows through bounded
HDF5 handles. Its source identity binds the immutable manifest, selected fields,
observation slices, H/F, camera order and declared target-transform version.
Only history RGB frames enter policy inputs; future geometry may construct
labels but never enters visual policy forward. Prefetch keeps one owned batch
and a private RNG reservation; only delivered sampling progress is committed.

`experiments/learning/sl/train_bc.py` owns the pinned offline PickPlace recipe,
development checkpoint selection and sealed final test. Task-specific pose
encoding/decoding and physical success remain in its task adapter. The collector
prepares a fixed corpus directly, independently of the SL learner. Formal
qualification requires a completed immutable corpus and unchanged physical,
camera, target and optimizer contracts.
