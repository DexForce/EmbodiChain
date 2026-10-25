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

## Rigid-body geometry models

`learning/models/rigidformer.py` lazily constructs a pinned optional upstream
model. `learning/sl/rigidformer.py` adapts its position/acceleration objective
and builds a `train-sl` recipe. `data_pipeline/datasets/rigid_body.py` samples
owned previous/current/next point windows with an explicit reference frame,
object/point lengths and bounded resident/returned payloads. Points correspond
across frames; units and step duration belong to the recorded corpus.

The optional `rigidformer` extra is isolated from ordinary SL imports. The
adapter supplies forward prediction and supervised parameter training; it does
not qualify action/state gradients for APG or bind a native simulation solver.
The fixed Box2D friction/collision comparison in `scripts/experiments/rigidformer/`
retains upstream numerical behavior, baseline limitations and full seed results.
Finite unequal-length point padding is supported only with a padded point axis
of at most 128; the pinned hierarchical encoder leaks padding above that size,
so the adapter rejects that layout. Supplied point counts, including placeholder
objects, must be at least `num_anchors`, and fixed anchors index valid points.
The acceptance runner verifies dataset/recipe/model source bytes against pinned
Git blobs and gates every paired loss, in addition to rollout endpoint metrics.
