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
  Under single-node `torchrun`, the workflow initializes a process group before
  the factory and resolves `trainer.device` to CPU/Gloo or `cuda:LOCAL_RANK`/NCCL.
  `distributed_run(device)` provides the same scoped setup for programmatic
  callers; it cleans up only groups it creates. Launch commands and a complete
  masked-regression factory live in `examples/learning/sl_ddp/`.

Objectives own loss masks, normalization, integration and multi-step graph
construction. The optimizer does not infer action semantics or detach histories.
Under DDP, `LossResult.loss` is a local mean and `weight` is its effective valid
supervision count; the learner forms the global weighted mean. Batch size is per
rank, the learning rate is unchanged, and rank-local sampling streams draw with
replacement, so samples may overlap. All ranks call fitting, saving and recovery;
only rank 0 writes artifacts or runs validation through the unwrapped model.
Schema, action/frame/timestep and numerical-policy identifiers belong in each
recipe's checkpoint signature. Tensor sources additionally bind content identity.
Fresh runs require artifact-free output directories. Restoring an older
checkpoint into a directory with later metric history requires a fresh output
directory so prior results are retained.
Distributed checkpoints retain each rank's RNG, sampler and buffers and require
the same world size for exact recovery. Inside rank 0's validation callback only,
a save uses the previously gathered committed-update snapshot without a new
collective; ordinary save calls require every rank to participate.
Validation callbacks use frozen models and restored RNG; only development data
may select checkpoints. Acquisition and producer lifecycles remain application-owned.

## Validation

`tests/learning/sl/` checks independent masked updates, graph propagation,
nonfinite update boundaries, owned samples, simulator-free imports and exact
CPU recovery with dropout, optimizer and sampling state. Model-specific
qualification belongs to each model's acceptance recipe. The distributed tests
exercise weighted global updates and fixed-world-size recovery with CPU/Gloo;
the example is validated by launch smoke runs rather than example-specific tests.
