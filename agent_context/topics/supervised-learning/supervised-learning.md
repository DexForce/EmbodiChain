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
  Under single-node `torchrun`, the workflow initializes a process group before
  the factory and resolves `trainer.device` to CPU/Gloo or `cuda:LOCAL_RANK`/NCCL.
  `distributed_run(device)` provides the same scoped setup for programmatic
  callers; it cleans up only groups it creates. Launch commands and a complete
  masked-regression factory live in `examples/learning/sl_ddp/`.
- The [NeRD recipe](../neural-dynamics/neural-dynamics.md) is an installable factory
  at `embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer`.
  The core does not import the model extension or example helpers. The NeRD
  showcase's default `train` command uses this same common recipe.

Objectives own loss masks, normalization, integration and multi-step graph
construction. The optimizer does not infer action semantics or detach histories.
Under DDP, `LossResult.loss` is a local mean and `weight` is its effective valid
supervision count; the learner forms the global weighted mean. Batch size is per
rank, the learning rate is unchanged, and rank-local sampling streams draw with
replacement, so samples may overlap. All ranks call fitting, saving and recovery;
only rank 0 writes artifacts or runs validation through the unwrapped model.
Schema, action/frame/timestep and numerical-policy identifiers belong in each
recipe's checkpoint signature. Tensor sources additionally bind content identity.
Distributed checkpoints retain each rank's RNG, sampler and buffers and require
the same world size for exact recovery. Inside rank 0's validation callback only,
a save uses the previously gathered committed-update snapshot without a new
collective; ordinary save calls require every rank to participate.
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
qualification belongs to each model's acceptance recipe. The distributed tests
exercise weighted global updates and fixed-world-size recovery with CPU/Gloo;
the example is validated by launch smoke runs rather than example-specific tests.
