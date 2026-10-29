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
- Optional `SequenceBatchSource` in `learning/sl/sequence.py` adapts a data-pipeline
  `BatchSource` using an explicit versioned encoder. It preserves canonical masks
  and provenance under reserved `metadata['sequence_source']`, excluding raw
  observations/actions there. Encoders select causal inputs and supervision;
  model families retain action/codec/objective semantics. Core SL imports do not
  load this bridge, ODS or the SDK. See `examples/learning/sl_data_source/`.
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
- The [RWM recipe](../robot-world-model/robot-world-model.md) is
  `embodichain_models.dynamics.rwm.sl_recipe:build_rwm_trainer`. Its model,
  state codec, objective, predictor and replay belong to that model family.

Objectives own loss masks, normalization, integration and multi-step graph
construction. The optimizer does not infer action semantics or detach histories.
Under DDP, `LossResult.loss` is a local mean and `weight` is its effective valid
supervision count; the learner forms the global weighted mean. Batch size is per
rank and the learning rate is unchanged. Sampling belongs to the source: tensor
sources draw with replacement, while epoch-traversal sources may shard windows.
All ranks call fitting, saving and recovery;
only rank 0 writes artifacts or runs validation through the unwrapped model.
Schema, action/frame/timestep and numerical-policy identifiers belong in each
recipe's checkpoint signature. Tensor sources additionally bind content identity.
The sequence bridge defaults to refusing exact resume. Explicitly immutable,
unbudgeted resident sources bind actual CPU content, schema, window, encoder
version and semantic metadata (including language); storage diagnostics do not.
Sampling checks declarations, quota and tensor versions; resume rehashes content
in bounded blocks. An optional source `validate_resume()` preflight runs inside
the collective checkpoint validation before any trainer state or output admission
changes. Live buffers and delivery credits are not restored: such sources may
train/save snapshots, but exact continuation is rejected. Acquisition, waits,
refill and delivery debits stay source/caller-owned; encoding failures do not
refund a delivered context.
Distributed checkpoints retain each rank's RNG, sampler, buffers and participant
state and require the same world size for exact recovery. Rank-local snapshots
are transferred to rank 0 in bounded chunks to limit device memory. Inside rank
0's validation callback only,
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
sequence tests reuse real CPU source fixtures for masks, provenance, owned output,
quota and recovery admission. Examples use launch smoke checks, not tutorial tests.

Factories may register ordered `SupervisedTrainingState` participants for source
cursors, schedules and model averages. They commit after optimizer success,
produce owned rank-local snapshots, validate before live restoration and restore
after validation callbacks. Empty participants retain existing checkpoint
compatibility. Model-specific participant math stays in its model family.
