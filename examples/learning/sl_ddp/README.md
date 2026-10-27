# Single-node distributed supervised learning

This example fits a `4 → 32 → 2` MLP to a fixed synthetic regression corpus.
It uses ordinary PyTorch and the common SL source, objective, learner and trainer;
no dataset download or simulator is needed. Run from the repository checkout in
an installed EmbodiChain `py311` environment.

The first target is a linear combination of inputs; the second adds a sine term.
Every example supervises the first scalar, while the second is masked according
to an input condition. The objective returns the local masked mean squared error
and `LossResult.weight = mask.sum()`. This makes the distributed update represent
`sum(weight_rank * loss_rank) / sum(weight_rank)`, including batches with different
numbers of valid targets. The independent fixed validation corpus is evaluated
on rank 0.

## Launch

Activate the environment and run two CPU workers using Gloo:

```bash
conda activate py311
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m embodichain train-sl --config examples/learning/sl_ddp/cpu.yaml
```

For two GPUs, select the available devices and use the CUDA recipe. The workflow
maps each worker to `cuda:LOCAL_RANK` within `CUDA_VISIBLE_DEVICES` and uses NCCL:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m embodichain train-sl --config examples/learning/sl_ddp/cuda.yaml
```

The same factory also runs on one CPU without a distributed process group. For
a separate run, copy `cpu.yaml` to `/tmp/sl-single-cpu.yaml` and set its
`trainer.output_dir` to `outputs/sl_single_cpu`:

```bash
python -m embodichain train-sl --config /tmp/sl-single-cpu.yaml
```

Each fresh run needs an artifact-free output directory. Rank 0 claims its durable
identity in `run.json` and writes `metrics.jsonl`, `step_100.pt`, `step_200.pt` and
`last.pt`; the validation
MSE appears every 50 updates as `validation/mse`. These launch commands illustrate
the execution path, without claiming a multi-GPU speedup for this tiny model.

## Batch size, sampling and recovery

`trainer.batch_size: 32` is per worker: two workers draw 64 examples per optimizer
update. Each worker draws independently with replacement from the same immutable
corpus, so samples can overlap across workers. The learning rate stays 0.001;
it is not automatically multiplied by the worker count. Changing world size
changes the effective batch and sampling sequence.

Every worker participates in fitting, saving and restoring checkpoints, while
only rank 0 writes artifacts or runs validation. The checkpoint retains the common
optimizer/update state and every rank's RNG, sampler and model buffers. Exact
recovery requires the same world size, recipe, source identity and training
configuration. This example's fixed corpus/model construction is the same on
all ranks; the shared trainer supplies different sampling streams.

To demonstrate continuation from update 100, copy the same recipe and change only
`trainer.output_dir`, for example to `outputs/sl_ddp_cpu_resumed`. Keep the 200-update
budget and all other settings. Then launch the same two workers:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m embodichain train-sl --config /tmp/sl-ddp-resume.yaml \
  --resume outputs/sl_ddp_cpu/step_100.pt
```

A fresh destination preserves the original 200-update result when restoring its
older 100-update snapshot. To resume an interrupted run from its latest snapshot,
use the original directory only when `run.json` matches the checkpoint's run
identity and its complete metric history is not ahead of that snapshot.
Checkpoints made before run identities were added require a fresh destination;
their next save records a run identity. Checkpoints are trusted local PyTorch files.

The installed NeRD factory uses this same launch path. After preparing the data
and checking the paths in its recipe, launch it with two available GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m embodichain train-sl \
  --config examples/learning/nerd/configs/cartpole/sl.train.yaml
```

Its factory is `embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer`.
The existing precision, causal attention, normalization and loss-mask settings
remain recipe-owned. Each rank loads the dataset and keeps its own resident
cache, so budget memory per worker. Validation runs on rank 0 with its unwrapped
model. If a validation callback calls `trainer.save()`, that save uses the
committed state captured from all ranks before validation; calling `save()`
elsewhere requires every rank to participate.

The CLI handles process-group setup and cleanup automatically. Programmatic
launchers can use `embodichain.learning.sl.distributed_run(device)` around model
and trainer construction; all ranks must enter that scope and participate in the
training calls. It yields the resolved `torch.device`, and destroys only a process
group that it created. A pre-existing group remains caller-owned.

See the pinned-version PyTorch references for
[`torchrun` in 2.7](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/distributed/run.py)
and [DistributedDataParallel 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html).
