# Multi-GPU Training

EmbodiChain supports distributed RL training across multiple GPUs using PyTorch DistributedDataParallel (DDP).

## Overview

- **One process per GPU**: Each GPU runs an independent process via `torchrun`.
- **Independent components per rank**: Each process creates its own environment, collector, buffer, and policy copy.
- **Gradient synchronization only**: All ranks synchronize gradients after each PPO/GRPO update; no rollout all-gather.

## Launch Commands

### Single-Node Multi-GPU

Use `torchrun` with `--nproc_per_node` equal to the number of GPUs, and add `--distributed`:

```bash
torchrun --nproc_per_node=2 -m embodichain.agents.rl.train --config <config_path> --distributed
```

Example:

```bash
torchrun --nproc_per_node=2 -m embodichain.agents.rl.train --config configs/agents/rl/push_cube/train_config.json --distributed
```

No config file changes needed; `device` and `gpu_id` are overridden automatically per rank.

### Specifying GPUs

Use `CUDA_VISIBLE_DEVICES` to select which GPUs to use. The processes will see only these GPUs as `cuda:0`, `cuda:1`, etc.:

```bash
# Use GPU 0 and 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m embodichain.agents.rl.train --config <config_path> --distributed
```

`--nproc_per_node` must equal the number of GPUs in `CUDA_VISIBLE_DEVICES`.

## Behavior

- **Device**: Each rank uses `cuda:{local_rank}`; the simulation and policy run on the assigned GPU.
- **Seeds**: Each rank uses `seed + rank` for different rollout diversity.
- **Total steps**: Scaled by `world_size`; e.g., 2 GPUs collect twice as many steps per iteration.
- **Logging**: Only rank 0 writes to TensorBoard, WandB, and console.
- **Episode stats**: `episode_reward_avg_100` and `episode_length_avg_100` are aggregated across all ranks via `all_gather` for accurate global metrics.
- **Evaluation**: Only rank 0 creates and runs the evaluation environment.
- **Checkpoints**: Only rank 0 saves; the underlying policy state (without DDP wrapper) is stored.

