# Temporal policy capacity verification

Measured on 2026-10-25 with an NVIDIA GeForce RTX 5090, Torch 2.7.1+cu128, Python 3.11.14 and FP32. These are synthetic tensor capacity measurements, not assembly training success or simulator throughput.

Reproduce:

```bash
python -m scripts.benchmark.learning.proprioceptive_policy --device cuda:0 --training-update
```

The workload uses 30 prior frames plus the current frame, 36 features per frame, 7 actions, a 64-feature privileged critic, and 128 contact points with 4 classes. This deliberately combines the largest frame and label dimensions. Actor timing uses 5 warmups, 20 repetitions, one CPU thread, resident inputs and explicit CUDA synchronization.

| Environments | Exported actor median (ms) |
| --- | --- |
| 1 | 0.115 |
| 64 | 0.143 |
| 256 | 0.142 |
| 1024 | 0.144 |
| 12288 | 0.666 |

A 12,288-environment, 32-step native rollout allocates 2261.14 MiB of tensor payload. A full native PPO epoch (393,216 transitions, mini-batch 4,096) with contact auxiliary supervision takes 1452.55 ms on its first update. A separate warmed update peaks at 6359.98 MiB in the Torch CUDA allocator, including persistent storage. Native simulator, renderer and non-Torch allocations are excluded.

All measured batches produce finite outputs. Exported actor and native deterministic action outputs match exactly. Encoder parameters change during PPO and contact supervision, all retained gradients are finite, and changing privileged critic inputs and training labels changes actor actions by 0. These checks establish implementation and capacity properties, not learned assembly competence.

The complete machine-readable output, raw timing samples and three-table report are generated under `outputs/benchmarks/`. Official task training, held-out geometry evaluation and paper success thresholds remain separately qualified.
