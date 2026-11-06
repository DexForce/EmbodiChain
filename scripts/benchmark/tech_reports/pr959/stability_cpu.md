# PR #959 stability policy CPU comparison

CPU: AMD EPYC 9K65 192-Core Processor; Python 3.11.14; Torch 2.7.0+cu128; CPU float32; one Torch thread.

The same benchmark harness ran five warmups and 40 completed-call wall samples per case. Each call checks a constant (B,4,4) pose and emits 75 holds over the declared 3 s / 0.04 s control grid. Construction, native acquisition, actual physics, GPU work and a full robot are excluded. Paired cases include terminal-only and per-step metadata reads. Python allocation peaks are measured separately by tracemalloc; they exclude native/Torch/RSS peaks.

Baseline runtime SHA-256: `8d79a0bc5e3996c01f8ce5de1ad9fc1d236e3a3a2a4452760f8aba9d167b8488` (pre-optimization runtime at `5846274ce`).
Candidate runtime SHA-256: `4efd343c5069fbb8478aa81d76787ace558bb2f17b610c253daefda96a9c0577`.

## Time & Memory

| B | Condition | Metadata reads | Before median ms | After median ms | Before p95 ms | After p95 ms | Before Python peak bytes | After Python peak bytes | GPU peak |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | placement | terminal_only | 9.7661 | 9.7704 | 9.8236 | 9.8834 | 4399 | 4065 | unmeasured |
| 1 | placement | every_step | 10.5351 | 10.5870 | 10.6011 | 10.6600 | 4767 | 4877 | unmeasured |
| 1 | hold | terminal_only | 20.9822 | 19.4463 | 21.0807 | 19.6120 | 5366 | 5354 | unmeasured |
| 1 | hold | every_step | 22.1337 | 20.5424 | 22.2579 | 20.6788 | 6816 | 6190 | unmeasured |
| 4 | placement | terminal_only | 9.9871 | 9.9517 | 10.0330 | 10.0293 | 4386 | 3939 | unmeasured |
| 4 | placement | every_step | 10.8782 | 10.8538 | 10.9123 | 10.9703 | 4870 | 4803 | unmeasured |
| 4 | hold | terminal_only | 21.2939 | 19.9333 | 21.3739 | 20.0032 | 5606 | 5594 | unmeasured |
| 4 | hold | every_step | 22.7785 | 21.2210 | 22.9236 | 21.3081 | 6390 | 6382 | unmeasured |

## Success & Other Metrics

All eight cases preserve the exact final JSON metadata hash, 75 commands and accepted masks. Fixture success is 1.0 for both implementations; it is not physical task qualification. Empty attachment selections reduce qpos reads from 227 to 150 per call; dual-part hold keeps all 154 FK calls while computing one inverse per pose. Shared-machine wall timing is observational, not a guaranteed application speedup. Placement timing is effectively unchanged; hold policy median is 6.4–7.3% lower. No lazy metadata format/cache was introduced.

## Leaderboard

| Implementation | overall_success_rate | Scope |
|---|---:|---|
| Baseline | 1.0 | Fixed CPU fixture acceptance and deterministic metadata |
| Candidate | 1.0 | Same fixture acceptance and exact metadata |

Reproduce with `python -m scripts.benchmark.task_program.stability --output /tmp/stability`.
