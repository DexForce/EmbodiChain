# Data pipeline and persistence

This topic owns online worker/sampling contracts and durable demonstration
recording. Generic manager dispatch belongs to
[manager-functor](../manager-functor/manager-functor.md); asset lookup/download
belongs to [data-assets](../data-assets/data-assets.md).

## Entry points

| Request | Owner |
|---|---|
| Online worker and sample selection | `embodichain/data_pipeline/engine/data.py`: `OnlineDataEngine` |
| Infinite PyTorch dataset | `embodichain/data_pipeline/datasets/online_data.py`: `OnlineDataset` |
| Commit/discard/finalization dispatch | `embodichain/lab/gym/envs/managers/dataset_manager.py` |
| Synchronous LeRobot persistence | `embodichain/lab/gym/envs/managers/datasets.py`: `LeRobotRecorder` |
| Background persistence | `embodichain/lab/gym/envs/managers/async_datasets.py`: `AsyncLeRobotRecorder` |
| Depth sidecars | `embodichain/data_pipeline/depth_video/writer.py`, `reader.py` |
| Offline collection retries and final partial batch | `embodichain/lab/scripts/run_env.py` |

Read [online sampling](online-sampling.md) for worker states, refill, shared
errors, continuity and DataLoader behavior. Read [persistence](persistence.md)
for fragment transactions, async ownership, depth output and shutdown.

## Online worker contract

The creator process owns `start()` / `stop()`. A forkserver worker fills CPU
shared memory: `CREATED → STARTING → READY`; `FAILED` and `STOPPED` are terminal.
Startup waits for initial fill with a timeout and demonstration generation has
bounded attempts. Worker errors are published to all consumers; failed engines
are replaced with new instances rather than restarted.

`sample_batch()` is READY-only. Selection excludes the currently written range,
checks validity/continuity/segment boundaries, and clones the result while
holding the shared write lock. Moving selection or cloning outside that scope
can expose partially overwritten rows. Refill requests are threshold-driven.

Distinguish a terminal worker failure from a sampling request with no eligible
unlocked window. The latter raises `RuntimeError` immediately without itself
setting `FAILED` or triggering a refill. `OnlineDataset` propagates both kinds
of exception; it does not supply an automatic retry loop.

## Persistence contract

`env.step()` buffers frames. An explicit reset/commit reaches
`DatasetManager.apply("save", env_ids)` before rows are cleared. Finalization
drains committed work; it does not commit a live rollout. Final partial vector
batches use `commit_env_ids` to select dataset rows while preserving a full
physical reset. The recorder requires an exactly representable integer FPS.

`dataset.save_episode()` is the LeRobot commit point. A later depth/sidecar
failure cannot roll back that episode. Fragment IDs provide same-recorder
deduplication and sticky partial-commit errors; they are not a crash-recovery journal.

Async persistence clones tensor payloads to CPU and copies metadata before
enqueue. One FIFO worker owns LeRobot access. Finalize rejects new work, drains,
joins and surfaces errors. The queue is unbounded: writer lag can grow memory.
Choose synchronous persistence or throttle production when memory bounds matter.

## Change sites and focused validation

| Change | Tests |
|---|---|
| Worker states, valid sampling, continuity, DataLoader | `tests/data_pipeline/test_online_data.py` |
| Sync commit, FPS, fragments and depth routing | `tests/gym/envs/managers/test_dataset_functors.py` |
| Clone isolation, FIFO, errors and drain | `tests/gym/envs/managers/test_async_dataset_functors.py` |
| Save/discard dispatch | `tests/gym/envs/managers/test_dataset_manager.py` |
| Depth codec/temp files/metadata | `tests/data_pipeline/depth_video/` |

For corrupt or duplicate recordings, establish whether failure preceded or
followed the LeRobot commit before changing retry logic. For engine failure,
inspect the shared worker error and attempt limit before adding consumer retries.
Benchmark throughput using `scripts/benchmark/data_pipeline/benchmark_lerobot_save.py`
with recorded hardware, image size, environment count and writer settings.
