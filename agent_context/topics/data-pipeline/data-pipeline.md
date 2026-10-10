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
| File-only validation and grouped quality splits | `embodichain/data_pipeline/datasets/inspection.py`; `embodichain dataset validate\|split` |
| Durable commit diagnostics and metadata repair | `embodichain/data_pipeline/recording/journal.py`; `embodichain dataset recover` |
| Isolated LeRobot 0.6.1 language/depth conversion | `tools/lerobot_export/` (Python >=3.12, independent dependencies) |

Read [online sampling](online-sampling.md) for worker states, refill, shared
errors, continuity and DataLoader behavior. Read [persistence](persistence.md)
for fragment transactions, async ownership, depth output and shutdown. Read
[offline dataset tools](offline-datasets.md) for lineage, quality filtering,
validation, and conversion across Python/LeRobot versions.

## Online worker contract

Online batches also carry engine-local, append-only task/subtask IDs. The shared
registry and `resolve_language(batch)` expose the frozen texts separately from
TensorDict data, without tying old samples to reused trajectory rows. See
[online sampling](online-sampling.md) for capacity and process boundaries.

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

Expert rollout actions use the environment's `ExpertActionSpec`, independently
of the policy action space. The default position mode preserves active-joint
qpos storage and rejects qvel/qf-only controller commands. Position-velocity
mode stores a flat `[qpos, qvel]` vector;
LeRobot feature width and ordered names follow that layout, while episode and
trajectory metadata publish its version, joint names, slices, mode, and
environment `step_dt`.

LeRobot recording preserves that existing expert schema when `action_contract`
is omitted. Expert contracts select `joint_position` or
`joint_position_velocity` and continue to derive width, names, and slices from
`ExpertActionSpec`. Policy contracts select `eef_pose_parallel_gripper` or
`joint_position_parallel_gripper`; their width, ordered names, and term slices
come from the live `ActionManager` descriptors. The descriptor sequence must
match the declared representation and is stored as
`embodichain.action_terms` in action-feature metadata and episode sidecars.
Policy recorders validate EEF finiteness and normalized gripper bounds before
ActionManager processing, then validate again before persistence. A
`ControllerAction` cannot be written into this policy schema.
Recorders consume the task-owned episode instruction snapshot. Program roots,
handwritten `task_instruction`, and explicit per-episode overrides are resolved
by `demo.py`; recorder `instruction` is only a legacy fallback. Snapshots and
segment annotation sources survive async enqueue and fragment export, including
unannotated dynamic programs whose goal remains explicitly unknown.

Language annotations are independent of the action contract: LeRobot `task` /
`task_index` retain the overall task and `subtask` / `subtask_index` identify the
current segment instruction through `meta/subtasks.parquet`. Natural-segment
fragments retain both levels. Missing segment instructions fall back to the
overall task. Executed controller commands are not a separate dataset feature.

`dataset.save_episode()` is the LeRobot commit point. A later depth/sidecar
failure cannot roll back that episode. Fragment IDs provide same-recorder
deduplication and sticky partial-commit errors. Per-episode fsynced commit
journals add durable evidence for offline diagnosis and conservative sidecar
repair; they cannot roll back an unknown SDK commit or resume unfinished videos.
Recording run/source/episode UUIDs link full episodes, fragments and optional
`.pt` replay artifacts. The host still owns state capture and restore.

Async persistence clones tensor payloads to CPU and copies metadata before
enqueue. One FIFO worker owns LeRobot access. Finalize rejects new work, drains,
joins and surfaces errors. Set ``async_queue_maxsize`` to a positive payload
count to apply producer backpressure when memory bounds matter; the default
``0`` preserves the historical unbounded queue. Queue depth and backpressure
time are available through the recorder's async stats.

## Change sites and focused validation

| Change | Tests |
|---|---|
| Worker states, valid sampling, continuity, DataLoader | `tests/data_pipeline/test_online_data.py` |
| Sync commit, FPS, fragments and depth routing | `tests/gym/envs/managers/test_dataset_functors.py` |
| Legacy/action-contract schemas and official task mapping | `tests/gym/envs/managers/test_dataset_functors.py` |
| Clone isolation, FIFO, errors and drain | `tests/gym/envs/managers/test_async_dataset_functors.py` |
| Save/discard dispatch | `tests/gym/envs/managers/test_dataset_manager.py` |
| Depth codec/temp files/metadata | `tests/data_pipeline/depth_video/` |

For corrupt or duplicate recordings, establish whether failure preceded or
followed the LeRobot commit before changing retry logic. For engine failure,
inspect the shared worker error and attempt limit before adding consumer retries.
Benchmark throughput using `scripts/benchmark/data_pipeline/benchmark_lerobot_save.py`
with recorded hardware, image size, environment count and writer settings.
The CPU-only `benchmark_segment_annotations.py` isolates the validated frame-map
lookup from simulation and storage costs.
