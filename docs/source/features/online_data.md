# Online Data Streaming

This page documents the online data streaming pipeline used for live training from simulation. The core pieces are:

- **OnlineDataEngine**: a process-safe shared buffer that stores trajectories coming from live simulation workers.
- **OnlineDataset**: a PyTorch `IterableDataset` that samples trajectory chunks from the engine in either item mode or batch mode.
- **ChunkSizeSampler**: an interface for drawing dynamic chunk sizes per iteration step.

These components live under `embodichain/data_pipeline/` and are designed to work with standard `DataLoader` patterns.

---

## OnlineDataEngine

For an end-to-end BC experiment, run
`python examples/data_pipeline/ods_pick_place_showcase.py` from an installed
checkout. The Franka task qualifies a physical expert, trains a geometry MLP
by default, and evaluates immutable student checkpoints in a separate process.
`--policy gru-pose` selects a GRU with the same measured geometry and pose-request
contract; `--policy mlp-joint` selects joint residuals. The older `--policy gru`
route retains its original observation and joint-command contracts. Cameras
are display-only; policy geometry comes from measured robot/object state.

The runner supports live ODS, frozen initial data, offline full-corpus training,
exact offline MLP minibatch replay and serial refresh. Exact replay requires
`--mode offline-matched`, a validated `--offline-data` corpus and a matching
`--batch-manifest`; it is not available for the geometry GRU. Serial refresh
uses an unmatched supervision schedule, so its timings alone do not isolate
the effect of overlap. Use `--stage overfit-one` or `--stage overfit-four` for
small-data diagnostics and `--smoke --no-viewer --no-video` for an engineering
check. These checks do not establish physical student success or a learning
advantage. Commands, results and the comparison protocol live under
`extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/`.

### Configuration and recorded-action meaning

The engine preserves expert-controller recording by default. Policy requests
have a separate schema and may have a different width from both robot state
and expert commands.

| Setting | Meaning |
| --- | --- |
| `state_dim` | Stored robot proprioception width. |
| `action_dim` | Optional stored action width; omitted values preserve the historical `state_dim` width. |
| `action_semantics` | `"expert_controller_command"` by default, or explicit `"policy_request"`. |
| `action_contracts` | Ordered ActionManager contract IDs; required with an explicit `action_dim` for policy-request recording. |
| `refill_policy` | `"sample_threshold"` by default, or owner-requested `"manual"` refill. |
| `include_provenance` | Add numeric frame identity and publication timestamps to sampled tensors. |
| `shutdown_timeout` | Finite, positive graceful-close wait in seconds; defaults to `5.0`. |

For an environment with six relative EEF dimensions followed by one normalized
parallel-gripper dimension, select the request schema explicitly:

```python
from embodichain.data_pipeline.engine import OnlineDataEngineCfg

cfg = OnlineDataEngineCfg(
    gym_config=your_gym_cfg,
    state_dim=9,
    action_dim=7,
    action_semantics="policy_request",
    action_contracts=("eef_pose.relative@1", "gripper.parallel@1"),
    refill_policy="manual",
    include_provenance=True,
)
```

Before generating transitions, the worker checks the live rollout semantics,
proprioception/action widths, ordered contracts, contiguous term slices and
controlled-joint order. It selects the environment's
`rollout_action_semantics` accordingly. Explicit policy recording preserves raw
requests through ActionManager processing, requires finite floating values
within the action-space bounds, and works without a DatasetManager. A direct
`ControllerAction` cannot be stored as a policy request. Expert-controller
recording continues to use `ExpertActionSpec`.

With manual refill, `request_refill()` returns a target generation and coalesces
an outstanding request; `wait_for_generation()` waits for complete publication
with a timeout. `get_stats()` reports production/consumption counters and excludes
the active write window from readable-frame counts.

With `include_provenance=True`, each sample carries `ods_run_id`,
`ods_episode_id`, `ods_generation`, `ods_frame_index` and `ods_accepted_at_ns`.
Frame indices are zero-based positions in the stored episode. The timestamp
uses Unix nanoseconds when a successful episode is accepted for publication;
it is not simulation time or a monotonic duration. The worker stamps provenance
under the publication lock and samplers copy it with the corresponding payload.
Invalid frames retain `-1` for every provenance field. Engine-local IDs identify
live samples; validated offline corpus manifests provide separate content
identities.

### Causal context and padding

`sample_batch_with_context(batch_size, history_length, prediction_horizon)`
copies a fixed causal history and future supervision around each valid anchor
under the producer lock. The current frame is at `history_length - 1`.
Episode ends, validity gaps and continuity changes bound the context; numeric
padding is zero, while language/provenance padding is `-1`. `valid` selects real
frames and `loss_mask` selects real current/future targets. Decode language only
for valid frames, as `resolve_language()` rejects padding IDs.

The PickPlace showcase uses ChunkBC with two measured observations and a masked
action-chunk loss. Live resume restores the model, optimizer and RNG state while
creating a fresh ODS source; it does not restore the previous shared cache.
Unsupported checkpoint parameterizations are rejected explicitly.

**Module:** `embodichain/data_pipeline/engine/data.py`

`OnlineDataEngine` manages an in-memory, shared buffer for streaming trajectory data. A typical usage pattern is:

1. Build and start the engine with `OnlineDataEngineCfg`.
2. Run simulation workers that continually push new experience into the engine.
3. Train by sampling trajectory chunks from the engine via `OnlineDataset`.

Key ideas:

- **Shared buffer**: the simulation producer and training consumers can read/write concurrently.
- **CPU shared memory**: the supported process-shared buffer uses CPU storage;
  sampled tensors can then be transferred to the learner's device.
- **Chunked sampling**: training samples fixed-length or dynamically sized chunks.
- **Transactional rows**: the current write window is locked until a complete,
  successful episode replaces it. Failed generation attempts never become
  sampleable.
- **Variable lengths**: every frame has a `valid` flag. Sampling constructs
  windows only from real frames and never reads zero padding or a stale tail.
- **Episode-uniform sampling**: an eligible episode row is selected uniformly,
  then a valid start offset is selected within that row. Longer episodes do
  not receive extra probability merely because they contain more windows.
- **Explicit lifecycle**: the engine progresses through `CREATED`, `STARTING`,
  `READY`, and a terminal `FAILED` or `STOPPED` state. Sampling is allowed only
  in `READY`, and a stopped or failed instance cannot be restarted.
- **Fail-fast errors**: initial-fill timeouts, rollout failures, hard worker
  exits, and shutdown/recorder failures are raised to the owner. The same
  stable error snapshot is visible to every forked or spawned data consumer.

Each row stores one complete task episode. A row may contain multiple semantic
segments. In addition to observations, actions, and rewards, the shared buffer
contains:

```text
valid, episode_step, segment_id, segment_step,
segment_start, segment_end, segment_accepted, segment_attempt_id,
continuity_id, terminated, truncated, task_index, subtask_index
```

Tasks use the same `create_demo_segments()` protocol as `run-task`; legacy
`create_demo_action_list()` tasks are treated as one segment. The worker checks
termination after every action and retries a failed episode up to
`max_generation_attempts` before reporting an error.

`start()` blocks until the initial fill is complete and is bounded by
`initialization_timeout`. If the worker exits or generation fails, `start()`
raises the worker error instead of waiting indefinitely. A later worker failure
is reported on the next `sample_batch()` call rather than silently serving a
permanently stale buffer.

### Minimal setup

```python
from embodichain.data_pipeline.engine import OnlineDataEngine, OnlineDataEngineCfg

cfg = OnlineDataEngineCfg(
    buffer_size=2,              # trajectories kept in the shared buffer
    state_dim=6,                # example state dimension
    gym_config=your_gym_cfg,    # parsed gym config for the task
    initialization_timeout=300, # maximum seconds for the initial fill
)
engine = OnlineDataEngine(cfg)
engine.start()
```

### Shutdown

Prefer a context manager so cleanup runs on both success and failure:

```python
with OnlineDataEngine(cfg) as engine:
    batch = engine.sample_batch(batch_size=32, chunk_size=64)
    train_step(batch)
```

For an explicitly managed lifecycle, call `stop()` in the process that created
the engine:

```python
engine = OnlineDataEngine(cfg)
engine.start()
try:
    train(engine)
finally:
    engine.stop()
```

`stop()` is idempotent after a successful cleanup. It waits up to
`shutdown_timeout` seconds for graceful worker close and raises any failure reported while the worker closes its environment or
flushes committed data. If graceful shutdown times out and the worker requires
`terminate()` or `kill()`, durability cannot be confirmed: the engine remains
`FAILED` and `stop()` raises instead of reporting a false `STOPPED` state. When
a `with` body and cleanup both fail, the body exception remains primary and the
cleanup failure is attached as a note.

Start the engine before constructing multiprocessing `DataLoader` workers.
Forked and spawned copies may call `sample_batch()`, but only the original
owner process may call `start()` or `stop()`; consumer destructors never signal
the shared producer.

---

## OnlineDataset

**Module:** `embodichain/data_pipeline/datasets/online_data.py`

`OnlineDataset` wraps a live `OnlineDataEngine` and exposes a PyTorch `IterableDataset`. It supports two modes:

### Item mode (default)
- Create the dataset with `batch_size=None` (default).
- Each iteration yields a single `TensorDict` of shape `[chunk_size, ...]`.
- Use `DataLoader(dataset, batch_size=B)` to let the DataLoader stack items into batches.

```python
from torch.utils.data import DataLoader
from embodichain.data_pipeline.datasets import OnlineDataset

dataset = OnlineDataset(engine, chunk_size=64)
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=OnlineDataset.collate_fn,
)
for batch in loader:
    # batch shape: [32, 64, ...]
    train_step(batch)
```

### Batch mode
- Create the dataset with `batch_size=N`.
- Each iteration yields a pre-batched `TensorDict` of shape `[N, chunk_size, ...]`.
- Use `DataLoader(dataset, batch_size=None)` to bypass auto-collation.

```python
dataset = OnlineDataset(engine, chunk_size=64, batch_size=32)
loader = DataLoader(
    dataset,
    batch_size=None,
    collate_fn=OnlineDataset.passthrough_collate_fn,
)
for batch in loader:
    # batch shape: [32, 64, ...]
    train_step(batch)
```

### Dynamic chunk sizes
Pass a `ChunkSizeSampler` instead of an `int` to `chunk_size` to sample a new length each iteration step.

```python
from embodichain.data_pipeline.datasets.sampler import UniformChunkSampler

sampler = UniformChunkSampler(low=16, high=64)
dataset = OnlineDataset(engine, chunk_size=sampler)
```

In batch mode, the sampler is called once per step so all trajectories in the batch share the same chunk length.

### Segment-aware sampling

Set `sampling_mode` according to the training objective:

```python
# Chunks may span adjacent subtasks (default).
episode_dataset = OnlineDataset(engine, chunk_size=64, sampling_mode="episode")

# Every chunk stays inside one pick/place segment.
segment_dataset = OnlineDataset(engine, chunk_size=32, sampling_mode="segment")

# Every chunk contains an internal transition between two segments.
boundary_dataset = OnlineDataset(engine, chunk_size=32, sampling_mode="boundary")
```

All three modes still require every sampled frame to be valid. `boundary`
requires a chunk size of at least two and raises a clear error when no internal
boundary can satisfy the requested length.

---

## ChunkSizeSampler

**Module:** `embodichain/data_pipeline/datasets/sampler.py`

`ChunkSizeSampler` is a small interface that returns a positive integer chunk size each time it is called.

Built-in samplers:

- `UniformChunkSampler(low, high)`: discrete uniform over `[low, high]`.
- `GMMChunkSampler(means, stds, weights, low, high)`: Gaussian mixture with optional bounds.

Example (GMM):

```python
from embodichain.data_pipeline.datasets.sampler import GMMChunkSampler

sampler = GMMChunkSampler(
    means=[16.0, 64.0],
    stds=[4.0, 8.0],
    weights=[0.6, 0.4],
    low=8,
    high=96,
)
```

---

## End-to-end demo

A runnable example that wires everything together is provided in:

- `examples/data_pipeline/online_dataset_demo.py`

It shows item mode, batch mode, and dynamic chunk sizes. Run it with:

```bash
python examples/data_pipeline/online_dataset_demo.py
```

---

## See Also

- [RL Architecture](../overview/rl/index.rst) — RL training pipeline
- [Expert Data Expansion Tutorial](../tutorial/data_expansion.rst) — Generating offline datasets
