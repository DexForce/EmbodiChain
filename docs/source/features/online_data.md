# Online Data Streaming

This page documents the online data streaming pipeline used for live training from simulation. The core pieces are:

- **OnlineDataEngine**: a process-safe shared buffer that stores trajectories coming from live simulation workers.
- **OnlineDataset**: a PyTorch `IterableDataset` that samples trajectory chunks from the engine in either item mode or batch mode.
- **ChunkSizeSampler**: an interface for drawing dynamic chunk sizes per iteration step.

These components live under `embodichain/data_pipeline/` and are designed to work with standard `DataLoader` patterns.

---

## OnlineDataEngine

**Module:** `embodichain/data_pipeline/engine/data.py`

`OnlineDataEngine` manages an in-memory, shared buffer for streaming trajectory data. A typical usage pattern is:

1. Build and start the engine with `OnlineDataEngineCfg`.
2. Run simulation workers that continually push new experience into the engine.
3. Train by sampling trajectory chunks from the engine via `OnlineDataset`.

Key ideas:

- **Shared buffer**: the simulation producer and training consumers can read/write concurrently.
- **GPU-friendly**: buffer is designed for efficient sampling and minimal copying.
- **Chunked sampling**: training samples fixed-length or dynamically sized chunks.
- **Transactional rows**: the current write window is locked until a complete,
  successful episode replaces it. Failed generation attempts never become
  sampleable.
- **Variable lengths**: every frame has a `valid` flag. Sampling constructs
  windows only from real frames and never reads zero padding or a stale tail.
- **Episode-uniform sampling**: an eligible episode row is selected uniformly,
  then a valid start offset is selected within that row. Longer episodes do
  not receive extra probability merely because they contain more windows.

Each row stores one complete task episode. A row may contain multiple semantic
segments. In addition to observations, actions, and rewards, the shared buffer
contains:

```text
valid, episode_step, segment_id, segment_step,
segment_start, segment_end, terminated, truncated
```

Tasks use the same `create_demo_segments()` protocol as `run-env`; legacy
`create_demo_action_list()` tasks are treated as one segment. The worker checks
termination after every action and retries a failed episode up to
`max_generation_attempts` before reporting an error.

If the simulation worker exits during initial fill, `start()` raises instead
of waiting indefinitely. A later worker failure is reported on the next
`sample_batch()` call rather than silently serving a permanently stale buffer.

### Minimal setup

```python
from embodichain.data_pipeline.engine.data import OnlineDataEngine, OnlineDataEngineCfg

cfg = OnlineDataEngineCfg(
    buffer_size=2,           # number of trajectories kept in the ring buffer
    state_dim=6,             # example state dimension
    gym_config=your_gym_cfg, # parsed gym config for the task (JSON or YAML)
)
engine = OnlineDataEngine(cfg)
engine.start()
```

### Shutdown

```python
engine.stop()
```

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
- [Data Generation Tutorial](../tutorial/data_generation.rst) — Generating offline datasets
