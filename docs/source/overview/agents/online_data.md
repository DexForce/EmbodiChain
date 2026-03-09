# Online Data Streaming

This page documents the online data streaming pipeline used for live training from simulation. The core pieces are:

- **OnlineDataEngine**: a process-safe shared buffer that stores trajectories coming from live simulation workers.
- **OnlineDataset**: a PyTorch `IterableDataset` that samples trajectory chunks from the engine in either item mode or batch mode.
- **ChunkSizeSampler**: an interface for drawing dynamic chunk sizes per iteration step.

These components live under `embodichain/agents/` and are designed to work with standard `DataLoader` patterns.

---

## OnlineDataEngine

**Module:** `embodichain/agents/engine/data.py`

`OnlineDataEngine` manages an in-memory, shared buffer for streaming trajectory data. A typical usage pattern is:

1. Build and start the engine with `OnlineDataEngineCfg`.
2. Run simulation workers that continually push new experience into the engine.
3. Train by sampling trajectory chunks from the engine via `OnlineDataset`.

Key ideas:

- **Shared buffer**: multiple producers (simulation workers) and multiple consumers (training workers) can read/write concurrently.
- **GPU-friendly**: buffer is designed for efficient sampling and minimal copying.
- **Chunked sampling**: training samples fixed-length or dynamically sized chunks.

### Minimal setup

```python
from embodichain.agents.engine.data import OnlineDataEngine, OnlineDataEngineCfg

cfg = OnlineDataEngineCfg(
    buffer_size=2,           # number of trajectories kept in the ring buffer
    state_dim=6,             # example state dimension
    gym_config=your_gym_cfg, # parsed JSON config for the task
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

**Module:** `embodichain/agents/datasets/online_data.py`

`OnlineDataset` wraps a live `OnlineDataEngine` and exposes a PyTorch `IterableDataset`. It supports two modes:

### Item mode (default)
- Create the dataset with `batch_size=None` (default).
- Each iteration yields a single `TensorDict` of shape `[chunk_size, ...]`.
- Use `DataLoader(dataset, batch_size=B)` to let the DataLoader stack items into batches.

```python
from torch.utils.data import DataLoader
from embodichain.agents.datasets import OnlineDataset

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
from embodichain.agents.datasets.sampler import UniformChunkSampler

sampler = UniformChunkSampler(low=16, high=64)
dataset = OnlineDataset(engine, chunk_size=sampler)
```

In batch mode, the sampler is called once per step so all trajectories in the batch share the same chunk length.

---

## ChunkSizeSampler

**Module:** `embodichain/agents/datasets/sampler.py`

`ChunkSizeSampler` is a small interface that returns a positive integer chunk size each time it is called.

Built-in samplers:

- `UniformChunkSampler(low, high)`: discrete uniform over `[low, high]`.
- `GMMChunkSampler(means, stds, weights, low, high)`: Gaussian mixture with optional bounds.

Example (GMM):

```python
from embodichain.agents.datasets.sampler import GMMChunkSampler

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

- `examples/agents/datasets/online_dataset_demo.py`

It shows item mode, batch mode, and dynamic chunk sizes. Run it with:

```bash
python examples/agents/datasets/online_dataset_demo.py
```
