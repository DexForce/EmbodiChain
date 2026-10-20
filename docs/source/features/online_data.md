# Online Data Streaming (ODS)

ODS supplies owned, causally aligned training batches from continuously
generated trajectories through a bounded CPU shared cache. The producer and
learner can run independently; explicit provenance and optional per-trajectory
delivery budgets make data use auditable. Shared APIs live in
`embodichain.data_pipeline` and do not depend on a particular learner or task.

| Component | Responsibility |
| --- | --- |
| `OnlineDataEngine` | Own the simulation producer, shared storage, publication, locked selection/copy and optional live sampling ledger. |
| `BatchSource` adapters | Present live, resident tensor and numeric LeRobot data under one explicit schema/context contract. |
| `SequenceDataset` | Iterate causal contexts from a source; the caller owns production, loading, retry and refill. |
| `OnlineDataset`, `ChunkSizeSampler` | Preserve item/batch chunk iteration and dynamic chunk lengths. |
| `configure_ods()` | Calculate bounded reuse, cache and pacing/refill targets from measurements and a versioned profile; the host applies the plan. |

## Guides and ownership

```{toctree}
:maxdepth: 1
:hidden:

ods/configuration
ods/evaluation
```

- [Throughput configuration](ods/configuration.md): rate units, reuse profiles,
  exact cache/rate limits and consumer enforcement responsibilities.
- [Evaluation and offline comparison](ods/evaluation.md): exposure, quality,
  timing, resource metrics and fair experimental controls.
- [Dataset/source architecture](https://github.com/DexForce/EmbodiChain/blob/main/docs/architecture/dataset-loading-design.md):
  public schemas, source ownership and implementation boundaries.
- [Simple BC demo](https://github.com/DexForce/EmbodiChain/blob/main/examples/ods/simple_bc_demo/README.md):
  one task/model binding, runnable commands and retained experiment results.

These guides own reusable ODS behavior. Task-specific parameters, checkpoint
results and historical validation remain with their examples.

---

## OnlineDataEngine

The engine owns live production and publication. The training process chooses
when to consume data, request manual refill and stop production. The engine
does not own an optimizer or infer when a policy has learned the task.

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
| `episode_sample_budget`, `episode_sample_hard_limit` | Optional per-source delivered-chunk budget and its configuration ceiling; supply both positive values or omit both. |
| `include_provenance` | Add numeric frame identity and publication timestamps to sampled tensors. |
| `lifecycle_event_capacity` | Optional bounded publication/eviction log; the owner must drain an enabled log during production and waits. |
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

## Per-trajectory delivery protection

Set both `episode_sample_budget` and `episode_sample_hard_limit` on
`OnlineDataEngineCfg` to positive integers with budget <= hard limit. Omit both
for unrestricted historical behavior. Strict process-shared budgets use CPU
storage and identify trajectories by immutable engine run/episode IDs, rather
than reusable cache slots.

One returned batch row/context consumes one selection, independent of chunk
length. Within-batch duplicate draws count separately. Each sampling call
selects, copies and debits its complete requested batch under the shared lock.
A failed copy spends no quota; downstream transforms, optimizer failures and
prefetch do not refund delivered data. A new Dataset iterator does not reset
its source's allowance. Live replacement retires unused credits.

`get_sampling_budget_snapshot()` exposes retained identities, assigned/delivered/
remaining allowances and lifetime delivered chunks. Its unlocked total requires
at least one valid continuity-qualified anchor, but does not apply every
requested length, segment or generation filter. It is a scheduling hint, not
proof that a specific batch can be sampled. Retired-source exposure and completed
optimizer counts require consumer accounting.

| Outcome | Required handling |
| --- | --- |
| Invalid schema/arguments | Fix the configuration; do not treat it as a data wait. |
| `DataUnavailableError` | No eligible unlocked window/anchor exists. A live caller may wait with a deadline if production can change that condition. Fixed data cannot repair it by waiting. |
| `DataBudgetExhaustedError` | Eligible data exists but cannot fund the requested full batch. Stop or have the owner schedule further supply within its declared production budget. No partial batch is silently debited. |
| Worker/read/shutdown failure | Propagate the failure; do not continue indefinitely with old data. |

`OnlineDataBudgetExhaustedError` is the engine-facing alias of common
`DataBudgetExhaustedError`. Eligibility is checked before quota sufficiency;
errors are immediate and Dataset wrappers do not silently retry. Worker health
errors take precedence. Quota depletion does not automatically request refill.
Use the [configuration guide](ods/configuration.md) to connect strict sampling
protection to an explicit host pacing/refill policy.

## Shared online and offline context sources

`OnlineDataSource`, `TensorDataSource` and `LeRobotDataSource` expose
`sample_context(batch_size, spec, generator=...)` under one `BatchSource`
contract. Every source requires an explicit observation/action schema, action
semantics or representation, and positive control period `step_dt`. Vector width
alone cannot establish compatible meaning, joint order, units or frame convention.

For history H and prediction horizon F, each context has H+F-1 positions and
its current frame is H-1. `obs`, `actions`, `valid`, `loss_mask` and `frame_ids`
retain the same axes across sources. Frame identity is `(source run, episode,
frame)`; padding IDs are -1. Numeric padding is zero, and the loss mask includes
only real current/future targets. Episodes, validity gaps, continuity changes,
segment mode and optional generation filtering constrain eligible contexts.
Feature conversion must preserve masks and restore zero padding if a nonlinear
conversion maps the neutral padded input to a nonzero value.

The following sketch assumes a truthful schema for the recorded numeric fields:

```python
from torch.utils.data import DataLoader
from embodichain.data_pipeline.datasets import (
    LeRobotDataSource,
    SequenceDataset,
    SequenceSpecCfg,
)

source = LeRobotDataSource(
    "/path/to/dataset",
    schema=schema,
    source_id="recording-v1",
    memory_limit_bytes=512 * 1024 * 1024,
)
sequence = SequenceSpecCfg(history_length=2, prediction_horizon=64)
dataset = SequenceDataset(source, sequence, batch_size=64, max_batches=1000)
loader = DataLoader(dataset, batch_size=None, num_workers=0)
for batch in loader:
    train_step(batch)
```

The LeRobot adapter uses official SDK numeric reads once, then samples resident
CPU tensors. The resident allowance does not bound SDK caches, temporary staging,
total process memory or native GPU allocations. Explicit mappings are required
when recorded field names differ; standard numeric data does not require the
BC demo sidecar. Image/video and out-of-core decoding are outside this backend.

For live data, use `OnlineDataSource(engine, schema=schema,
observation_transform=feature_fn)` with the same truthful feature/action contract.
The caller starts/stops the engine and schedules any refill/wait. The adapter
owns none of those lifecycle decisions. The transform must accept copied
observations and their `valid_mask` and produce the declared numeric features.

`SequenceDataset` yields items when `batch_size=None` and complete batches when
it is specified; use DataLoader `batch_size=None` for prebatched iteration.
`max_batches` limits one iterator, not source cardinality or training epochs.
Explicit generators and finite iteration require owner-process consumption;
protected resident sources also reject unsupported worker copies. Start with
`num_workers=0`. `OfflineDataset` is a compatibility alias of `SequenceDataset`.

The [Dataset/source design](https://github.com/DexForce/EmbodiChain/blob/main/docs/architecture/dataset-loading-design.md)
provides architecture and boundary details. The following `OnlineDataset`
interface remains available for existing chunk consumers.

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

## Runnable examples

A runnable example that wires everything together is provided in:

- `examples/data_pipeline/online_dataset_demo.py`

The local [Franka BC demo](https://github.com/DexForce/EmbodiChain/blob/main/examples/ods/simple_bc_demo/README.md)
adds a complete learner, expert, checkpoint/evaluator and fixed-data controls:

```bash
python -m examples.ods.simple_bc_demo.train --smoke --no-viewer --no-video
```

Its architecture bindings, task-specific throughput packet, artifacts and
learning results belong to the example. The generic chunk example shows item
mode, batch mode and dynamic chunk sizes. Run it with:

```bash
python examples/data_pipeline/online_dataset_demo.py
```

---

## See Also

- [RL Architecture](../overview/rl/index.rst) — RL training pipeline
- [Expert Data Expansion Tutorial](../tutorial/data_expansion.rst) — Generating offline datasets
