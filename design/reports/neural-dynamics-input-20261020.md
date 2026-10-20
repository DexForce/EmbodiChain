# NeRD bounded input pipeline — 2026-10-20

**The synchronous input path is faster without changing sampled data, optimizer
order or checkpoint semantics.** The final benchmark observes 11–23× speedups for
batch 512 on the genuine small datasets. A distinct-trajectory, many-shard stress
case improves 1.86×; it remains substantially slower than GPU model computation.
These measurements do not qualify a full training run or reduce the previously
measured 607.52 GPU-hour compute-only lower bound.

## Implementation and resource model

`DynamicsWindowDataset.get_batch(indices)` replaces individual window reads plus
`default_collate` in statistics computation, training, offline evaluation and the
original-trainer transport adapter. The original `__getitem__` path remains intact
as the independent comparison baseline and ordinary Dataset API.

The method groups its I/O schedule by shard, trajectory and fixed time page,
then writes results directly into owned batch arrays in the **original requested
order**, including duplicate indices. Requests never cross a trajectory or split.
Startup prefixes retain the same right padding, masks and valid lengths.

Each field read spans at most `read_page_steps + history_length` frames: the
benchmark uses 256 + 10 = **266 frames**. Sparse requests in a long trajectory
therefore cannot cause the whole trajectory to be loaded. Public HDF5 hyperslab
reads avoid repeatedly constructing high-level Dataset/File wrappers. The memory
space is created from the actual owned contiguous destination array, and file
shape/range is checked before reading. Empty fields retain their shape and dtype.

Storage is the requested output batch, O(batch-size) indexing, one bounded source
field block and one per-field gather. The file-handle LRU remains capped at four
per reader/worker. There is **no retained trajectory cache, whole-dataset load,
background worker or speculative prefetch**. Native HDF5 caches remain separate
from the explicitly measured Python/NumPy allocations. The existing O(trajectory
count) metadata index is unchanged.

The sampler advances only after an optimizer update, exactly as before. Returned
rows do not alias one another when indices repeat, and later calls or file closure
do not overwrite previously returned tensors.

Changed production owners:

- `embodichain/data_pipeline/dynamics/shards.py`: bounded grouped/hyperslab reader.
- `embodichain/learning/dynamics/trainer.py`: batched statistics/train/evaluation.
- `embodichain/learning/dynamics/reference.py`: identical batch transport into the
  untouched official trainer methods.

## Measurements

CPU read + collation + shared codec encoding, one CPU thread, history 10. Each
implementation uses the same seeded index sequence, two warmup batches and ten
timed batches. Source verification, index generation and correctness checks are
outside timing. Filesystem pages are warm; this is not a cold-storage benchmark.
Python 3.11.14, Torch 2.7.1+cu128 and h5py 3.15.1 were used. No GPU was used.

| Dataset / batch | Individual windows, ms | Grouped reads, ms | Speedup |
|---|---:|---:|---:|
| Current Newton Cartpole / 1 | 0.742 | 0.443 | 1.68× |
| Current Newton Cartpole / 16 | 8.425 | 3.720 | 2.26× |
| Current Newton Cartpole / 512 | 257.874 | 12.749 | 20.23× |
| Current Newton Ant / 1 | 1.142 | 0.909 | 1.26× |
| Current Newton Ant / 16 | 9.336 | 5.249 | 1.78× |
| Current Newton Ant / 512 | 285.034 | 25.883 | 11.01× |
| Official Warp Cartpole / 1 | 0.618 | 0.358 | 1.73× |
| Official Warp Cartpole / 16 | 7.592 | 3.229 | 2.35× |
| Official Warp Cartpole / 512 | 241.613 | 10.622 | 22.75× |
| Official Warp Ant / 1 | 1.097 | 0.816 | 1.34× |
| Official Warp Ant / 16 | 8.795 | 4.504 | 1.95× |
| Official Warp Ant / 512 | 271.967 | 20.767 | 13.10× |
| 128-shard distinct-trajectory stress / 1 | 1.304 | 1.066 | 1.22× |
| 128-shard distinct-trajectory stress / 16 | 11.353 | 7.237 | 1.57× |
| 128-shard distinct-trajectory stress / 512 | 337.134 | 181.723 | 1.86× |

The current Newton datasets have 57 training trajectories across 16 shards; the
official datasets have 32 training trajectories in one shard. Their batch-512
requests consequently reuse many trajectories, explaining the larger gains.

The stress dataset repeats an actual Ant trajectory under 512 independent IDs
across 128 shards. Every batch-512 sample chooses a **distinct trajectory** and
the file cache holds only four shards. This tests a weak-reuse case beyond cache
capacity, not physics accuracy. Cold-batch file opens fall from **503 to 128**;
retained handles never exceed four. Output/encoding tensors match bitwise.

For stress batch 512, returned tensor data is 7,316,480 bytes. Tracemalloc's
Python/NumPy peak falls from **10,646,618 to 8,207,876 bytes**. These numbers exclude
native HDF5/Torch allocator memory and must not be presented as total process RSS.
For the small genuine Ant datasets, grouped peaks are about 7.48 MB. No retained
data cache scales with the dataset's total sample count.

The stress result still spends roughly 182 ms preparing a batch, compared with
about 14 ms of Ant GPU training computation in the earlier separate benchmark.
The small-dataset 11–23× gains must therefore **not** be extrapolated to full-size
random sampling. Packed layouts or controlled prefetch would need a separate,
representative experiment before changing the full-budget duration estimate.

## Correctness and provenance checks

Focused tests cover:

- Exact tensor/metadata equality with individual reads for B=1/16/512, histories
  1/3/10, partial prefixes, repeated/negative indices and more shards than cache.
- Independent duplicate rows, cross-call ownership and validity after close.
- Sparse windows across a 4,097-step trajectory with an observed 39-frame bound
  (page 32 + history 7).
- Float64, float32, integer, Boolean, multidimensional, zero-width and UTF-8 fields.
- Shape validation before a low-level buffer write, invalid indices before I/O,
  worker-local file handles and existing teardown behavior.
- Bitwise-identical 20-step training weights and logs between individual and
  grouped reads with dropout, plus the existing exact-resume and independent
  official-trainer tests.

A separate provenance fix in `workflow.run_train` closes a relabeling hole:
manifest annotations may add unrelated keys, but cannot replace/drop existing
dataset metadata or invent a missing deployment-contract field. Protected fields
include physics signature/schema, gravity, ordered joints, asset identity, codec,
timestep and contact schema. An explicitly empty override is rejected. Fine-tuning
may initialize from another physical model with compatible codec/timestep; its
result retains the **new measured dataset's** identity. Negative tests cover
changed, omitted and invented signatures, schema/gravity/assets, while positive
tests cover annotations and legitimate fine-tuning.

The combined scoped suite passed **70 tests** with the explicitly enabled real
reference checkout: storage, training/resume, original trainer and original CPU
collection, experiment/report helpers, CLI and packaging checks. The changed
Python files pass the scoped Black check. No GPU tests were needed for this
storage/transport change.

## Reproduction and artifacts

Benchmark entry point: `scripts/benchmark/learning/dynamics_input.py`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m scripts.benchmark.learning.dynamics_input \
  --dataset /tmp/embodichain-nerd-v2-qualification/cartpole/data \
  --dataset /tmp/embodichain-nerd-v2-qualification/ant/data \
  --dataset /tmp/nerd-official-real-cartpole-20261020/canonical \
  --dataset /tmp/nerd-official-real-ant-20261020/canonical \
  --stress-dir /tmp/nerd-input-new-stress-run \
  --output /tmp/nerd-input-new-benchmark.json
```

The stress destination must be new. Full measured samples, dataset fingerprints,
allocation scope and handle counts are saved in
`/tmp/nerd-input-final-benchmark-20261020.json`; console output is in the adjacent
`.log`. No multiweek training job or new GPU workload was launched for this change.
