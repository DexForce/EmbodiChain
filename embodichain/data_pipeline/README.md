# Expert dataset tools

EmbodiChain records with its existing Python >=3.10 runtime and LeRobot 0.4.x.
Offline validation, quality filtering, and recovery do not start a simulator.
LeRobot 0.6.1 conversion lives in the independent
[`tools/lerobot_export`](../../tools/lerobot_export/) package and runs in its own
Python >=3.12 environment. Installing the converter does not require installing
EmbodiChain or changing its dependencies.

## Validate a recording

```bash
embodichain dataset validate /path/to/dataset --output /tmp/validation.json
embodichain dataset validate /path/to/dataset --check-media --media-frame-counts
```

The inexpensive pass checks Parquet/schema/episode consistency, action values,
segment spans and references, lineage identities, depth metadata, and commit
journal evidence. Media checks additionally inspect referenced files; counting
decoded frames is optional because it can be expensive. Legacy recordings may
lack lineage or journal fields and are reported accordingly. Validation reports
separate errors from warnings and return a nonzero exit status on errors.

## Create train/validation/test splits

```bash
embodichain dataset split /path/to/dataset /tmp/split.json \
  --fractions 0.8 0.1 0.1 --seed 42 --success-only
embodichain dataset split /path/to/dataset /tmp/scene-split.json \
  --group-by scene --scene-field scene_id --fractions 0.8 0.1 0.1
```

A split manifest references existing episodes; it does not rewrite the dataset.
Source episodes, their fragments, and derivatives share a partition. Grouped
splits can differ from requested frame/episode ratios when groups are large.
Legacy records without sufficient source identity are conservatively grouped;
`--allow-legacy-independent` explicitly permits independent legacy episodes
when the caller has verified that their derivations cannot leak across splits.
Optional recovery-count and physical-validation filters require corresponding
quality evidence; missing measurements are not silently interpreted as success.

## Recording identity and replay

Each recording has a `run_uuid`. Full episodes have an `episode_uuid`, and their
`source_episode_uuid` points to that same source identity. Natural segment
fragments preserve the source UUID and receive their own deterministic UUID.
Configuration/program fingerprints and runtime timing/version metadata support
traceability across conversion and dataset assembly.

Enable the existing `record_trajectory` and trajectory-saving configuration to
save replay state. The `.pt` artifact and LeRobot episode sidecar share source
episode identities and an explicit external artifact path. Preserve that replay
artifact when moving or archiving a dataset. A fragment links back
to its full source trajectory and source frame range. States are captured before
corresponding actions. A seed alone does not reconstruct a scene, and state
replay does not guarantee bit-exact physics across versions/backends.

## Diagnose and repair interrupted commits

Stop the recorder before inspection or repair:

```bash
embodichain dataset recover /path/to/dataset
embodichain dataset recover /path/to/dataset --repair
```

Atomic, fsynced JSON records under `meta/embodichain_commits/` track progress from
preparation through SDK, depth, and sidecar commits. Diagnosis is read-only.
Repair may restore missing episode sidecars from journal evidence once the
committed frame/depth artifacts verify. Unknown SDK commit outcomes, missing
videos, and conflicting metadata remain unresolved; the tool does not replay
frames, truncate data, or resume an unfinished video writer. Keep the report
with the recording when manual investigation is required.

Media evidence is checked by decoding referenced RGB/depth videos with PyAV,
including frame counts, FPS, timestamps, and episode spans. A nonempty file
alone is insufficient. Shared video shards are decoded once per inspection;
unreadable or incomplete videos block metadata repair.

## Recording throughput

`metadata_buffer_size` controls LeRobot episode-metadata batching and defaults
to the existing value of 1. A positive `async_queue_maxsize` bounds pending
payload count for asynchronous persistence. Larger batches/queues trade memory
and visibility of recent writes for producer throughput; use representative
camera sizes and episode lengths to select them.

Segment annotations use one validated frame map per episode. Run the CPU-only
microbenchmark to compare it with repeated range scans:

```bash
python -m scripts.benchmark.data_pipeline.benchmark_segment_annotations \
  --frames 10000 --segments 100 --repeats 5 \
  --report /tmp/segment-annotations.md
```

The benchmark verifies exact annotation equivalence and reports lookup time and
sampled process RSS. It does not measure simulation throughput or disk saving.
The existing `benchmark_lerobot_save` script measures complete recording with a
real environment.
