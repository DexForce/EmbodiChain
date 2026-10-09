# Export recordings for LeRobot 0.6.1

This standalone tool reads finalized EmbodiChain LeRobot v3.0 datasets from
disk. It runs in Python 3.12 independently of the simulator and does not install
or import the EmbodiChain package. EmbodiChain keeps its Python 3.10+ support
and `lerobot>=0.4.4,<0.5` recording dependency.

Create a separate environment from the repository root:

```bash
uv venv --python 3.12 /tmp/embodichain-lerobot-export
uv pip install --python /tmp/embodichain-lerobot-export/bin/python \
  './tools/lerobot_export[test]'
/tmp/embodichain-lerobot-export/bin/python -m embodichain_lerobot_export \
  /absolute/path/to/recording /absolute/path/to/new-dataset
```

The destination must be new and outside the source dataset. The source is
never edited. A temporary sibling directory holds the conversion until the
official LeRobot 0.6.1 reader and language recipe pass checks at the first,
last and segment-transition frames of every episode. Failed conversions remove
their temporary output. Finalize the recorder first: changes to the source
during the initial snapshot cause conversion to fail.

## Language and provenance

- `task_index` and `meta/tasks.parquet` retain the overall task. For legacy
  recordings that used segment text as the task, `instruction` in
  `meta/embodichain_episodes.jsonl` restores the overall task.
- `language_persistent` uses the official Arrow schema with assistant rows of
  style `subtask`. Each episode broadcasts its timestamped segment transitions
  across all of its frames. Existing `subtask_index` and `meta/subtasks.parquet`
  remain available for older consumers.
- Segment descriptions come from the sidecar where provided, then from the
  existing subtask table, and otherwise fall back to the overall task. Cropped
  fragment time starts at zero; source crop boundaries and UUIDs are preserved
  in the sidecar. Missing legacy episode UUIDs receive reproducible local
  identities derived from the source fingerprint and episode index. Missing
  source lineage remains unknown, with `lineage_unknown` and a shared source
  fingerprint so splitters conservatively group those episodes together.
- Existing actions, action contracts and RGB encoding are preserved. Changing
  timestamps adjusts the RGB video offset to retain the same source frames.
- `meta/task_subtask_recipe.json` is a ready-to-use official `TrainingRecipe`
  (JSON is also valid YAML). It renders the overall task as a user message and
  the active segment as an assistant training target.
- `meta/embodichain_export.json` records the conversion version, source
  metadata/data fingerprint, depth features and consumer check count.
- Existing recording journals must be complete and agree with their original
  episode sidecars. They are archived under
  `meta/embodichain_source_commits/<source-fingerprint>/` because their snapshots
  describe the source storage, rather than the converted language/depth layout.
  Recovery therefore never treats source commit evidence as an unfinished
  converted recording.

Datasets already containing `language_persistent` are rejected to avoid
discarding annotations. Input using pre-v3 storage needs a separate official
storage migration first. Unresolved changing task labels without an overall
sidecar instruction are rejected rather than guessing the intended task.

## Native depth

Existing `depth_meta.json` sidecars become official
`observation.depth.<sensor>` video features with `is_depth_map`, `depth_unit`
and explicit 12-bit quantizer parameters. Per-episode HEVC streams become
official video shards; their encoded bytes are copied without decoding and
re-encoding. Every depth frame is decoded for shape, range, FPS, timestamp and
frame-count checks, and statistics are recomputed using the official decoder
in metres. Episode metadata records each depth shard and video time range.

The converted dataset reads directly through the latest SDK:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/export", root="/absolute/path/to/new-dataset",
    video_backend="pyav", depth_output_unit="m",
)
sample = dataset[0]
depth_metres = sample["observation.depth.camera"]
```

The SDK defaults to millimetres when `depth_output_unit` is omitted and scales
depth statistics accordingly. Values that were clipped during original
quantization remain clipped; conversion cannot recover raw floating-point
depth or invalid-value masks that were never saved. Original depth metadata
is retained as `meta/embodichain_source_depth.json`; duplicated legacy video
files are removed from the converted dataset.

Only the existing HEVC `gray12le`, 12-bit code contract is supported. Missing
depth for an episode is an error because official video features must have
valid episode indexing. The tool does not export Mimic or robomimic data.

## Validation

Run the standalone tests in the isolated environment:

```bash
/tmp/embodichain-lerobot-export/bin/python -m pytest \
  -c tools/lerobot_export/pyproject.toml tools/lerobot_export/tests
```

These tests use the real pinned SDK, public language recipe renderer and
PyAV depth reader. They exercise both quantizers, metre/millimetre statistics,
fragment timestamps and lineage, legacy overall-task restoration, unchanged
actions, source immutability and atomic failure cleanup. Export keeps at most
one frame-data shard in memory while rewriting; annotation columns and episode
metadata remain resident, and the destination initially needs space for a full
copy of the source dataset.
