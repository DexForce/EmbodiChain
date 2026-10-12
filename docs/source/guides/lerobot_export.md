(lerobot-offline-export)=

# Export expert data for LeRobot 0.6.1

The standalone exporter converts finalized EmbodiChain LeRobot **v3.0 storage**
into a dataset consumable by the official LeRobot **0.6.1 library**. Storage
version and Python-library version are separate concepts. Collection remains
in the SDK's existing environment:

| Component | Python | LeRobot dependency |
| --- | --- | --- |
| EmbodiChain collection | `>=3.10` | `>=0.4.4,<0.5` |
| Standalone exporter | `>=3.12` | `lerobot[dataset]==0.6.1` |

The exporter lives under `tools/lerobot_export/` and does not import or install
EmbodiChain. Its separate environment avoids changing the simulator's Python
and LeRobot dependencies. First follow {doc}`offline_expert_datasets` for
validation, interrupted commits, and quality partitions.

## Install and convert in a separate environment

From the repository root, using `uv` (CI uses version `0.12.5`):

```bash
uv venv --python 3.12 .venv-lerobot-export
uv pip install --python .venv-lerobot-export/bin/python --torch-backend cpu \
    ./tools/lerobot_export \
    'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' 'torchcodec==0.11.1+cpu'

.venv-lerobot-export/bin/embodichain-lerobot-export \
    /absolute/path/to/finalized-source \
    /absolute/path/to/new-dataset
```

This is the CPU dependency combination exercised by the export CI. No simulator
or GPU is required. The package also exposes
`python -m embodichain_lerobot_export SOURCE DESTINATION` in that environment.
There are two required positional arguments; conversion behavior is derived
from the source metadata.

Use an exact finalized source root containing `meta/info.json`. The destination
must be new and outside the source. The exporter stages a sibling directory,
checks the official consumer, and publishes the destination only on success.
Failed conversion removes its staging output. Source snapshots detect changes
during conversion, so keep all source writers stopped.

The command prints a JSON conversion manifest and exits `0` on success, `1` on
conversion failure, or `2` for invalid command-line arguments. The initial
destination needs space for a complete copy of the source. Frame data is
rewritten one shard at a time; annotations and episode metadata remain in memory.

## Preserve both language levels

| Input | Exported meaning |
| --- | --- |
| Overall `task_index` / task table | Retained as the episode goal |
| Overall sidecar `instruction` | Restores the goal for legacy recordings that used changing segment text as `task` |
| Segment sidecar instruction | Preferred fine-grained description |
| Old subtask table | Fallback when a sidecar segment has no description |
| No finer description | Fall back to the overall task |

Segment transitions become the official `language_persistent` Arrow feature,
with timestamped assistant rows of style `subtask`. The episode's timeline is
broadcast across its frames; the recipe selects the active description at the
sample timestamp. Existing `subtask_index` and `meta/subtasks.parquet` are
retained for older consumers, but the 0.6.1 reader does not automatically turn
that old table into a `sample["subtask"]` string.

The export writes `meta/task_subtask_recipe.json`, an official `TrainingRecipe`
example. It renders the overall task as a user message and the current segment
as an assistant target. Training code must explicitly load/apply the recipe;
conversion preserves action features and does not automatically change a
policy's objective or training pipeline.

### Read and render the current subtask

Run this example in the independent Python 3.12 environment:

```python
import json
from pathlib import Path

from lerobot.configs.recipe import TrainingRecipe
from lerobot.datasets.language_render import render_sample
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset_root = Path("/absolute/path/to/new-dataset")
dataset = LeRobotDataset(
    "local/export",
    root=dataset_root,
    video_backend="pyav",
    depth_output_unit="m",
)
recipe = TrainingRecipe.from_dict(
    json.loads((dataset_root / "meta/task_subtask_recipe.json").read_text())
)
sample_index = 0
sample = dataset[sample_index]
rendered = render_sample(
    recipe=recipe,
    persistent=sample["language_persistent"],
    events=None,
    t=float(sample["timestamp"].item()),
    sample_idx=sample_index,
    task=sample["task"],
)
assert rendered is not None
print(sample["task"])
print(rendered["messages"])
```

Use the stored sample timestamp when rendering. Reconstructing a different
timestamp from an index can choose the previous subtask at a float32 transition
in a long episode. Cropped fragment frame/language time starts at zero. Source
cadence validation accounts for declared floating-point precision, while source
crop origins and frame ranges remain in the sidecar. RGB video offsets are
adjusted to preserve the same source frames and encoding.

## Read native depth with explicit units

Existing `depth_meta.json` streams become official
`observation.depth.<sensor>` video features. Their metadata includes
`is_depth_map`, `depth_unit`, explicit 12-bit quantizer parameters, and valid
episode/shard time ranges. Encoded HEVC bytes are copied without re-encoding.
Every depth frame is decoded for shape, code range, FPS, timestamp, and frame
count checks. Statistics are recomputed with the official decoder in metres.

Continuing from the reader above, for a sensor named `camera`:

```python
depth_metres = sample["observation.depth.camera"]
print(depth_metres.shape, depth_metres.min(), depth_metres.max())
```

Use the actual feature name from the dataset. The official SDK defaults to
millimetres when `depth_output_unit` is omitted and scales depth statistics
accordingly. The explicit `"m"` option above keeps the example in metres.

Only the existing HEVC `gray12le`, 12-bit contract is supported, including its
uniform/log quantizer settings. Every episode must have a valid depth reference
for a declared depth feature. Values clipped during original quantization remain
clipped; raw floating-point depth and invalid-value masks that were not saved
cannot be reconstructed.

## Inspect the output and preserve source evidence

| Output | Purpose |
| --- | --- |
| Frame Parquet with `language_persistent` | Official timestamped subtask annotations alongside preserved actions |
| `meta/tasks.parquet`, `meta/subtasks.parquet` | Overall and legacy subtask vocabularies |
| `meta/task_subtask_recipe.json` | Ready-to-load language recipe example |
| `meta/embodichain_episodes.jsonl` | Identities, source ranges and episode/segment provenance |
| `meta/embodichain_export.json` | Conversion version, source metadata/data fingerprint, depth features and `official_samples_checked` consumer-check count |
| `meta/embodichain_source_depth.json` | Original depth metadata when depth was migrated |
| `meta/embodichain_source_commits/<source-fingerprint>/` | Archived source journals when present |

The official reader and recipe are checked at the first, last, and
segment-transition frames of every episode before publication. Existing actions,
action contracts, and RGB encoding are preserved. Legacy episodes without UUIDs
receive reproducible identities, but missing source lineage remains explicitly
unknown and shares a source fingerprint for conservative split grouping.

Existing source journals must be complete and match their original sidecars.
They are archived because their evidence describes the source storage rather
than the converted layout. External replay paths remain external: retain those
artifacts when moving datasets. Validate a moved or converted root and generate
its split manifest there instead of silently reusing an old `dataset_root`.

## Handle unsupported or incomplete inputs

| Failure | Required action |
| --- | --- |
| Destination already exists or lies inside the source | Choose a new external directory |
| Source writer changed files during conversion | Stop writers, finalize, and retry from a stable source |
| Unfinished/conflicting source journal | Diagnose/recover the source before export |
| Pre-v3 dataset storage | Perform a separate official storage migration first |
| Existing `language_persistent` feature | Use an appropriate annotation-preserving workflow; this exporter rejects replacement |
| Changing task labels without an overall sidecar instruction | Provide verified task semantics; the exporter does not guess the overall goal |
| Missing depth episode or unsupported codec/quantizer | Supply a complete supported recording |
| Official consumer check fails | Investigate the error; the staged destination is not published |

This tool does not export Mimic/robomimic data, recover lost raw depth, or restore
physical scene state. The producer and consumer APIs remain separate Python
environments even though both use LeRobot v3.0 storage.
