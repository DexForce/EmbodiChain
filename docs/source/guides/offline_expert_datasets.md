(offline-expert-datasets)=

# Validate, split, and recover expert datasets

Use the offline dataset tools after collecting expert demonstrations to check
their consistency, create quality-filtered training partitions, and diagnose
interrupted commits. They read recorded files without starting a simulator or
loading PyTorch. Numeric scans use NumPy/PyArrow; video decoding uses PyAV.
These lightweight imports do not change the dependencies of a normal SDK install.

Task language is authored before collection. See {doc}`expert_task_language`
for overall and segment instructions, and {doc}`/tutorial/data_expansion` for
collection. Export to the independently installed LeRobot 0.6.1 runtime is
covered in {doc}`lerobot_export`.

## Start with a finalized dataset

Finish collection and finalize the recorder before validation or splitting.
For an interrupted run, stop every writer before inspection or repair. A
recorder's `finalize()` drains committed work; it does not save a live episode.

Use the **exact dataset directory** printed by the runner, containing
`meta/info.json`, rather than its parent of auto-numbered datasets. The
`dataset` subcommands do not have the preview command's `--latest` option.

```bash
EXPERT_DATASET=/absolute/path/to/finalized-dataset

embodichain dataset validate "$EXPERT_DATASET" \
    --output /tmp/expert-validation.json

embodichain dataset split "$EXPERT_DATASET" /tmp/expert-split.json \
    --fractions 0.8 0.1 0.1 --seed 42 --success-only
```

If validation reports an incomplete commit, follow the recovery procedure
below and validate again before splitting. A split manifest references the
original episodes; it does not create new dataset directories. Reports and
manifests must not overwrite dataset-owned files under `meta/`, `data/`,
`videos/`, `images/`, or `depth_videos/`.

`python -m embodichain dataset ...` is equivalent to the console command. Use
`embodichain dataset validate --help`, `split --help`, or `recover --help` for
the argument list.

## Validate structure and media

The basic scan checks actual frame and episode Parquet against the declared
features and EmbodiChain sidecars:

| Check | Examples of detected problems |
| --- | --- |
| Episode/frame indexing | Missing episodes, wrong lengths, interleaved global ranges, discontinuous frame indices |
| Timing | Non-finite timestamps or timestamps inconsistent with the frame/FPS grid |
| Numeric features | Missing fields, invalid shapes, non-finite actions, normalized gripper values outside `[-1, 1]` |
| Language references | Unknown `task_index` or `subtask_index`, inconsistent segment labels |
| Segment annotations | Overlapping/out-of-range spans or inconsistent start/end markers |
| Recording evidence | Pending journals, contradictory sidecars, invalid replay links or depth decoder metadata |

Validation establishes file and annotation consistency. Physical success still
depends on measured evidence recorded by the task; a successful scan does not
measure whether a grasp, placement, or liquid transfer succeeded.

```bash
embodichain dataset validate "$EXPERT_DATASET" \
    --check-media --media-frame-counts \
    --batch-size 4096 --output /tmp/expert-media-validation.json
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--check-media` | Off | Check referenced RGB/image/depth files |
| `--media-frame-counts` | Off | Decode referenced videos to check frame counts and cadence; implies media checks and requires PyAV |
| `--batch-size` | `4096` | Positive maximum number of numeric frame rows read per Arrow batch |
| `--output` | None | Write a separate JSON report; the report is also printed |

Decoding can be substantially more expensive than inspecting metadata. Shared
video shards are probed once per inspection, without retaining all decoded
frames. The recovery tool always requires usable media evidence for repair;
its media checks are not disabled by these validation flags.

### Read the diagnostic report

Reports contain `ok`, `root`, `episodes`, `frames`, `errors`, `warnings`, and
`issues`. Each detailed issue has `severity`, `code`, `location`, and `message`.
For example, `language_reference` identifies a vocabulary-reference problem,
while `pending_commit` indicates that a recording needs diagnosis before use.

```python
import json
from pathlib import Path

report = json.loads(Path("/tmp/expert-validation.json").read_text())
print(report["ok"], report["errors"], report["warnings"])
for issue in report["issues"]:
    print(issue["severity"], issue["code"], issue["location"], issue["message"])
```

Issue details are capped at 1,000 entries; aggregate counts can be larger.
Missing optional lineage in an old recording can produce warnings, while
contradictory references or incomplete commits produce errors. `ok` means no
errors, so review warnings before combining legacy data with new recordings.

| Command exit status | Meaning |
| --- | --- |
| `0` | Validation found no errors, splitting succeeded, or recovery inspection is complete |
| `1` | Validation errors or unresolved/incomplete recording evidence |
| `2` | Invalid arguments or a request that cannot be performed |

## Filter quality and keep related episodes together

```bash
embodichain dataset split "$EXPERT_DATASET" /tmp/expert-quality-split.json \
    --fractions 0.8 0.1 0.1 --seed 42 \
    --success-only --max-recoveries 1 --require-physical-valid
```

Only use the extra quality filters when the recording contains their evidence.
The command fails if no episodes satisfy the requested filters.

| Argument | Meaning |
| --- | --- |
| `--fractions TRAIN VALIDATION TEST` | Three finite nonnegative episode fractions summing to one; default `0.8 0.1 0.1` |
| `--seed` | Deterministic group assignment; default `0` |
| `--success-only` / `--failure-only` | Keep explicitly successful or failed episodes; mutually exclusive |
| `--max-recoveries N` | Nonnegative limit on execution recovery/replanning counts; unknown counts are excluded |
| `--require-physical-valid` | Require explicit accepted physical evidence; unknown evidence is excluded |
| `--group-by lineage` | Default: preserve source/parent/expansion relationships |
| `--group-by scene --scene-field scene_id` | Also join episodes with the same explicit scene identity; every episode needs that field |
| `--allow-legacy-independent` | Explicitly assume unidentified legacy episodes are independent; record the assumption in the manifest |

Recovery counts refer to execution retries/replans and workflow recovery
cycles, not the number of times an operator repaired a recording. Canonical
Task Program traces support row-local counts without double-counting retries.
Physical filtering recognizes explicit `physical_valid` or `physical_validation`
results, and successful `ordered_stable_regions` objectives with valid measured
evidence. A declared objective or projected effect alone is insufficient.

Grouping happens **before filtering**. If B derives from A and C derives from B,
all three remain in one partition even when B is excluded. Shared
`source_episode_uuid`, multiple sources, parent identities, and recorded
expansion references establish these transitive groups. Scene grouping joins
additional groups; it never separates derivatives of the same source.

Unknown legacy origins are grouped conservatively using their source
fingerprint, run identity, or the whole dataset. This can leave validation or
test empty. Fractions are targets, because source groups are indivisible and
episode lengths differ. Supply `--allow-legacy-independent` only when you have
established that their derivations cannot leak between partitions.

### Use a split manifest in a reader

The JSON records `splits`, episode/frame counts, group assignments, excluded
episodes with reasons, quality settings, and provenance assumptions. The
training application must explicitly apply the episode selection. In the
recording environment with LeRobot 0.4.x:

```python
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

manifest = json.loads(Path("/tmp/expert-split.json").read_text())
dataset_root = Path(manifest["dataset_root"])
train_episodes = manifest["splits"]["train"]
if not train_episodes:
    raise ValueError("The grouped split has no training episodes")

train_dataset = LeRobotDataset(
    repo_id=dataset_root.name,
    root=dataset_root,
    episodes=train_episodes,
)
print(train_dataset[0]["task"], train_dataset[0]["subtask"])
```

The split command checks episode metadata and pending commits; it does not
replace a full numeric/media validation scan. If you export or relocate the
dataset, validate that destination and generate a manifest for its root.

## Trace sources, fragments, and replay artifacts

New recordings add stable identities and configuration/runtime fingerprints
to `meta/embodichain_episodes.jsonl`:

| Field | Meaning |
| --- | --- |
| `run_uuid` | Collection run identity |
| `episode_uuid` | Identity of this full episode or fragment |
| `source_episode_uuid` / `source_episode_uuids` | Original source or sources |
| `parent_episode_uuid` | Parent of a derived record, when present |
| `config_hash`, `program_hash` | Stable configuration/program fingerprints; program information is conditional on a Task Program |
| `provenance` | Available SDK/Python/LeRobot/PyTorch versions, physics backend and control timing |
| `expansion` | Candidate records and selected source/case identities, when expansion is used |
| `replay_artifact`, `trajectory_path` | Explicit link to optional replay state and its source alignment |

A fragment retains the original source identity and gets a deterministic
derived UUID for its span, attempt, and continuity region. Language vocabulary
indices identify descriptions, not episodes or source trajectories. Fingerprints
support traceability but do not replace the full initial scene state.

Enable the existing replay recorder when that state is needed:

```bash
embodichain run-env \
    --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml \
    --headless --max_episodes 1 \
    --record_trajectory --trajectory_save_dir outputs/replay
```

Replay files are atomically saved before the dataset recorder snapshots their
paths, including async snapshots. A fragment points to the full source
trajectory and its source frame range. Preserve external `.pt` files when
moving a dataset; offline inspection/recovery checks their links without
loading pickle state. A seed or fingerprint alone does not reconstruct a
scene, and replay does not guarantee identical physics across versions/backends.

## Diagnose and repair interrupted commits

After stopping all writers:

```bash
embodichain dataset recover "$EXPERT_DATASET"
```

Each episode has atomic, fsynced evidence under `meta/embodichain_commits/`.
Its prepared snapshot includes the frozen task/segment language and provenance
before frames are submitted.

| Journal phase | Last confirmed progress |
| --- | --- |
| `prepared` | Metadata captured; SDK commit has not begun |
| `lerobot_committing` | SDK save was entered; an interruption requires disk inspection |
| `lerobot_committed` | SDK save returned |
| `depth_committed` | Applicable depth resources finalized |
| `complete` | Episode sidecar appended and the journal advanced |

These phases are not a transaction across Parquet, videos, and JSONL. SDK
buffering can leave even a `complete` journal without durable frame/episode
files. Inspection verifies the actual global/local frame identities, SDK ranges,
sidecar ownership, replay references, and decodable RGB/depth coverage.

| Reported commit status | Operator action |
| --- | --- |
| `complete` | Verified evidence and matching sidecar are present |
| `repairable` | Committed artifacts verify; the missing sidecar can be restored |
| `retryable` | Preparation has no committed frames; the tool does not execute the retry |
| `unresolved` | Missing, contradictory, unreadable, or insufficient evidence needs investigation |

Request repair explicitly:

```bash
embodichain dataset recover "$EXPERT_DATASET" --repair
embodichain dataset validate "$EXPERT_DATASET" --media-frame-counts
```

Repair can append a missing episode sidecar from verified journal evidence and
advance its journal. It does not reconstruct Parquet/videos, rewrite SDK
metadata, truncate malformed JSONL, load replay state, or resume collection.
Duplicate/conflicting sidecar indices and unknown SDK outcomes remain
unresolved when disk evidence is insufficient. Preserve the report and original
files for investigation. A live recorder with an uncertain SDK write blocks
later submissions rather than retrying an ambiguous append.

## Configure recording buffers

This fragment is merged into an existing handwritten task's Gym config:

```yaml
env:
  task_instruction: Pick and place the object.
  dataset:
    lerobot:
      func: AsyncLeRobotRecorder
      mode: save
      save_failed_episodes: false
      params:
        save_path: outputs/lerobot/expert_demos
        use_videos: true
        metadata_buffer_size: 1
        async_queue_maxsize: 4
```

For a Task Program, keep overall language at its program root instead.
`metadata_buffer_size` is a positive integer, defaults to `1`, and controls
LeRobot episode-metadata batching for both recorders. Larger values can delay
visibility of recent metadata. The async-only `async_queue_maxsize` limits
pending payload count, not bytes; its default `0` is unbounded. Payload sizes
depend on camera count, resolution, and episode length. See
{doc}`/overview/gym/dataset_functors` for image-writer and depth settings.

Segment lookup builds a validated frame map once per episode instead of
scanning every segment for each frame. The CPU microbenchmark checks exact
annotation equivalence and reports lookup time and sampled process RSS:

```bash
python -m scripts.benchmark.data_pipeline.benchmark_segment_annotations \
    --frames 10000 --segments 100 --repeats 5 \
    --report /tmp/segment-annotations.md
```

It does not measure full simulation or disk throughput.

## Python API and common failures

The file-only API mirrors the CLI:

```python
from embodichain.data_pipeline.datasets.inspection import (
    create_split_manifest,
    validate_dataset,
)
from embodichain.data_pipeline.recording import inspect_recording

report = validate_dataset(dataset_root, media_frame_counts=True)
report.raise_for_errors()
split = create_split_manifest(dataset_root, "/tmp/python-split.json", seed=42)
commit_report = inspect_recording(dataset_root)
print(split["episode_counts"], commit_report["ok"])
```

Here `dataset_root` is the finalized root used above. Full signatures and report
types are in {doc}`/api_reference/embodichain/embodichain.data_pipeline.datasets`
and {doc}`/api_reference/embodichain/embodichain.data_pipeline.recording`.

| Symptom | Check |
| --- | --- |
| `meta/info.json` not found | Use the exact auto-numbered dataset root |
| `language_reference` or segment errors | Inspect task/subtask tables, sidecar spans and frame annotations together |
| `pending_commit` | Stop writers, inspect/recover, then revalidate |
| Quality selection returns no episodes | Check whether success/recovery/physical evidence is actually present |
| A partition is empty | Inspect indivisible source groups and legacy-origin assumptions |
| Media blocks repair | Install PyAV and verify the complete referenced streams; nonempty files are insufficient |
| External replay path is missing | Retain or correctly relocate the linked source trajectory |

For official persistent-language and native-depth conversion, continue with
{doc}`lerobot_export`. Automatic physical-state restore, suffix continuation,
and Mimic/robomimic export are outside these tools.
