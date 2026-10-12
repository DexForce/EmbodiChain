# Offline expert dataset tools

[Topic overview](data-pipeline.md). This page owns the file-only inspection,
identity, and cross-version conversion boundaries.

## Runtime ownership

The main SDK retains Python >=3.10 and LeRobot 0.4.x. Its `data_pipeline` and
`datasets` packages load online/simulation components lazily; offline inspection
and journal recovery do not import the simulator or PyTorch.

`tools/lerobot_export/` is a separately installable Python >=3.12 tool pinned to
LeRobot 0.6.1. It reads files and must not import/install the main SDK. The
converter and its official reader/recipe tests use an isolated environment;
changing main-package dependency markers is not a compatibility strategy.

LeRobot 0.6.1's reader does not automatically resolve the old subtask table.
Conversion preserves the overall task and exports segment descriptions through
its official persistent-language timeline. The source recording is unchanged,
and a staged destination is published only after validation succeeds. Native
depth features must preserve quantization, units, video offsets, and statistics,
not merely rename a sidecar file. The tool README and tests own exact options.

## Identity and replay

Run/episode/source UUIDs and configuration/program fingerprints are created by
the recorder/host. A natural fragment retains its full source episode identity,
receives a deterministic derived UUID, and preserves source frame offsets.
Description-table indices are labels, not episode or segment identities.

Optional `.pt` replay artifacts reuse host-owned state capture. Their explicit
external paths and source UUIDs link training samples to source trajectories;
retain those files when moving a dataset. Fragment offsets identify the state
inside the full source trajectory, not a new state capture at fragment export.
No offline checker or recovery tool deserializes replay pickle data.

## Validation, quality, and splits

`datasets/inspection.py` owns `validate_dataset` and `create_split_manifest`;
`embodichain dataset` is the public CLI adapter. Validation reads Parquet in
batches, checks schema/indices/timing/segment/action invariants, and optionally
checks media files and decoded frame counts. Diagnostics carry stable codes and
locations. Missing legacy metadata is distinct from contradictory evidence.

Grouping happens before quality filtering. Shared source/parent identities
form transitive groups, and optional scene grouping can only join them.
Unknown legacy origins are conservatively co-grouped unless the caller
explicitly assumes independence. Quality filters exclude unknown recovery or
physical-validation evidence instead of treating it as a successful zero.
Split manifests reference source episodes without rewriting data or metadata.

## Interrupted recording

`recording/journal.py` owns durable commit evidence. Journal writes use atomic
replacement and fsync; LeRobot, videos, and JSONL are still separate resources.
Stop the writer before diagnosis or repair. `recover_recording` is read-only
unless `repair=True`; it can restore missing sidecars from verified committed
artifacts and never replays frames or guesses an unfinished SDK write succeeded.
Inspection uses PyAV to decode referenced RGB/depth videos and verify cadence,
frame counts, and episode coverage before treating media as committed evidence.
Each file is decoded once per inspection; only probe results are cached.
Missing PyAV, unreadable/truncated media, malformed sidecars, inconsistent
evidence, and unknown outcomes remain unresolved. This is metadata recovery,
not automatic collection resume or a cross-process multi-file transaction.

## Focused validation

- `tests/data_pipeline/test_dataset_inspection.py`: real Parquet/JSONL fixtures,
  media checks, quality filters, and grouped splits.
- `tests/data_pipeline/test_offline_imports.py`: no-site-package discovery/help
  and command exit statuses.
- Recording journal/provenance tests under `tests/data_pipeline/`, plus existing
  sync/async recorder and Gym trajectory tests.
- `tools/lerobot_export/tests/`: isolated official 0.6.1 reader, recipe, and depth
  round trips; `.github/workflows/dataset-tooling.yml` separates Python runtimes.
