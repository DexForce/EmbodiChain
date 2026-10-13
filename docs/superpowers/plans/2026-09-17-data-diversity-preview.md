# Data diversity P0/P1 preview implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Work only in the existing isolated worktree; do not push or merge.

**Goal:** Deliver a working Gradio/Viser data-analysis preview populated by an existing simulation task, with traceable collection, honest failures and testable statistics.

**Architecture:** A CPU-only `embodichain.data_analysis` package owns contracts, SQLite catalog and Polars queries. A simulation-only collection adapter records existing task execution into portable JSON/NPZ artifacts; the UI and offline viewer consume these without running simulation.

**Tech Stack:** Existing Gradio, Viser 1.0.21, Polars 1.31.0, NumPy, Matplotlib, SQLite, LeRobot.

**Spec:** `docs/superpowers/specs/2026-09-17-data-diversity-p0-p1-acceptance.md`

## Global constraints

- Preserve all source data, main checkout, task ownership and file-owned physics backend.
- No React, Vite, ECharts, new Three.js code, DuckDB or independent FastAPI application.
- Imported historical metadata may be unknown; do not infer successful physical execution from projected task effects.
- Use Apache headers, future annotations, annotated public APIs and `__all__`.
- Python interpreter: `/home/dex/miniconda3/envs/open/bin/python`.
- Feature tests must run CPU-only where possible. Live task qualification is separate and never substituted by mock data.

## Shared interfaces

An episode record is a JSON mapping with `schema_version=1`, `episode_id`, `run_id`, `candidate_id`, integer `attempt_id`, `task_id`, `robot_id`, `status`, `reason`, `seed`, `parents` (list of mappings), `dimensions` (mapping from feature name to measurement), `artifacts` (mapping to file paths), `segments` (list), `metrics` and `provenance`.

A measurement is `{value, source, unit, frame, scope}`; unknown measurements use `value: null`, `source: unknown`, `missing_reason`. Feature keys: `asset`, `pose_x`, `pose_y`, `pose_z`, `yaw`, `affordance`, `approach`, `material`, `light`, `trajectory_family`, `duration_s`. Catalog rejects non-finite JSON and unknown schema versions. Sources are `measured`, `configured`, `annotated`, `derived`, `estimated`, `unknown`.

Statuses are `proposed`, `planning_failed`, `rollout_failed`, `rejected`, `pending_write`, `partial_commit`, `committed`. Identity is immutable; equal updates are idempotent; legal lifecycle updates are explicit and transactionally recorded. Artifacts use stable paths and are verified before committed coverage.

Portable replay uses `trajectory.npz`: `timestamps(T)`, `qpos(T,D)`, `tcp_position(T,3)`, `object_position(T,3)`, optional `actions(T-1,D)`, optional `scene_positions(T,N,3)`, `scene_wxyz(T,N,4)`, `scene_visible(T,N)`. A separate `scene.json` serializes existing SceneManifest nodes/geometries/cameras. Optional camera frames/video include timestamps. NPZ loading disables pickle.

## Task 1: Contracts, catalog and analysis

Files: `embodichain/data_analysis/{__init__,schema,catalog,statistics,importers}.py`; `tests/data_analysis/test_{catalog,statistics,importers}.py`.

Produces `Catalog(path)`, `upsert(record)`, `get(episode_id)`, `records(filters=None,status=None)`, `summary()`, `snapshot(name)`, `snapshot_records(snapshot_id)`, `import_jsonl(path)`; pure `distribution(records,dimension)` and `joint_distribution(records,x,y)` returning JSON rows with count. Filters support categorical values and numeric min/max. The UI uses only these interfaces.

- [ ] Write tests for lifecycle counts, idempotence, identity conflict, incomplete commits, unknowns and literal 3/4 coverage. Example expected contract:
  ```python
  catalog.upsert(record(status='partial_commit'))
  assert catalog.summary()['statuses']['partial_commit'] == 1
  catalog.upsert(record(status='committed'))
  catalog.upsert(record(status='committed'))
  assert catalog.summary()['statuses']['committed'] == 1
  ```
- [ ] Run tests and establish failure before implementation.
- [ ] Implement durable catalog, strict record validation, immutable snapshots and LeRobot metadata adapter; test missing metadata explicitly.
- [ ] Run focused tests; provide evidence and interface notes for integration review.

## Task 2: Existing-task collection and portable recording

Files: `embodichain/data_analysis/{collection,recording,cli}.py`, `embodichain/cli/main.py`; production recorder tests under `tests/data_analysis/`.

Consumes the shared record contract and existing task program executor. Produces JSONL records and portable scene/trajectory artifacts; root owns CLI registration.

- [ ] Write recorder tests for aligned timestamps, file-write failure, unavailable camera and strict artifact reading.
- [ ] Build one existing task through its production config loader, execute a bounded trial, record actual object and joint motion.
- [ ] Implement a reusable wrapper that observes real `env.step()` calls and exports SceneExporter snapshots on the simulation thread.
- [ ] Persist candidate status before execution and finalized records after artifact verification. Store all failed attempts.
- [ ] Run an initial existing-task smoke, then collect varied episodes with a bounded budget; measure object lift and placement rather than trusting projected effects.

## Task 3: Offline Viser replay and import isolation

Files: `embodichain/data_analysis/replay.py`, targeted lazy import changes in `embodichain/lab/__init__.py` if necessary; replay and import tests.

Produces `ReplayViewer(host='127.0.0.1',port=0)`, `load(record,compare_record=None)`, `url`, `close()`. Reuses SceneManifest/SceneFrame and ViserBackend; one owner thread handles all Viser mutations.

- [ ] Write tests for nearest recorded time, episode switching/close, malformed NPZ, and blocking DexSim imports during offline viewer import.
- [ ] Reconstruct protocol values from serialized scene and timed samples; allow trajectory-only replay for historical records.
- [ ] Publish camera/state/phase from one timeline; no Gradio tick for every frame.
- [ ] Run focused tests and real viewer smoke with collected artifacts.

## Task 4: Gradio workbench and integrations

Files: `embodichain/data_analysis/{ui,cli}.py`, minimal GenSim entry, optional dependencies; `tests/data_analysis/test_ui.py`.

- [ ] Write query/controller tests proving visible table, chart and exported IDs share filters.
- [ ] Build overview, filters, joint coverage, selected record lineage, Viser iframe, snapshot comparison and export controls using the shared service.
- [ ] Add standalone CLI and lazily loaded GenSim panel; scope worker/viewer lifecycle per browser session.
- [ ] Verify app config generation and real browser operations with actual task data.

## Task 5: Qualification and documentation

- [ ] Run affected CPU regression tests, offline isolation check, real-task recording, UI browser checks and metadata performance measurements.
- [ ] Run Black 26.3.1, API documentation coverage and relevant context checks; update public API and context ownership.
- [ ] Review diff for correctness and spec gaps; fix actionable findings.
- [ ] Write an acceptance report distinguishing passed, partial and unexecuted criteria, with real artifact paths and reproducible commands. Keep the local preview available for user inspection.

## Progress

Implementation authorized by the user requesting a first version and an existing task acceptance run. The preview is a vertical implementation slice; report any remaining full P0/P1 criteria explicitly rather than claiming completion from a UI screenshot.

## Execution result

The runnable preview and real-task qualification are delivered. See `../reports/2026-09-17-data-diversity-preview.md` for exact scope, 277-test evidence, real data, browser checks, performance and explicitly incomplete full P0/P1 criteria. The unchecked original checklist above remains the broader work breakdown, not a claim of full completion.
