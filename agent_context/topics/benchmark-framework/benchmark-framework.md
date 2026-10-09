# Benchmark framework

Shared execution/artifact contracts live in `scripts/benchmark/core/`; shared
aggregation and comparison helpers live in `scripts/benchmark/reporting/`.
`rendering-formal` is the document-aligned rendering consumer: artifact schema
version 1, experiment definition version `2.0`. Its 82 declared cases retain
unselected, unsupported, failed and interrupted work. `camera-pilot` and
`rendering-suite` are compatibility routes with their original R-series meanings.
Commands, matrices and runtime requirements are in the
[rendering operator guide](../../../scripts/benchmark/rendering/README.md).

## Entry points and ownership

| Request | Start here |
|---|---|
| Public dispatch | `embodichain/cli/main.py` → `scripts/benchmark/__main__.py` |
| Shared definitions, run identities and lifecycle states | `scripts/benchmark/core/contracts.py`, `records.py` |
| Budgets, repeat ordering, subprocesses and interruption | `scripts/benchmark/core/planning.py`, `execution.py` |
| Timing and standard artifacts/provenance | `scripts/benchmark/core/measurement.py`, `artifacts.py`, `provenance.py` |
| Aggregate populations and comparison gates | `scripts/benchmark/reporting/aggregation.py`, `comparison.py` |
| Formal cases, profiles, approval and view identities | `scripts/benchmark/rendering/formal.py` |
| Formal selection, runtime environment and recovery | `scripts/benchmark/rendering/formal_runner.py` |
| Worker capture, freshness, samples and cleanup | `scripts/benchmark/rendering/formal_worker.py`, `capture.py` |
| Approved scene manifests, assets and twenty views | `scripts/benchmark/rendering/scenes.py`, `scene_cases.py`, `preview.py` |
| Frozen CPU replay and external deformation validation | `scripts/benchmark/rendering/replay.py` |
| Native APIs and resolved renderer evidence | `scripts/benchmark/rendering/backends/scene_embodichain.py`, `scene_isaaclab.py` |
| Process/device memory and delivery/overhead diagnostics | `scripts/benchmark/rendering/resources.py`, `diagnostics.py` |
| Frozen policy, reference renderer and quality metrics | `scripts/benchmark/rendering/quality_protocol.py`, `reference.py`, `quality.py` |
| Scene-defined shadow/reflection/edge diagnostic crops | `scripts/benchmark/rendering/quality_regions.py` |
| Measured photometric calibration and held-out verification | `scripts/benchmark/rendering/calibration.py` |
| Verified calibration import and per-frame target binding | `scripts/benchmark/rendering/calibration_binding.py` |
| Append-only offline evaluation events | `scripts/benchmark/rendering/evaluate.py` |
| Formal offline Markdown, CSV, SVG and PDF | `scripts/benchmark/rendering/publication_report.py`, `_publication_figures.py` |
| Within-backend R-05/C3 versus R-04/C1 increments | `scripts/benchmark/rendering/camera_scaling_report.py` |
| Optional ManiSkill raster reference points | `scripts/benchmark/rendering/maniskill_reference.py`, `maniskill_reference_worker.py`, `backends/scene_maniskill.py` |
| Canonical GPU RGB delivery transfer | `scripts/benchmark/rendering/color.py` |
| Legacy camera / R-series pilot | `scripts/benchmark/rendering/run_benchmark.py`, `suite_runner.py`, `suite_report.py` |
| G-03 attempts, receipts and production-session adapter | `scripts/benchmark/expert_generation/` |

## Shared boundaries

Core/shared reporting import only the standard library. Formal list, dry-run and
report-only paths remain independent of simulator/array imports; dry-run does
not prepare assets or create directories. Simulator adapters load only in their
workers. Keep Isaac-specific comparison code under `scripts/benchmark/` and use
an external frozen Isaac checkout; do not alter production simulator APIs or
the external checkout to implement the comparison.

`ExperimentDefinition` freezes matrix, budget, quality protocol and invariants.
`RunSpec` contains a resolved command, not simulator objects. Core reserves one
run and attempt together, writes not-run records before execution, and reaps
isolated process groups on timeout/interruption. Invalid worker JSON and logs
are retained. `ArtifactStore` accepts identical replayed records but rejects
conflicting identities; existing experiment directories are not overwritten.

Formal selection freezes the full case universe separately from selected
runnable attempts. Exclusions and CPU replay-preparation failures remain
reportable not-run rows; renderer construction errors remain execution failures.
Do not infer full-matrix completion from a filtered or smoke command. Approval,
canonical assets and needed replays are copied into experiment roots; case hashes
exclude host paths. Source pinning and lexical virtualenv paths belong to the
runner, not the simulator adapter.

Domain callbacks own GPU completion and validation. Latency excludes validation;
whole-window throughput includes validation and record bookkeeping. Warmup,
initialization and persistence are outside the measured denominator. A callback
return is not proof of GPU completion. Missing measurements remain null with
reasons; failures stay in trial counts. Hardware/config groups and paired repeat
identities cannot be merged into a cross-workload leaderboard.

## Rendering decisions and limits

Layout approval binds reviewed scene/camera hashes; preview completion and
nonempty images do not certify quality. Static scenes and 30 Hz kinematic replay
do not step physics or establish physical grasp success. Formal R-05 has three
cameras; the legacy pilot's R-05 measures modalities. External deformation cells
remain unsupported without the actual approved topology and frozen input.

`capture.py` owns device-ready, blocking host-ready, duplicate real D2H and
additional CPU-copy boundaries. Exposures count environments × cameras, without
a modality multiplier. Camera-frame and output-pixel rates use the whole window.
Native camera-group chunk assembly remains inside capture timing; its batching
semantics are recorded rather than hidden by reducing N.
Initial or changed native state is published explicitly; unchanged static captures
submit fresh camera-group renders without repeating `World.update(0)`.
Publication-policy profile hashes distinguish earlier measurements that included
that synchronization on every capture. GPU colour-transfer work remains timed.
Application observation-ready latency uses one monotonic clock; logical replay
time and unavailable native camera-source age remain separate. Adapter assembly
counters cover executed tensor organization, not unknown renderer-internal
copies. Evidence collection is outside callback latency, inside whole-window rate.

`rendering-preview --render-profile` copies and hashes an explicit profile without
starting a timing matrix. Isaac Clamp delivery verifies disabled colour transforms
and converts linear bytes to sRGB on device. Formal light contracts explicitly
author UsdLux version/normalization and disable extra preset ambient illumination;
image similarity and measured calibration remain independent requirements.

`maniskill-reference` runs only the declared N=1/16 raster points in a separate
experiment, without equal-quality ray-tracing ratios. Its v3 profile
requires ManiSkill commit `62ff3a58` and SAPIEN 3.0.3, recording VCS
metadata/source hashes rather than treating a version-only wheel as that commit.
Its selected-device setup
initializes PhysX GPU pose storage with one source-documented warm-start step;
capture never steps physics. Missing runtime libraries fail before measurement,
and workers do not download them. Preserve native shader gamma evidence instead
of labeling it exact IEC sRGB without verification.

Profiles freeze explicit per-renderer settings. High-quality Isaac capture uses
PT16 with accumulation reset per exposure. Realtime DLAA may enlarge internal
work at low output resolutions. Read `renderer.resolution_evidence` and native
settings; output/atlas shape and lack of a warning do not establish internal
native resolution. CPU capture latency and CUPTI traces do not provide a complete
native graphics kernel timer. Native-render seconds remain null when unverified.

`ResourceSampler` uses one selected physical NVML device and deduplicates graphics
and compute entries for the PID. Process memory, whole-device usage, CPU RSS and
Torch allocator peaks have different scopes. Missing NVML support must not become
zero or a fabricated UUID. Diagnostic R-09 cases save traces and matched minimal,
shared-capture and common-loop windows after primary measurements. The 3% overhead
target is a target, not an automatic pass from a few in-process windows.

Reference rendering and quality evaluation run separately on CPU. Known immutable
policies bind requested modalities, thresholds, frame/sequence coverage and
checksummed evidence. The original full policy and the explicit 30 Hz temporal
policy remain distinct. Unrequested metrics are explicit; missing requested
metrics, calibration, reference noise evidence or temporal coverage cannot pass.
Moving performance workers may freeze a full 100-frame quality replay after the
timing/resource window; old final-frame samples remain insufficient. RGB temporal
quality uses reference geometry for visibility without claiming candidate geometry
quality. Semantic crops are geometry/material-selected diagnostics, and temporal
curves retain missing pairs. Neither changes metric thresholds or qualification.
Calibration receipts bind real training and disjoint held-out artifacts; a manual
qualified status plus an arbitrary file is insufficient. Normal-specific quality
remains unassessed. A calibration import copies the verified source closure and
binds matching actual controls/reference definitions to target frames; it never
certifies target image or sequence quality by itself. Evaluation preserves original
execution snapshots, timing fields and `raw.jsonl`, appending quality events and
updating only derived quality projections. The report verifies proof bindings and
policy requirements before any equal-quality ratio.

Declared large-N cells are capacity probes, not support promises. Generic failures
and interruptions do not prove a maximum supported N; retain explicit renderer-limit
or OOM evidence and the surrounding tested successes separately. RTX PRO 5000 smoke work
does not satisfy the external RTX 5090 acceptance requirement. No publication
performance conclusion follows from implementation coverage or short smoke runs.

G-03 acceptance still requires completed execution, measured validation, task
success and a confirmed persistence receipt. `session_adapter.py` translates the
production session lifecycle; benchmark code does not create another generation
coordinator, physics controller or dataset sink.

## Validation

Use the matching CPU contracts under `tests/benchmark/core/`,
`tests/benchmark/reporting/` and `tests/benchmark/rendering/`. The rendering guide
separates simulator-free contract tests, optional learned-metric/reference tests
and serial GPU smoke commands. Keep `tests/test_main.py` for public dispatch
regressions and Python `-S` checks for offline routes. Tutorial scripts are
examples rather than production test subjects.
