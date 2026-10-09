# Formal rendering v2 and legacy pilots

Use `python -m scripts.benchmark rendering-formal` for the document-aligned
rendering experiments. The common artifact schema remains **1**; formal
experiment definitions are **2.0**. `camera-pilot` and `rendering-suite` retain
their legacy configurations and R-series meanings.

The implementation declares 82 cells and preserves explicit exclusions,
unsupported inputs, failures, OOM and interruption. It does not promise that
every declared scale fits every renderer or GPU. There are currently no
statistically and quality-qualified publication performance results.

`maniskill-reference` is a separate, explicitly unqualified raster reference:
the approved Franka scene at N=1/16, C=1, 128×128 RGB. For a serial smoke, run
`.venv-maniskill-reference/bin/python -m scripts.benchmark maniskill-reference --scene-root "$SCENE_ROOT" --gpu-id 5 --smoke`;
`--num-envs 1`, `--shader minimal`, and `--delivery host_ready` select a point,
raster variant, or one blocking D2H boundary. The default is the `default` shader
and device-ready delivery, with three independent repeats outside smoke mode.
The isolated Python 3.11 environment pins ManiSkill 3.0.0b22/SAPIEN 3.0.1 and an
explicit, checksum-verified PhysX GPU library (`--physx-gpu-library`); workers
never download stock assets or runtime libraries. PhysX pose-buffer warm-start
is setup-only, and capture does not step physics. Native gamma-2.2 RGB is retained
without claiming exact IEC sRGB or equivalence to the formal ray-tracing profiles.
`--list`, `--dry-run`, and `--report-only EXPERIMENT_ROOT` work offline; the separate
report preserves failures, exclusions, run hashes and independent-repeat
uncertainty. CPU tests validate contracts; actual GPU support and image quality
require their own evidence before any stronger claim.

## Formal matrix

Counts below are workload cells, before backend selection and independent process
repeats. `formal.py` is the authority for case IDs, matrices and profile payloads.

| Track | Cells | Frozen workload |
|---|---:|---|
| R-01 | 8 | Calibration/workshop; 20 fixed views; 512×512; N=1/64; realtime/high_quality; RGB/depth/seg quality captures |
| R-02 | 4 | Franka workstation; 100 frames at 30 Hz; two rigid three-camera profile variants and two external-deformation placeholders |
| R-03 | 48 | Calibration/Franka; static/moving; 256², 512² or 1920×1080; RGB or RGB/depth/seg; both profiles |
| R-04 | 5 | One camera; 128² RGB; N=1/16/64/256/1024; realtime, static, device-ready |
| R-05 | 5 | Three cameras at the same N values and output size |
| R-09 | 8 | N=16, three cameras; four delivery modes × ordinary/diagnostic treatments |
| modality-ablation | 4 | 256²; RGB, RGB+depth, RGB+normals, RGB+depth+seg |

The common scenes are custom EmbodiChain fixtures, not official Isaac Lab tasks.
Both adapters consume the approved manifests, canonical triangle buffers,
materials, camera model and source transforms. Source textures are replaced by
recorded common material overrides. The three-camera layout is external-front,
external-side and wrist. Robot FK/IK and scripted object/hand motion are prepared
on CPU; rendering does not instantiate a physical rollout or step physics.
Kinematic grasp/release replay is not evidence of physical task success.

The two deformation cells require actual frozen external deformation data with
the approved topology. The current formal launcher does not supply such data;
workers retain `failed` / `failure_reason_code=unsupported` before GPU startup.
No synthetic substitute qualifies those cells.

## Inspect, approve and select

Listing and dry-run work without site packages or simulator imports:

```bash
python -S -m scripts.benchmark rendering-formal --list --experiment R04
python -S -m scripts.benchmark rendering-formal --list --json --show-excluded
python -S -m scripts.benchmark rendering-formal --experiment R04 --num-envs 1 --dry-run
```

`--list` displays selected cases and a summary. `--show-excluded` and `--json`
are explicit listing options. Dry-run always emits the full structured selection
and never reads scene assets, prepares replay, or creates output directories.
`--case-id` is repeatable; `--num-envs` accepts one or more values. `--max-cases`
limits the selected prefix explicitly. Every exclusion reason remains in the
frozen plan and report denominators; an excluded cell is never called passed.

Review scenes first, using independent runtime environments:

```bash
EMBODICHAIN_PYTHON=/path/to/embodichain-python
ISAACLAB_ROOT=/path/to/frozen/IsaacLab
ISAACLAB_PYTHON=/path/to/matching/isaaclab-env/bin/python
SCENE_ROOT=outputs/rendering-review/new-review-id

"$EMBODICHAIN_PYTHON" -m scripts.benchmark rendering-preview \
  --output "$SCENE_ROOT" --gpu-id 1 \
  --embodichain-python "$EMBODICHAIN_PYTHON" \
  --isaaclab-root "$ISAACLAB_ROOT" --isaaclab-python "$ISAACLAB_PYTHON"
```

`rendering-preview` saves RGB/depth/instance arrays, contact sheets, geometry
checks and renderer/software evidence. Its `review.json` remains pending user
review; it neither approves the layout nor certifies image quality.
`--prepare-only --scene calibration` needs no simulator; `--reuse-scenes`
verifies source hashes and requires fresh worker output directories.

After an explicit review decision, persist an `approval.json` receipt at the
scene root, using the approved hashes in `review.json`. Its required structure is:

```json
{
  "status": "approved",
  "scope": "scene_layout_and_camera_scheme",
  "scenes": {
    "calibration": "<approved scene SHA-256>",
    "workshop": "<approved scene SHA-256>",
    "franka_workstation": "<approved scene SHA-256>"
  }
}
```

`--approval /path/to/receipt.json` selects a separate receipt. The launcher
verifies all original scene/asset hashes before execution and copies the canonical
inputs and receipt into each experiment root. Layout approval is independent of
photometric calibration, image metrics and comparison eligibility.

## Runtime requirements

Isaac Lab is an **external frozen dependency**. This comparison targets the
`v3.0.0-beta2.patch1` checkout at
`ffff603eafc6b74264a5261cc0183d6a65390d78` with its matching Isaac Sim runtime
(the exercised runtime line is Isaac Sim `6.0.1.0`). Keep its environment separate
from EmbodiChain/DexSim. Record the actual source revision, distribution versions,
CUDA/runtime/driver and dirty-source hashes; a package version alone is insufficient.

All our Isaac-specific comparison adapters and setup logic stay under
`scripts/benchmark/`. Neither production `embodichain` modules nor the external
Isaac checkout should be patched to implement this benchmark.

```bash
git -C "$ISAACLAB_ROOT" rev-parse HEAD
git -C "$ISAACLAB_ROOT" status --short
"$ISAACLAB_PYTHON" -c "import sys; from importlib.metadata import version; print(sys.version); print(version('isaaclab')); print(version('isaacsim'))"
```

Formal workers launch with the explicitly selected Python executable. The runner
pins `PYTHONPATH` to this repository and each frozen Isaac `source/*` directory,
clears inherited CUDA visibility/Conda selection, and sets PCI bus ordering.
`VIRTUAL_ENV` comes from the executable's lexical environment path, preserving
virtualenv symlinks. `--gpu-id` is explicit; inspect the recorded selected GPU UUID
rather than assuming an inherited device selection.

Install optional NVML Python bindings in **both worker environments**, outside
measurement. The distribution is `nvidia-ml-py`; its import name is `pynvml`:

```bash
"$EMBODICHAIN_PYTHON" -m pip install nvidia-ml-py==13.595.45
"$ISAACLAB_PYTHON" -m pip install nvidia-ml-py==13.595.45
```

`resources.py` samples graphics and compute process lists on the selected physical
NVML device, deduplicating the PID. GPU process peak/measurement median, whole-device
peak, CPU RSS/high-water and Torch allocation are separate scopes. Samples can
miss short peaks; whole-device usage includes other processes. Missing bindings,
permissions or counters yield null plus reasons. Hardware comparisons require a
valid matching UUID; memory claims require the relevant measured counter or an
explicit missing result. Earlier smoke rows missing Isaac NVML identity are not
retroactively made eligible by installing the package later.

## Execute a selected matrix

```bash
"$EMBODICHAIN_PYTHON" -m scripts.benchmark rendering-formal \
  --scene-root "$SCENE_ROOT" --experiment R04 --num-envs 1 \
  --smoke --gpu-id 1 \
  --embodichain-python "$EMBODICHAIN_PYTHON" \
  --isaaclab-root "$ISAACLAB_ROOT" --isaaclab-python "$ISAACLAB_PYTHON" \
  --output outputs/benchmarks/rendering-formal
```

Default performance trials use three fresh process repeats, 30 warmups and 300
measured captures. `--publication` raises selected performance repeats to at least
ten; it does not certify the resulting data. R-01/R-02 quality acquisition defaults
to one process per backend. `--smoke` uses one repeat, two warmups and four timed
performance captures. It preserves N, resolution, modalities, profiles and the full
R-01/R-02 view/sequence requirements. `--smoke` and `--publication` are exclusive.
Use `--backend E`, `I`, `embodichain`, `isaaclab` or `both`; the default is both.
Backend order alternates between repeats and execution is serial. The default
900-second per-process timeout includes startup.

Each invocation receives a new timestamped directory, with `index.json`,
`acceptance.json` and the full `selection.json`. Each selected experiment receives
its own standard directory. CPU replay preparation occurs once per needed scene
before GPU workers; preparation failures remain selected `not_run` rows with reasons.
Worker crashes, unsupported operations, timeout and interruption retain evidence.
Budgeted attempts reserve only selected runnable work; full matrix counts remain
separate. A new run never overwrites an existing experiment.

## Timing, profiles and counting populations

A batch begins with frozen state publication when applicable, then render submission,
selected delivery and backend/device completion. The primary loop does not step
physics. Initialization, warmup, sample persistence, reference rendering and quality
evaluation are excluded. Per-capture latency excludes validation; whole-window
throughput includes validation and metadata bookkeeping. Neither timer is a native
GPU rendering-kernel timer.
Native initial/changed state is published before exposure. Unchanged static state
does not repeat `World.update(0)`; every capture still submits all native camera
groups and advances their rendering/reconstruction history. Older profile hashes
include per-capture synchronization and must not be pooled with this boundary.

Let B be completed batch captures, T the complete measured window in seconds, N
environments, C cameras per environment and W×H the output resolution:

- `batch_steps_per_s = B / T`.
- `camera_frames_per_s = B × N × C / T`.
- `gpixel_per_s = B × N × C × W × H / (T × 10^9)`.

Modalities do not multiply exposures. Gpixel/s counts **output pixels**, not an
inferred internal shading workload. P50/P95 describe capture latency; reports
aggregate independent process summaries, not frames as independent repeats.

| Delivery | Completed application boundary |
|---|---|
| `device_ready` | Original CUDA observation buffers, with backend/device synchronization |
| `host_ready` | One blocking D2H per modality, then completion |
| `duplicate_d2h` | Two actual readbacks from the same original device buffers |
| `host_copy` | One actual readback plus an independent CPU copy |

Declared D2H and CPU-copy bytes count different operations. Native/pose transfers
may exist outside these observation counters. Static identical pixels are valid;
freshness probes perturb the view/state outside timed windows.

Profiles are full versioned payloads and hashes in `formal.py`, not merely names.
Realtime uses native hybrid with tiled DLAA and Isaac balanced/DLAA settings.
High quality uses native fast-rt with tiled DLAA at 16 spp and explicit Isaac PathTracing at 16 spp;
Isaac accumulation resets each exposure rather than silently converging across
warmups or frames. Resolved settings and accumulation-reset evidence are saved.
Native DLAA supplies an output-resolution, pixel-centre public geometry pass.
The earlier NRD profile exposed jittered depth/labels; FastRT/NRD additionally
exported range rather than optical-axis depth. Those measurements are retained
as diagnostics, and the adapter rejects FastRT/NRD canonical depth requests.
RGB transfer, lighting and tone-response equivalence still require measured
qualification. Isaac uses explicitly authored UsdLux 25.05 lights, zero extra
scene ambient intensity, Clamp output and verified GPU sRGB conversion. The
earlier ACES profiles remain separate historical measurements. Neither the
current light contract nor the native DLAA choice certifies matching RGB.
Native Hybrid/DLSS packs linear UNORM8 while FastRT emits display-sRGB bytes;
the Hybrid adapter applies the IEC sRGB transfer on device before delivery.
Its time and temporary buffers remain inside the device-ready boundary, and
the original quantization cannot be recovered. Native light controls compensate
the engine's environment percentage and directional-power quarter scaling;
source-derived unit conversion is still separate from measured calibration.
The native adapter uses declared 32-environment camera-group chunks and assembles
the full N×C batch on device inside the timed boundary. It does not reduce N to
avoid backend limits. Native changed-state publication includes the explicit
render-update commit cost, without physics steps or an extra exposure.

Realtime DLAA has been observed enlarging internal input work at low resolutions.
`renderer.resolution_evidence` records requested dimensions, observed output/atlas,
resolved settings and resize warnings. Internal resolution remains unknown unless
actually measured; equal output shape or no warning is not proof of native-size
rendering. `native_render_s` remains null when native timestamp queries and their
window scope are not validated.

## R-09 trace and overhead diagnostics

Only R-09 cells marked `diagnostic` run the additional diagnostics automatically.
Find their IDs with `--list --experiment R09`; for example, select the CPU-copy
cell explicitly:

```bash
BENCH_R09_CASE=$(python -S -c "from scripts.benchmark.rendering.formal import experiment_cases; print(next(c.case_id for c in experiment_cases('R-09') if c.delivery == 'host_copy' and c.diagnostic))")
"$EMBODICHAIN_PYTHON" -m scripts.benchmark rendering-formal \
  --scene-root "$SCENE_ROOT" --experiment R09 --case-id "$BENCH_R09_CASE" \
  --smoke --gpu-id 1 --embodichain-python "$EMBODICHAIN_PYTHON" \
  --isaaclab-root "$ISAACLAB_ROOT" --isaaclab-python "$ISAACLAB_PYTHON" \
  --output outputs/benchmarks/rendering-diagnostics
```

Diagnostic phase timing adds explicit synchronizations and is a separate treatment.
After primary timing/resources and sample persistence, the worker runs a three-capture,
zero-warmup delivery trace and three paired overhead repeats, using at most 100
iterations per overhead window and the configured warmup. The delivery/synchronization/
state-publication/validation policies match the selected cell.

`diagnostics/delivery_trace/` contains `delivery_diagnostics.json`, `cpu_ranges.json`
and an optional `torch_trace.json`. `diagnostics/framework_overhead.json` retains
minimal/direct/common windows; `diagnostics/summary.json` and `result.diagnostics`
link evidence and missing reasons. These rates and windows never replace primary
throughput metrics. Optional profiler failure preserves primary capture evidence.

Trace `record_function` ranges identify render, synchronization, readback, state
publication and extra CPU copy. Observed D2H/H2D/D2D events/bytes are attributed only
through matching launch evidence; unknown and unattributed events stay explicit.
CUDA/CUPTI cannot establish complete Vulkan/RTX/native graphics visibility. CPU
ranges and CUDA API call durations are not GPU kernel durations, and absence of
copy events does not prove zero copies.

`overhead_ratio` compares the common loop with minimal direct adapter calls plus
the same mandatory validation. `common_loop_overhead_ratio` separately compares
with a loop sharing capture, validation and metadata recording. Alternating windows,
counts, specifications and ratio MAD are retained. The **3% target is not an automatic
pass**; a few in-process windows are not independent process publication repeats.

## Independent reference and quality evaluation

Run quality work only after GPU timing has finished. Use a separate compatible
CPU environment with Torch, torchvision and NumPy, then install the optional
[reference requirements](requirements-reference.txt):

```bash
REFERENCE_PYTHON=/path/to/reference-env/bin/python
"$REFERENCE_PYTHON" -m pip install -r scripts/benchmark/rendering/requirements-reference.txt
"$REFERENCE_PYTHON" -c "from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1; squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)"
```

The explicit setup command prefetches the official SqueezeNet 1.1 ImageNet
backbone. Evaluation itself never downloads weights or uses random substitutes.
It verifies `torch.hub.get_dir()/checkpoints/squeezenet1_1-b8a52dc0.pth` and the
LPIPS package's learned `weights/v0.1/squeeze.pth` against the frozen hashes in
`quality_protocol.py`. `TORCH_HOME` can select the cache root. LPIPS requires at
least 64 pixels on each image axis. Missing dependencies, weights or mismatched
hashes remain missing/not-qualified.

Mitsuba CPU references reuse canonical scene meshes/materials/camera state and
native ray-intersection depth/IDs. Two independent seeds record a reference noise
floor; requested spp is not proof that noise qualification passed. The default
256 spp can fail that gate. Increasing reference spp is a new recorded evaluation,
not a reason to edit frozen thresholds or previously measured timings.

```bash
EXPERIMENT_ROOT=/path/to/invocation/R-04/experiment-timestamp
"$REFERENCE_PYTHON" -m scripts.benchmark.rendering.evaluate \
  --experiment-root "$EXPERIMENT_ROOT" --spp 256 --threads 4 --seed 0
```

`--run-id` is repeatable for staged evaluation. Supply
`--calibration-evidence /path/inside/experiment/calibration.json` only when real,
case/backend/reference/profile-bound lighting/colour evidence exists. Scene approval
is not this calibration. No fitted exposure/registration or arbitrary renderer output
is promoted into an independent reference.

`calibration.py:create_calibration_receipt` is the measured calibration owner.
It takes a frozen backend/profile and real training/held-out frame/reference
pairs, binds their files, native lighting controls and reference/RGB policy, and
recomputes held-out quality/noise evidence. `verify_calibration` checks that evidence
against the candidate and independent reference. Training/held-out overlap and a
manual `qualified` status with arbitrary attachments are rejected. Creating a
receipt does not guarantee that its qualification passes.

`quality_protocol.py` freezes SSIM, learned LPIPS, optical-depth RMSE/coverage,
instance/silhouette measures and temporal rules. RGB/RGBD variants retain the same
thresholds for requested metrics; unrequested rows are explicitly `not_requested`.
The original full policy's 60 Hz sequence and `temporal30-v1` are distinct contracts;
formal 30 Hz timestamps are never rescaled to fit the former. Every required frame,
view and sequence group must be represented. A moving performance singleton cannot
prove 100-frame temporal quality. RGB+normals checks storage shape/type/finite values,
but normal-specific quality remains unassessed and cannot support equal-quality ratios.

Evaluation preserves original execution snapshots and raw timing fields, appends
`quality_events.jsonl` and immutable evaluation directories, verifies artifact hashes,
and updates derived quality projections and reports. It never rewrites `raw.jsonl`.
The report requires matching hardware/config/scene/views/profile/delivery/completion,
known policy hashes, complete checksummed qualification and paired independent
repeats before returning an equal-quality ratio.

## Artifacts and offline rebuilding

Per-experiment files include `definition.json`, `effective_config.yaml`,
`assets_manifest.json`, `approval.json`, `selection.json`, `manifest.json`,
`runs.json`, `raw.jsonl`, `quality.json` and `artifact_index.json`. Case folders
contain `worker-config.json`; workers retain `result.json`, `worker.log`,
`capture.json`, `resources.json`, `frames.json`, representative arrays/metadata
and diagnostics when requested. R-01 saves required view evidence; R-02 retains
all sequence states and camera streams, with optional review-video artifacts.

```bash
python -S -m scripts.benchmark rendering-formal --report-only "$EXPERIMENT_ROOT"
```

The standard-library-only report emits `report.md`, `summary.json`, `metrics.json`,
`metrics.csv`, `plot_data.csv`, `figure_data.csv`, `comparisons.csv`, `capacity.csv`
and `figures/*.svg` / `*.pdf`. Figures embed source run IDs, configuration hashes
and source artifact hashes. Medians/MAD use independent process runs. Reports keep
full-plan, selected, runnable, started and completed denominators separate, group
hardware/workloads, and preserve missing metrics and all failure reasons. Capacity
figures show observed endpoints; a failure or untested larger N is not a certified
maximum. OOM is distinct from unsupported inputs, dependency errors and crashes.

`acceptance.json` keeps the external **RTX 5090** requirement explicit. Work on the
available RTX PRO 5000 72 GB machine cannot be relabelled as 5090 evidence.

## Current validation boundary

CPU contracts cover selection/recovery, capture/copy policy, fake worker lifecycles,
resources, profiler fallback, matching overhead windows, replay, independent references,
quality gates and simulator-free report reconstruction. Real learned LPIPS and
Mitsuba paths have also been exercised; this is implementation validation.

Completed short GPU smoke coverage includes both backends at R-04 N=1, R-05 N=16/C=3,
and moving R-03 256² RGB/depth/seg in both profiles. Isaac R-05 N=64/256/1024/C=3
smokes also completed after its NVML dependency was installed. These used two warmups
and four timed captures, not publication repetition. Older Isaac smoke rows lacking
UUID/process-memory evidence remain ineligible for hardware comparisons.
Each result belongs to its recorded profile hash; completion under an earlier
profile does not qualify a later revision or justify pooling by profile name.

Larger native hybrid probes encountered multiview/sampler limits before camera-group
chunking; those failed/interrupted records remain evidence, not a supported-capacity
claim. Native R-04 N=64 subsequently completed smoke and all-camera freshness after
chunking and explicit state publication. The subsequent NRD three-camera N=64
probe still exceeded Vulkan sampler capacity and failed freshness; explicit
capacity messages are now retained in `renderer_validation.json`, including on
failed runs. A nominally completed capture with these errors is also rejected.
Larger three-camera native capacity and
the complete R-01/R-02 matrix still require qualification. PT16 configuration/reset
and cleanup have been probed. R-01 calibration/N=1 has complete 20-view captures
in both profiles/backends; the native DLAA profiles pass the measured depth/label
thresholds in that scope, while RGB remains unqualified. R-02 realtime has
complete 100-frame, three-camera captures and videos under its earlier profile.
A higher-spp
independent reference has passed its noise check, while photometric/SSIM/LPIPS quality
has not qualified the cross-backend comparison. There are no publication performance
numbers or equal-quality speedup claims here.

Targeted CPU validation can run without global simulator test setup:

```bash
python -m pytest -q -c /dev/null --noconftest \
  tests/benchmark/core tests/benchmark/reporting \
  tests/benchmark/rendering/test_formal_runner.py \
  tests/benchmark/rendering/test_rendering_capture.py \
  tests/benchmark/rendering/test_publication_report.py \
  tests/benchmark/rendering/test_rendering_resources.py \
  tests/benchmark/rendering/test_rendering_diagnostics.py
```

Run optional reference/quality/evaluation tests in the CPU reference environment.
Real GPU tests are separate, serial and must inspect the retained images, freshness,
renderer settings and failure evidence. CPU fixtures do not certify live GPU capacity.

## Legacy compatibility routes

`camera-pilot` is the small procedural table/three-box, single-RGB example;
`rendering-suite` retains the older pilot catalog:

| Legacy ID | Meaning |
|---|---|
| R-03 | 128/256/512 output resolution |
| R-04 | N=1/4/16 small camera batches |
| R-05 | RGB, RGB+depth, RGB+normals |
| R-06 | Static versus moving temporal capture |
| R-09 | Legacy device/host/repeated readback variants |

```bash
python -m scripts.benchmark camera-pilot --help
python -S -m scripts.benchmark rendering-suite --list
python -S -m scripts.benchmark camera-pilot --report-only /path/to/camera-pilot-run
python -S -m scripts.benchmark rendering-suite --report-only /path/to/legacy-suite-run
```

Legacy `suite*.py`, `workload.py`, `run_benchmark.py` and their adapters/reports remain
available. Their IDs, delivery labels, geometry, defaults and artifact versions are
not interchangeable with formal v2. In particular, legacy modality R-05 is never a
substitute for formal three-camera R-05 or for the separate modality-ablation track.
