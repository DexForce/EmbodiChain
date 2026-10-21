# ODS data-stream capability demo

Generate real Franka PickPlace trajectories and measure how they reach a
consumer. There is no learner, optimizer or student evaluation. The experiment
compares ODS with standard LeRobot files and real video decoding, independently
of whether a policy can learn the task.

The [recorded study](results/2026-10-21/RESULTS.md) contains 27 main runs and nine
appendix diagnostics validated in the py311 environment, with raw measurement
logs, per-run tables and figures. It includes the current visual-throughput
limitation and process-exit cleanup warnings.
The subsequent [sampling-copy optimization](results/2026-10-21-copy-optimization/README.md)
records CPU component measurements and bitwise equivalence checks separately.

ODS's primary training-efficiency question is how long it takes to generate
the same data and complete the same real training work when generation and
training overlap. This no-model demo diagnoses delivery costs and correctness;
the 36 runs are not evidence of end-to-end training acceleration. The separate
[equal-work training protocol](../../../docs/architecture/ods-data-stream-showcase-design.md#6-follow-up-equal-work-end-to-end-training-time)
is designed but has not been implemented or measured yet.

Run commands from the repository root. The complete protocol, including the
timing boundaries and comparison limits, is in the
[design](../../../docs/architecture/ods-data-stream-showcase-design.md).

## Compared paths

| Arm | Source availability | Per-source chunk budget |
| --- | --- | --- |
| `lerobot` | Finalized standard Parquet/MP4 corpus; SDK/DataLoader/TorchCodec access | K |
| `streaming` | ODS initial fill, followed by asynchronous cache replacement | Unlimited |
| `protected` | Same ODS refill recipe | K |
| `lerobot_unbounded` | Full finalized LeRobot corpus, decoded on demand; appendix diagnostic | Unlimited |
| `frozen` | ODS initial cache only; appendix diagnostic | K |
| `lerobot_same_cache` | First C LeRobot episodes; appendix diagnostic | K |

Each arm supports `state`, `single` and `dual`: no camera, fixed third-person
camera, and the same camera plus a wrist-mounted camera. Visual output is
256x256 RGB float32 in [0,1], with all context frames delivered. ODS native
RGBA-to-RGB conversion is timed, as are LeRobot decoding and collation.

The formal defaults are C=8 cached episodes, P=2 producer environments,
N=32 accepted episodes, B=32 chunks, history=2, future=32, K=128. Each chunk
contains 33 context frames and costs one credit, independent of its camera
count. K is a demonstration value, not an optimal learning-reuse setting.
An evicted source's unused credits are retired, never transferred to another
source. The main suite uses seeds 19001, 19002 and 19003; appendix diagnostics
use the first seed only.

## Requirements

Use a source installation of EmbodiChain with a matching DexSim runtime,
Franka assets and a supported CUDA simulation device. State-only acquisition
uses `no-render`; visual acquisition still needs native headless rendering.
The standard reader requires the pinned SDK/decoder and an FFmpeg build
supporting the selected AV1 encoder:

Validation used PyTorch 2.7.0+cu128 with the SDK/decoder versions below. The
isolated DataLoader lifecycle hook is tested against that PyTorch version;
use a matching simulator and Torch/TorchCodec build.

```bash
python -m pip install -e .
python -m pip install 'lerobot==0.4.4' 'torchcodec==0.5'
python -m examples.ods.data_stream_demo.run --help
```

Select idle physical GPUs and CPU thread settings before starting Python:

```bash
export CUDA_VISIBLE_DEVICES="$(nvidia-smi --id=7 --query-gpu=uuid --format=csv,noheader)"
export ODS_GPU_UUID="$CUDA_VISIBLE_DEVICES"
export SDL_VIDEODRIVER=dummy
unset DISPLAY WAYLAND_DISPLAY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

This selects physical GPU7 by its NVML UUID. Pass logical
`--sim-device cuda:0` for the single run, or use `--gpu-ids 7` for a suite.
The matching native runtime maps the primary CUDA UUID to the primary Vulkan
UUID. Qualify compute **and graphics** placement with the actual ODS subprocess
path before running the matrix. On the validation host, Vulkan instance/driver
initialization additionally creates a small graphics footprint on GPU7, so
GPU7 is the qualified strict single-GPU example there. Other machines require
their own device qualification; unmasked physical ordinals are not a fallback.
GPU7 is an example selection, not a reservation. Check that the selected
device is idle and has enough memory. Do not silently shrink an unsupported
workload. The default dual-camera native image cache alone occupies about
2.34 GiB; a returned RGB float32 batch contains about 1.55 GiB of image
payload, before decoder, normalization and prefetch temporaries. Shared
memory, RAM and VRAM are checked separately.

The acquisition reuses the existing BC demo's physical task and expert but
does not import its trainer or modify its sensor-off defaults. This example
owns its additional camera profiles and decoded data adapter.

The qualified camera profile is `v3`: fixed third-person and side-mounted
wrist views, `fast-rt`, SPP=1, offscreen OptiX spatial denoising, no window
denoising or super-resolution. V1's self-occluded wrist view and noisy raw
rendering were rejected before timing. V3 also separates the inherited
cube/goal material UIDs so their original orange/green colors remain distinct;
it changes no physical property or success criterion.

## Run and inspect

Use a fresh output directory for every run. The stable single-run entry is:

```bash
python -m examples.ods.data_stream_demo.run \
  --suite single --arm protected --vision state --sim-device cuda:0 \
  --output outputs/ods-stream/protected-state

python -m examples.ods.data_stream_demo.run \
  --suite single --arm lerobot --vision single --sim-device cuda:0 \
  --output outputs/ods-stream/lerobot-single
```

These commands select paths; an engineering run alone is not a completed
27-run report. Freeze calibration, rates, hardware and profile before comparing
formal runs. The matrix is three main arms by three visual conditions by
three seeds, with nine separately identified appendix diagnostics. Do not mix
different profiles in one report input directory.

Start with the small engineering suite before allocating the formal budget:

```bash
python -m examples.ods.data_stream_demo.run \
  --suite smoke --gpu-ids 7 --output outputs/ods-stream-smoke
```

`--gpu-ids` accepts physical GPU IDs. If omitted, the launcher selects devices
with three zero-utilization readings and at least 16 GiB free, then assigns at
most one child per selected GPU and eight disjoint CPUs per child. It sets
both `CUDA_VISIBLE_DEVICES` and `ODS_GPU_UUID` to the selected physical UUID,
then passes logical `cuda:0`. Reproduction does not require several GPUs.

The parent periodically audits compute and graphics contexts belonging to
its child PID tree or process group. An owned context on another GPU fails
the cell and terminates only that owned process group. Context samples are
retained below `_supervision/`; unrelated processes are never terminated.
There is no global GPU0-idle prerequisite. Periodic checks can miss short-lived
contexts, so they supplement qualification of the matching native UUID mapping
and do not establish an exclusive-device reservation. Previously blocked
calibration attempts remain failed evidence and do not become a usable protocol.

For the qualified validation-host driver, an optional auxiliary reservation
allows another idle primary GPU while accounting for that graphics footprint:

```bash
python -m examples.ods.data_stream_demo.run \
  --suite calibrate --gpu-ids 1 3 4 5 6 --aux-graphics-gpu-id 7 \
  --output outputs/ods-stream-calibration-aux
```

The auxiliary GPU must pass the same three idle readings and 16 GiB admission,
is excluded from every primary assignment, and is shared by this suite's jobs.
Only its exact UUID with pure `G` type and known per-PID memory at most 8 MiB is
allowed. Unknown memory, compute contexts, larger graphics footprints or any
other GPU fail the job. Primary and auxiliary owned memory are both recorded,
deduplicating `(PID, UUID)` records before summing across devices. This scope
uses primary GPUs **plus a shared reserved auxiliary GPU**, and is not a
single-GPU result. Low memory and `G` type alone do not prove no computation;
the permission relies on the separate controlled native-runtime qualification.

An independent minimal libvulkan probe, without CUDA/SDL/DexSim or device/queue submission,
reproduced the 6 MiB GPU7 footprint after `vkCreateInstance` and its disappearance
after instance destruction. A real dual-camera ODS episode then qualified
primary GPU3 plus the explicit GPU7 footprint with graceful process cleanup.
These observations support a bounded **Vulkan instance/driver initialization**
scope, not a claim of zero overhead or a general graphics exception.

The default remains strict. Calibrate, report and appendix must each receive
the same explicit auxiliary flag when opting in; the frozen protocol records
its UUID, model, memory limit and the offscreen environment. The launcher uses
`SDL_VIDEODRIVER=dummy` with `DISPLAY`/`WAYLAND_DISPLAY` unset. This removes the
X11 dependency; it does not remove the Vulkan graphics footprint.

Smoke checks all nine main cells with one seed, C=P=2, N=8, B=4, K=8,
30 seconds of consumption and a 16 chunks/s offered load. Images and context
length remain unchanged. Smoke verifies paths and cleanup, not the formal load
or generation coverage.

Calibrate capacity, then freeze the resulting protocol for the main suite:

```bash
python -m examples.ods.data_stream_demo.run \
  --suite calibrate --gpu-ids 7 \
  --output outputs/ods-stream-calibration

python -m examples.ods.data_stream_demo.run \
  --suite report --gpu-ids 7 \
  --protocol outputs/ods-stream-calibration/calibration.json \
  --output outputs/ods-stream-main
```

The protocol fixes the common offered rate and observation duration based on
measured generation and ready-source throughput; it does not select a
learning-optimal K. Formal runs do not silently shorten an expensive visual
condition. Calibration and smoke results remain separate from main estimates.
Use a custom `--suite single --duration-seconds ...` run to explore a different
window. The smoke recipe and formal protocol have fixed windows.

Calibration performs six cells: protected ODS and unbounded standard decoded
LeRobot for each visual condition. It requires all 32 accepted episodes and
four completed generations. The packet records generation rates, observed
batch-service estimates, actual decoded-reader throughput, source hashes and
camera profile. It is not a throughput guarantee or a learning-budget search.
Code/profile changes invalidate the packet and require recalibration.

After the main suite, run the nine appendix diagnostics using the same packet
and its seed 19001 corpora:

```bash
python -m examples.ods.data_stream_demo.run \
  --suite appendix --gpu-ids 7 \
  --protocol outputs/ods-stream-calibration/calibration.json \
  --dataset-root outputs/ods-stream-main \
  --output outputs/ods-stream-appendix
```

For appendix, `--dataset-root` is the **main report root**. The launcher looks
for `lerobot/<vision>/seed19001/lerobot_dataset/` below it. Frozen starts a new
physical producer; the two SDK diagnostics reuse those finalized files.
For `--suite single`, `--dataset-root` instead names one concrete finalized
LeRobot dataset directory.

Each child has a finite watchdog and its own process group. The parent
passes the same-host monotonic launch timestamp so cold first-batch timing
includes Python/library startup and instrumentation setup. Exact commands,
device/CPU assignments and stdout are stored below `_supervision/`; final
per-run summaries remain below `<arm>/<vision>/seed<seed>/`. Failures,
timeouts, cancellation and unsupported cells remain in `suite-summary.json`.
The launcher kills only a process group it created, and never retries a
failed experimental cell automatically.

The LeRobot writer uses the official SDK and real MP4 video features. The
reader calls the public Dataset temporal API in spawned DataLoader workers,
with actual CPU TorchCodec decoding. It does not load a decoded corpus into
resident tensors. The parent owns delivered and outstanding K reservations;
recreating an iterator does not replenish them. Normal OS file caches,
decoder instances and bounded prefetch are retained and disclosed. The
experiment does not clear system-wide caches.

Closing first stops new dispatch and drains existing prefetch without counting
it as delivered. Every reader and quality-audit worker must exit with code zero;
joining an aborted worker or forcing termination fails the run. Summaries retain
discarded reservations and shutdown time. Iterator restart keeps already
delivered K usage.

## Render results

Rendering reads retained logs and performs no acquisition, decoding or new
timing experiment:

```bash
python -m examples.ods.data_stream_demo.report outputs/ods-stream-main \
  --output outputs/ods-stream-report --animation
```

The input must be the exact experiment root containing run directories, or
one run's `summary.json`. Do not point it at an unrelated results tree. It
reads `summary.json`, `lifecycle.jsonl` and `batches.jsonl`; malformed records
raise an error instead of being silently dropped.

Every suite renders its report automatically after all children stop. A manual
render can add the optional animation or combine retained results deliberately;
keep smoke, calibration and different protocol versions clearly identified.

Output includes one Markdown report, `report.json`, and four figures in PNG,
SVG and PDF:

1. Cache residency, generation and delivered-source timeline.
2. Per-source delivered chunks, including evicted identities and the K line.
3. First-batch latency, consumer p95 and delivered rate versus offered demand.
4. State/single/dual throughput, logical bandwidth, memory and storage costs.

`--animation` additionally replays logged protected-run deliveries as a GIF.
It is generated after timing and is not a simulation recording. Late complete
batches remain in lifetime K accounting and are marked with crosses in the
timeline; they do not enter observation-window throughput or latency estimates.
Raw camera references and decoded image quality remain part of the run artifacts.

Filled points and median/range aggregates use only complete, explicitly
qualified `report`/`appendix` runs. Failed, unsupported, unqualified, smoke
and calibration runs remain visible; measured points are hollow and missing
metrics are unknown, never zero. One-seed appendix results remain diagnostics.
The report does not select favorable seeds or rank a performance winner.

## Read the evidence correctly

Launch-to-first-batch includes generation and the chosen availability barrier:
LeRobot waits for N finalized episodes, whereas ODS starts after C episodes.
Source-ready first-batch starts from available data, but LeRobot still reads
and decodes. These two timing scopes overlap and must not be added. Their
entire difference cannot be attributed to shared-memory transport.

Wait counters overlap only the source-ready observation window; cold generation
waits remain in `lifetime_data_wait_seconds`. Request throttling and terminal
supply-exhausted idle time are separate. Controller recovery includes the final
successful retry service, so wait and service counters are not mutually
exclusive phases and must not be summed as a wall-time breakdown.

Protected streaming can deliberately wait for new data after exhausting the
current budget. Report this separately from slow copying or decoding. A
source ID is not a measure of state coverage; delivered chunks are not
optimizer updates. AV1 is lossy, so throughput comparisons retain quality
and encoding settings. Fixed ODS cache bytes do not imply constant total
process memory or lower memory than a bounded DataLoader.

This demo can establish measured availability, reuse protection, provenance
and resource tradeoffs for these workloads. It does not establish policy
quality, training acceleration, a universal ODS speedup, or that the broader
LeRobot ecosystem cannot stream data.

## Planned end-to-end training comparison

The follow-up compares fresh collect-then-train standard LeRobot against
ODS generation concurrent with actual BC updates. An additional ODS run
alternates generation and training to isolate the contribution of overlap.
The clock starts before producer/learner launch and stops only when accepted
data, all required optimizer steps, the common final checkpoint and worker
cleanup are complete. Video encoding/finalization is included where needed;
asynchronous stage times are never summed to fabricate total time.

The first protocol reuses the existing state-only `ChunkBC`, with N=32
accepted episodes and q=32 registered legal anchors per episode. Four passes
over this finite anchor set give 4,096 trained chunks, 128 B=32 updates and
131,072 valid future action targets at F=32. One registered-anchor epoch
means visiting each registered anchor once, not every possible trajectory
frame. All paths must train every registered source/anchor/pass, with matching
batch order, initial weights, normalization, optimizer and loss work. K=128
is an upper delivery guard; merely staying under K does not prove equal
training coverage. Retained data and prefetch are bounded, and generation
waits rather than overwriting unfinished required work.

Report launch-to-completion time, paired speedup, first real update, measured
generation/training overlap, data waits and whole-pipeline resource usage.
Give each path the same CPU/GPU allocation and account for contention. The
source-ready LeRobot decoding rate remains a secondary diagnostic. A visual
training result needs a common visual model and causal image-input contract;
the existing geometry policy and full-context delivery figures do not supply
that result. These are follow-up acceptance requirements, not extra CLI
options already implemented in this example. See the linked design for the
retention, persistence, matching and timing contracts.
