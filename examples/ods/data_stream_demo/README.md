# ODS visual training: diversity and end-to-end cost

The main experiment asks: **with the same actual training budget and hardware,
how much distinct data reaches the model, and how long does the complete
pipeline take?** It runs real BC training from scratch on the existing Franka
PickPlace task. Only third-person (`single`) and third-person plus wrist
(`dual`) camera policies participate in the training comparison.

Accepted data volume and trained-state coverage may differ. The common budget
is optimizer updates, batch size and valid supervision, not epochs over
unequal datasets. All paths save a standard LeRobot dataset, checkpoint and
an auditable training manifest. A faster result is interpreted together with
the achieved data coverage and independent policy evaluation.

The complete contract is in the
[fixed-budget visual design](../../../docs/architecture/ods-data-stream-showcase-design.md#6-fixed-training-budget-visual-comparison).

The [completed visual study](results/2026-10-21-visual-training/README.md)
retains all 30 formal attempts: 29 qualified and one audit-process shutdown
failure. ODS starts updates earlier, but its complete pipeline is slower than
Offline-32 at comparable observed coverage under this budget. The short
training budget does not produce a reliable PickPlace policy.

The [budget-aware follow-up](results/2026-10-22-budget-aware-prefetch/README.md)
retains six capacity preflights and eight matched pilot outcomes. It removes
eight untrained tail episodes and reduces complete time by 16–19% versus the
eager control, and 4–5.5% versus Offline-32 in this single-seed measurement.
Trained coverage remains similar, and serial is slightly faster than overlap;
the pilot establishes neither a diversity advantage nor a positive isolated
generation/training-overlap benefit. The three-stage guide explains how to
measure and choose producer capacity without raising the protected reuse cap.

## Primary training paths

| CLI path | Data | Purpose |
| --- | --- | --- |
| `--arm offline --offline-episodes 8` | Finalize 8 episodes, then SDK read/decode and train | Small corpus, low acquisition cost |
| `--arm offline --offline-episodes 32` | Same, with 32 episodes | Intermediate time–coverage point |
| `--arm offline --offline-episodes 128` | Same, with 128 episodes | Larger corpus, higher acquisition cost |
| `--arm ods_overlap` | Start after C episodes, generate and record the next block during training | Main ODS demonstration |
| `--arm ods_serial --trace ...` | Regenerate overlap sources and replay its training manifest without generation/training overlap | Attribution control |

Defaults are U=128 updates, B=32, H=2, F=32, C=8 shared episodes, P=2 producer
environments and K=128 per-source lifetime ODS deliveries. Qualified runs
complete 4,096 chunks and 131,072 valid future action targets. K is a guarded
engineering setting, not a demonstrated best learning-reuse value. Offline
paths reuse their fixed corpora as required and disclose the resulting reuse.

ODS switches to a ready new block at a batch boundary after at least one
completed batch has covered every current source. It keeps training the
current block while production runs, within K, and waits when credits are
exhausted. The default `--prefetch-policy budget` requests new data when the
remaining credits cannot fund the remaining updates, or measured generation,
recording backpressure and snapshot times predict a refresh before training
ends. The latter decision uses recent fast, complete batch/update times, not
the first cold CNN step. Unknown timing permits quota-required requests only.
K remains a maximum: a fast producer can replace a block before K is spent.
`--prefetch-policy eager` retains the earlier request-after-every-switch rule
as a diagnostic control; it can accept data that never reaches training.
ODS does not repeat early data without a bound. At update U it stops
requesting new generations, completes any in-flight generation and records
all accepted data, including untrained final sources. Those sources do not
increase diversity metrics.

The model follows the pinned RLInf ResNet10/GroupNorm/spatial-embedding design,
with a shared trainable CNN, eight spatial embeddings and 256-dimensional
projection, followed by a 512/512 BC head. Both camera histories share encoder
weights. It starts from scratch, uses only two causal RGB images per camera
and two 27-feature robot states, and predicts 32 future TCP/rotation/gripper
targets. Cube geometry is retained only for supervision/auditing, not policy
input. ImageNet normalization runs once on RGB float32 [0,1]. Reference
alignment, gradients, image-dependent small-data fitting and reload checks
must pass before formal measurements.

## Run the visual training experiment

Use the py311 environment with matching DexSim/Franka assets, PyTorch 2.7,
torchvision 0.22, LeRobot 0.4.4 and TorchCodec 0.5. The example requires two idle
primary GPUs with sufficient memory: one for simulation, one for CNN training.
Use the qualified native device mapping and disclose any explicitly reserved
auxiliary graphics device. Physical IDs below are examples, not reservations;
replace them with idle devices on the current machine.

```bash
conda activate py311
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[vision]' 'lerobot==0.4.4' 'torchcodec==0.5'
python -m examples.ods.data_stream_demo.train --help

python -m examples.ods.data_stream_demo.train \
  --suite smoke --gpu-ids 2 3 --output outputs/ods-visual-smoke

python -m examples.ods.data_stream_demo.train \
  --suite calibrate --gpu-ids 2 3 --output outputs/ods-visual-calibration

python -m examples.ods.data_stream_demo.train \
  --suite pilot --gpu-ids 2 3 --evaluation-scenes 0 \
  --output outputs/ods-visual-pilot

python -m examples.ods.data_stream_demo.train \
  --suite report --gpu-ids 2 3 --output outputs/ods-visual-formal
```

Freeze model, camera profile, hardware and training budget after preflight;
do not increase epochs or insert delays to manufacture overlap. Formal
execution has 24 primary runs: four paths×two camera modes×three fixed seeds,
plus six matching serial replays, for 30 runs total. Primary runs regenerate
their data, rotate path order and retain failed/slower attempts. Preflight
measures real producer and learner throughput, not a sleep consumer.

### Select a configuration from measured capacity

Preflight preserves the requested C/P/B/K instead of substituting a fixed
producer configuration. Its default update count is
`max(16, ceil(C*K/B)+8)`, so it can observe new-generation use after the first
block's credits run out. `--calibration-updates` overrides that diagnostic
budget; a short run without later-generation training is explicitly marked
insufficient for steady-supply conclusions.

Measure three capacities in each visual condition: accepted generation
`gamma_gen` (episodes/s), SDK write/encode busy-service `gamma_record`
(episodes/s), and ready batch preparation + transfer + synchronized training
`mu_ready` (updates/s). The pure model update rate is retained separately.
Use `gamma_effective = min(gamma_gen, gamma_record)` as a bottleneck estimate,
then compare `gamma_effective*K/B` with `mu_ready`. This is a service-capacity
diagnostic, not an end-to-end speed prediction: staging, startup, shutdown and
shared resource contention still cost time. Never add overlapping stage
durations and call the sum pipeline time.

Sweep only feasible P values that divide C, holding camera settings, hardware,
CPU affinity, B, K and model fixed. Select the smallest measured P reaching
90% of the best measured effective supply for that condition. Incompatible
hardware/recipes or unqualified runs cannot enter that selection. Increasing P
cannot solve a recorder bottleneck. If one producer setting must serve both
camera conditions, use the smallest tested P reaching that threshold in both,
and retain each condition's individual choice. The implied K for full learner
utilization
is reported as a diagnostic, **not** permission to raise the reuse guard or a
recommended learning setting. Keep K bounded and report unavoidable waiting
when production cannot support the actual model workload.

For example, measure P=2/4/8 on the same admitted GPU pair and compare the
saved calibration packets. Recheck device availability before each command:

```bash
python -m examples.ods.data_stream_demo.train --suite calibrate \
  --producer-envs 2 --gpu-ids 2 3 --output outputs/ods-capacity/p2
python -m examples.ods.data_stream_demo.train --suite calibrate \
  --producer-envs 4 --gpu-ids 2 3 --output outputs/ods-capacity/p4
python -m examples.ods.data_stream_demo.train --suite calibrate \
  --producer-envs 8 --gpu-ids 2 3 --output outputs/ods-capacity/p8
python -m examples.ods.data_stream_demo.training_throughput \
  outputs/ods-capacity/p2 outputs/ods-capacity/p4 outputs/ods-capacity/p8 \
  --output outputs/ods-capacity/selection
```

The one-seed `pilot` runs Offline-32, budget-aware ODS overlap and its exact
numeric/source-anchor serial replay, plus a separately reported eager control,
in both camera modes. Each mode keeps the same GPU pair and eight CPU cores
across all four paths. It holds the actual training budget fixed, retains
full-corpus audits, and writes its primary report from `main/` so the eager
control cannot contaminate baseline aggregates. It is a scoped engineering
comparison, not the three-seed formal study. Zero evaluation scenes explicitly
means no new policy-quality evidence. Run the full matrix only after this
pilot establishes the intended resource and scheduling behavior.

The supervisor also owns device selection for single runs. It masks the
selected UUID pair, assigns logical `cuda:0` to generation and `cuda:1` to
training, and audits the owned process group:

```bash
python -m examples.ods.data_stream_demo.train \
  --suite single --arm ods_overlap --vision single --gpu-ids 2 3 \
  --output outputs/ods-visual/overlap-single

python -m examples.ods.data_stream_demo.train \
  --suite single --arm offline --offline-episodes 32 --vision dual \
  --gpu-ids 2 3 \
  --output outputs/ods-visual/offline32-dual
```

Do not reuse already finalized files as the cold offline baseline. Both
pipelines include fresh acquisition, failed expert attempts, encoding,
initialization, actual optimizer work, checkpoint writing, finalization and
normal shutdown. Last-update time is recorded separately from total time,
so pending recording work cannot disappear from the comparison.
All paths permit recording to overlap the next generation within the same
bounded storage. Request the next shared-cache fill before waiting for old
owned episodes to enter the writer queue; release those old snapshots before
taking new ones. Offline still starts training only after its full corpus is
finalized. Serial still starts each generation after the preceding optimizer
updates finish, and starts its next updates only after generation completes.
An extra recorder-submission barrier must not make either control slower.

The supervisor then launches a separate CPU corpus/decoder audit and the
independent policy evaluation. Their durations are saved and excluded from
pipeline time. A failed audit or evaluation retains the original measured
pipeline time and disqualifies that run for comparative claims.
After the stage leader exits, the supervisor allows up to two seconds for its
owned process group to disappear. Remaining members, including zombies, still
fail the stage; PID, parent, state and termination evidence are retained in
`process_group_shutdown`. This wait belongs to the stage, including pipeline
time for the training stage. The formal study predates this shutdown fix and
keeps its original failure; later validation is recorded separately.
Actual runs remain unqualified while validation is pending. The supervisor
persists the requested evaluation scene count; formal reports require all
requested scenes and normal evaluation closure, or an explicit zero-scene
`not_requested` status.

On a qualified driver, the new training supervisor can accept an explicit
`--aux-graphics-gpu-id` for the native driver's passive graphics footprint.
Outside each pipeline's primary pair it permits only owned pure-G contexts of
at most 8 MiB per PID, requires 16 GiB free and records other board activity.
The flag alone grants no compute capacity. If the same board is explicitly
listed in `--gpu-ids`, it may also serve as another pipeline's primary after
passing all three idle readings. Every process group remains limited to its
own two primary GPUs for computation. This exception belongs to the training supervisor.
The historical delivery supervisor below retains its stricter requirement
that an auxiliary board also pass three idle readings.

## Training outputs and interpretation

Render only retained training logs after workers exit:

```bash
python -m examples.ods.data_stream_demo.training_report \
  outputs/ods-visual-formal --output outputs/ods-visual-report
```

Per-run `summary.json` contains the fixed work ledger and `training_metrics`.
`training-events.jsonl` records immutable completed-update source/anchor IDs,
losses and monotonic stage timestamps. `training-sources.json` retains exact
per-source frame counters, including retired sources. The report writes
JSON/Markdown and four measured figures, each in PNG/SVG/PDF:

1. Total time versus initial-state coverage and effective trained sources.
2. Coverage growth against wall time and completed optimizer updates.
3. Generation, recording, data preparation, training and cleanup timeline.
4. Lifetime source use, including retired/unused sources and the ODS K line.

`writer-sdk-events.jsonl` streams the recording thread's actual SDK service
intervals. They include writing, encoding and recording bookkeeping, while
prefetch-worker SDK fetch spans include temporal reads, decoding and validation.
The timeline distinguishes both from owner copying and waiting; overlapping
worker durations are not additive pipeline time or isolated codec CPU time.
`dataset_bytes` and `checkpoint_bytes` record finalized logical file sizes,
excluding temporary disk peaks and later audit/evaluation artifacts.

The post-timing audit checks every numeric frame, an SDK H2 window from every
source, sparse RGB PSNR/MAE references, and sequential software decoding of
**every saved AV1 frame**, including untrained tails. Full-video frame hashes
cover decoded YUV420p planes; they do not imply lossless RGB or exhaustive SDK
random-seek equivalence. Fresh audit receipt/artifact paths are required, so
rechecking a historical run preserves the original evidence:

```bash
python -m examples.ods.data_stream_demo.training_audit \
  --run-dir outputs/ods-visual/overlap-single \
  --output outputs/ods-visual/overlap-single/post-audit-new.json
```

Only completed optimizer updates increase training diversity. Metrics include
unique sources, entropy-based effective sources, fixed 1 cm initial XY bins,
2 cm cube-state voxels at trained anchors, unique anchor/history/target frame
IDs and duplicate-target fraction. New source IDs alone do not demonstrate
physical-state novelty; overlapping prediction windows are deduplicated.
A registered-but-untrained source contributes to acquisition cost, not coverage.

Failed, unsupported or unqualified runs remain in the report but are excluded
from ratios. Formal same-vision/seed ratios require matching initial-model and
normalization hashes, the common initial-C numeric prefix and camera order,
the exact declared/completed U/B/H/F budget, the same assigned generation and
training GPU UUIDs, and the same eight CPU cores. Supervisor launch masks,
thread budgets, placement audits, normal-exit records and total times must
agree; missing evidence excludes the comparison. Smoke/calibration and legacy
CPU metric fixtures do not produce formal speedup claims. AV1 is lossy;
retain PSNR/MAE audits and the independent
20-state development evaluation beside timing. A smaller offline corpus may
finish sooner or a larger one may cover more states. No universal speedup or
policy-learning improvement is assumed.

## Historical delivery diagnostic

The original no-model entry point below remains available to isolate cache,
delivery, conversion and standard video-decode costs. Its state-only condition
belongs to that diagnostic, not the visual training matrix above. It does not
import a learner or produce policy-performance evidence.

The [2026-10-21 study](results/2026-10-21/RESULTS.md) contains 27 main runs and
nine appendix diagnostics validated in py311, raw logs and figures. It records
the visual delivery limitation and teardown cleanup warnings. The subsequent
[copy optimization](results/2026-10-21-copy-optimization/README.md) records CPU
component timings and bitwise checks. These historical measurements are not
relabelled as end-to-end visual training results.

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
