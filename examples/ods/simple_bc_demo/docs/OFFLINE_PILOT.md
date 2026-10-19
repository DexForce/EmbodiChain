# ODS and LeRobot offline comparison: integration pilot v1

Status: executed at `3cd6379a092f8106f08ea6073f13bec9a4bd6f63`,
2026-10-19 (Asia/Shanghai). See [results and limitations](OFFLINE_PILOT_RESULTS.md).
The subsequent [counter optimization](ODS_ACCOUNTING_AUDIT.md) was tested
separately and was not applied to these four runs. Freeze the
execution commit, CLI, GPU UUID, CPU affinity and measurement packet before the
first timed run. This document preserves the executed four-run development
protocol and its original task ID. A balanced multi-seed study and fresh held-out
verification remain
future work; the pilot does not qualify formal efficiency claims.

## Question and scope

Measure the cost and learning behavior of training while demonstrations are
generated, versus collecting them into a real LeRobot dataset before training.
Report an equal-quota comparison and unrestricted offline baselines. The task
may already be learnable from 32 demonstrations; retain that strong baseline.
No outcome is required to favor ODS. K=2660 chunks per trajectory is an already
tested reference budget, not an optimal or universally qualified setting.

## Fixed arms

Run these arms sequentially on the same physical GPU and local NUMA CPU block.
The order below is fixed before execution; single-block order/cache effects
remain a limitation, to be balanced in a later multi-block study.

| Order / arm | Accepted data cap | Data available to learner | Sampling protection |
| --- | --- | --- | --- |
| 1. `lerobot256_unrestricted` | 256 | Independently collected direct LeRobot dataset; all episodes resident | Conventional repeated sampling, no per-trajectory cap |
| 2. `ods256_protected` | 256 | Live 32-row shared cache, existing per-row replacement | K=2660 and hard limit=2660 per trajectory |
| 3. `lerobot32_unrestricted` | 32 | Independently collected direct LeRobot dataset; all episodes resident | Conventional repeated sampling, no per-trajectory cap |
| 4. `lerobot256_protected` | 256 | Direct LeRobot write, finalize, SDK load once to resident CPU tensors | Same K and hard limit |

All arms use `--rounds 1 --updates-per-round 2500`, a maximum of 2500
optimizer updates and batch size 256. Offline collection uses explicit
`--offline-episodes 32|256`, independently of training rounds. A protected
arm may stop earlier when a full batch cannot be drawn within the remaining
quotas; retain that outcome and its unused quota. Unrestricted arms deliberately
may repeatedly sample a trajectory beyond K. They are the practical baselines,
not quota violations. The 256-episode quota ceiling is 2660 updates before row
retirement and partial-batch losses; the shared 2500-update cap lies below it.
No arm receives additional updates after viewing its development results.

The equal-quota arms compare the complete systems under a matched exposure
constraint. Their resident data, retirement, order of samples and wall-time
availability differ. They are not an isolated shared-memory-versus-file
transport experiment. Parallel expert generation benefits both offline and
online arms and cannot alone be credited to ODS.

## Task, learner, data and random states

Keep `PickPlace-Franka-v1`, default physics, real contact, current handwritten
expert and measured success unchanged. Use P=8 producer environments and C=32
production/cache rows for every arm. The offline learner intentionally retains
all collected data and reports that memory cost. Reuse a matching expert
qualification only after existing source/config checks pass; record its common
campaign cost separately from each arm.

Freeze ChunkBC, measured geometry inputs, two-frame history, hidden widths
512/512, prediction horizon 64, action horizon 16, DLS action processing,
AdamW learning rate 3e-4, weight decay 1e-6, gradient norm cap 1, normalized
observation noise 0.01 and existing masked loss. No expert/controller stage
label enters student inference. Fit all observation and target normalization
buffers from each arm's first 32 accepted trajectories only, then freeze them.

- Learner seed: 19001. Producer seed: 1019001. Sampling seed: 29001.
- Development scenes: 810000–810019 in the same order for every checkpoint.
- Final-test scenes: zero. Seeds 610000–610199 remain sealed.
- Every arm independently generates its corpus using the same producer recipe.
  Record accepted/rejected episodes and source identities. Equal seeds do not
  establish equal numeric payloads in vector simulation.
- Audit first-generation payload, initial model and normalization hashes across
  arms. System comparison requires identical procedures and explicit disclosure
  of observed payload differences. Claim exact-data pairing only if ordered
  payload hashes, schema, masks and all normalization/model hashes match.

The online auto-configuration packet reuses the previous paced engineering
measurement, G=0.93264 accepted trajectories/s and T=14218.41 chunks/s, with
fixed provisional K=2660, B=256, P=8, minimum C=32, 9.5-second first-publication
p95 and safety factor 1.25. These are not newly measured saturation capacities.
Retain the complete packet and SHA256 in the manifest; do not retune it after
seeing offline results. Separate observed supply/consumption rates from inputs.

## Offline persistence and loading

Accepted numeric episodes go directly to the official installed LeRobot SDK
through a bounded writer queue. There is no intermediate `.pt` dataset save/load
or image/video encoding. Finalization drains the writer and publishes metadata;
training starts only after successful finalization and SDK numeric readback.
Readback must preserve the ordered feature/action schema, valid frame lengths,
causal observation/action pairing and episode/continuity boundaries exactly.

Record queue bounds, copies, backpressure, write/finalization durations, bytes
and numeric payload hashes. Set `HF_DATASETS_CACHE` to the arm’s own
`output/dataset_cache` before importing the SDK. Include this cache and the
generated dataset in final disk bytes; keep temporary-write peak bytes unknown
unless measured. Do not purge global software, filesystem or SDK caches. The SDK is loaded once into resident CPU tensors;
subsequent sampling is not a measurement of per-minibatch LeRobot disk reads.
Each cold arm pays its own collection and persistence costs, including failures.
Existing-data use is a separate estimand, never obtained by omitting costs from
a cold-start headline. Checkpoints may use `.pt`; the ban concerns dataset
round trips, not checkpoint serialization.

## Evaluation and timing

Save initialization and optimizer steps 500, 1000, 1500, 2000, 2500, plus the last
completed policy if early stopping occurs between scheduled steps. Evaluate
exactly these unique checkpoint identities after training and producer shutdown,
in a fresh evaluator process/world per panel with B=1. Disable generation-based
panels through `--deferred-evaluation-schedule updates`. Use no viewer or video.

Distinguish these time origins and outcomes:

1. The launcher records process start and complete process wall duration. The
   application records its monotonic origin before per-run simulator/producer
   startup. Align application timestamps to the measured process origin. The
   runner reconstructs its origin from the process creation time; their offset
   is an alignment quantity and can be negative. It is not measured import
   latency. This corrects the prospective protocol's original interpretation;
   the execution revision and raw timestamps are preserved in the results.
2. Cold first-update latency includes worker startup, generation and, for offline
   arms, writing/finalization/loading, normalization and model initialization.
3. Checkpoint readiness is its actual save-completion timestamp. A checkpoint
   later measured at >=80% development success has a retrospectively qualified
   readiness time, not an online time-to-80 or certified result-delivery time.
4. Report generation, writer service/backpressure/finalization, SDK load,
   model/normalizer setup, active training, quota waiting and missing-data waiting
   separately. Do not add overlapping phase intervals to estimate total time.
5. Report training completion, posthoc evaluation, reporting and final cleanup
   separately. Full process wall time remains available as total delivery cost.

Report development success as integer successes/20, lift rate, final goal error,
episode length, latest and best success, checkpoint-to-checkpoint regressions,
actual updates/chunks/valid supervised targets, per-trajectory exposure and
consumed generations. A single development seed block is descriptive evidence,
not statistical proof or held-out generalization.

Record allocated GPU-hours, sampled native per-process VRAM, process-tree PSS,
observed CPU time, dataset bytes and checkpoint/log bytes. Label sampled peaks
and incomplete counters honestly. Observe descendants from process startup and
verify physical GPU placement and process exit; Torch allocator memory alone
does not measure simulator/native GPU usage. No unknown metric becomes zero.

## Execution gates and failure policy

Before the four arms, pass a separate tiny SDK smoke with two generations
(C=4, D=8, P=2), including capped offline sampling and exact payload readback.
This is engineering evidence only and is excluded from the pilot's learning
and timing comparison. Retain its cost separately.

Use GPU3 (`GPU-afe5dd63-2838-06db-b0fa-5d98d4b1eec5`) and CPUs32–47,
a local NUMA block. Common SDK/runtime engineering smoke warms the software
caches before the pilot block; these are process-cold, software-cache-warm
runs. The single pilot order is not counterbalanced.

Before each arm, confirm the chosen GPU has no external compute processes and
zero utilization on three readings
spaced two seconds apart and at least 16 GiB free. Set its full UUID in
`CUDA_VISIBLE_DEVICES` before Python imports, and use local `cuda:0`. Keep a
fixed NUMA-local 16-logical-CPU block and one OMP/MKL/OpenBLAS thread. Record
GPU/CPU topology, source commit and file hashes, interpreter/dependency versions,
asset/runtime overlay identity, qualification hash and storage location. Shared
host interference can remain even after idle checks and must be disclosed.

Freeze a 5400-second application budget for all arms as a failure guard, not a
claim that all spend equal wall time. An independent launcher watchdog triggers
at 7200 seconds from process launch, including synchronous SDK/native work and
posthoc evaluation. Each arm starts its own process session. On timeout, verify
that its process-group ID equals its main PID, mark the run failed and send
SIGTERM only to that owned group; after a 20-second grace, use SIGKILL with a
further 5-second cleanup bound. Record all forced-cleanup signals, PIDs and
survivors, retain partial artifacts and stop the queue. Cleanup grace is separate
from the 7200-second execution deadline. Unrelated processes are never selected
by GPU allocation or name for termination. Do not alter the source while a queue is
running. Persist failures, partial outputs and shutdown evidence. A pipeline or
learning failure is not silently retried; external failure retries require a
separate recorded attempt and explanation. Never launch a duplicate run because
an aggregate report has not yet updated.

Pilot completion requires all four arms accounted for, SDK round-trip audit,
quota audit for protected arms, the shared checkpoint schedule, first-generation
model/normalizer/payload comparison and no surviving owned worker/evaluator.
Any unsupported comparison remains explicitly unqualified. Only then revise
and freeze a formal multi-seed protocol; do not automatically open final tests.

The pilot produces one report with time/memory, quality/exposure and a descriptive
development-success ordering covering every arm, including failed or missing
ones. That ordering is not a recommendation of a globally best pipeline.

Implementation notes: offline geometry features are computed once while collecting
accepted snapshots, whereas the live ODS path converts raw sampled observations.
These preprocessing-placement costs belong to this system comparison; it is not
a byte-identical transport benchmark. Direct SDK loading is bounded/batched and
records fallback behavior and actual SDK cache files. Protected offline resume
is rejected until saved per-source quota restoration exists.
