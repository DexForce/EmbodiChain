# ODS versus offline data: controlled experiment protocol v2

Status: complete study design; execution prerequisites remain. Date: 2026-10-17,
UTC. The machine-readable design is [BENCHMARK_PROTOCOL.json](BENCHMARK_PROTOCOL.json),
which is not a runnable CLI configuration. This replaces the earlier frame-BC
protocol. It does not claim that the complete experiment has already run.

The reference implementation is `10a7718e`: the selected online checkpoint
achieved 99/100, while the checkpoint after refresh achieved 65/100. The task is
learnable; refresh is not automatically beneficial. The [working evidence](WORKING_SHOWCASE.md)
is a feasibility result, not a speed comparison. In particular, the previous
21.4-minute online run and 9.4-minute data-ready offline run have different
acquisition costs and cannot form a valid speedup ratio.

## 1. Questions and permissible claims

The primary question is **how quickly a pipeline produces a useful policy at a
fixed resource budget**, with selected-policy quality and later regression
reported alongside time. Separate these conclusions:

- **System result:** end-to-end ODS versus practical offline pipelines under the
  same task, producer recipe, learner and budgets. Independent regeneration is
  paired by reset/attempt seeds; payload hashes establish whether stronger exact
  data matching was actually obtained. Pipeline preprocessing/lifecycle costs
  belong to this result.
- **Delivery mechanism:** shared-memory sampling versus resident offline
  sampling, using identical bytes, masks, tensor layout, preprocessing placement
  and ordered contexts. This is a data-ready component experiment.
- **Overlap mechanism:** parallel versus serialized generation/learning with a
  predetermined valid consumption/publication schedule. The current `serial`
  mode is unmatched and cannot establish this causal claim.
- **Refresh effect:** quality, retention and cost when G2 replaces G1, versus
  continuing to train on G1. This intervention includes production contention
  and changed data exposure; it is not a pure memory-transport experiment.

A win over Offline-64 alone supports an advantage over full precollection.
Offline-32 must also be shown, because the task may already be solvable from G1.
No result is required to favor ODS. Sustained-training superiority additionally
requires a good latest checkpoint; an earlier good checkpoint does not conceal
later regression. No external dataset or pretrained model is introduced.

## 2. Freeze the task, learner and source

Use the existing single-cycle `PickPlace-Franka-v1`: real grasp contact, 5 cm
cube, unchanged initial robot posture, XY randomization of +/-4 cm, fixed goal,
240 Hz physics / 30 Hz control and 600-step horizon. Success still requires
measured lift, placement within 3 cm at table height, open fingers, withdrawn TCP
and 0.3 seconds of stability. Both expert and student use that predicate.

Freeze the revised cube-symmetric expert orientation, DLS controller parameters,
software/assets and action schema. Reuse a matching 40-scene qualification only
when its source/config hashes pass; record that common campaign cost separately.
Failed production attempts consume time and remain in the acceptance log.

Freeze `ChunkBC`: 51 measured features, two-frame history, MLP-512x2, 64 absolute
TCP/gripper targets, 16-step execution, 7D environment requests, AdamW at 3e-4,
weight decay 1e-6, gradient norm 1 and normalized observation noise 0.01.
No phase/progress/success label enters inference. Both input and target statistics
come only from the first 32 accepted episodes. Freeze the normalization rule,
ordered features, transforms, sampling distribution, masks and all four buffers.

The learner remains unchanged throughout the study, even if an arm performs
poorly. Changing its optimizer, teacher, sampler or representation creates a new
protocol revision and requires a new untouched final panel.

## 3. Four primary cold-start arms

| Arm | Data and availability | Main comparison |
| --- | --- | --- |
| ODS-online | Generate G1=32; train; request G2=32; retain the existing bounded 32-row replacement behavior | Actual online system |
| ODS-frozen32 | Generate the same G1 recipe; no refill; train on the live shared cache | Whole refresh intervention relative to ODS-online |
| LeRobot-32 | Generate/persist 32 accepted episodes; finalize; official SDK load to resident CPU tensors; then train | Strong small offline baseline |
| LeRobot-64 | Generate/persist 64 accepted episodes; finalize; SDK load; sample all 64 during training | Strong full-data offline baseline |

All arms permit 10,000 updates, batch size 256 and the same context horizons.
There is one simulation producer environment and one active B=1 evaluation
environment. The online row capacity stays 32. Offline-64 may retain all 64 in
RAM: this is an intentional practical baseline, with its memory cost reported.
All arms sample episodes uniformly and then eligible anchors uniformly.
Padding is excluded using the same loss-mask semantics. Report actual valid
future targets; 10,000 chunk updates are not 10,000 individual-frame updates.

For the online arm, preserve the current intervention: request G2 after the
step-5,000 evaluation/confirmation; continue updates while refill progresses;
wait for full G2 publication before update 6,251. Per-row publication can expose
G2 earlier. Record the actual generation mixture rather than assuming a sharp
switch. Do not replace this with accumulated replay, DAgger or adaptive collection.

The LeRobot baselines use the official SDK, state/action features and explicit
schema/provenance. Write accepted episodes through a bounded asynchronous queue
(two episodes initially), and drain/finalize before training. Avoid an artificial
intermediate `.pt` save/load just to export LeRobot. Include all encoding,
copying, writing, failed attempts, finalization and loading costs. No image/video
decoding is charged to this state-only policy. Exact numeric read-back must pass.
The existing `.pt` loader is an additional optimized-format sensitivity control,
not a substitute for the LeRobot arms.

## 4. Fresh splits, seeds and pairing

Previously used development, engineering and test scenes, including
50000-50099, are **historical** and cannot be called untouched test data again.
Use the following namespaces for this study:

| Purpose | Fixed seeds / rule | Use |
| --- | --- | --- |
| Formal learner initialization | 11001, 11002, 11003, 11004, 11005 | Five paired trial blocks; every arm runs every seed |
| Expert attempt resets | `1000000 + 1000 * trial_index + attempt_index` | Same ordered attempt stream within a block; G1 is its first 32 accepted episodes |
| Selection development | 210000-210019 | 20 scenes for each scheduled checkpoint |
| Candidate confirmation | 310000-310049 | 50 different development scenes for the same frozen candidate |
| Final test | 610000-610199 | 200 scenes, opened only after checkpoints/configuration are frozen |
| Timing/integration pilots | Learners 91001/91002; attempts from 9000000/9001000 | Never contribute to formal aggregates or access final scenes |

Trial indices are zero-based, 0-4; attempt indices are zero-based, 0-319.
Allow at most 320 expert attempts per trial (five times the largest 64-episode
budget), counting failures. Failed attempts are not replaced with handpicked
initial states. Record each reset seed, attempt ordinal, accepted ordinal,
initial observations, payload hash and rejection reason. The shorter arms stop
after their first 32 accepted episodes. Learner, producer and sampling RNG streams
must be separate; the current single `--seed` coupling needs an implementation
extension. Replay diagnostics also isolate augmentation RNG from data sampling.

Every arm in a trial block has the same parameter initialization seed. G1 payload
and normalization hashes are checked, not assumed equal from matching seeds.
For system comparisons, use the same initialization/statistics procedures and
report payload/value differences as part of acquisition variability; exact data
identity is a stronger, separate label. Mechanism comparisons require identical
payloads, model state and normalization values. Any difference in policy settings,
normalization *rule*, task, controller, precision or resource budget invalidates
comparability in both categories. The reporter must distinguish these categories
explicitly rather than silently relaxing its existing strict matching gate.

Use a fresh evaluator process/world per evaluation panel, a fixed scene order
and reset hidden state per episode. This prevents previous checkpoints or
confirmation calls from changing a later panel's simulator history. Record
initial measured states; matching observable resets is not a claim that every
latent backend state is bitwise identical. Model selection never reads final-test
outcomes. Do not reuse test results to revise the learner and retest on the same
panel as if it were new.

## 5. Budgets, evaluation and candidate selection

Use one physical GPU per timed run and a fixed 16-logical-CPU affinity allocation,
64 GiB host-RAM ceiling, Torch/OMP/MKL thread settings and local storage path.
Pin the actually available core IDs and GPU UUID before execution. Set
`CUDA_VISIBLE_DEVICES` before process creation and use local `cuda:0` for all
owned CUDA clients. Verify with per-PID device accounting that native libraries
have not allocated a second physical GPU. All timed arms use the same allocation.
Disable Viser, cameras and videos. Run display demonstrations separately.

Optimization stops at 10,000 updates, terminal failure or a common wall-time
limit H, initially 3,600 seconds. Each arm separately caps data production;
reaching that cap does not stop optimization on available data. Pilot timing must show that Offline-64 has
meaningful training time inside H. If a pilot shows H is infeasible, set
`H = 300 * ceil(max(3600, 1.25 * max_pilot_complete_seconds) / 300)` for **every**
arm before formal protocol freeze. This adjustment uses runtime, not final quality.
Never extend one failing run after its cap. No main trial runs before H is frozen.

Scheduled selection panels occur at updates 0, 2,500, 5,000, 6,500, 7,500 and
10,000 in every arm. The G2 publication barrier before update 6,251 therefore
precedes the 6,500 panel in the online arm. Keep comparable batching/checkpoint/evaluation policy across arms;
data waits are a measured outcome, not an excuse to remove an evaluation cost.
If a cap arrives before a result is published, it cannot affect in-budget model
selection. Partial panels do not certify success. Include process startup and
panel teardown in resource/time accounting; cleanup after a deadline is separate.

Use a **same-checkpoint confirmation gate**, replacing the old requirement for
two different consecutive checkpoints. When selection success first reaches
18/20 and no candidate has been confirmed, evaluate that unchanged checkpoint
on the 50 confirmation scenes. At least 45/50 confirms it. Allow at most two
confirmation attempts at distinct scheduled checkpoints; costs of failed attempts
remain in the run. Once confirmed, freeze that first-ready checkpoint. Later
training continues to measure retention, without replacing this time-to-readiness
candidate. Confirmation is development data, not a population success guarantee.

At the stopping boundary, freeze three policy identities:

1. the first independently confirmed candidate, if any;
2. the best checkpoint whose complete selection result was published by the cap
   (success, lift, negative goal error, then earlier publication as tie-break);
3. the latest completed checkpoint saved by the cap, whether or not it looks good.

Hash model parameters, normalization, execution settings and controller schema.
Evaluate all unique frozen policies on the 200 final scenes after training.
Identical policy identities are tested once, not counted as independent evidence.
Passing the task target requires at least 160/200 complete successes. No failed
first-ready candidate can be replaced using test results. Test failures do not
invalidate an independently selected best policy's result; both are reported.

## 6. Time origins and primary metrics

**Cold end-to-end:** t0 precedes per-run worker/simulator startup, task-data
production, task-specific warm-up and dataset writing/loading. Common software,
assets and one documented expert-qualification campaign can precede t0 for every
arm. Record cache/JIT state and balance run order; do not call reused software
caches a completely cold machine. **Data-ready:** a separate workload with files
already available, starting before loader/learner initialization. **Resident
microbenchmarks:** preparation is excluded for both providers and reported
separately. These are different estimands and never share one speedup headline.

| Metric | Definition / decision role |
| --- | --- |
| First-update latency | t0 to completion of optimizer update 1; includes readiness and initialization |
| Candidate-ready time | t0 to publication of a complete selection + confirmation pass for the same frozen checkpoint |
| Verified candidate attainment | Whether that candidate also gets >=160/200 in final evaluation; otherwise verified time is null |
| Best-policy success at budget | Final success of the preselected best checkpoint; primary quality guardrail |
| Latest-policy success and regression | Latest final success, plus best-minus-latest percentage points; never hide forgetting with best selection |
| Quality/time area | Left-constant integral of best published selection success divided by H; no interpolation/look-ahead; an early-finished policy remains available to H |
| GPU/CPU/memory/storage cost | Allocated physical GPU-hours, owned CPU core-seconds, sampled process-tree PSS/native VRAM, dataset bytes and retained checkpoint/log bytes separately |
| Supervision/sample efficiency | Completed valid future targets and distinct supervised source frames through candidate readiness and through the stopping boundary |

Candidate-ready time confirmed retrospectively by the final test is not the
wall-clock time at which final test certification was delivered. Report final
verification elapsed time/GPU-hours separately and report actual full-run-to-final
publication time when discussing delivered results. Do not simply add overlapping
production, training, evaluation and waiting intervals.

For non-attainment record `no_development_candidate`, `confirmation_failed`,
`candidate_failed_final_test`, `budget_limit` or `pipeline_failure` explicitly.
Do not invent T=0, discard failures or treat failed test verification as an
ordinary successful finite time. Report all seed-level results. A failure-penalized
cost may use ready-time for verified successes and H for every non-attainment;
label it a composite score, not an estimate of the true censored learning time.
Paired time ratios are descriptive for verified-success pairs only unless all
predeclared pairs attain the target. Show attainment counts beside any such ratio.

Operational diagnostics include batch-service p50/p95, blocked-data time,
producer accepted/rejected frames per second, generation exposure, publication-to-
consumption age, unused frames at eviction and G1/G2 teacher-state error before/
after refresh. Missing measurements stay null. The latter retention audit uses
training data only and performs no optimizer update. Failure videos/records are
retained; visual examples do not select the reported evaluation episodes.

## 7. Systems/HPC measurement contract

Follow `implement-performance` for actual hot-path changes and `benchmark` for
component measurements. The microbenchmark convention of excluding setup does
not apply to the primary cold-start experiment. No optimization may change
precision, masks, data ordering or the task's acceptance conditions to gain speed.

Profile the actual B=256, history=2, future=64 workload, plus B=64/1024 and
capacities 32/64/160 for component scaling. Separate:

`selection/lock -> copy -> geometry encoding -> H2D -> target encoding ->
forward/backward/optimizer -> completed-update publication -> telemetry`.

Currently ODS packs 51 geometry features from structured raw observations per
batch, while offline corpora already contain those features. Their direct batch
latencies therefore include different preprocessing work. Keep those costs in
practical system results and label them. For pure delivery comparisons, use the
same raw or packed schema, layout/strides, transform placement, masks and copy
ownership in both paths. Do not disable an efficient offline cache to manufacture
a disk-I/O bottleneck. Conversely, do not move ODS-specific preparation outside
its timed interval. The same requested model input must be numerically identical.

Use monotonic host timestamps and CUDA events/completed-update windows. Record
H2D separately from GPU computation. Do not measure asynchronous CUDA work using
unsynchronized dispatch time, or claim actual kernel overlap from host intervals
alone. An overlap claim needs device traces for all participating processes;
without them report host scheduling overlap as a distinct proxy.

Measure telemetry overhead with and without detailed provenance tracing. The
current per-target Python unique-frame accounting visits roughly 148 million
labels in a reference run and is a candidate for profiling, not an assumed-free
measurement. If material, use bounded/vectorized exact bookkeeping with tests
for gaps, padding, overwritten rows and source identity; verify it against the
reference accounting. Apply the same instrumentation policy to every arm and
measure its overhead. Full manifest tracing belongs to a labeled diagnostic run.

Measure both framework allocator peaks and native per-device usage. PSS samples
avoid summing shared-memory RSS repeatedly; report sampling frequency and avoid
calling a sampled peak exact. Distinguish data memory from full-process memory.
Include writer queues, temporary exports, snapshots and any replay instrumentation
in their respective memory/storage totals. Record actual GPU UUIDs, CPU affinity,
thread pools, background load, software/source hashes and local cache conditions.
Run primary trials sequentially by default. Parallel campaign execution requires
disjoint GPUs/CPU allocations and a pilot showing its contention impact; it is
not automatically equivalent to the uncontended deployment.

## 8. Mechanism diagnostics and reuse

**Chunk trace replay:** freeze an ordered anchor manifest with episode/content
identity, frame index, context shape, valid/loss-mask digest, generation and an
independent augmentation RNG stream. Store episode validity/continuity maps once;
avoid repeating whole contexts or per-target Python tuples just to record identity.
Replay exactly the same input/target/mask bytes in live/shared and LeRobot-resident
providers. Check model/optimizer parity over fixed deterministic updates before
timing the diagnostic. Initial parameters and all normalization values must match.
Current exact-manifest replay supports MLP frames only; this is new work.

**Controlled overlap:** freeze the consumption schedule using a separate pilot.
Before overwriting a row, a producer retirement guard must ensure its last planned
old-generation use has completed. Consumers wait for required new publications;
old data is never silently substituted. A serialized arm disables production/
optimization overlap while preserving that event order and every sampled context.
This requires bounded coordination, not unbounded replay. Verify data hashes and
update parity. If independent regeneration or row availability cannot satisfy
this contract, retain only a conditional trace-replay diagnostic and withdraw
the isolated causal-overlap claim. The current `--mode serial` is not that arm.

**Data reuse:** report K=1,5,10 workloads. Offline pays collection/persistence once
and loading/training per job; fresh ODS pays live generation per job. Account for
actual makespan and allocated resources, not a sum of overlapped intervals. Mark
K=10 as a projection unless actually executed. If ODS archives data for reuse,
measure an archive-enabled variant with its queue, copying and persistence costs.
Ephemeral ODS and durable LeRobot provide different services; zero dataset disk
usage is not a like-for-like durability win. State-only results do not establish
image-policy or video-decoding performance.

## 9. Statistics, claims and expected output

Run five independent paired trial blocks and all four arms in each block. Freeze
a balanced/randomized arm order before measuring. Publish per-seed data, medians,
ranges and paired differences. Resample whole trial blocks for paired uncertainty;
do not treat 200 evaluation episodes as 200 independent training seeds. Show
within-policy binomial uncertainty separately. Five seeds have limited power:
wide intervals mean insufficient evidence, not permission to drop a seed. Use 10,000 paired block-bootstrap replicates (RNG seed 81721) and 95%
percentile intervals, explicitly labeled exploratory with only five blocks. Any
extension is declared before opening its final results and labeled separately.

An optional practical target is >=20% lower median verified-candidate time than
a baseline, with no more than 5 percentage points of selected-policy quality
loss. These are preregistered decision thresholds, not promised outcomes. Require
all five pairs to attain the verified target for an unqualified paired time ratio;
show uncertainty and do not claim superiority if it includes no gain or a material
quality loss. Report Offline-32 and Offline-64 separately. A sustained-learning
claim additionally requires latest-policy success >=80% and regression <=5 points;
the existing 65/100 later checkpoint would fail this rule.

The final report contains: all-arm cost/memory and quality/retention tables,
claim-eligibility evidence, quality versus cold wall time and GPU-hours, a
producer/learner/evaluator timeline, generation exposure/retention plots and
reuse-cost curves. Component benchmark scripts follow the repository's standard
three-table report. End-to-end reporting retains the larger study's failure,
selection and verification evidence rather than a single opaque ranking.

## 10. Implementation readiness and execution order

Implemented: the physical task/teacher, chunk learner, ODS context masks and
provenance, manual refill, frozen/numeric offline modes, LeRobot SDK export/load,
model/normalization fingerprints, checkpoint/resume, basic timelines and failure
records. The current fixed-split example is not a complete v2 experiment runner.

Required before formal execution:

1. Protocol loader/freeze manifest; independent learner/attempt RNGs; configurable
   fresh panels; same-checkpoint confirmation; per-panel fresh evaluators;
   deadline-safe publication; best/latest/first-ready identity freezing.
2. Direct accepted-episode LeRobot persistence with bounded async writing, exact
   read-back and no compulsory `.pt` bounce. Preserve the optimized in-memory
   reader for the state-only baseline.
3. System-versus-mechanism comparability rules and all-arm result aggregation,
   including latest-policy regret, failed confirmation/verification and workload
   reuse. Extend chunk context replay before claiming exact supervision matching.
4. Component profiling, telemetry-overhead audit, complete GPU/CPU accounting and
   preprocessing attribution. Keep unimplemented age/eviction/overlap metrics null.
5. Tiny numerical/process tests for context identity, masks, writer error/drain,
   generation gates, failures/censoring, hash mismatches and teardown. Reconfirm
   the reference policy path on development data after any execution-layer change.

Each of the two timing pilot blocks exercises all four arms without final-test
access. Freeze the five-block arm order from the companion JSON; the first four
blocks balance each arm across execution positions.

Order: engineering preflight -> two timing pilot blocks without final tests -> freeze
source/protocol/resource hashes and H -> 20 primary runs -> freeze candidate
identities -> final evaluation -> locked reporting. Do not fix regression and
repeat the same final panel until it passes. Keep failed runs; retry a documented
external infrastructure failure at most once: require an independently logged
host interruption or unrelated device failure; retain both attempts and charge
the failed attempt to campaign cost. ODS worker/writer errors remain pipeline
failures. Never retry a policy failure as an infrastructure error.

At H=60 minutes the training-side allocation ceiling is 20 GPU-hours. There are
at most 60 unique policies for final validation (20 runs x 3 candidates), hence
up to 12,000 final episodes. At an illustrative 6-12 seconds per episode this
adds 20-40 GPU-hours, before pilots and cleanup. This 40-60 GPU-hour planning range
is an extrapolation, not a measured study duration; update it from timing pilots.
Confirmatory evaluation can dominate such a small state learner. No long study
is launched merely by saving this design.
