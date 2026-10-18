# ODS production, reuse and stability tuning

Status: implementation and staged experiments in progress, 2026-10-18
(Asia/Shanghai). This study precedes [the formal offline comparison](BENCHMARK.md).
It does not establish a universal optimal configuration or an ODS speed advantage.
See [initial findings and limitations](TUNING_FINDINGS.md) for measured evidence.

## What is controlled

Keep the measured Franka task, revised expert, DLS controller, ChunkBC, 64-step
prediction/16-step execution, G1-only normalization and optimizer unchanged.
Use one producer process with P vector simulation environments, one learner,
and the existing full-cache, per-row replacement. Evaluation runs after training
in fresh B=1 processes so it does not alter production/learning contention.
No partial replacement, historical replay, DAgger or changed success threshold.

Vary P, cache C, batch B and anchor reuse budget r. An anchor is the current
trajectory frame selecting two causal observations and up to 64 future targets.
One draw is one chunk, not an entire episode. Overlapping chunks may supervise
the same frame repeatedly. Padding and invalid/continuity gaps are excluded.

## Measurements and configuration rule

Track sampled and optimizer-completed usage separately, with immutable
run/episode/frame identity. Record per-episode anchor histograms and supervised
frame histograms, including zero-use frames at eviction. Report p50/p95/max,
unique coverage, generation exposure, first-use delay and sample age. Sampling
is episode-uniform, so shorter episodes can have higher per-frame reuse.

Record accepted trajectories/s, valid anchors/s, attempted/failed trajectories,
whole-vector batch failures, startup, first-publication and full-refill latency.
The worker currently requires every vector environment to succeed; rejected good
members of a failed batch are not counted as useful published throughput.

Let lambda_f be actual published anchors/s, u actual completed updates/s and B
batch size, measured while producer and learner share the chosen GPU:

- Natural reuse estimate: `r_natural = B * u / lambda_f`.
- Sustainable update rate for a chosen reuse budget: `u_target = r * lambda_f / B`.
- Initial update allowance: `floor(r * A0 / B)`, where A0 is the initial anchor count.

These are steady-state workload estimates, not convergence guarantees. A
resident synthetic trainer measures contention separately; its updates/s must
not replace the full learner's rate in a deployment recommendation.

The optional anchor-budget scheduler credits each publication once by r times
its eligible anchors; caps stored credit at max(B, r*A0); debits only completed
updates; and waits when a batch cannot be funded. Request refill when remaining
trainable time is at most 1.25 times p95 request-to-first-publication latency.
Cold-start estimation and actual measurements are labeled separately. This is
an aggregate budget, not a hard per-episode or per-frame reuse cap.

The current legacy threshold counts chunk draws. It is not a frame epoch count.
Manual scheduling of 5,000 updates at B256 on 32 episodes draws 1.28 million
chunks, averaging 40,000 selections per episode before the first refill.
Whether this is excessive must be established by physical evaluation.

## Staged study

1. Qualify P=1,2,4,8 on 40 scenes (at least38 successes), then test actual ODS
   generation. Measure 64 accepted episodes, three repeats, with and without
   a resident ChunkBC contention workload. Use one physical GPU per cell.
2. Controlled resident-data diagnostic: G1/G2 each32 accepted episodes; initial
   reuse8/32/128; branch the identical model/optimizer/RNG state to continue G1
   or replace with G2. Each branch receives32*A_G1 additional chunk draws,
   rounded up by at most one batch. Two initialization seeds12001/12002.
3. Live screen C=16/32/64 and r=8/32/128, B256, one screen seed13001, C>=2P.
   Each cell caps production at256 accepted episodes and training at50,000
   updates or90minutes. Include C32 fixed-5,000-update refresh as reference.
4. Confirm the top two plus reference with new seeds14001/14002/14003. The
   latest checkpoint must reach80% on100 development scenes in every seed,
   lose at most5points relative to its development-selected best, and consume
   at least four generations. Test B64/1024 as secondary sensitivities; an
   alternative needs the same three-seed confirmation before recommendation.

Before selecting confirmation candidates, run the secondary batch sensitivity
on the exploratory C32/r8/P8 candidate: B256 control, B64, then B1024, each with
256 accepted episodes and screen seed13001. Freeze one source revision and use
the same GPU with local NUMA CPU affinity for all three. This repeats the B256
control because the original grid used an earlier reporting revision and its
GPU5 queue used remote NUMA affinity. Use `--eval-every 50000` for all three,
leaving initial, last-three-generation and latest policy panels; otherwise B64
would get more intermediate evaluations and model-selection opportunities.
Fixed r holds the chunk budget nearly
constant, not the optimizer-update count; report both. This exploratory choice
does not establish C32/r8 as the confirmed winner.

Primary deployment is shared single-GPU production/training. Separate GPUs are
only a diagnostic for contention. Pin GPU UUID through CUDA_VISIBLE_DEVICES
before spawning and verify native allocation. Concurrent independent runs use
disjoint CPU affinities; record host interference and do not claim uncontended
measurements without supporting evidence.

Record GPU-to-NUMA topology and distinguish logical CPUs from physical cores.
Do not change affinity or executable source midway through a frozen queue.
Timing from remote-NUMA or externally contended runs remains diagnostic and
must not silently be pooled with local-NUMA confirmation measurements.

Development selection uses810000-810019; configuration confirmation uses
820000-820099. Formal comparison panels210000,310000,610000 remain untouched.
Historical100-scene panels50000 and710000 are already exposed. The resident
causal study reuses the archived numeric corpus from the feasibility run and
therefore is a mechanism diagnostic, not an independent data-acquisition trial.

The initial diagnostic fixes continuation reuse at32. A separately declared
exploratory extension uses initial128/continuation128 for both branches and the
same two seeds, to distinguish a short adaptation interval from persistent
regression. Its prefix model/normalizers must match the original128 checkpoint
bitwise. It reuses development scenes and is not an independent confirmation.

## Executable diagnostics

Use the repository's configured Python/runtime. Set CUDA_VISIBLE_DEVICES to an
idle physical GPU before the process starts; commands use local cuda:0.

```bash
python -m scripts.benchmark.data_pipeline.benchmark_ods_usage --help
python -m scripts.benchmark.data_pipeline.benchmark_ods_reuse_causality \
  --phase train --source /path/to/two-generation/corpus.pt \
  --output /path/to/new-run --seed 12001
python -m scripts.benchmark.data_pipeline.benchmark_ods_reuse_causality \
  --phase evaluate --output /path/to/new-run --scenes 20
```

The causal driver loads a schema/hash-validated corpus, freezes G1 normalization,
uses independent CPU sampling and CUDA augmentation RNGs, restores identical
optimizer/RNG states at each fork, and evaluates each immutable checkpoint in a
fresh simulator. Training time excludes acquisition and post-training evaluation.
It retains every failure. Report tables include all evaluated policies; missing
memory measurements are explicitly unmeasured.

The reporting revision records checkpoint-save completion separately from
optimizer completion and evaluation completion. A policy later found to pass
80% is a retrospective result, not an online time-to-certification. Older runs
without save events retain unknown save times. TensorBoard uses the evaluated
checkpoint's policy step even when evaluations happen after training. Report
the evaluated steps alongside maximum observed drawdown: unequal panel counts
do not provide equally sensitive regression detection.

Resource sampling runs every five seconds and at explicit lifecycle boundaries,
including before worker startup. It records service time and missing processes.
Process-tree PSS is a sampled peak; CPU counters omit work after the last sample;
Torch allocator counters belong only to the learner and exclude native simulator
allocations. Unavailable memory is null, never zero. Forced sampling and periodic
sampling overhead are measured separately; component overhead checks do not
establish the combined instrumentation cost.

## Evidence already available, not a tuning conclusion

A separate, read-only historical campaign under `/tmp/ods-budget-ablation`
compared a1,000-selection-per-episode cap. With parallel production, the B256
capped cell reached29/100 latest success, whereas B32 reached99/100 using the
same64,000 total contexts but2,000 instead of250 optimizer updates. Its unlimited
parallel cell reached97/100 latest success. The runs used one learner seed,
serial generation barriers and independently regenerated data.

This is evidence that a sampling cap also changes optimization opportunity;
it does not prove high reuse always hurts or that batch32 is optimal. The new
study records both total supervision and update count and tests live overlap.

## Final deliverable and recommendation gate

The final guide must supply measured P/C/B/r ranges, first-fill cost, refill lead
time, achieved reuse distributions, quality/retention, hardware and source hashes.
Choose among configurations passing stability first; compare wall time/resource
cost second. Differences under5% prefer fewer production environments, then a
smaller cache. If none pass, state that tuning within full replacement has not
yet produced a stable recommendation; do not relax the task or hide late failure.

Publish raw records, failed runs and independent-seed results, not only a winning
configuration. Formal comparison is re-frozen after tuning. Shared learner
changes apply to every comparison arm; offline acquisition receives the same
resource budget and validated producer recipe. Tuning cost is reported separately.
