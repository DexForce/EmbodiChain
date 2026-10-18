# Protected ODS configuration: regime and learning tests

These tests validate the configuration method and source-level sampling bounds.
They do not select a universal optimal tuple. Real learning uses frozen revision
`6d2ff529`; a separate default-device compatibility fix is `69d65ecd`.

## Engineering coverage

The new regime suite passes 580 tests: 576 deterministic input combinations plus
public-sampler boundary cases. Of the matrix cases, 414 satisfy the constraints
and 162 are explicitly infeasible. Exact rational and independent finite-search
checks cover supply-limited, learner-limited, latency/memory-limited and integer
budget regimes. A combined core/configuration/regime run passes 792 tests.

An independent review found CPU quota temporaries inherited the application's
PyTorch default device. Two new regressions fail on the old implementation and
pass with explicit CPU/candidate-device factories. Existing CPU sampling payloads
(except deliberately fresh run IDs), RNG state and delivery counts match exactly
before/after the narrow fix, in both nonbinding and depleted-quota cases. The
running learning trials were never edited or restarted for this fix.

## Fixed learning protocol

Use Franka single PickPlace, ChunkBC pose H64/action16, P8/C32/B256 and 256 accepted
trajectories. Assign each trajectory a hard budget of 1024,2660 or 8192 chunks;
run learner seeds 16001 and 16002 for each. Producer/sampler seeds are independently
derived and fixed across quotas. Initial model parameters and all normalization
buffers match bitwise across the three settings within each seed.

Each run evaluates initialization, the last three generation checkpoints and
latest policy on 20 fixed development states 810000–810019, in fresh processes.
No formal test or 100-state confirmation panel is used. Two training seeds reuse
the same 20 states; they must not be pooled as 40 independent evaluation scenes.
Profiles remain explicitly provisional.

| Per-trajectory cap | Seed | Latest dev20 | Best dev20 | Largest observed drawdown | Updates | Assigned budget consumed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16001 | 18/20 | 20/20 | 10 pp | 984 | 96.09% |
| 1024 | 16002 | 18/20 | 19/20 | 20 pp | 988 | 96.48% |
| 2660 | 16001 | 20/20 | 20/20 | 0 pp | 2559 | 96.20% |
| 2660 | 16002 | 20/20 | 20/20 | 0 pp | 2556 | 96.09% |
| 8192 | 16001 | 20/20 | 20/20 | 5 pp | 7953 | 97.08% |
| 8192 | 16002 | 20/20 | 20/20 | 5 pp | 7954 | 97.09% |

All six original runs pass the hard-cap, source-identity, delivery/completion,
final-checkpoint and cleanup audits. Every checkpoint panel retains its failures.
The four latest-policy failures across the two 1024 runs all lifted the cube,
but final goal errors were 3.11–4.26 cm, exceeding the unchanged 3 cm task criterion.
This establishes a failed placement condition, not a complete causal explanation
of the policy regression. No success criterion was relaxed.

## What the configuration method must account for

1. **Execution of limits and learning sufficiency are separate.** All caps hold,
   including when a low-budget checkpoint drops to 12/20. Throughput alone cannot
   qualify a learning-safe lower bound.
2. **Assigned budget is an upper bound, not guaranteed supervision.** Across these
   runs about 96–97% of assigned chunks are delivered. Sequential full replacement
   retires unused allowances; a final remainder smaller than a batch cannot be
   consumed. Use delivered/completed counts in efficiency comparisons.
3. **Separate measured capacity from enforced rate.** First-seed wall consumption
   closely tracks its requested pace. This validates pacing execution, not an
   independent proof that the original two capacity estimates are invariant.
4. **Re-measure when actual supply changes.** The1024/16002 run observes 0.7835
   accepted trajectories/s against the input 0.9326 and accumulates 44.22 s of quota
   waiting. The engine waits rather than violating source caps. A dry-run
   recalculation using the new observed rates preserves K=1024 and lowers target
   consumption from 955.0 to 802.3 chunks/s. That corrective configuration has not
   itself been validated in a new run.
5. **Preserve attribution limits.** An external native GPU process was observed
   only at startup of 1024/16001. No external native GPU process was observed in
   1024/16002; the cause of its slower supply remains unknown. A preliminary
   progress report conflated the two, and the diagnostic amendment records the
   correction without rewriting the original declaration. Other host activity
   was not fully isolated. Timing labels are monitoring scopes, not exclusivity
   guarantees.

## Targeted repeat

One same-seed, same-budget repeat of 1024/16002 on another idle GPU was declared
separately. It preserves all model/data/sampling seeds and input settings, and
is not a third independent training seed or a replacement for the original
failure. The comparison tests sensitivity to run/placement conditions; it is
not a proven contended-versus-clean comparison. The repeat finished with20/20 latest and best success, but its full curve was
0/20 →20/20 →14/20 →18/20 →20/20: a30-point observed drawdown. The original same
seed finished18/20 with a20-point observed drawdown. The repeat's initial model
and normalization match exactly; asynchronous sampling and checkpoint-update
positions do not. A perfect final checkpoint therefore does not establish
training stability, and the original slower supply is not necessary for observing
regression at this budget.

Repeat productive supply was0.9422 trajectories/s, wall consumption954.7 chunks/s
against target955.0, with no quota/data waits. No external native GPU processes
were observed. All checks and normal shutdown passed. These observations do not
identify a single cause of the original slowdown or the learning regressions.

## Recommendation boundary

Keep numerical profiles provisional. The two larger tested budgets pass this
small development screen more consistently; that does not qualify every value
between them, other models/tasks, longer trajectories or unseen initial states.
Finite training capacity also matters: for N published trajectories, the most
possible completed updates is floor(N*K/B), and actual updates are lower after
unused quota retirement. With N=256 and B=256, K1024 permits at most 1024 updates,
while K2660 and K8192 permit 2660 and 8192. Throughputs determine the pace of this
work, not how much learning is necessary. Choose the total production/training
budget from a task learning protocol; do not assume that lowering K preserves
an earlier update budget or that legacy defaults remain sufficient.

The lower tested budget needs further qualification before serving as a stable
recommended minimum. The configurator should continue honoring its explicit
profile and hard cap, expose observed supply/consumption and wait reasons, and
require recalibration rather than silently raising reuse to fill the GPU.

## Reproduction and audit

Seven runs generated1,792 accepted trajectories and consumed6,137,856 chunks.
Every per-source cap, model update, exact development panel, source snapshot and
final checkpoint reconciled independently. All70 observed owned process
identities exited. Allocated run wall time totaled2.199 GPU-hours, including
startup, training, posthoc evaluation and cleanup; CPU tests and other campaigns
are excluded. No further runs are queued.

The configuration report now exposes `training_budget`: assigned chunk ceiling,
quota-limited update ceiling, requested update cap and maximum additional
updates. This is an accounting bound, not a convergence target, and unused
retired allowances reduce actual work.

[Machine-readable evidence](ODS_QUOTA_TEST_RESULTS.json) includes the complete
protocol, amendment/correction, six-run and replication results, source and raw
artifact identities, rate recalculation, test counts and cleanup audit. Original
commands and logs remain under `/tmp/ods-quota-validation/`.

Project context was reviewed: the device fix enforces the existing CPU shared-
buffer contract; no task registration, RL or Task Program behavior changed.
The existing online-sampling context remains accurate.
