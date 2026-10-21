# Supported-environment NeRD evaluation data — 2026-10-20

**Historical data-provenance report.** The later user instruction stopped the
forty-cell full-budget queues and their evaluation waiters, preserving all
artifacts. The [small-budget original-entry report](neural-dynamics-small-budget-20261020.md)
is authoritative for the current task's completed training and accuracy results.
This report records the earlier supported-environment data work and historical
execution plan. Project-wide trained-model qualification remains unclaimed.

## Corrected evaluation boundary

The preceding workflow generated official training data in Python 3.8/Torch 2.2,
but the historical evaluator regenerated its physical teacher trajectories under
the model-evaluation Torch 2.7 process. The pinned Warp version alone did not
establish that those comparison data came from the supported environment.

The new standalone `official_evaluation_data.py` runs unchanged original
sampling, physics and torque APIs under **Python 3.8.20, Torch 2.2.2+cpu,
NumPy 1.24.4**, using Warp 1.8.0 for Cartpole and Warp 1.5.1 for Ant. It loads only
the explicitly hashed pure Torch coordinate/observation files under an isolated
namespace; it does not import the EmbodiChain package, DexSim or Newton. The
original checkout remains at `370e89d2` with no tracked source changes.

Every shard preserves original raw state and applied controls, canonical q/qd
and canonical controls. All requested outcomes remain, including physical
failure and NaN/Inf; there is no rejection or replacement of evaluation rows.
Generation stores actual physics arrays, source/asset/code hashes, versions,
logical trajectory identities and independent per-trajectory seeds.

Both robots' passive, random-effort and fixed-PD corpora are completed: six
original-domain corpora × 2,048 trajectories × 1,000 steps. Combined with the six
already frozen current-Newton corpora, this is **24,576 trajectories and
24,576,000 macro steps**. These are data-generation results, not model accuracy.

All shard hashes were rechecked. For each domain/robot the initial states agree
across all three scenarios; original raw initial states also agree exactly.
The source-domain holdout audit reads all 47,528 train/validation/test initial
states across the four frozen datasets and finds zero duplicates against the
new 2,048 initial states per robot. The native-domain audit has the same result.
Seeds remain 1,000,000 through 1,002,047. Exact duplicate checks do not establish
statistical independence.

## Historical replay behavior

`reference_rollout` accepts `teacher_dataset` for these supported-environment
traces. It checks versions and isolation, original source/asset identity,
trajectory order/count, timestep/codec, measured physical arrays, raw/canonical
state agreement, and recorded feedback actions. Rehashed inconsistent controls,
coordinates or missing measured gravity are rejected.

The original neural integrator resets from the recorded initial raw state and
then evolves independently. It recomputes closed-loop actions from its own
state. Later ground-truth states never enter the neural state. Original teacher
states and controls are read from the frozen data rather than simulated again
under a different Torch version.

Model evaluation remains in the declared EmbodiChain Torch 2.7 environment with
the original pinned-Warp integrator. Data-generation versions and model-runtime
versions are recorded separately. This is a controlled training-port comparison;
it does not claim historical Torch 2.2 training or literal paper reproduction.

## Executed validation

The affected reader, historical live/replay runtime, comparison and reporter
tests passed **55 checks** in 18.27s, including actual trusted-source execution
for both robots and all three scenarios. They cover incomplete/incorrect
provenance, reordered rows, raw shard tampering, rehashed dtype/length/control
or coordinate changes, missing physics, padded partial batches, ownership and
retention of nonfinite states. The older broader 276-pass regression is separate
evidence; overlapping test counts are not added.

Additional 8-trajectory / 1,000-step diagnostics used the earlier 1,000-update
source-trained pilot exports, outside the formal 120-case matrix. Both replayed
the exact supported-environment teacher-content prefixes. Measured raw-to-
canonical transport differences were **zero** for both robots, and model/source
physical verification completed. Neither diagnostic produced nonfinite states.
Poor pilot physical outcomes remain visible: Cartpole teacher/student failed
7/8 versus 7/8; Ant 7/8 versus 8/8. These are pipeline diagnostics and do not satisfy
the full accuracy requirement.

Black 26.3.1 completed over the worktree; the standalone producer also compiles
under Python 3.8. The read-only API gate reports 2748/2748 exports documented,
missing 0. Context and whitespace checks pass. No Sphinx source change was
required for the private reader or the existing evaluator's optional setting.

## Previous frozen execution, superseded by the user

The evaluation-only snapshot is `implementation-eval-v3` with 1,572 hashed files.
All 16 training-critical model/codec/cache/sampling/trainer/native-collection
files were byte-identical to the training `implementation-v2`. The old frozen
specifications, including the 4,995,000-update budgets, remain unchanged as
historical artifacts; their processes have now been stopped.

The v3 protocol retains 120 cases: 40 full-budget models × 3 scenarios, each using
2,048 trajectories at horizons 1/10/100/1000. All 120 teacher-data readiness guards
passed. The old idle v2 evaluation waiter was stopped before any case began;
the subsequent v3 waiter and finalizer were also stopped under the new instruction.
Their full-budget guards were not relaxed to relabel partial runs. Large final
rollout traces retain `/workspace/exp_data` access paths backed by separate NVMe.

The previous GPU 0–6 allocation and deferred reservations remain archived. The
new scope stopped eight training processes and fifteen scheduling/waiting
processes owned by this task, without changing other tasks. All eight committed
checkpoints were inventoried with hashes and saved update counts. No old queue
or full-budget evaluation remains active.

The larger forty-model/120-evaluation, five-seed qualification is future work
outside the new round's completion criteria. The new round uses four main
1,000-update runs and a small fixed held-out rollout cohort. Its measurements
cannot be relabeled as the old full-budget acceptance.

The learning-curve reporter now retains the first/final observations and
samples by update spacing. Its previous modulo rule discarded nearly all sparse
500-step logs; an actual 1,614-record file yielded one plotted point. The fix
passed 33 report/comparison checks, and both completed pilot comparison JSON
files remained byte-identical before/after the plot-only change.

The now-stopped independent figure finalizer was designed to wait for all
full-budget models and 120 verified evaluations. It acquires the writer's lock before
refreshing shared report images, and requires unchanged statistical report
bytes. The final bundle adds timestep q/qd/base/quaternion errors, failure and
nonfinite rates, fixed-PD returns, PNG/SVG and exact per-model CSV. Seed ranges
are explicitly labeled min–max, not confidence intervals; undefined values
remain gaps. Its actual pilot rendering is marked diagnostic and unqualified.

[Machine-readable evidence](neural-dynamics-supported-reference-evaluation-20261020.json)
binds the actual data, versions, snapshots, guards and observed processes.
The [reproduction workflow](../../scripts/experiments/neural_dynamics/README.md)
documents the two isolated environments and commands. This report supersedes
the evaluation execution details in the
[CUDA alignment report](neural-dynamics-cuda-alignment-20261020.md); its training
arithmetic and earlier measured evidence remain valid.
