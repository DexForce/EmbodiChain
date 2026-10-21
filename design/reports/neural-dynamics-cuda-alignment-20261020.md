# NeRD CUDA alignment and full-budget evaluation — 2026-10-20

**Full model-accuracy qualification is still pending.** The real Ant CUDA
training-entrypoint residual is corrected, exact resumed execution is verified,
and the unchanged 40-cell full-budget training matrix is running. The official
supported environment generated the reference data; both trainers execute in
the clean EmbodiChain environment to control framework differences.

The [supported reference-evaluation follow-up](neural-dynamics-supported-reference-evaluation-20261020.md)
now records the active v3 evaluation protocol: historical teacher trajectories
are generated once in the supported environment and replayed, replacing the
v2 evaluator's live Torch2.7 teacher regeneration. Running training is unchanged.

## Actual training arithmetic

The earlier CPU comparison used byte-identical canonical inputs and obtained
zero Ant feature/target/prediction differences. A subsequent real CUDA probe
used the actual full-data training entry points, B512/H10 and original-width
network. Although the first loss was identical, the first Adam update differed
by **2.563931e-5** in one parameter. Target arithmetic differed by up to
9.536743e-7. Thus CPU forward parity alone did not establish CUDA training parity.

The corrected path retains the original scripted quaternion/body-transform
execution and flattened contiguous layout. Target construction and decoding
reuse the current unnormalized embedding, avoiding a second frame transform.
The reference adapter transports raw data to the model device before coordinate
math. The loss uses the original literal reciprocal-sqrt and native MSE
reduction. Cartpole wraps each position before differencing, as the source does.

CUDA preprocessing also specializes its scripted graph after initial calls.
Both trainers now perform three real-layout preprocessing warmups before any
optimizer update or restoration of RNG state. Checkpoints bind this execution
contract; optimizer resume from an older incompatible contract is rejected.
Direct checkpoint comparisons also match optimizer state and the Torch RNG
streams used by training; unused Python/NumPy global states are recorded separately.

| Actual diagnostic | Maximum parameter/buffer difference |
|---|---:|
| Ant, first CUDA optimizer update after correction | 0 |
| Ant, ten uninterrupted CUDA updates, original vs migrated trainer | 4.432135e-6 |
| Original Ant trainer, uninterrupted vs resumed ten updates | 0 |
| Migrated Ant trainer, uninterrupted vs resumed ten updates | 0 |
| Cartpole, ten CUDA updates, original vs migrated trainer | 0 |

The unchanged absolute update tolerance is 1e-5. These are implementation and
resume diagnostics, **not** trained-model accuracy results. The full data,
initial tensors, normalizers, sampling streams, update counts and acceptance
thresholds were not reduced to obtain them.

## Full training and reference scope

The immutable `implementation-v2` includes the corrections above. Two robot
plans each contain two teacher domains × two trainers × five seeds. Every cell
retains **4,995,000 updates**, B512/H10, six layers, twelve heads and the declared
192/384 embedding widths. All 40 cells remain assigned exactly once. Experiments
use physical GPUs **0–6** only. The interrupted GPU7 cell resumed from its actual
80,000-update checkpoint on GPU6; its four queued cells were also reassigned.

Reference training data remain the actual original-repository outputs generated
in Python3.8.20 / Torch2.2.2 / NumPy1.24.4, with Warp1.8.0 for Cartpole and
Warp1.5.1 for Ant. Separate train/validation/test raw files and their measured
physical arrays and hashes are retained. The original sampler actually produced
1,024,000 training transitions per robot. Current-Newton training uses the
separate frozen corpus, with 1,005,600 training transitions per robot.

Both trainers execute under Python3.11 / Torch2.7.0+cu128 in the resolved
EmbodiChain environment, with the unmodified numerical `VanillaTrainer` methods
as the reference. The original-source data environment is distinct from this
controlled trainer-comparison environment. This does not establish a historical
Torch2.2 training run or literal paper-recipe reproduction.

Fresh readiness reports find no full-profile discrepancies, verify both teacher
corpora and accept measured pinned-main numerical evidence. They correctly mark
training and rollout qualification **pending**. No partial checkpoint is used by
the full evaluation driver.

Actual partial validation curves are retained at
`/workspace/exp_data/embodichain-pr822-accuracy-20261020/qualification-v2/training-validation-progress.png`,
with the exact plotted records in the adjacent JSON. These seed0 curves show
training progress; they are not the five-seed final test comparison.

## Frozen evaluation data

Repeated GPU Newton teacher runs were not byte-identical even at equal seeds.
Therefore the native evaluation now generates one common corpus per robot and
motion scenario, then reuses its ordered state/control content for every seed
and both trainers. All 2,048 requested trajectories are retained, including
physical failures and nonfinite states. For fixed PD feedback, every recorded
teacher action is independently checked against its own teacher state and
sealed policy identity. The neural simulator still computes each action from
its own evolving state.

All six corpora have now completed: **12,288 trajectories / 12,288,000 macro
steps**. Every shard hash was rechecked, initial states agree across the three
motions for each robot, all60 native-case teacher guards accept them, and their
physics signatures match the training corpora despite different batch sizes.

The superseded evaluation seed500 overlaps the current training candidates'
per-trajectory seeds0–12,499. Before observing full-model evaluation results,
the v2 protocol changes evaluation to **seed1,000,000**, with trajectories seeded
through1,002,047. The audit reads actual recorded trajectory seeds rather than
only the dataset's top-level seed. It also reads the initial q/qd of every row
in all four train/validation/test corpora: each robot's 2,048 new initial states
has **zero exact canonical-state overlap** with either domain. Signed zeros and
Ant quaternion sign are normalized for this check. This is a duplicate audit,
not a claim of statistical independence.

The frozen evaluation has **120 cases**: 40 validation-selected full-budget
models × passive/random-effort/fixed-PD scenarios, each using 2,048 trajectories
and horizons1/10/100/1000. Historical models run through the original isolated
Warp integrator; current-domain models run the installed native NeRD provider.
Historical paired runs must also identify identical ordered teacher content.
The fixed PD baseline remains explicitly distinct from the paper's policy.

The driver verifies training completion receipts, full budgets, seeds, dataset
and sampling identities, model hashes, and native common-corpus hashes before
attaching measurements. Actual guard probes reject a real 1,000-update pilot,
an incorrect corpus seed/population/domain and physical GPU7. Final paired
error, failure, nonfinite and policy-return gates retain their original margins.

Separate native diagnostics also executed all four horizons through1,000steps
using these new PD traces and the earlier1,000-update workflow models. Their
selected teacher-content hashes match the exact eight-trajectory prefixes, and
the neural actions differ from teacher actions because each is evaluated from
its own state. These are pipeline diagnostics, not formal full-model evaluations:

| Pilot diagnostic at1,000steps | Teacher failed | Neural failed | q MAE | qd MAE |
|---|---:|---:|---:|---:|
| Cartpole,8trajectories | 8/8 | 8/8 | 1.96837 | 16.84159 |
| Ant,8trajectories | 6/8 | 7/8 | 0.64865 | 1.17569 |

Neither diagnostic produced nonfinite states. The poor physical outcomes remain
visible; completion of inference is not evidence of useful model accuracy.

The stable cache root is
`/workspace/exp_data/embodichain-pr822-accuracy-20261020`. Its overlay filesystem
had only46GiB free when checked. The still-empty v2 rollout directories were
hash-verified and linked to separate local NVMe storage with about1.8TiB free;
all120 specification bytes and workspace access paths remain unchanged. Active
training and teacher-generation files were not moved.

## Validation and remaining work

The final focused affected-code suite passed **276 tests, with 8 optional
reference tests skipped** in193.14s. Actual trusted-source update/resume tests
and CUDA loss/transport checks were exercised separately; overlapping counts
are not added. An earlier broader unrelated URDF integration crashed inside the
SDK, so this scoped result must not be presented as an all-repository CI pass.

The remaining work is completion of all40 full-budget training cells, all120
full-population evaluations, trace verification, paired five-seed confidence
bounds, learning/error curves and the final human-readable qualification report.
Unfinished, failed or undefined results remain visible and cannot be treated as
acceptance. This report supplements the
[earlier trained-model/workflow evidence](neural-dynamics-trained-comparison-20261020.md).

[Machine-readable evidence](neural-dynamics-cuda-alignment-20261020.json) binds the
actual CUDA diagnostics, dataset audits, immutable plans, guard checks and
observed processes. Local evidence paths are not download URLs.
