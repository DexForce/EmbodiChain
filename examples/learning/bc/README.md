# Behavior cloning for PickPlace

The default showcase trains state-based policies on a physical Franka
pick-and-place task through `embodichain.learning.sl`. The input includes
simulator-provided cube and goal state, so state results do not establish
camera-based or real-world performance. The task requires lifting, placing,
opening and withdrawing, followed by 0.3 seconds of stable placement.

Two state recipes use the same action and task contracts:

- `maniskill_action`: one 51-feature state predicts the current seven-component
  relative-pose/gripper request with a 256×256 MLP and Adam.
- `legacy_chunk`: two states predict 64 absolute pose/gripper targets with a
  512×512 MLP; the executor applies 16 targets before replanning.

Both use accepted demonstrations and the shared supervised trainer. Recipe A
uses identity normalization; recipe B freezes statistics from training data.
Policy observations and recorded actions refer to
the same pre-action step. Per-frame sampling for `maniskill_action` is uniform
with replacement; the official ManiSkill sampler instead shuffles frames within
epochs. The `legacy_chunk` recipe uses episode-uniform causal contexts, valid
target masks and training-only observation noise.

## Collect and train state BC

Use an installed EmbodiChain distribution with its matching simulation runtime
and an explicitly selected physical GPU. The commands below run from the
repository root and write to fresh directories; adapt the paths and seed to the
planned experiment.

```bash
embodichain run-example learning/bc environment \
  --output outputs/bc/environment.json --device cuda:0
embodichain run-example learning/bc qualify \
  --output-dir outputs/bc/qualification --scenes 40 --device cuda:0
embodichain run-example learning/bc collect \
  --output-dir outputs/bc/demonstrations --episodes 32 --device cuda:0
embodichain run-example learning/bc replay \
  --manifest outputs/bc/demonstrations/manifest.json \
  --output-dir outputs/bc/replay --episodes 32 --device cuda:0
embodichain run-example learning/bc prepare-state \
  --manifest outputs/bc/demonstrations/manifest.json \
  --output-dir outputs/bc/state-view --episodes 32
embodichain run-example learning/bc train \
  --recipe maniskill_action --view outputs/bc/state-view/manifest.json \
  --environment outputs/bc/environment.json --output-dir outputs/bc/run \
  --seed 20400011 --device cuda:0 --prepare-only
embodichain run-example learning/bc train \
  --recipe maniskill_action --view outputs/bc/state-view/manifest.json \
  --environment outputs/bc/environment.json --output-dir outputs/bc/run \
  --seed 20400011 --device cuda:0 --train
embodichain run-example learning/bc evaluate \
  --checkpoint outputs/bc/run/training/step_10000.pt \
  --output outputs/bc/evaluation.json --seed-base 20500000 \
  --scenes 20 --device cuda:0
```

Select `--recipe legacy_chunk` for the historical-style state baseline.
`replay` checks saved policy requests in the current physical environment.
Evaluation stops at the first termination or timeout, records `success_once`
and `success_at_end` at that actual episode boundary, and saves per-scene
state/action traces. The recorded episode-end convention differs from
ManiSkill's fixed-horizon, ignore-termination evaluation.

The state view binds the source manifest, per-episode content hashes, action
semantics, observation timing and normalization. An immutable resident source
can be resumed exactly. Caller-owned live ODS data can train through the same
model and objective and save snapshots; live delivery state is not restored by
the trainer. The host owns ODS lifecycle and waits.

The [state acceptance record](research/state-acceptance.md) distinguishes the
completed first-32 results, the ongoing fresh-256 fallback, and the gated
independent test. It also records source and data identities, the task-success
predicate, all three seeds, and both checkpoints of both recipes. Neither recipe
passed the first-round development gate.

## Historical PR828 visual recipe

The historical visual recipe uses the physical PickPlace task pinned by
PR828 (`51f1f0c`), with an immutable offline corpus and the shared SL learner.
The production model, objective and HDF5 reader do not import this task or any
ODS example. The evaluation adapter reuses the original task measurements and
success predicate; it never constructs an expert during student execution.

The historical first-eight-episode learnability check completed exactly
2,000 updates. Fixed-panel normalized MSE fell 98.38%, but grasp-phase TCP p95
remained 7.51 mm against the preregistered 5 mm gate; the other four phases
passed. The planned sampling comparison and formal training therefore did not
start. The visual recipe remains unqualified. A separate diagnostic with the original
20k checkpoint scored 2/4 successes; its new scenes/runtime do not establish a
training gain. See the [current diagnostics and stopping decision](research/improvement-v2.md)
and [preserved historical experiments](research/acceptance.md). The original
three-seed formal failures and sealed test remain unchanged.

## Numerical migration

From the repository root in `py311`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
embodichain run-example learning/bc reference-parity \
  --reference-root /path/to/pr828-checkout \
  --output outputs/sl/bc-reference-parity.json
```

This compares actual PR828 forward outputs, its arm/gripper loss, every parameter
gradient and one clipped AdamW update with the generic SL implementation. The
declared tolerance is `atol=1e-5`, `rtol=1e-4`; a mismatch exits nonzero.

## Historical PR828 fixed corpus

`_support/_collect.py` owns expert qualification and bounded collection. Its
completed manifest records each accepted episode's content hash, geometry51,
raw policy-request actions7 and dual-camera lossless uint8 RGB data. Production
training rejects `complete=false`. HDF5 handles are bounded and checked files
are immutable during a run; only two historical RGB frames enter each sample.
Future geometry is used to construct supervision only.

Dataset identity binds both the manifest and its model-facing view: observation
field/slice, target fields and explicit transform version, camera order and H/F
windows. Changing that view rejects recovery before model state is modified.
The fixed reader preallocates NCHW image output and batches the pure target
transform; a frozen real-data fixture verifies bitwise-identical sampled values.
One optional prefetched batch overlaps CPU decoding with GPU work. Its worker
uses a private RNG, and only delivered batches advance the caller's sampling
cursor, so an unused pending batch need not be serialized for exact recovery.

The formal recipe uses exactly 1,024 accepted training episodes, with input and
target statistics frozen from the first 32. Qualification, training generation,
development and final test have separate declared seed populations. Corpus
preparation remains separate from learner timing and does not use ODS.

## Historical PR828 training and evaluation

The frozen protocol below documents the historical experiment. All three seeds
completed 20,000 updates and every development checkpoint scored 0/20 successes.
The independent test panels remain sealed. Do not rerun the full protocol as a
namespace check; see [the failure evidence and controlled plan](research/acceptance.md).
Use explicitly assigned physical devices for any new experiment.

This frozen acceptance protocol uses one training process and global batch64.
It rejects DDP before accessing the corpus or creating experiment artifacts.
The shared BC model, objective and sources can be used in a separate SL DDP
recipe; its global batch, budget and learning acceptance must be declared
separately rather than changing this protocol.

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
embodichain run-example learning/bc pr828-train \
  --manifest outputs/sl/bc-corpus/manifest.json \
  --output outputs/sl/bc-seed20261025 --seed 20261025 \
  --evaluation-visible-devices 4 --sim-device cuda:0 --policy-device cuda:0
```

The historical training seeds were `20261025`, `20261026` and `20261027`, using
the same frozen corpus.
The recipe uses dual 256×256 cameras, 27 measured robot features, two historical
frames, 32 future targets, eight executed targets, batch64 and 20,000 updates.
It uses FP32 AdamW, weight decay1e-6, gradient norm1, 500-update warmup and cosine
decay from3e-4 to3e-5. Targets are absolute arena TCP position, rotation6D and
normalized gripper commands; the task adapter converts them to the original
relative-pose/gripper action interface.

The initial policy and each2,000-update checkpoint run on the same20 development
states (seed20271025). The final two updated checkpoints must each achieve at
least18/20 successes. Selection uses greatest development success, then smallest
goal error and earliest update. `selection.json` is frozen before the one
independent100-state test (seed20281025), which requires at least80/100 successes
for every training seed. A failed development gate leaves the final panel sealed.
An unmet learning target exits nonzero and preserves all recorded outcomes.

The worker budget is six hours; each evaluator has at most one hour and only
the remaining worker budget. Evaluators run in owned process groups and their
logs, requests and partial outcomes remain available on timeout or failure.

`--prepare-only` validates the finished corpus and writes the resolved recipe
and normalization without policy updates. The emitted `recipe.json` is owned by this example and its bootstrap. Its trusted
checkpoint can
restore optimizer, LR schedule, sampling RNG and committed model updates with
`--resume FILE`. The formal acceptance driver starts fresh experiments; it does
not claim restoration of elapsed deadlines or interrupted physical evaluations.

Outputs include immutable development checkpoints, `metrics.jsonl`, per-scene
reports, normalization, frozen recipe, checkpoint selection, final-test request
and `summary.json`. Learning success and engineering completion are recorded
separately. Generated data, weights and logs stay outside Git.


## Installable model family

Use `from embodichain_models.policy.bc import ChunkBCModel, ChunkBCModelCfg` for
state or RGB policies. The same family owns `ChunkBCObjective`, `BCWindowDataset`,
`HDF5BCSource`, `PrefetchedBCSource` and `ChunkBCExecutor`. Install the `vision`
extra for RGB (`pip install 'embodichain[vision]'`); numeric inference loads no
vision or simulation modules. Shared MLP/normalizers and RL interfaces stay in
core.

The standalone factory
`embodichain_models.policy.bc.sl_recipe:build_supervised_trainer` works from any
working directory with `embodichain train-sl --config FILE`. The example
[stored-target configuration](configs/stored-targets.sl.yaml) expects a caller
prepared immutable HDF5 manifest with physical `targets` and four frozen
normalization tensors. The raw frozen PickPlace corpus stores geometry/actions
and uses a separate task-owned transform; it is not a stored-target corpus.
Derive new artifacts in a fresh destination rather than altering frozen data.

`embodichain run-example learning/bc --help` is lazy. The state commands are
`collect`, `train` and `evaluate`; the historical visual equivalents are
`pr828-collect`, `pr828-train` and `pr828-evaluate`. Private `_support` modules
own collection, geometry targets, controller decode, acceptance and reference
research. They are ordinary installed example data, outside the model package.

The reader's explicit `include_partial_targets=False` default preserves the
frozen full-future sampling/view. Generic recipes may opt in to complete-history
tail conditioning, finite zero right padding and `target_valid` masks; the
source/recipe identity changes. The formal driver continues using the default.
The [tail-conditioning configuration](configs/tail-conditioning.candidate.yaml)
and its private `window-candidate` adapter are preserved from the completed
comparison. Historical `--prepare-only` reports remain unchanged at zero
updates; actual training and held evaluation now have separate immutable
records. The paired run stopped after failing its auxiliary gate. Coverage of
terminal conditioning alone did not produce task success. Review the [result
and stop rule](research/acceptance.md) before considering a separately declared
next recipe; do not repeat or enlarge this run.
