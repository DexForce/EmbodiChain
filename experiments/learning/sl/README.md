# Shared supervised BC acceptance

This directory uses the physical PickPlace task and visual recipe pinned by
PR828 (`51f1f0c`), with an immutable offline corpus and the shared SL learner.
The production model, objective and HDF5 reader do not import this task or any
ODS example. The evaluation adapter reuses the original task measurements and
success predicate; it never constructs an expert during student execution.

## Numerical migration

From the repository root in `py311`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.learning.sl.bc_reference_parity \
  --reference-root /path/to/pr828-checkout \
  --output outputs/sl/bc-reference-parity.json
```

This compares actual PR828 forward outputs, its arm/gripper loss, every parameter
gradient and one clipped AdamW update with the generic SL implementation. The
declared tolerance is `atol=1e-5`, `rtol=1e-4`; a mismatch exits nonzero.

## Fixed corpus

`collect_pick_place.py` owns expert qualification and bounded collection. Its
completed manifest records each accepted episode's content hash, geometry51,
raw policy-request actions7 and dual-camera lossless uint8 RGB data. Production
training rejects `complete=false`. HDF5 handles are bounded and checked files
are immutable during a run; only two historical RGB frames enter each sample.
Future geometry is used to construct supervision only.

The formal recipe uses exactly 1,024 accepted training episodes, with input and
target statistics frozen from the first 32. Qualification, training generation,
development and final test have separate declared seed populations. Corpus
preparation remains separate from learner timing and does not use ODS.

## Train and evaluate

Run one seed per fresh output directory. The example assigns physical GPU1 to
training and GPU7 to the independent evaluator's simulation; pass the GPU IDs
allocated to your run:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m experiments.learning.sl.train_bc \
  --manifest outputs/sl/bc-corpus/manifest.json \
  --output outputs/sl/bc-seed20261025 --seed 20261025 \
  --evaluation-visible-devices 7,1
```

Repeat training seeds `20261026` and `20261027` against the same frozen corpus.
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
and normalization without policy updates. The emitted `recipe.json` can also
be run through `embodichain train-sl --config FILE`. Its trusted checkpoint can
restore optimizer, LR schedule, sampling RNG and committed model updates with
`--resume FILE`. The formal acceptance driver starts fresh experiments; it does
not claim restoration of elapsed deadlines or interrupted physical evaluations.

Outputs include immutable development checkpoints, `metrics.jsonl`, per-scene
reports, normalization, frozen recipe, checkpoint selection, final-test request
and `summary.json`. Learning success and engineering completion are recorded
separately. Generated data, weights and logs stay outside Git.
