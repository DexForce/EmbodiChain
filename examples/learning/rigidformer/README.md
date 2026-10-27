# RigidFormer author implementation

This showcase integrates the author's RigidFormer implementation at
`160fe0852c51cffae7b7440452b58d0b49db6df6` from
[Frank-ZY-Dou/Dynamics-Modeling](https://github.com/Frank-ZY-Dou/Dynamics-Modeling/tree/160fe0852c51cffae7b7440452b58d0b49db6df6/RigidFormer).
The model lives at `embodichain_models.dynamics.rigidformer` and trains through
the common SL trainer. The community `rigidformer` package is not a runtime dependency.

## Setup and ownership

Use a Python 3.11 environment and a CUDA-capable Torch installation:

```bash
conda activate py311
python -m pip install 'embodichain[rigidformer]'
embodichain run-example learning/rigidformer --help
```

The standard native attention backend is Torch SDPA. The explicit `flash`
backend requires a Flash-Attention build compatible with Torch, CUDA and the
GPU. Reference evidence records the actual dependency/backend versions.
Imports, architecture construction and checkpoint inspection work on CPU;
geometry prediction uses the author's CUDA KNN path and needs at least two
valid objects per scene.

Model/config/checkpoint/data/objective/recipe code is installable under the
model family. Downloading assets, conversion and reference experiments remain
showcase-owned. Private author-derived source retains MIT attribution in
`THIRD_PARTY_LICENSES`. Official scenes are CC BY 4.0; collision meshes and
derived geometry are CC BY-SA 4.0. Those licenses remain attached to prepared
data. Data, checkpoints, source checkouts and reports are not bundled in the wheel.

## Prepare official assets

```bash
embodichain run-example learning/rigidformer prepare --dataset movi_a
```

Data preparation is independent of pretrained weights. Download those separately
with `prepare --dataset movi_a --assets weights`, or select `--assets all`.
Training from scratch requires no pretrained checkpoint.

The downloader pins dataset revision
`e78514b479b11acfa5d23d28f1211d7830646ff9` and weights revision
`8f2c0a3a157b64e4b310c64a01eee6e4e7e25af6`. It verifies published checksums.
Use `movi_sphere`, `movi_a`, or `movi_b`. Numeric scenes and collision meshes
are sufficient; videos are not downloaded.

Prepared files use the author's packed layout, with aligned material points,
complete object identities (including recovered objects), physical properties,
reference anchors/FPS order, and a content-hashed manifest. Stored vertices are
**unscaled asset vertices transformed into world coordinates**. The size sidecar
applies each object's metadata size about its origin exactly once during reads.

By default preparation requires all 960 training, 120 validation and 120 test
scenes. `--limit-per-split N` creates a clearly labelled debug subset.
`--local-files-only` requires already downloaded, verified assets.
Every output dataset needs a fresh directory. Failures after conversion starts
remain recorded; checksum preflight rejects bad assets before creating output.
Readers reject altered payloads and geometry omitted from the manifest. FPS and
anchor caches are committed inputs, rather than untracked files generated during
training. Debug preparation verifies collision meshes as well as scene shards.

## Train with shared SL

```bash
embodichain train-sl --config examples/learning/rigidformer/configs/sl.movi_a.yaml
```

Paths resolve from the caller's current working directory. The preset recipes
use eight-frame windows, four anchors, variable integration strides 1/5/10,
author augmentation, anchor-only before/after-Kabsch position and acceleration
losses, grouped AdamW, gradient clipping, epoch scheduling and FP32 EMA.

The author model integrates with **model dt=1**. Physical time is
`step_size / 240` seconds. Loss scaling and conditioning use the recorded
step size; do not replace the model integration unit with seconds.
FP32 geometry augmentation precedes BF16 network autocast; integration and
rigid registration retain the author's numerical policy.

Each source shuffles and shards its epoch traversal, drops an incomplete batch,
and stages the cursor until optimizer success. The source-sampling stream is
explicit and recoverable; the legacy research DataLoader's worker-RNG trajectory
is not claimed bitwise identical. Checkpoints preserve sampling, model/optimizer,
scheduler, EMA and per-rank state. A changed source/split/scale/recipe/world size
is rejected before restoration.

For multiple GPUs, use the same launcher:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m embodichain train-sl --config examples/learning/rigidformer/configs/sl.movi_a.yaml
```

Batch size is per rank, and learning rates are not scaled automatically.
The released MOVi-A/Sphere training layout uses four GPUs x 32 scenes;
MOVi-B uses six x 21. A smaller training run is useful for integration checks
but is not equivalent to that reference training budget. Native DDP preserves
the author's synchronized encoder statistics and uses a global valid-anchor
loss mean exactly once.

Resume through the common entry point; change only the output directory when
continuing an older checkpoint into a fresh run destination:

```bash
embodichain train-sl --config my-recipe.yaml --resume outputs/run/last.pt
```

## Export and evaluate

```bash
embodichain run-example learning/rigidformer export \
  --checkpoint outputs/run/last.pt --weights ema --output outputs/model-ema.pt

embodichain run-example learning/rigidformer evaluate \
  --checkpoint outputs/model-ema.pt \
  --data-root outputs/rigidformer/data/MOVi-A \
  --split validation --output outputs/rigidformer/eval-validation
```

Official weights can be passed directly to `evaluate`. Exported native weights
use the same named tensor/config layout; importing weights does not recover an
official optimizer process. EMA/raw selection is explicit, and an absent EMA
does not silently fall back to raw weights.

The fixed-checkpoint evaluator uses four ground-truth warmup frames, step sizes
1/5/10, and the author's HopNet-aligned position/rotation metrics at physical
frames 25/50/75/100/200/300/400. Off-grid checkpoints are not interpolated.
Reports include corpus scope and dependency/source/checkpoint identities.
A debug-subset result does not establish complete benchmark accuracy.

New native runs do not select a checkpoint from test scores. Use validation to
select, freeze the checkpoint, then evaluate test. The author's default research
training recipe monitors test metrics; that selection-protocol difference must
remain visible in any retraining comparison.

## Independent reference comparison

Supply an unmodified author checkout's `RigidFormer` directory at the pinned
revision. The runner verifies source hashes in `REFERENCE_SOURCES.json` before
importing the independent original model and Trainer.

```bash
embodichain run-example learning/rigidformer compare \
  --reference /path/to/Dynamics-Modeling/RigidFormer \
  --checkpoint outputs/model-ema.pt \
  --data-root outputs/rigidformer/data/MOVi-A \
  --attention-backend flash --updates 3 --output outputs/rigidformer/compare
```

Comparison checks prediction, free rollout, losses, gradients and paired optimizer
updates on identical fixed inputs and randomness. `flash` executes the genuine
Flash-Attention package. Explicit `sdpa` uses a documented compatibility operator
for the reference's unmasked attention; its results do not qualify Flash execution.
Reports distinguish official released weights from other explicitly provided weights.
They also compare optimizer moments/groups, buffers, EMA and epoch schedules,
and record selected frames, source hashes and the executed numerical runtime.
Each paired update represents one fixed-batch comparison epoch; it is separate
from the source's normal training-epoch length.

Check exact continuous versus resumed training using the same installable recipe:

```bash
embodichain run-example learning/rigidformer check-recovery \
  --config my-four-update-recipe.yaml --stop-after 2 --output outputs/recovery-check
```

This creates fresh full/split/resumed runs, compares complete committed state
exactly, and verifies raw/EMA exports. The [acceptance protocol](research/acceptance.md)
also covers epoch/warmup boundaries, two-rank NCCL and installed-wheel checks.

This is a numerical-integration check. Full published-accuracy reproduction
additionally requires frozen official weights/data/evaluation, or complete
retraining with the declared compute and selection protocol. No short-run
comparison is labelled a full training reproduction.

