# Native RigidFormer qualification

This protocol qualifies the installable author model, prepared MOVi inputs and
shared SL training. Published accuracy and full-budget retraining are separate
experiments. A debug subset or a short paired run cannot satisfy either claim.

## Frozen inputs

- Author source: `Frank-ZY-Dou/Dynamics-Modeling`, revision
  `160fe0852c51cffae7b7440452b58d0b49db6df6`. The comparison verifies all 23
  files in the model package's `REFERENCE_SOURCES.json` before importing the
  independent original model, Trainer and EMA implementation.
- Data: `frankzydou/RigidFormer-MOVi` at
  `e78514b479b11acfa5d23d28f1211d7830646ff9`. Use the scene's complete object
  identities, collision mesh vertices, recorded size, four FPS anchors and
  physical parameters. Preparation checks raw scene and mesh hashes, including
  debug preparation. Model integration uses dt=1; physical time is stride/240 s.
- Weights: `frankzydou/RigidFormer` at
  `8f2c0a3a157b64e4b310c64a01eee6e4e7e25af6`, with the checksum selected by
  `_prepare.py`. Use Sphere weights on Sphere scenes for the evaluation smoke.
  Cross-dataset runs can establish operator agreement, not dataset accuracy.

Prepared readers verify actual payload hashes and manifest coverage, including
anchor/FPS caches. An omitted payload, changed split, changed size sidecar or
changed geometry is not an exact-resume-compatible source.

## Required checks

| Check | Acceptance |
|---|---|
| Independent author comparison | Prediction, free rollout, loss, clipped gradients, parameters, buffers, optimizer moments and EMA agree at atol=1e-5, rtol=1e-4 for at least three updates. Optimizer groups, scalar policy and scheduler state agree exactly. |
| Attention | Run the genuine Flash-Attention package for Flash qualification. SDPA compatibility of the reference is reported separately. |
| Full architecture recovery | Continuous four-update training equals two updates plus resume plus two updates, including model, optimizer, EMA, schedule, sampler and Python/NumPy/Torch/CUDA RNG. Raw and EMA exports retain every named tensor. |
| Epoch recovery | The small model/data fixture runs six updates over three epochs with two warmup epochs, resuming after update three. Every saved value must match exactly. |
| CUDA/NCCL | Two ranks execute SyncBatchNorm, unequal 3/2-object supervision and exact rank-local recovery. The weighted-update oracle uses a concatenated batch outside any process group, with stochastic dropout disabled on both sides. Recovery retains author dropout. |
| Fixed weights evaluation | Complete 480-frame rollout on the declared debug scenes, with four warmup frames and strides 1/5/10. Persist scope and physical-frame metrics without interpolating off-grid checkpoints. No accuracy threshold is assigned to a debug subset. |
| Packaging | Each dependent layer builds and installs its own wheel. Imports, configs and licenses resolve from the installed wheel without a repository PYTHONPATH. The final wheel includes runnable showcase resources, without downloaded assets. |

The numerical comparison uses one fixed batch per comparison epoch to exercise
epoch scheduling on each update. Seed 42, selected scene/frame identities,
executed source hashes, dependency versions and asset hashes are recorded.
This does not reproduce the research DataLoader's worker-RNG trajectory or the
released four/six-GPU training budget. Native sources own an explicit recoverable
epoch traversal, and exact recovery requires the same world size and recipe.

## Reproduce

Install the optional dependencies and use a compatible CUDA/Torch/Flash/NCCL
stack. Put all downloaded assets, prepared data and reports outside the checkout.

```bash
embodichain run-example learning/rigidformer prepare --dataset movi_sphere \
  --assets all --cache /path/to/assets --output /path/to/prepared --limit-per-split 2

embodichain run-example learning/rigidformer compare \
  --reference /path/to/Dynamics-Modeling/RigidFormer \
  --checkpoint /path/to/assets/model/rigidformer_movisphere.pth \
  --data-root /path/to/prepared/MOVi-Sphere --attention-backend flash \
  --seed 42 --updates 3 --output /path/to/fresh-comparison

# Use the Sphere recipe with external data paths, batch_size=1, total_updates=4.
embodichain run-example learning/rigidformer check-recovery \
  --config /path/to/recovery.yaml --stop-after 2 --output /path/to/fresh-recovery

python -m pytest tests/learning/sl tests/models/dynamics -m 'not gpu' -q
python -m pytest tests/models/dynamics/test_rigidformer_official.py \
  tests/models/dynamics/test_rigidformer_training.py --run-gpu -m gpu -q
CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
  tests/models/dynamics/test_rigidformer_distributed.py --run-gpu -m gpu -q

python -m scripts.benchmark.learning.rigidformer_index --output /path/to/index.json
```

The index benchmark measures CPU metadata construction at 1/32/960 scenes,
excluding data IO, epoch permutations, model work and GPU execution. It reports
median latency and separately measured Python allocations; these are not total
process-memory or training-throughput measurements.

## Evidence

The checked-in [qualification index](qualification.json) identifies the executed
reports and their hashes. Reports and binary inputs remain external artifacts;
historical reports are not relabeled after source or environment changes.
Every reported backend must be the backend actually executed. In particular,
Torch's compile-time NCCL version can differ from a process-local loaded runtime.

The development Blackwell host required a locally built NCCL 2.29.7 (CUDA 12.8,
sm_120) for model-layout broadcasts; its inherited 2.26.2 runtime failed the same
broadcast in an independent module containing only the parameter/buffer shapes.
The qualification records this runtime override and source/binary identities.
No NCCL transport-throughput claim follows from these correctness checks.

To reproduce that host's two-rank runtime, build outside the checkout and load
the library for the validation process. The recorded source commit is
`b91894bd5b190c874d98a017f93f5daa515b65d0`:

```bash
git clone --depth 1 --branch v2.29.7-1 https://github.com/NVIDIA/nccl.git /path/to/nccl
git -C /path/to/nccl checkout --detach b91894bd5b190c874d98a017f93f5daa515b65d0
make -C /path/to/nccl -j16 src.build NVCC_GENCODE='-gencode=arch=compute_120,code=sm_120'
LD_PRELOAD=/path/to/nccl/build/lib/libnccl.so.2 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 \
  CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
  tests/models/dynamics/test_rigidformer_distributed.py --run-gpu -m gpu -q
```
