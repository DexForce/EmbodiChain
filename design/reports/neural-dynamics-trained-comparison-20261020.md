# NeRD trainer comparison and workflow qualification — 2026-10-20

**The full accuracy objective remains active.** This report records actual
from-scratch paired training, the Ant conversion correction and a complete CLI
workflow in a clean EmbodiChain dependency environment. It does not substitute
these bounded results for the declared full-budget qualification.

## Ant conversion correction

The codec now preserves the original NeRD quaternion multiplication and vector
rotation operation order. The official-reference adapter also delegates the
native-COM to world-origin velocity conversion to the same codec. Reversing the
cross-product operands and subtracting had introduced additional FP32 rounding.
A frozen real-data fixture catches the previous behavior without loading a
simulator or depending on an external checkout during ordinary tests.

Both implementations are evaluated on byte-identical canonical-frame input
windows. The oracle is the unchanged official model/preprocessing in Python
3.8.20, Torch 2.2.2, NumPy 1.24.4 and Warp 1.5.1. Ant features, targets and
predictions agree exactly for 64 windows at each history length 1/9/10.
The original `atol=1e-5, rtol=1e-4` gate is unchanged.

Comparing the reference in Y-up with its globally rotated Z-up inputs separately
measures the original implementation's finite-precision frame non-invariance.
The recorded 3.5e-4–5.2e-4 prediction changes persist in the reference itself;
they are retained as that diagnostic, not counted as migration error between
identical inputs. No average loss replaces an elementwise numerical gate.

Actual official-main checkpoint probes in the robot-specific old Warp process
also report **zero forward, input-gradient, parameter-gradient and three-Adam
parameter differences** for both robots. Codec residuals stay below 1.2e-7.

## Trained model comparison

The official supported environment generated separate train/validation/test
corpora: Python 3.8.20 / Torch 2.2.2+cpu, Warp 1.8.0 for Cartpole and 1.5.1 for
Ant. Each robot has 25,600 training and 6,400 validation/test transitions in this
pilot. Raw HDF5, measured physics, seeds and file hashes are preserved. The known
uninitialized gravity transport feature is explicitly corrected on both sides;
the raw source files are retained unchanged.

The two trainers then run in the same clean EmbodiChain Torch environment.
This controls framework/runtime differences when assessing the training port;
it is not a claim that training also ran in the historical Torch 2.2 environment.
Each seed starts with a **new randomly initialized** official-width network;
no supplied pretrained weights are used. The original `VanillaTrainer` methods
execute without source modification. Both sides share the exact dataset,
normalizers, continuous sampling stream, update budget, scheduler and validation
checkpoint selector. There are 2 robots × 2 trainers × 5 seeds = **20 completed
runs**, each with 1,000 optimizer updates, B512/H10 and six Transformer layers.
The shared protocol remains a controlled noninferiority experiment rather than
literal reproduction of the paper's outer data-loader/validation recipe.

| Robot | Primary metric | One-sided paired 95% degradation upper bound | Pilot margin |
|---|---|---:|---:|
| Cartpole | q error norm | 1.6979% | ≤5% |
| Cartpole | qd error norm | −1.0592% | ≤5% |
| Ant | q error norm | 0.2475% | ≤5% |
| Ant | qd error norm | 0.0900% | ≤5% |

Statistics use the five paired seed results and 10,000 bootstrap resamples with
seed zero. Conditional pilot agreement does not prove converged accuracy,
long-rollout stability or the 4,995,000-update full-budget result.

## Clean official dependency environment

A new Python 3.11 virtualenv was installed with all declared EmbodiChain
requirements and the requested MR !1483 SDK wheel, whose runtime commit is
`b61e920fbb222e50213574ad57d885e7b96148b9`. The source SDK pin now matches its
0.5.1 metadata. Installation used dependency resolution, not `--no-deps` or a
wheel-metadata overlay. `uv pip check` validates all 273 installed packages.
Newton is 1.7.0, Warp 1.17.0, Torch 2.7.0, TorchCodec 0.5 and LeRobot 0.4.4.

On this Blackwell host, the default PyPI Torch2.7 CUDA12.6 wheel was actually
reproduced failing with `no kernel image` for sm_120. The official
`torch==2.7.0+cu128` and matching torchvision CUDA12.8 wheels retain Torch2.7,
provide sm_120 kernels, pass GPU matrix/backward execution and pass `pip check`.
For Blackwell installation, select that official PyTorch binary index before
running the CLI; this is a hardware wheel choice, not a model-version change.
The private SDK artifact is tested branch evidence, not a published SDK release.

Both robots executed, in this clean environment:

```text
embodichain dynamics generate → train → export → eval(mode=rollout)
```

Each generated 128 × 100 current-Newton transitions using `warp_reference`,
trained a fresh full-width model for 1,000 updates, exported the validation-chosen
checkpoint and executed the registered `nerd` solver at horizons 1/10/100 on 16
trajectories. Old-Warp reference libraries are not injected into this workflow.
The selected Ant checkpoint is update600; training actually completed update1000.
Exports remain **unqualified** because this proves workflow execution, not final
model accuracy. Reported numerical/task failures remain in the rollout evidence.

## Full-budget execution prerequisites

- Exact sampling streams now use RNG checkpoints instead of a dense index
  matrix. Measured 4,995,000×512 storage is **24,663,610 bytes**, versus
  20,459,520,000 dense bytes. Generation took 6.81s. Random seeks and resumed
  training reproduce the original indices and final weights exactly; existing
  `.npy` orders remain supported.
- Optional raw-transition residency has an explicit byte limit and bounded host
  staging. It preserves all values, repeated-row ownership, trajectory/split
  boundaries and prefix padding. On the measured Ant B512/H10 workload, completed
  gather+encoding latency falls from **171.85ms to 3.20ms** using 57.4MB of raw
  GPU fields. Startup was 0.35s. Full-corpus residency must still be measured on
  the final corpus; the configured limit fails before over-allocation. This
  bounded timing predates the full-v2 CUDA arithmetic revision and is not a
  measurement of current full-training throughput.
- Full configs log every500 updates while retaining validation/final records.
  This avoids millions of redundant log records without changing training.
- The original trainer now has atomic optimizer/RNG/best-model checkpoints and
  explicit resume; actual-source interruption/resume tests reproduce both best
  and last model weights exactly. The selected venv executable is preserved
  rather than symlink-resolved to the base interpreter.

The full matrix keeps the declared 4,995,000 updates per cell and five
seeds, on frozen official-version and current-Newton datasets. Both 20-cell
robot plans passed their complete artifact-hash checks. The current full-v2
matrix uses the immutable implementation containing the additional CUDA Ant
target/update and scripted-graph warmup corrections, with all40 cells assigned
only to physical GPUs0–6. Finite updates are observed; no full cell has completed
yet. The
[CUDA alignment and full-v2 execution report](neural-dynamics-cuda-alignment-20261020.md)
is authoritative for current execution and the frozen, disjoint-seed evaluation
protocol. The earlier full-v1 launch was explicitly superseded; its historical
identities remain in
`/root/.cache/embodichain/pr822-accuracy/full-training-launch.json`.

The formal readiness reports verify both teacher corpora and record no full
architecture/optimizer-profile discrepancies. The numerical gate accepts the
pinned official-main result with its source identity retained. Split-source
receipts verify actual raw-file hashes and the supported historical runtime;
they do not rely on the matrix's teacher label. Reporter and comparison checks
pass 31 focused tests, including HDF5 import and raw-source tamper detection.
This reporter revision is separately hashed in `full-readiness-verification.json`;
running training code remains the frozen snapshot.

The official
public `train/cfg` one-million-transition recipe identifies the reference data
scale; the original sampler's 512-replica batches actually produce 1,024,000
training transitions, recorded as measured. This is not a claim of matching the
unavailable data fingerprints behind the supplied 5M/10M pretrained checkpoints.
Each original validation/test split has 512 trajectories (51,200 transitions).
The current Newton corpus has 12,500 trajectories per robot, with 1,005,600 /
124,500 / 119,900 transitions in its train/validation/test splits. Each paired
trainer comparison uses the same split artifacts within its teacher domain.
Full offline, 2048-trajectory long-rollout/failure/nonfinite/return gates remain
pending until those runs finish and their artifacts are verified.

## Evidence and validation

The frozen implementation is identified by
`/root/.cache/embodichain/pr822-accuracy/implementation-snapshot.json`; execution
imports were checked against that snapshot. Clean-environment affected tests:
**258 passed, 8 optional-reference tests skipped**, no failures. The skipped
external checkpoint/trainer surfaces were exercised separately with explicit
trusted source and dependency paths; their overlapping counts are not added.

[Machine-readable trained comparison](neural-dynamics-trained-comparison-20261020.json)
contains per-seed metrics, numerical probes, complete workflow receipts and
measured performance. Raw logs/data/checkpoints remain under
`/root/.cache/embodichain/pr822-accuracy/`; local paths are not download URLs.
