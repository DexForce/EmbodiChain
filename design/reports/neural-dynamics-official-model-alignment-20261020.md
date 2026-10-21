# Official NeRD model alignment — 2026-10-20

The reference for this follow-up is the actual pretrained NeRD model executed
from the unchanged official repository in its supported version environment.
Earlier Newton-teacher loss, parameter reconstruction and fine-tuning runs are
auxiliary diagnostics. They do not satisfy this model-comparison requirement.

## Frozen reference

- Repository: `NVlabs/neural-robot-dynamics`, commit
  `370e89d2b8ce68aa3578aa7812859c681b7502bc`; tracked source verified unchanged.
- Official execution: Python **3.8.20**, Torch **2.2.2+cpu**, NumPy **1.24.4**.
  Cartpole uses Warp **1.8.0**; Ant uses Warp **1.5.1**, as required by the
  repository's pretrained-model reproduction note.
- The official objects loaded from the provided `model.pt` run directly.
  Neither EmbodiChain nor DexSim/Newton is imported into reference execution.
- Each robot generates **32 × 100** transitions through the original sampler,
  with seed 500. The original model produces frozen predictions for 64 real
  windows at each history length 1/9/10, plus a 32-trajectory 100-step neural
  rollout with history rollover. Both sides use identical weights, normalizers,
  initial states, commands and feature order.
- CPU execution maps the original integrator's CPU CUDA-stream request to
  Torch's no-op `None` stream. Numerical source methods remain unchanged.
- Raw HDF5 retains the upstream sampler's uninitialized non-up-axis gravity
  feature. Both comparison sides explicitly replace that transport feature
  with the measured physical gravity direction; the replacement is recorded.
  This is not an undocumented repair of the original dataset.

## Results

The migrated reference run uses Python 3.10.12, Torch 2.2.2+cpu, NumPy 1.24.4
and the same robot-specific old Warp. Python 3.10 satisfies EmbodiChain's
minimum Python version while keeping the numerical Torch/Warp dependencies
matched to the official oracle.

| Robot | Same-feature forward maximum difference, H1/9/10 | Original-runtime neural rollout maximum difference, 32 × 100 | Status |
|---|---:|---:|---|
| Cartpole | 0 | 0 | Bounded model/history port agrees exactly |
| Ant | 0 | 0 | Bounded model/history port agrees exactly |

This verifies migration of the provided pretrained weights. It does **not**
verify a newly trained model against the official trained model or complete
full-budget training noninferiority.

The complete canonical-state conversion path is measured separately:

| Robot | Maximum feature difference | Maximum prediction difference | Existing elementwise tolerance |
|---|---:|---:|---|
| Cartpole | 2.38e-7 | 1.31e-6 | Passed |
| Ant | 2.15e-6 | 4.58e-4 | **Not passed** |

Ant's normalized-MSE relative changes are only −0.002825% through +0.000298%
in these windows, but that average does not override the failed elementwise
check (`atol=1e-5`, `rtol=1e-4`). Frame transport and twist/COM conversion remain
a precision investigation target. No tolerance was relaxed.

A separate run retains the current environment, Python 3.11.14 / Torch
2.7.0+cu128, while still executing CPU old-Warp physics and the same reference
inputs. Its maximum same-feature output differences are 9.54e-7 for Cartpole
and 3.86e-5 for Ant. Over 100 original-runtime neural steps, raw-state maximum
component differences are 8.96e-5 and 7.82 respectively; mean absolute
component differences are 2.11e-6 and 0.00533. These are mixed-coordinate raw
state diagnostics, not meter-valued position errors. All compared trajectories
remain finite. Environment-dependent accumulation must be explained before
asserting full current-runtime equivalence; the matched-version zero result
cannot replace this check.

## Reproduction and artifacts

The independent official worker is
`scripts/experiments/neural_dynamics/official_model_reference.py`; the migrated
worker is `scripts/experiments/neural_dynamics/compare_official_model.py`.
Use explicit trusted source/checkpoint paths and a fresh output directory.
The official worker enforces the exact Python/Torch/Warp versions above.

```bash
# OFFICIAL_PY points to Python 3.8.20 with Torch 2.2.2 and the correct Warp.
# For Ant, put the isolated Warp 1.5.1 installation first on PYTHONPATH.
"$OFFICIAL_PY" scripts/experiments/neural_dynamics/official_model_reference.py \
  --source "$NERD_SOURCE" --checkpoint "$OFFICIAL_MODEL_PT" \
  --robot ant --output "$REFERENCE_OUTPUT" --trust-external-code

# PORT_PY points to supported EmbodiChain Python with the chosen Torch version.
"$PORT_PY" scripts/experiments/neural_dynamics/compare_official_model.py \
  --source "$NERD_SOURCE" --reference "$REFERENCE_OUTPUT" \
  --model "$PORTABLE_MODEL" --output "$COMPARISON_JSON" --trust-external-code
```

Reference receipts include source commit, checkpoint and data/prediction hashes,
actual numerical dependencies, transport decisions and an assertion that the
current simulators were not imported. The comparison verifies those hashes
before reading inputs. The original model normalizes by replacing dictionary
entries; fixture generation preserves physical features before model execution
so they are not normalized twice.

Executed artifacts are retained under
`/root/.cache/embodichain/pr822-accuracy/strict-official-{cartpole,ant}-v3/` and
`*-official-model-{matched,current}-torch.{json,npz}`. The aggregate results are
[checked in as JSON](neural-dynamics-official-model-alignment-20261020.json).
The local paths are not published download URLs.

## Remaining acceptance work

1. Diagnose the measured Ant canonical conversion residual without weakening
   the existing numerical gate; preserve the exact official oracle.
2. Separate Torch/runtime numerical accumulation from data/parameter and model
   migration effects. Compare physical state metrics as well as raw components.
3. For **training quality**, train/evaluate the original and migrated trainers
   on the same official-environment datasets, normalizers, seed matrix, update
   budget and checkpoint-selection protocol. The earlier Newton-domain fine-tune
   is not evidence for that requirement.
4. Complete the declared five-seed, full-data/full-budget and 2048-trajectory
   long-horizon qualification before changing that acceptance status.
