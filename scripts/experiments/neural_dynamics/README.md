# Original NeRD data and matched training

Use two isolated environments. The original data worker uses Python 3.8.20,
Torch 2.2.2, NumPy 1.24.4 and Warp 1.8.0 for Cartpole or Warp 1.5.1 for Ant. The
EmbodiChain environment owns current-Newton collection, controlled training,
export and native inference. Use the matching SDK build for the named solver
and functional Scene contracts. Keep the unchanged original checkout at commit
`370e89d2b8ce68aa3578aa7812859c681b7502bc`.

## Original-entry comparison with a small budget

`official_small_budget.py` runs the actual upstream `train/train.py` through
`runpy`, including its constructor, DataLoaders, FP32 RunningMeanStd, loss,
epoch-zero validation, rollout evaluation and independent checkpoint selectors.
It observes the emitted batches rather than generating an approximate shuffle.
Use the supported original environment above. CPU execution maps the upstream
CUDA-stream request to `None`; the original numerical methods remain unchanged.

Provide a JSON plan with `source`, `robot`, `output_dir`, `training_config`,
`seed: 0`, `steps: 1000` and `cpu_threads: 8`. Start from the original robot YAML,
change the existing dataset paths and set `algorithm.num_iters_per_epoch: 1000`.
Keep `num_epochs: 1000` and the original optimizer/model fields. The observer
stops before the second trained epoch, after the original validation, simulation
evaluation and checkpoint selection complete. Thus all 1,000 updates retain the
original first-trained-epoch rate, 0.0009991. Existing output directories are
rejected. The original source checkout stays unmodified.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$NERD_WARP_ROOT" "$NERD_DATA_PYTHON" scripts/experiments/neural_dynamics/official_small_budget.py --config original-small-plan.json --trust-external-code
```

Import the same raw split files for the migrated trainer. For Ant, preserve the
original Y-up world coordinates in this experiment (`world_rotation_xyzw=None`)
so extra world rotation does not enter the comparison. This does not change the
native Newton deployment coordinate contract. The official run supplies the
actual initial weights, normalization statistics and consumed sample indices.

```bash
CUDA_VISIBLE_DEVICES="" "$EMBODICHAIN_PYTHON" scripts/experiments/neural_dynamics/matched_small_budget.py --official-dir /workspace/exp_data/ant/official --dataset /workspace/exp_data/ant/original-frame-data --output-dir /workspace/exp_data/ant/migrated --device cpu
```

The migrated experiment calls production `DynamicsTrainer.step` and export APIs;
its outer loop reproduces the captured original sampling and validation selector.
This differs from the default CLI sampling/statistics protocol. Compare both CPU
runs first, and record any separate CUDA run as an environment diagnostic.

New first-batch observations are saved **before** the original model forward,
with `first-preprocessed-batch.json` identifying physical features and binding
the tensor file's SHA256. The original model normalizes its input dictionary in
place. Earlier observations without that sidecar contain already-normalized
features; the matched reader preserves that legacy interpretation and records
the observed stage. Rebuild physical inputs from raw data for gradient probes,
rather than passing a legacy normalized snapshot through the model again.

`official_small_budget_evaluate.py` evaluates the selected original checkpoint on
its untouched test split, records fixed validation curves and produces forward
golden tensors under Python3.8/Torch2.2. `official_small_budget_rollout.py` reuses
fixed prefixes of the supported-environment evaluation corpora below, evaluates
both trained weight sets through the original integrator in the same supported
runtime, and records endpoint/prefix errors at 1/10/100 steps. A migrated export
must first be restored into the original model format, preserving all weights
and normalizers. All failed trajectories remain in the measurements, and each
feedback trajectory uses its own predicted state.

Historical-data model exports and native SDK models have different deployment
provenance. Verify the current SDK generate/train/export/infer workflow with
native-data models; never replace a historical model's missing native physics
signature to bypass deployment checks.

## Reference/math exported-model inference

`official_math_rollout.py` evaluates an explicitly `reference`/math portable
export through the unchanged original model and Transformer integrator. This is
the reusable entry used for the full-budget model's supported-environment
accuracy checks. The original-default baseline remains separately evaluated by
`official_small_budget_rollout.py` with its native official checkpoint.

Use the original Python 3.8.20/Torch 2.2.2 environment and the robot's pinned Warp.
Provide fixed official teacher trajectories, the portable export and the exact
pure transport modules whose hashes were recorded when the teachers were
created. Source, asset, raw-state/action, physics and transport identities are
checked; changing a teacher or transport requires its own experiment identity.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$NERD_WARP_ROOT" "$NERD_DATA_PYTHON" scripts/experiments/neural_dynamics/official_math_rollout.py --source "$NERD_REFERENCE_SOURCE" --portable "$NERD_PORTABLE_MODEL" --scope checkpoint_accuracy --teacher "$NERD_TEACHER_DATA" --transport-root "$NERD_FROZEN_TRANSPORT" --output "$NERD_NEW_RESULT_JSON" --count 16 --trust-external-code
```

The output contains H1/10/100 endpoint and prefix metrics, complete NPZ traces,
source/runtime/physics provenance and the actual attention operator. Passive and
random-effort runs replay the same controls; PD recomputes each neural branch's
controls from its own predicted state. Failed trajectories remain included.
`--scope smoke` is reserved for zero-update pipeline checks. This tool neither
starts training nor qualifies historical weights for current-Newton deployment.

`_portable_reference.py` and `_historical_physics.py` are standalone helpers for
this old Python environment. They do not import the installed EmbodiChain package
or SDK, and do not modify the original source or default baseline.

## Cross-version numerical attribution

Treat training implementation, runtime kernels and data transport as separate
controls. On shared canonical data, the current-environment original numerical
methods and migrated trainer match through 1,000 updates for both robots. This
does not remove the Ant raw-spatial/native-COM target-rounding distinction.

CPU SDPA dispatch and the final FP32 gradient-norm reduction can differ between
Torch 2.2 and 2.7. A jointly specified math-attention and fixed-order total-norm
policy is a separate controlled variant. It changes the original default
numerical path and must not replace the unchanged official default baseline or
silently change production settings. Record optimizer tensor equality separately
from version-specific metadata such as `decoupled_weight_decay=False`.

The [numerical attribution report](../../../design/reports/neural-dynamics-numerical-attribution-20261020.md)
records bounded operator probes, the 1,000-update controls and their limits.

## Full-budget continuation with an explicit numerical policy

Production `DynamicsTrainCfg.numerical_profile="reference"` selects math SDPA
and ordered FP64 accumulation of FP32 per-parameter gradient norms. Portable
exports persist `attention_backend="math"`; changing the policy requires a new
fine-tuning run. The original-method `run_reference` adapter supports only its
default policy and rejects math initial exports rather than silently ignoring
their execution metadata. Keep the original default baseline separate.

For source-fidelity imports, `import_legacy_datasets` accepts
`preserve_legacy_velocity=True` with `world_rotation_xyzw=None`. It preserves
original FP32 velocity before COM conversion, verifies bit-exact transition
continuity, and supplies current/next raw velocity directly to production
encoding. This option does not change native deployment physics qualification.

Keep `train.total_steps=4995000` for both initial and resumed phases. Use only
top-level `stop_after=5000` for the first checkpoint; then remove `fine_tune` and
`statistics`, provide `resume`, and retain the entire training configuration.
The exact source-shuffle descriptor includes the original DataLoader's epoch
tail dropping and source permutation hashes. Independent seeded permutations
or random-with-replacement orders do not reproduce that stream.

The [full-budget progress report](../../../design/reports/neural-dynamics-full-budget-20261021.md)
binds the two formal runs, official zero-step initialization, GPU recovery gates,
fixed-test metrics and original-supported-runtime rollout comparisons. Both
actual committed cursors must reach the full budget before completion is claimed.

## Training data and paired training

`official_model_reference.py --generate-only` executes the unmodified original
sampler without a model. `NERD_DATA_PYTHON` is the original environment's
interpreter; `NERD_WARP_ROOT` is the robot's pinned Warp dependency directory:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$NERD_WARP_ROOT" "$NERD_DATA_PYTHON" scripts/experiments/neural_dynamics/official_model_reference.py --source /path/to/neural-robot-dynamics --robot ant --output /workspace/exp_data/ant/reference-train --count 10000 --num-envs 512 --horizon 100 --seed 0 --generate-only --trust-external-code
```

Generate validation/test separately with different seeds and output paths. The
original sampler rounds counts to replica batches; inspect the measured count.
`import_legacy_datasets` in `embodichain.learning.dynamics.reference` imports raw
split files with canonical transport and explicit gravity-transport repair.
Preserve unchanged raw files and actual split identities, versions and hashes.

Prepare the controlled matrix with
`embodichain.learning.dynamics.comparison.prepare_comparison`, supplying both
immutable teacher datasets, original source/dependency paths, five seeds and
complete model/train configuration. `run_comparison` executes frozen commands.
Both trainers share initialized tensors, normalizers, sample indices, updates
and the validation selector. Use distinct output directories and explicitly
resume incomplete work from its own optimizer/RNG checkpoint. This controlled
protocol does not establish literal paper-recipe reproduction.

## Freeze supported-environment evaluation data

`official_evaluation_data.py` retains all requested outcomes, including physical
failures and NaN/Inf, with no rejection/resampling. It runs original physics and
sampling unchanged. Pure Torch coordinate/observation files load directly under
an isolated namespace; their paths/hashes are recorded separately. The worker
does not import the EmbodiChain package, DexSim or Newton.

Example `original-eval-data.json`:

```json
{
  "source": "/path/to/neural-robot-dynamics",
  "robot": "ant",
  "output_dir": "/workspace/exp_data/ant/original-eval-passive",
  "num_envs": 64,
  "trajectories": 2048,
  "horizon": 1000,
  "seed": 1000000,
  "initial_states": "sample",
  "motion_mode": "passive"
}
```

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$NERD_WARP_ROOT" "$NERD_DATA_PYTHON" scripts/experiments/neural_dynamics/official_evaluation_data.py --config original-eval-data.json --trust-external-code
```

Use new directories for the other scenarios. Set `motion_mode: random_effort`,
or set `motion_mode: fixed_policy` with Ant policy
`{"kind": "pd", "kp": 20.0, "kd": 1.0}`. Cartpole PD uses
`{"kind": "pd", "gains": [5.0, 50.0, 5.0, 20.0]}`. These are declared baselines,
not the paper's learned policy. Original raw states/efforts are preserved beside
canonical q/qd/efforts and measured physics.

Choose seeds outside every stored training/validation/test candidate range and
audit actual initial states. Current collection uses generation seed plus
candidate index, so comparing top-level seeds alone is insufficient. Reuse one
immutable corpus per robot/scenario for both trainers and all model seeds.

## Evaluate both exports against one corpus

Run the historical evaluator in a fresh EmbodiChain Python process, with the
old Warp dependency directory prepended to `PYTHONPATH`. Do not replace installed
current Warp or import DexSim/Newton into that process. Example spec:

```json
{
  "source": "/path/to/neural-robot-dynamics",
  "robot": "ant",
  "model": "/workspace/exp_data/ant/official-seed0/best-export",
  "teacher_dataset": "/workspace/exp_data/ant/original-eval-passive",
  "output": "/workspace/exp_data/ant/official-seed0/passive.json",
  "device": "cpu",
  "num_envs": 64,
  "trajectories": 2048,
  "horizons": [1, 10, 100, 1000],
  "seed": 1000000,
  "initial_states": "sample",
  "motion_mode": "passive",
  "cpu_threads": 4
}
```

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$NERD_WARP_ROOT" "$EMBODICHAIN_PYTHON" -m embodichain.learning.dynamics.reference_rollout --spec historical-eval.json --trust-external-code
```

The evaluator verifies generation versions/source/physics, raw/canonical state
transport, ordered teacher contents and feedback actions. The neural simulator
starts from the recorded initial state, evolves independently, and computes
feedback from its own state. Later teacher states are never copied into it.
Model-runtime versions and teacher-generation versions are recorded separately.
Omitting `teacher_dataset` retains live teacher generation for diagnostics.

For the migrated export change only model/output. Repeat for both robots, all
three scenarios and five seeds, retaining complete budgets. Use
`embodichain dynamics` for separate current-Newton deployment evaluation; it also
accepts a verified `teacher_dataset`, including fixed-PD feedback traces.

Attach results with `annotate_rollout_evidence`, then call
`write_comparison_report`. Training completion, numerical parity, hashes and all
error/failure/nonfinite/return gates are separate checks. Completed commands and
short pilots cannot substitute for full trained-model accuracy qualification.

## Optional reference tests

For the opt-in checkpoint tests, set `NERD_REFERENCE_ROOT` to a pinned checkout
with real LFS weights. When that checkout is original Warp main, also set
`NERD_REFERENCE_DEPS` to the matching isolated Warp dependency directory. A
checkout containing LFS pointer files cannot supply the pretrained checkpoint.
For the fixed-teacher tests, `NERD_SUPPORTED_EVAL_DATA` contains directories named
`cartpole-passive`, `ant-passive`, and the corresponding `random_effort` and
`fixed_policy` directories. These tests inspect original generation provenance
separately from the model-inference runtime.
