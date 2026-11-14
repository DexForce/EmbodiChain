# Cartpole / Ant APG–PPO policy benchmark

This benchmark trains APG and PPO in the **same frozen NeRD dynamics model**, then
runs each saved policy with deterministic feedback in its matched analytical
Newton Scene. It reuses the registered `DifferentiableEnv` tasks and production
RL trainers. PPO has no access to dynamics gradients. This compares policy
training algorithms and transfer; it does not compare NeRD with analytical
training, retrain NeRD, or reproduce a paper's convergence budget.

## Prepare models and reference assets

Run from a source checkout with the project's dependencies installed. Keep
resources and results outside Git. The example's ordinary `prepare` command also
downloads supervised datasets; the policy benchmark only needs the two exports
and reference assets. Download just those pinned resources with:

```python
from pathlib import Path
import json
import shutil
from huggingface_hub import hf_hub_download
from examples.learning.nerd._resources import (
    _MODEL_REPO, _MODEL_REVISION, _PAYLOAD_MANIFEST_SHA256,
    _require_hash, prepare_reference_assets,
)

prepare_reference_assets()
for robot in ("cartpole", "ant"):
    target = Path("outputs/nerd-policy/models") / robot
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "weights.pt"):
        source = hf_hub_download(
            repo_id=_MODEL_REPO, revision=_MODEL_REVISION,
            filename=f"neural_dynamics/nerd/{robot}/v1/{name}",
        )
        shutil.copy2(source, target / name)
    _require_hash(target / "manifest.json", _PAYLOAD_MANIFEST_SHA256["model"][robot])
    manifest = json.loads((target / "manifest.json").read_text())
    _require_hash(target / "weights.pt", manifest["weights_sha256"])
```

The pinned model revision is `78e581526570069410bf269b38c0c52c800b24a7`.
Custom exports are accepted only if the reference Scene can reconstruct their
recorded physical domain. The benchmark validates the actual physical signature,
gravity, timestep, joint order, contact schema and codec. Matching the profile
name alone is insufficient. Historical runtime identity is retained in the
export; current evaluation records its actual runtime separately.

## Run the frozen seed matrix

The committed [recipe](nerd_policy.yaml) defines 3 seeds × 2 tasks × 2 algorithms:

| Shared condition | Value |
| --- | --- |
| Training seeds | 0, 1, 2 |
| Vector environments / rollout steps | 64 / 16 |
| Updates / transitions per run | 512 / 524,288 |
| Actor / initial log standard deviation | 64–64 tanh, tanh output / −2 |
| Optimizer / learning rate / maximum gradient norm | Adam / 0.00025 / 0.5 |
| Discount / entropy coefficient | 0.99 / 0 |
| APG backpropagation segment | 16 steps |
| PPO epochs / minibatches / GAE lambda / clip | 4 / 4 / 0.95 / 0.2 |
| Evaluation checkpoints | 0, then every 65,536 transitions, including final |
| Held-out evaluation starts | 32, fixed seed 10000, shared across all runs |
| Episode limits | Cartpole 500 / Ant 1000 control steps |
| Control interval / dtype / CPU threads | 1/60 s / float32 / 2 |

Both algorithms instantiate the same actor-critic policy with identical initial
weights, distribution bounds and no observation normalization. APG leaves the
critic unused; PPO optimizes it. Reset states are indexed by world and episode,
so different completion orders cannot change either world's sequence of starts.
CPU sampling fixes the reset bank across GPUs. No hyperparameters are tuned per
seed, and the final checkpoint is reported without selecting the best result.

```bash
python -m scripts.benchmark.rl.nerd_policy \
  --models outputs/nerd-policy/models \
  --output outputs/benchmarks/nerd-policy \
  --device cuda:0

python -m scripts.benchmark.rl.nerd_policy \
  --output outputs/benchmarks/nerd-policy --report
```

Use `--seed 0`, `--robot cartpole` or `--algorithm apg` to run a matrix subset.
For independent GPUs, start one process per seed with a different
`CUDA_VISIBLE_DEVICES`, the same configuration and the same output root. Each
process runs paired algorithms sequentially; order reverses on odd seeds.
Avoid other training on those GPUs during timing. Aggregate after every process
finishes. The report rejects incomplete matrices or mismatched actor/model,
reset-bank, evaluation-start or recipe identities.

Use `--config FILE.yaml` with a smaller budget for smoke checks; report those
separately. Each run retains checkpoints and partial JSON progress if interrupted.
Policy exports are explicitly **not resumable trainer checkpoints**. Completed
results cannot be overwritten; use a fresh destination for repeated experiments.

## Outputs and interpretation

The output root contains:

- `report.md`, `summary.json`, `recipe.json` and PNG/SVG curves for each task;
- `training.csv`: per-update losses, gradients and completed-episode statistics;
- `evaluation.csv`: all checkpoints, analytical returns, lengths, failures,
  quality and synchronized evaluation/training time;
- `episodes.csv`: every requested evaluation episode, including invalid outcomes;
- `<task>/<algorithm>/seed-N/`: model/source/runtime identities, full run JSON,
  partial progress and policy-only checkpoints, including initialization.

Return and time tables show **mean ± sample standard deviation across training
seeds**. Plot bands have the same meaning; they are not confidence intervals.
Training curves use the trainers' rolling last-100 completed-episode statistics.
Evaluation runs each held-out start exactly once, without auto-resetting or
replacing early failures. Evaluation happens after policy training in a separate
Scene, so it does not consume training RNG or mutate the training environment.

Training time includes collection, optimization and trainer bookkeeping, with
Torch **and** Warp synchronization at timing-window boundaries. Two throwaway
updates warm compilation and optimizers; their environment, policy and RNG are
then reconstructed. Pre-training setup, checkpoint/report I/O and analytical evaluation are
excluded from training time. Pre-training setup and evaluation are reported
separately; post-training analytical Scene creation is excluded from evaluation
timing. Published NeRD pretraining
and dataset generation were not rerun, and their historical cost is unavailable.
Torch peak allocator memory excludes Warp and native allocations.

Cartpole's original reward is nonpositive, so an early failed episode can have a
less negative return. Read return together with episode length, failure rate and
mean absolute pole angle. Ant's standing/heading bonus can reward survival
without locomotion; inspect mean forward velocity as well. Task rewards and
failure rules are unchanged. Nonfinite episodes are preserved and counted; a
checkpoint containing one has an undefined mean return rather than dropping it.

The fixed budget establishes a reproducible baseline, not convergence or a
paper-scale accuracy claim. Results remain tied to the recorded physical domain,
weights, source and runtime. Seeds control RNG streams; GPU kernels can differ in low-order floating-point
results across runs or GPUs, and bitwise replay is not claimed.
