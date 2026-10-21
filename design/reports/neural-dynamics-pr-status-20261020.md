# NeRD implementation and evidence index

Snapshot: 2026-10-21. The filename is retained for existing PR links.

Cartpole has completed its **4,995,000-update training budget** and passed the
fixed official-domain offline acceptance protocol. Its validation-selected best
is update 4,940,000. Ant has no final-budget acceptance in this source snapshot.
Current-Newton deployment accuracy and broader multi-seed qualification remain
pending. The stopped 40-cell training plan is preserved as historical evidence.

The [design contract](../neural-dynamics-solvers.md) describes supported behavior.
The [full-budget report](neural-dynamics-full-budget-20261021.md) records the
latest measured results and their limits. Earlier dated reports retain their
original scope; a pilot result does not become full-budget evidence.

## Implementation map

| Responsibility | Owner |
|---|---|
| Named solver registration | `pyproject.toml`: installed `dexsim.newton_solvers` entry point `nerd` |
| Native and functional inference, reset and history | `embodichain/lab/sim/physics/solvers/nerd.py` |
| State, velocity, contact and relative-target encoding | `embodichain/compute/dynamics/codec.py` |
| Measured reference assets and analytical collection | `embodichain/lab/sim/physics/solvers/{_reference_assets,collection}.py` |
| HDF5 shards, trajectory splits and budgeted raw-data residency | `embodichain/data_pipeline/dynamics/{shards,_resident}.py` |
| Training, resume, validation and export | `embodichain/learning/dynamics/{trainer,workflow}.py` |
| Original sampling replay and explicit numerical policy | `embodichain/learning/dynamics/{_sample_order,_numerics}.py` |
| Original datasets and historical rollout verification | `embodichain/learning/dynamics/{reference,reference_rollout,_reference_traces}.py` |
| Model, frozen normalizers and portable checkpoint | `embodichain/learning/models/nerd/` |
| CLI and runnable task configurations | `embodichain/cli/dynamics.py`; `extensions/embodichain_tasks/configs/tasks/classic_control/{cartpole,ant}/` |
| Reproducible original-entry comparisons | [Experiment runners](../../scripts/experiments/neural_dynamics/README.md) |
| Sampling benchmark | `scripts/benchmark/learning/dynamics_sampling.py` |

`embodichain dynamics generate/train/eval/export` owns the installed workflow.
Native data generation uses current Newton, including when
`physics_profile=warp_reference` requests measured legacy parameters. The formal
Cartpole/Ant training round instead imports data generated in the original
supported Warp environments. Historical model exports do not gain native-Newton
physics compatibility by changing annotations.

The solver uses the installed provider name through `solver_type: nerd`, with
an exported checkpoint in `options`. It validates timestep, joint order, gravity,
contact schema and measured physics signature before stepping. It supports
homogeneous Cartpole/Ant scenes and first-order functional derivatives; CUDA
Graph, second-order derivatives, external wrenches and heterogeneous coupling
remain outside the implemented contract.

## Runtime and training identity

| Component | Identity |
|---|---|
| Original source | `NVlabs/neural-robot-dynamics` commit `370e89d2b8ce68aa3578aa7812859c681b7502bc` |
| Original supported runtime | Python 3.8.20, Torch 2.2.2, NumPy 1.24.4; Cartpole Warp 1.8.0, Ant Warp 1.5.1 |
| Formal migrated training runtime | Python 3.11.14, Torch 2.7.0+cu128, Warp 1.17.0, Newton 1.7.0 |
| SDK used by the formal round | DexSim 0.5.1, MR !1483 build `b61e920fbb222e50213574ad57d885e7b96148b9` |
| Required SDK integration | [DexSim MR !1483](http://192.168.3.16/Engine/dexsim/-/merge_requests/1483); version 0.5.1 alone does not identify an extension-enabled build |
| Formal model budget | One seed per robot, original model width, 999 trained epochs × 5,000 updates |
| Formal sampling and numerics | Captured original DataLoader shuffle/drop-last stream; frozen original initial weights/RMS; explicit `reference`/math profile |

The standalone [math-evaluation entry](../../scripts/experiments/neural_dynamics/official_math_rollout.py)
was rerun in Python3.8/Torch2.2/Warp1.8 with the accepted Cartpole best export;
all saved NPZ arrays exactly match the original accepted passive-motion evidence.
This publication check covers 16 trajectories at horizons 1/10/100.

The original-default pretrained baseline remains unchanged. The explicit math
profile is a documented controlled variant, not an untouched official-default
training reproduction. Older default exports/checkpoints retain their behavior.

## Evidence by question

| Question | Report |
|---|---|
| Full-budget progress and Cartpole final acceptance | [Full-budget report](neural-dynamics-full-budget-20261021.md) |
| What changes across Torch/Warp versions? | [Numerical attribution](neural-dynamics-numerical-attribution-20261020.md) |
| Original training entry versus migrated training | [Small-budget comparison](neural-dynamics-small-budget-20261020.md) |
| Were teachers generated in supported original environments? | [Supported reference evaluation](neural-dynamics-supported-reference-evaluation-20261020.md) |
| Published model versus its restored portable representation | [Official-model alignment](neural-dynamics-official-model-alignment-20261020.md) |
| CUDA, targets and preprocessing attribution | [CUDA alignment](neural-dynamics-cuda-alignment-20261020.md) |
| Physical parameter differences | [Parameter measurements](neural-dynamics-parameter-alignment-20261020.json) |
| Earlier trained-model comparison | [Trained comparison](neural-dynamics-trained-comparison-20261020.md) |
| Current-Newton generate/train/export/infer smoke | [Implementation and v2 smoke](neural-dynamics-implementation-v2-20261020.md), [receipts](neural-dynamics-v2-smoke-20261020.json) |
| Current publication checks: 595 unique tests, API, formatting and wheel | [Publication validation](neural-dynamics-publication-20261021.json) |
| Initial API and integration verification | [Initial PR validation](neural-dynamics-pr-validation-20261020.json) |
| Initial network/gradient/update parity | [Cartpole](nerd-cartpole-parity-20261020.json), [Ant](nerd-ant-parity-20261020.json) |
| Historical 40-cell, 100-update controlled experiment | [Five-seed pilot](neural-dynamics-bounded-five-seed-20261020.md), [summary](neural-dynamics-five-seed-20261020.json) |
| Earlier closed-loop and teacher pairing | [Historical rollout](neural-dynamics-historical-rollout-20261020.md), [pairing](neural-dynamics-bounded-pairing-20261020.json) |
| Bounded input-performance measurements | [Input pipeline](neural-dynamics-input-20261020.md) |
| Historical paired installation artifacts | [Local artifacts](neural-dynamics-local-artifacts-v2-20261020.json) |

## Acceptance boundaries

- Cartpole final and validation-selected exports have lower aggregate q/qd
  endpoint and prefix means than the published default baseline for the fixed
  16 trajectories per mode at horizons 1/10/100. This does not imply improvement
  on every trajectory, universal stability or a distribution-wide guarantee.
- The published Cartpole configuration uses 5M training transitions; this round
  uses approximately 1M. Published model update completion is unknown. Data,
  runtime and numerical-policy differences prevent a single-factor causal claim.
- Cartpole final offline normalized test MSE is approximately `1.52e-7`.
  Offline accuracy and completed training budget do not establish current-Newton
  deployment qualification. Native inference still enforces its physics contract.
- The current-Newton native workflow has bounded smoke evidence. Ant final
  acceptance and larger multi-seed/long-rollout qualification remain separate.
- Reports and aggregate JSON are reviewable source artifacts. Local `/workspace`,
  `/root/.cache` and `/tmp` links identify retained operator artifacts; they are
  not remote download URLs. Dataset shards, weights and complete logs remain in
  the experiment storage rather than the source checkout.
