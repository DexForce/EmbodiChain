# Historical NeRD rollout evidence — 2026-10-20

Executed 12 bounded CPU runs using the newly trained seed-0 exports: two robots × two trainers × three motion modes. All runs completed, all six pairs used identical teacher trajectories and policy identities, and all original-model restoration probes passed with maximum absolute state error **0**. This establishes executable historical-domain comparison; it does not establish full-budget accuracy, paper reproduction, or deployment of historical physics into current Newton.

## Protocol

- Original untouched NeRD main commit `370e89d2b8ce68aa3578aa7812859c681b7502bc`; original `NeuralEnvironment`, Featherstone teacher and `TransformerNeuralIntegrator` student.
- Cartpole uses Warp 1.8.0 / 5 teacher substeps; Ant uses Warp 1.5.1 / 16 teacher substeps. Each run is a separate CPU process with CUDA hidden. The scoped CPU stream adapter maps the original CUDA stream request to `None`; source math remains unchanged.
- Both trainers completed 100 updates on the historical physical-v2 data. These are bounded training checkpoints, not converged models. Best-validation export from each trainer is evaluated without changing weights or manifests.
- Batch size 2, 3 trajectories (actual batches 2 + 1), original sampled initial states with seeds 500–502; horizons 1, 10 and 12. Passive and random effort share commands within a pair. Fixed PD recomputes teacher and student controls from their respective states; trace inspection confirms their commands diverge.
- Each run additionally compares reconstructed original `ModelMixedInput` with the portable model inside the same original integrator for 11 steps, covering history length 10 and its first rollover. All 12 probes were finite and exactly equal.
- Live source physical arrays, gravity `[0, -9.81, 0]`, axis, timestep, substeps, asset hash and codec metadata matched the recorded historical contract. No recorded physical field was missing.
- Metrics use shared canonical Z-up / COM velocity transport and shared task reward definitions. Ant q MAE covers its scalar joints; base position and quaternion angular errors are reported separately. Horizon state errors average steps 1 through H; returns sum until the first failure. All requested trajectories remain in evidence.

## Measured results at horizon 12

All 12 runs had 3/3 numerically valid trajectories and **0/3 teacher failures, student failures, teacher nonfinite states and student nonfinite states**. No rows were rejected or resampled. The two trained models are not numerically identical; the table reports both.

| Robot | Motion | Trainer | q MAE | qd MAE | Base position | Base angle (rad) | Fixed-PD return |
|---|---|---|---:|---:|---:|---:|---:|
| cartpole | passive | official | 0.06185331 | 0.5009031 | 0 | 0 | — |
| cartpole | passive | embodichain | 0.06182037 | 0.5007337 | 0 | 0 | — |
| cartpole | random_effort | official | 0.08175079 | 0.5582347 | 0 | 0 | — |
| cartpole | random_effort | embodichain | 0.08169176 | 0.555877 | 0 | 0 | — |
| cartpole | fixed_policy | official | 0.06957492 | 0.5151157 | 0 | 0 | -170.8377 |
| cartpole | fixed_policy | embodichain | 0.06958081 | 0.5154237 | 0 | 0 | -170.8884 |
| ant | passive | official | 0.3156452 | 2.652962 | 0.1061554 | 0.247806 | — |
| ant | passive | embodichain | 0.3172099 | 2.657473 | 0.1043622 | 0.2473033 | — |
| ant | random_effort | official | 0.2701225 | 2.734384 | 0.08245032 | 0.1912716 | — |
| ant | random_effort | embodichain | 0.2708459 | 2.725167 | 0.08210292 | 0.192527 | — |
| ant | fixed_policy | official | 0.2945366 | 2.536861 | 0.1055715 | 0.2423552 | 1.458784 |
| ant | fixed_policy | embodichain | 0.2932671 | 2.54465 | 0.1017988 | 0.2401732 | 1.572607 |

Fixed-PD teacher returns: Cartpole **−163.68427**, Ant **9.7618113**. These fixed baselines are explicitly not the paper policies. The short Ant learned rollouts have materially lower return than their teacher; zero failures over 12 steps must not be read as accuracy qualification.

## Pairing and evidence integrity

Recomputed every trajectory content digest from the saved teacher state/control arrays and logical seed/ID, then recomputed the ordered selected-teacher digest. All six pairs match exactly despite distinct model artifacts. Verified trace, JSONL, model weights and manifest hashes against the result receipts. Each pair also matches teacher origin, dtype, count and policy hash.

| Robot | Motion | Ordered teacher SHA-256 | Policy SHA-256 |
|---|---|---|---|
| cartpole | passive | `1754aa11691ee45e4f285796f5323b859b14a747250af4a2efca1053048118e5` | none |
| cartpole | random_effort | `b28a63ba9822549a910ac640953ef581c0f0fe5a9cd0ef029f2221341c192d71` | none |
| cartpole | fixed_policy | `427bcb6f013d43eb31c00e0f071ce085f2b079b5c134263a8b0bbe1d91549b86` | `5f373a5c1aab23985abc978b15ed04891beed8a47e5350e6064f2d9d5ab5930a` |
| ant | passive | `4522f97dbaf9841a2df00630161927836633ac3f4639097e8f2689b6ac423956` | none |
| ant | random_effort | `16e537821f84443b3ec7328bc820049ce3b198b96fc5ad5b08fb83965447b113` | none |
| ant | fixed_policy | `c87a0b7ed58a157ec0cf6818d0e01f3ee1847f6083768b4a4bb807c6ad32ec83` | `2dfde4898331fc07566d72fdfd90e78cbf60eeb6f006e9d151c5d636992f937a` |

Artifacts are local verification receipts; they must be retained when sharing this report:

- [Cartpole specs, logs, JSON metrics, HDF5 traces and JSONL trajectories](/tmp/nerd-historical-paired-cartpole-20261020/).
- [Ant specs, logs, JSON metrics, HDF5 traces and JSONL trajectories](/tmp/nerd-historical-paired-ant-20261020/).
- [Independent pairing/integrity receipt](/tmp/nerd-historical-paired-validation-20261020.json); SHA-256 `e8b6bcbba7034a6c4b7eeb33c92fd8d2bc7c573ea50398fdc80f4728f7deb4bb`.
- Model roots: `/tmp/nerd-five-seed-{cartpole,ant}-20261020/official_warp/seed-0/{official,embodichain}/best-export`. Adjacent `result.json` files record each completed 100-update training run.
- [Harness](/home/dex/.codex/worktrees/e142/EmbodiChain/embodichain/learning/dynamics/reference_rollout.py); [focused real-runtime tests](/home/dex/.codex/worktrees/e142/EmbodiChain/tests/learning/dynamics/test_reference_rollout.py), **5 passed** with both pinned legacy environments enabled.

Reproduce one receipt from its `*-spec.json` using `python -m embodichain.learning.dynamics.reference_rollout --spec <spec> --trust-external-code`, with the matching legacy dependency directory first on `PYTHONPATH` and a fresh output destination.

## Remaining qualification

Only seed 0 and three trajectories per mode were evaluated here. Five-seed aggregate rollout gates, 2,048 trajectories through horizon 1,000, full training budgets and paper-policy comparisons remain unexecuted in this report. Historical runtime measurements must be compared within the historical physical domain; they do not authorize loading these old-physics models into the current-Newton provider.
