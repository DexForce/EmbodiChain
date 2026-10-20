# Five-seed bounded NeRD comparison — 2026-10-20

**Status: all 40 short training cells completed; full qualification remains pending.** Each cell ran 100 updates with batch 512/history 10. This does not replace the 4,995,000-update budget or full-size data/evaluation.

## Scope and reproducibility

- Two robots × two teacher domains × two trainers × seeds 0–4 = 40 cells, 4,000 optimizer updates total.
- Networks use the official 192/384 embedding, six layers, twelve heads and 64-unit output head in FP32. Actual initial artifacts preserve the required feature order; sorted plan mappings are not used to infer it.
- Current Newton datasets: 64 trajectories × 32 transitions, splits 57/1/6, default initial states; Ant collection uses effort scale 0.05. Historical datasets: 48 × 100, splits 32/8/8, measured gravity and historical physical-v2 provenance.
- Each paired row shares exact initial weights, statistics, indices and checkpoint-selection controls. Both teacher origins and source hashes are preserved.
- This is controlled matched-data comparison, not literal paper-recipe reproduction. See [the protocol audit](neural-dynamics-official-protocol-audit-20261020.md) for sampling/statistics/validation differences.
- Observed sequential matrix time was about 98 s for Cartpole and 139 s for Ant on this workstation, excluding preparation. This small-data timing must not be extrapolated to the full corpus.

## Paired offline results

The table shows the one-sided 95% paired bootstrap upper bound on relative degradation, using five seeds and 10,000 resamples. Every short-run offline metric is below the predeclared 5% margin. These conditional results do not establish final-budget quality. q/qd norms can combine coordinates with different units; they are not Cartesian distances.

| Robot | Teacher | Metric | Upper bound | Short-run criterion |
|---|---|---|---:|---|
| cartpole | official_warp | q_error_norm | 0.7563% | passed |
| cartpole | official_warp | qd_error_norm | -0.0300% | passed |
| cartpole | current_newton | q_error_norm | 0.0609% | passed |
| cartpole | current_newton | qd_error_norm | 2.1676% | passed |
| ant | official_warp | q_error_norm | 0.0786% | passed |
| ant | official_warp | qd_error_norm | 0.0514% | passed |
| ant | official_warp | base_position_error | 0.9660% | passed |
| ant | official_warp | base_orientation_error_rad | 0.6761% | passed |
| ant | current_newton | q_error_norm | 0.0042% | passed |
| ant | current_newton | qd_error_norm | 0.0515% | passed |
| ant | current_newton | base_position_error | -0.1524% | passed |
| ant | current_newton | base_orientation_error_rad | 0.1560% | passed |

The largest upper bound is about **2.17%**, for current-Newton Cartpole velocity error. This is evidence of short-run port agreement, not a claim that the learned dynamics are accurate enough for deployment.

## Bounded closed-loop and rollout evidence

Seed 0 was additionally evaluated in both domains: 24 measured runs, three trajectories each, horizons 1/10/12, two environments with a partial final batch. All 12 paired teacher-content and policy identities match. Source, weights, manifest, HDF5 and JSONL hashes were verified for this bounded scope. The formal report still requires 2048 trajectories, horizon 1000 and five rollout seeds.

Historical execution uses the actual original Warp environment/integrator; current-domain execution uses the registered Newton provider. See [historical rollout results](neural-dynamics-historical-rollout-20261020.md). The models are not numerically identical and neither domain is declared physically qualified.

| Current-Newton scenario | Learned failure rate, both trainers | Teacher failure rate |
|---|---:|---:|
| Cartpole passive | 0.6667 | 0.3333 |
| Cartpole random_effort | 0.6667 | 0.6667 |
| Cartpole fixed_policy | 0.3333 | 0.0000 |

For historical Ant fixed PD, teacher return was 9.7618 versus 1.4588/1.5726 for the official/migrated trained models. Current-Newton Ant returned 4.3053 versus 2.2850/2.3616. These are material teacher errors despite close trainer-to-trainer results. Cartpole negative returns also stop at failure, so a less negative return must not be interpreted as superior control while ignoring the failure count.

## Evidence and remaining requirements

- [Machine-readable five-seed summary](neural-dynamics-five-seed-20261020.json).
- [Verified bounded attachments and paired identities](neural-dynamics-bounded-pairing-20261020.json).
- Specs, command/stdout/stderr receipts, common weights/statistics/order and training/paired-error plots are retained under `/tmp/nerd-five-seed-{cartpole,ant}-20261020`.
- Raw current-domain rollout traces are under `/tmp/nerd-current-paired-{cartpole,ant}-20261020`; historical traces are under the matching `nerd-historical-paired-*` directories.
- The exact pre-runtime-metadata reference/workflow sources used by these runs are preserved with hashes in `/tmp/nerd-five-seed-source-snapshot-20261020`.
- Future runs now capture per-worker runtime/driver identity. Existing 40 receipts were not backfilled; they retain their original Torch/device/source records and the documented workstation context.

Remaining: full-size data, declared full training budget, complete long-rollout and stability/return gates, and an explicitly identified paper data/validation recipe before any literal reproduction claim. No full-budget job was launched.


## Latest installation and code checks

The latest paired local EC/SDK wheel is identified by
[its artifact manifest](neural-dynamics-local-artifacts-historical-20261020.json).
Actual imports, named-provider metadata and exact dependency matching were checked.
The installed EC package ran both historical CPU cases without importing or
attempting to import DexSim/Newton in those isolated processes.

One tiny installed CUDA probe per historical robot also completed on RTX 5090:
B1/count1, passive/default initialization, horizons 1/3. Cartpole Warp 1.8.0
portable/original maximum absolute error was zero; Ant Warp 1.5.1 was 1.4782e-5,
within atol=1e-5/rtol=1e-4. Both physical contracts matched. Old Warp emitted a
cuDeviceGetUuid initialization diagnostic, retained in the logs. This is a
bounded version-compatibility check, not a broad GPU or long-horizon qualification.

Focused reference/report/comparison/shared-metric tests: 36 passed with actual
legacy references enabled. Runtime-provenance tests: 31 passed; an actual CUDA
identity-query smoke also passed without changing Torch RNG state. Context tests:
40 passed. Public API coverage: 2748/2748. Updated neural-dynamics Sphinx pages
built with warnings treated as errors. Black 26.3.1 and diff checks passed.

No formal-budget or long-evaluation job is running. Full acceptance remains
pending the full experiment scope and compute-budget decision; the existing
budget question has not been answered. Code and bounded evidence are available
for review while that decision is outstanding.
