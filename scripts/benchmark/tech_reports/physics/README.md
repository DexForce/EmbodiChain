# Physics comparisons (P-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
P-01／P-02 已完成本轮质量实验，P-03 已完成质量部分；性能缺口及 P-04～P-07 计划见当前快照。

本 draft 导入已有实现。机器路径／冻结依赖适配仍未完成，
[当前快照](../reporting/releases/simulation-report-2026-10-28/README.md)记录已验证的离线入口与数据上传状态。

- `experiments/`: experiment definitions, physical scenarios, solver/step
  matrices, quality protocols, and comparison orchestration.
- `platforms/`: platform-specific stepping and evidence-collection adapters.
- `../assets/physics/`: small scenario inputs, input descriptions and references.

For new physics measurements, Align timestep, solver,
geometry, materials, and other declared invariants before comparisons. Record
analytical error, convergence, physical/task quality and quality-qualified
performance separately. Changed geometry or dynamics requires physical
execution validation. Unsupported conditions retain reasons instead of zero
measurements. Each platform runs in its own process and Python environment.

Production physics, environment components, and task validators retain their
existing owners. Run artifacts belong in `outputs/benchmarks/`.
See the [benchmark layout](../README.md#technical-report-comparison-layout).
