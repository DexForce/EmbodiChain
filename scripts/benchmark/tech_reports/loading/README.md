# Loading comparisons (L-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
L-02 sleep=False 公平性能待补；L-03 三机器人本轮完成；L-04 仅部分资产预检；L-05 待开展。全部采用多 arena。

本 draft 导入已有实现。机器路径／冻结依赖适配仍未完成，
[当前快照](../reporting/releases/simulation-report-2026-10-28/README.md)记录已验证的离线入口与数据上传状态。

- `experiments/`: experiment definitions, scene-size matrices, cold/warm/reset
  protocols, and comparison orchestration.
- `platforms/`: platform-specific loading, rebuild, and reset execution adapters.
- `../assets/loading/`: small authored input descriptions and asset references.

For subsequent loading measurements, separate
stages and cold/warm conditions, record latency, memory and failure points, and
use matched asset/configuration hashes across platforms. Run each platform in
its own process and Python environment. Write run artifacts under
`outputs/benchmarks/`; reuse the planned common measurement/reporting contracts.

See the [benchmark layout](../README.md#technical-report-comparison-layout).
