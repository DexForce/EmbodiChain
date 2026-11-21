# S02 已完成

批次：outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/。第一组 S01 保持完成状态，314 个归档文件哈希复核未变。

- 固定 s00/nominal、10 子步/40 VBD 迭代，两平台各 5 次独立无相机采样，共 10 次，全部通过，无重试/替换。
- 独立媒体运行两侧各 1 次，也全部通过；30 fps 四画面对比视频共 494 帧，时间映射误差为 0。
- EmbodiChain 同步 step：36.476 ± 1.116 ms；Isaac Lab：39.002 ± 1.297 ms。均排除最初 30 帧，为五次均值 ± 样本标准差。
- 放置中心误差：1.277 ± 0.986 mm / 2.899 ± 2.109 mm，全部节点投影在目标矩形内。
- Isaac Lab r04 最低提起高度 86.897 mm，仍超过 80 mm 门槛；该波动保留在原始数据、表格和五次重复图中。
- 同后端原生 DexSim–EmbodiChain 适配检查通过：物理模型数组一致、阶段运动和任务指标在约定容差内。原生数据及日志已清理，不进入统计或归档。
- 本轮未新增跨 DexSim 构建版本实验；engine/dexsim 是用户提供的源码与运行参考位置。
- 12 项 focused CPU tests 通过，变更 Python 格式/源码语法/相对链接检查通过。

## 产物

- [报告及曲线](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/REPORT.md)
- [对比视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/videos/s02_comparison.mp4)
- [五次重复图](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/results/s02_repetitions.png)
- [测量汇总 CSV](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/results/s02_measurements.csv)
- [主采样高度曲线 CSV](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/results/s02_height_curves.csv)
- [媒体响应曲线 CSV](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/videos/s02_response_data.csv)
- [全部正式尝试](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s02_fixed_five/attempts.json)

所有主采样与媒体运行的 states.npz/timing.csv/config.json/manifest.json 均完整，输入和源码快照已保留。
比较限定为固定初态下的整体平台和几何响应，不单独归因于求解器，也不作鲁棒性排名。
旧 dexuni_benchmark 已于 2026-10-27 删除；旧 S02 试跑的源码、配置及日志留在清理证据 ZIP，旧二进制中间结果已清理。
