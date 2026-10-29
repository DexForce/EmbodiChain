# S03：SDK 集成后 10×10 五次正式性能测量完成

2026-10-22，用户确认低预算视频并回复“可以，按 10×10 测量”；DexSim SDK 修改后按指令重测。当前批次 `outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift`；两平台各五个独立无相机进程，10/10完整运行，无性能样本替换。实际10子步/10次VBD迭代、自碰撞及CUDA Graph均已验证；全部Graph状态为captured。每次793外步、794个有限状态帧、13.216666667秒，前30步排除。

| 五次均值 ± 样本标准差 | EmbodiChain | Isaac Lab |
| --- | --- | --- |
| 同步step，ms | 12.722 ± 0.434 | 13.463 ± 0.316 |
| 进程显存采样最大值，MiB | 758 ± 0 | 706 ± 0 |
| 从step推导的吞吐，steps/s | 78.680 ± 2.699 | 74.308 ± 1.733 |

同期对照中 EC 外步耗时低 5.51%，五组成对差值均由 EC 更快。EC 的收益含参考配置对齐（自碰撞检测间隔 -1、三色调度、缓存），不是同调度下的纯内核加速；EC 状态读取仍更贵，结论只限物理外步。原生控制、计时范围、Newton版本及固定空连杆差异见报告；最终布形态只作描述，不评价质量优劣，也不筛选性能样本。SDK 重测前的批次（EC 17.766 ± 0.451 / IL 14.784 ± 0.212 ms）与 10×40 基线同属旧记录，归档在 `history/formal_before_sdk`，不与本次混合计算降幅。

- [SDK 正式三表报告与性能图](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/REPORT.md)
- [最新并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/videos/s03_comparison.mp4)
- [10次性能CSV](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/measurement/results/runs.csv)
- [全部7930条step计时CSV](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/measurement/results/steps.csv)
- [验证与初始适配偏差说明](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/VALIDATION.md)
- [SDK 重测证据](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/analysis/SDK_REMEASUREMENT.md)


首个临时原生适配检查因布角保持阶段差超2 mm停止；随后一次同EC诊断及两次原生复查通过，阈值和物理代码未改。首个性能样本及失败摘要保留，原生结果和日志删除。协议CPU测试10项通过；全部输入/视频哈希、无相机运行、有限值状态和绘图数据核验通过；SDK最终267项相关回归通过，并用S04五次记录的形变状态做180次局部核对照，位移/Hessian误差均为0。


按用户最新要求，只保留当前有效版本；10个旧S03顶层条目已删除，释放约1.68 GiB。当前报告、视频和复现输入合并在outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift，60个原始测量文件SHA256逐一一致，S01/S02文件哈希一致。
