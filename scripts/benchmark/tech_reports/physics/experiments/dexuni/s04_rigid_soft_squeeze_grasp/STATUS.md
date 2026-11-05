# S04：SDK 集成后五次正式性能测量完成

[报告](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/REPORT.md) · [两平台并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/videos/s04_comparison.mp4) · [每次测量](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/measurement/results/runs.csv) · [验证与审计](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/VALIDATION.md) · [SDK 重测证据](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/analysis/SDK_REMEASUREMENT.md)

用户接受修正版视频并回复“可以，按修正版重测”；DexSim SDK 修改后，两平台各五次独立无相机进程已完成：10子步×12次VBD迭代、所有Graph captured、每次实际12秒/720外步/721有限状态帧；前30步预热，后690步统计。输入和模型与预览全部一致。

EmbodiChain 14.839 ± 0.396 ms；Isaac Lab 15.098 ± 0.365 ms，均值±跨进程样本标准差；EC 外步耗时低 1.72%，五组成对差值（Isaac−EC）全部为正、均值 0.259 ± 0.041 ms。NVML进程显存采样最大值 EC 826 MiB、IL 712 MiB。结论仅限物理外步，EC 状态读取仍更贵；软件版本与控制实现差异在报告中披露。

EC默认十倍外步的4个完整进程和1个初始化中断保留在 `measurement/invalid_cadence` 作错误审计；SDK 重测前的修正节奏批次归档在 `history/formal_before_sdk`。两者均不计入当前比较；未按几何结果过滤/补跑。最终代码通过 267 项相关回归和 30 项基准检查；720 外步×10 次运行的原始计时与绘图输入核对通过，所有产物仍位于同一任务目录。
