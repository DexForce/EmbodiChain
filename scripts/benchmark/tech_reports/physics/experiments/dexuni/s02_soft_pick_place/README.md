# S02 软块大变形夹取、提起与定点放置

本轮沿用固定配置五次重复的整体平台比较：EmbodiChain + DexUni 对 Isaac Lab + Newton MuJoCo Warp/VBD。
第一组 S01 已完成且不重跑；旧 dexuni_benchmark 已删除；当前 S02 正式数据保留，旧试跑记录见清理账本。

## 当前配置

FR3 + Franka Hand，60 mm 软块，2197 节点/10368 四面体，E=30 kPa，密度 250 kg/m³。
固定 s00/nominal，10 子步、40 VBD 迭代、CUDA Graph，动态机器人和 two_way 耦合。
夹持宽度压缩目标 35%，验收带 30%–40%；提起 120 mm，保持 2 s，移至 (0.57, 0.08)，降低后松爪、退开并观察。

质量判据：保持和运输结束最低节点离台≥80 mm；保持末段下降≤10 mm；
放置检查时节点几何中心 XY 误差≤20 mm，目标矩形内节点比例≥95%，最低节点离台≤6 mm；
台面节点下穿≤4 mm，状态有限。宽度恢复误差单列，不用它推断材料塑性。

## 原生后端适配检查

正式继续采样前，用相同 Python/DexSim 后端版本和同一输入临时运行原生 DexSim。
核对质量/材料/拓扑/关节驱动数组、阶段对齐的粒子运动及任务指标。
阶段采样粒子 RMS 与节点中心差≤2 mm，块宽差≤2 mm（与粒子运动同为毫米级容差）；松爪接触回弹阶段用阶段平均宽度和恢复终点（≤1 mm），仍保留粒子运动阈值。
该选择基于 EmbodiChain 自身重复也存在单帧回弹差异，不能把随机接触事件直接判为适配缺陷。
抓取压缩差≤2 pp，提起/下降/放置误差差≤2 mm。任何系统偏差先定位适配，阻止后续采样。
DexSim 仅做临时诊断，不进入 attempts、表格、绘图或归档，临时数据/日志退出即清理。

## 采样和产物

每平台独立 5 次无相机采样，轮换平台先后；每平台另录一次媒体运行。
最初 30 帧为预先规定的启动/初始稳定窗口，后续同步 step 单独计时，读回/记录和进程总耗时分列。
保存逐帧粒子、宽度、关节/TCP、阶段、timing、配置与输入/源码快照。
媒体保持 30 Hz，按仿真时间对齐，额外阶段图像不插入固定帧率视频。
绘图除了原始 NPZ/CSV，还导出 response_data.csv，便于另行绘图。

```bash
.venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_s02.py
```

全部输出在 outputs/benchmarks/tech_reports/dexuni/<batch>/；[执行状态](STATUS.md)。
[总计划](../PLAN.md) / [运行说明](../SCRIPT_CONTRACT.md)。

## 本机后端源码参考

`/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim` 是供 EmbodiChain 使用的构建来源。
`/home/oem/code/engine/dexsim` 可用于阅读原生源码和参考独立运行方式；用户未要求额外做跨版本比较。
本轮适配检查继续使用同一后端安装，隔离 EmbodiChain 封装的影响。
