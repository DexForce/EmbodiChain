# S01 执行完成

批次：`outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/`（相对仓库根目录）。

## 完成结果

- S01-C/T × 2 平台 × 5 次独立主采样，共 20 次，全部通过；无重试或失败替换。
- 另录 4 次媒体运行，也全部通过，媒体计时不混入主样本。
- 压缩 step：EmbodiChain 41.124 ± 1.385 ms，Isaac Lab 52.770 ± 1.704 ms。
- 拉伸 step：EmbodiChain 35.177 ± 0.751 ms，Isaac Lab 37.000 ± 0.716 ms。
- 上述为五次运行均值 ± 样本标准差，排除初始 30 帧。保留原生控制，比较整体平台 step。
- 本轮 Isaac Lab 独立进程尝试总耗时更短；step 更快不能等同于整个脚本执行更快。
- 两段 30 fps 对比视频均已完成，源帧映射时间误差为 0；拉伸短片一侧结束后显示 ended。

## 入口

- [三表报告及响应曲线](../../../../../../outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/REPORT.md)
- [压缩视频](../../../../../../outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/videos/s01c_comparison.mp4)
- [拉伸视频](../../../../../../outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/videos/s01t_comparison.mp4)
- [全部尝试与指标](../../../../../../outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/attempts.json)
- [执行日志](../../../../../../outputs/benchmarks/tech_reports/dexuni/s01_fixed_five/experiment.log)
- [运行及重建命令](SCRIPT_CONTRACT.md)

## 验证与边界

7 项 focused CPU tests 通过；8 个变更 Python 文件格式检查通过；26 个源码 Python 文件 AST 检查通过；源码 Markdown 相对链接检查通过。
24/24 返回码为 0；输入 SHA256、执行脚本与源码快照、有效 CUDA Graph 状态、每格五个独立 run 目录、三表报告数值类型均已核验。
所有新输入快照、测量、报告和视频在 outputs/benchmarks/tech_reports/dexuni/。启动前本目录没有旧测量数据可删除。
旧 dexuni_benchmark 已于 2026-10-27 按用户要求删除；目录外历史快照和当前 S02 正式结果保留。
两侧 Newton 版本不同（1.7.0 / 1.6.1），详见报告；材料未标定，质量表只描述几何任务响应，五次固定初态重复不作为可靠性排名。

## 第二组 S02

已完成，独立状态与产物入口见 [S02 STATUS](s02_soft_pick_place/STATUS.md)。
第一组本页及原始结果保持有效。

## 第三组 S03

历史v6及10×40基线均已完成，原始绘图数据保留。低预算10×10已完成两平台各五次性能测量，全部完整运行且CUDA Graph实际状态均为captured；预算搜索结果与报告入口见 [S03 STATUS](s03_cloth_corner_lift/STATUS.md)。

## 第四组 S04

DexSim场景已确认，两平台实现和并排视频已完成；待当前两平台视频确认后才测量。见[S04 STATUS](s04_rigid_soft_squeeze_grasp/STATUS.md)。
