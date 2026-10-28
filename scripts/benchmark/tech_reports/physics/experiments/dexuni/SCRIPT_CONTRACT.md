# 当前实验运行与重建

有效结果从[runs索引](../../../../../../outputs/benchmarks/tech_reports/dexuni/README.md)进入。S01/S02分别使用run_s01.py/run_s02.py，S03唯一有效目录为outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift（10×10，五次/平台）。现有目录不重新启动测量。

## 离线重建S03

从仓库根目录运行，不启动模拟器：

```bash
.venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_s03.py --batch outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift --rebuild-report
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/plot_s03_repetitions.py --batch outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_s03_media.py --batch outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/preview --video-only
```

preview/videos链接到同批次videos，源视频、状态和时间映射在同一有效目录。原始config与输入哈希保持不变；旧绝对URDF位置已经移除时，加载器验证本批次机器人资产哈希并解析到inputs/s03/franka_emika_panda。

## S04确认流程

先在原生DexSim搭建W1双侧刚块/软块夹持，只录一段预览。用户确认后才实现EmbodiChain和Isaac Lab；两平台预览再次确认后才测五次。未确认的场景调整使用同一固定任务输出目录，替换临时视频，避免版本堆积。

## 数据与统计

固定一套配置，每平台五个独立进程，交替执行。正式性能运行不启用相机，初始30step不进入稳态统计；同步计时包含已完成GPU工作，控制/读回/保存另列。最终布/软体形态只描述，不筛选或补跑性能样本。报告保留耗时/显存、完整运行及描述性观察、性能汇总三表。

输入、执行源码快照、状态、逐步计时、日志和尝试记录均保存在有效任务目录。原生DexSim适配诊断结果退出即删除；用户要求的原生场景预览视频可以保留。旧 dexuni_benchmark 已按用户要求删除；当前 S02 正式结果保留。
