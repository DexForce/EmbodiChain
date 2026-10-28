# DexUni 离线报告与媒体

当前报告由 `physics/experiments/dexuni/run_s01.py`、`run_s02.py`、`run_s03.py` 读取明确 run 路径的原始状态生成，支持 --rebuild-report。
`build_s01_media.py` 合成 S01-C/T 对比视频、仿真时间映射和响应曲线；两侧视频结束时间不同则显示 ended。
新报告保留 Time & Memory、Success & Other Metrics、Leaderboard 三表，整体平台比较不做可靠性排名。

`s01c/`、`s01t/` 的 analyze.py 中 task_metrics 继续提供几何判据复算；旧 main 的目录筛选、全帧计时和视频合成不用于本轮。
`s02/`、`s03/` 的 analyze.py 提供各任务指标；build_s02_media.py/build_s03_media.py 与
plot_s02_repetitions.py/plot_s03_repetitions.py 合成独立媒体和全五次重复图，同时导出绘图 CSV。
parity、compare_models 保留历史诊断用途；旧工作区已删除，旧 S02 试跑证据见清理账本。

命令见 [SCRIPT_CONTRACT](../../physics/experiments/dexuni/SCRIPT_CONTRACT.md)，当前范围见 [PLAN](../../physics/experiments/dexuni/PLAN.md)。
所有生成产物在 outputs/benchmarks/tech_reports/dexuni/；原始数据不写回源码/资产目录。

## 已发布结果与原始数据

[13 项任务对比](RESULTS.md)提供各任务报告和视频入口，并明确正式测量、摸测及失败条件。
[Hugging Face 报告](https://huggingface.co/datasets/DexForceAI/embodichain_reports/tree/main/reports/dexuni-physics-2026-10-27)
同时提供原始 ZIP、可浏览的 JSON/CSV、输入和逐文件哈希。
复算前按[恢复命令](../../physics/experiments/dexuni/SETUP.md)下载所需结果；无需重新仿真。
