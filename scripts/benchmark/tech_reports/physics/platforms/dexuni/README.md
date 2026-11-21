# DexUni 平台 adapter

- `embodichain/`：SimulationManager + DexUni，正式目标平台。
- `isaaclab/`：Isaac Lab + Newton MuJoCo Warp/VBD，指定对照。
- `dexsim/`：DexSim 原生场景，仅用于适配诊断。
- `_lib/benchlib/`：原显存采样工具，保留原版权/许可，不移动到生产计算层。

各平台独立解释器/进程，不互相 import；参数来自外部只读输入快照。
实际入口均为 `{s01c,s01t,s02,s03}.py`，命令见
[SCRIPT_CONTRACT](../../experiments/dexuni/SCRIPT_CONTRACT.md)。

本轮按 [PLAN](../../experiments/dexuni/PLAN.md) 进行固定配置五次重复的整体平台比较。
S01-C/T manifest 标记 native_platform_comparison，保留平台原生控制差异，不标 matched_replay。
最初 30 帧不进入稳态窗口；逐帧状态与读回/记录/采集耗时分列。视频 RGB 仅在固定采样节拍写入。
S02/S03 均已完成；S03 使用三角布网格和固定材料角点验收。原生 DexSim 仅做临时适配检查，
不保留其测量或日志，不加入比较报告。旧工作区已删除；当前 S02 正式结果保留，旧试跑文字证据见清理账本。
DEXUNI_ALLOW_PILOT=1 仍是兼容执行开关，不构成材料校准或控制等价证明。
