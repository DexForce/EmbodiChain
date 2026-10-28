# S01 小变形压缩/拉伸与恢复

S01 是两个独立子实验，不在同一 episode 混算：

- [S01-C 压缩](../s01c_compress/README.md)
- [S01-T 拉伸](../s01t_stretch/README.md)

主目标都是 5%，10% 延后。机器人为实际 FR3 + Franka Hand；主耦合为 two_way/dynamic。
旧 pilot 已归档，但控制、计时、失败分母、视频的评审问题尚未通过修复验证，**新正式矩阵未启动**。
[PLAN](../PLAN.md) 的 G0–G6 是准入条件；[SCRIPT_CONTRACT](../SCRIPT_CONTRACT.md) 给出当前可执行的 pilot 命令与待实现项。

旧数据保持原样在仓库 `outputs/benchmarks/dexuni/pilot_v1/s01_soft_compress_stretch/`。
不把旧的“每帧快 21%/7%”或“夹持更牢”复制为新协议结论，也不删除旧失败。
