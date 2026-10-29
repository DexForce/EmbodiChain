# S01-T：5% 拉伸与恢复（重新规划）

状态：旧 pilot 为 EC 10/10、Isaac 9/10；只是旧执行配置的观察，不能推出可靠性排名。
本轮使用精简整体平台比较。遵循 [PLAN](../PLAN.md)、[准入清单](../PARITY_CHECKLIST.md) 与 [脚本契约](../SCRIPT_CONTRACT.md)。

## 任务与候选输入

- FR3 + Franka Hand；长块 120×40×40 mm、48 g，2025 节点/9216 四面体，E=30 kPa、ν=0.35。
- 近端真实静态碰撞夹具，初始夹口内节点 15% 预压，rest 形状另存；不 pin、不附着。
- 远端手指闭合目标 0.012 m，指垫形状 μ=1.4、粒子 μ=0.6，混合后 μ=1.0；这些是候选设计，不在正式 episode 内修改。
- 闭爪后静置到至少 2 s 且通过稳定检查，建立 L0；沿 X 拉伸、保持 1 s、回起点、松爪、恢复 2 s。
- 正式 matched_replay 冻结目标/阶段；原按中心线反馈积分的版本另列 matched_policy，不能用两侧不同阶段长度的全程节点差证明求解器收敛。
- 5% 主档，10% 延后。候选验收：5±2 pp、恢复误差≤5%、两端漂移≤3 mm、节点几何检查≤4 mm、无超时且状态有限。保持窗的瞬时失稳/峰值及端部全程最大漂移单列，不只看最后一帧。

## 测量与限制

自由段 x=0.44–0.49 的 11 个固定材料节点按三维相邻距离求折线长度；中心线主指标，另存四条表面线。
固定端相对夹具、远端相对指尖的漂移均需工具/夹具坐标定义，区别整体位移、材料剪切和真实滑脱。
L0 在夹紧受载后测量，松爪改变边界条件；2–3% 恢复误差不能直接解释成塑性或材料不恢复。
需另存夹紧前/夹紧后基线与卸载边界，正式材料结论须通过 S01-C 台架及本任务验证。

## 当前执行范围

本轮按 [PLAN](../PLAN.md) 执行：固定 s00/nominal，每平台独立重复五次，另录媒体。
保留原生控制，不强制共同 replay、10 初态或参数扫描；严格研究要求见 [完整协议](../FULL_PROTOCOL.md)。

## 文件与入口

- 配置：`assets/physics/dexuni/s01t/s01t_config.json`（相对 tech_reports 根）。
- 生成/编排：本目录 `tools/prepare_inputs.py`、`tools/run_matrix.sh`。
- 平台：`physics/platforms/dexuni/<platform>/s01t.py`。
- 旧分析：`reporting/dexuni/s01t/`，只用于 pilot。
- 旧档案：仓库 `outputs/benchmarks/dexuni/pilot_v1/s01_soft_compress_stretch/s01t_stretch/`。
- 新运行默认：`outputs/benchmarks/tech_reports/dexuni/development/s01t/`。

真实命令见 [SCRIPT_CONTRACT](../SCRIPT_CONTRACT.md)。迁移不代表旧驱动/计时/视频算法已修好。
