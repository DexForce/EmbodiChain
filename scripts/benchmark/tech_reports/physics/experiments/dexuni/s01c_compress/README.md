# S01-C：5% 压缩与恢复（重新规划）

状态：旧 pilot 已完成；本轮使用精简整体平台比较。旧 10/10 与速度表不是当前正式结果。
遵循 [PLAN](../PLAN.md)、[准入清单](../PARITY_CHECKLIST.md) 和 [脚本契约](../SCRIPT_CONTRACT.md)。

## 任务与候选输入

- FR3 + Franka Hand，桌面 z=0.86；60 mm 软块，ρ=250 kg/m³，E=30 kPa、ν=0.35。
- 旧网格 729 节点/3072 四面体，100 g 压板包含卡口与导向壁，初态已夹持。
- 新轮先保留这套明确标记的夹具压缩场景；不将它称为自然夹柄全过程。圆角是否建模、质量/惯量与完整几何在 G0 冻结。
- 静置至少 1 s 且达到预先规定的稳定阈值，建立 h0；接近、压缩、保持 1 s、卸载、恢复 2 s。
- 正式 matched_replay 使用冻结轨迹及阶段，5% 是验收目标而非运行中改轨迹的开关；原闭环版本只属于 matched_policy pilot。
- 候选成功线：压缩目标 5±2 pp、恢复误差≤5%、桌面内节点下穿≤4 mm、压板相对手的全程漂移≤3 mm，状态有限且无超时。冻结时明确保持窗每帧/均值判据，不能靠平均掩盖瞬时失稳。
- 压缩 h=max(z)-min(z)，恢复误差=abs(h-h0)/h0；滑移在工具坐标系测量，不能只比较世界 Z 后宣称三维不滑。

## 当前执行范围

本轮按 [PLAN](../PLAN.md) 执行：固定 s00/nominal，每平台独立重复五次，另录媒体。
保留原生控制，不强制共同 replay、10 初态或参数扫描；严格研究要求见 [完整协议](../FULL_PROTOCOL.md)。

## 文件与入口

- 配置：`assets/physics/dexuni/s01c/s01c_config.json`（相对 tech_reports 根）。
- 生成/编排：本目录 `tools/prepare_inputs.py`、`tools/run_matrix.sh`。
- 平台：`physics/platforms/dexuni/<platform>/s01c.py`。
- 旧分析：`reporting/dexuni/s01c/`，仅用于 pilot 重建。
- 原始档案：仓库 `outputs/benchmarks/dexuni/pilot_v1/s01_soft_compress_stretch/s01c_compress/`。
- 新运行默认：`outputs/benchmarks/tech_reports/dexuni/development/s01c/`。

真实命令见 [SCRIPT_CONTRACT](../SCRIPT_CONTRACT.md)，不要执行历史档案 README 中的旧路径命令作为新正式实验。
