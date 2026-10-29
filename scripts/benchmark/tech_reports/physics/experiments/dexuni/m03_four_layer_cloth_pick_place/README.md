# M03 双臂夹取四张布料并铺放

当前仅保留 [最新一套运行](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place/README.md)，视频已获用户确认，两平台各5次性能测量已完成，结果见最新运行目录REPORT.md。

四张独立的 300×300 mm 布料，每张 18 g、961 节点；W1 V030 双 Pika 夹住近边两端，抬升、保持、搬运，再依次铺放、张开和退出。所有布料节点均为动态，无绑定或 pin。材料、接触参数与命名关节轨迹在平台间共用。

当前配置为机器人 one_way / kinematic；DexUni surface-fast 基础8次与 Isaac Lab 普通 VBD 20次，均10子步、60 Hz、CUDA Graph。该配置不验证机器人受载反馈；本组属于统一场景、材料、控制轨迹及时间步配置下的求解器级优化比较，fast基础迭代、自适应回退与普通VBD迭代数均属于各自求解策略。用户所说 dynamic 指 two_way；本轮按用户要求优先修复轨迹几何和布料铺放，不改 SDK。

- [当前对照视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place/preview/comparison.mp4)、[原生视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place/preview/dexsim.mp4)、[场景记录](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m03_four_layer_cloth_pick_place/REPORT.md)。
- 原生入口：`physics/platforms/dexuni/dexsim/m03.py`；默认配置：`assets/physics/dexuni/m03/m03_config.json`（相对仓库根目录）。
- 复现顺序：原生 `--prepare-only` → 原生 `--reuse-inputs` → EmbodiChain / Isaac Lab `m03.py` → `reporting/dexuni/build_m03_comparison.py`。平台入口均归 `physics/platforms/dexuni/<platform>/`。
- 原始输入、机器人和 USD 资产归 `inputs/`；最新媒体和逐层状态归 `preview/`；运行源码快照归 `source/`；日志归 `logs/`。不创建调参版本或归档目录。
- 检查每层最低高度、中心轨迹、释放后中心误差、目标区内节点比例及固定中央材料线的反向段。反向段只是局部折叠诊断；不以平台间最终形态差异作质量排名。
- 机器人碰撞检查入口 `tools/check_clearance.py`，结果归运行根目录 `clearance.json`；检查所有子步采样姿态，不能代替连续碰撞证明。
- 本轮改善铺放高度、姿态和开爪顺序；用户已确认本版视频；一组冻结参数、两平台各5次求平均，不开展大参数矩阵。测量入口 `physics/experiments/dexuni/run_m03.py`，用 `--resume` 继续已完成前缀，`--rebuild-report` 离线重建报告。
