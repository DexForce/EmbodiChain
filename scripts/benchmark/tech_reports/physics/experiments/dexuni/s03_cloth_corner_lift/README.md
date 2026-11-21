# S03：动态夹持布角、提起和放回

当前唯一有效结果在[outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/REPORT.md)：10子步、10次VBD迭代，每平台五次独立无相机运行，全部完整且CUDA Graph实际captured。最终布形态只作观察，不评价质量优劣或筛选样本。

当前固定场景：30 cm方布、31×31节点、3 mm粒子半径、Z=0.868 m；45°对角夹持、finger PD6000/80、effort100、闭爪1 s后停0.5 s；举180 mm、保持2 s、下降3 s、TCP离桌约71 mm松爪。
物理材料来自指定DexSim franka_grasp_cloth demo：面密度0.002 kg/m²、tri_ke/ka1000、tri_kd1e-5、edge_ke/kd0.1/0.05、粒子接触10000/0.01/0.3；保留自碰撞，半径/裕量2 mm、rest排除5 mm。
纯蓝布双面、默认灰桌、源机器人材质，一个斜视相机。模型/材质/驱动不随预算候选改变。

两边请求的主参数共用JSON；控制子步插值/速度目标、固定空连杆处理及Newton/Python版本仍存在差异；两边实际mimic约束一致，详见当前批次参数核对记录。不能称完整动力学模型/求解器逐位一致，也不能预设EC一定更快。

- [当前状态](STATUS.md)
- [报告与全部绘图数据](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/REPORT.md)
- [已批准的并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/videos/s03_comparison.mp4)
- [SDK 集成前的初始化参数核对（历史记录）](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/history/formal_before_sdk/PARAMETER_CHECK.md)

旧S03版本已按用户要求删除。原生DexSim只作同安装的临时适配检查，不保留原生性能结果。S01/S02 当前正式结果保留；旧 dexuni_benchmark 已按用户要求删除。
