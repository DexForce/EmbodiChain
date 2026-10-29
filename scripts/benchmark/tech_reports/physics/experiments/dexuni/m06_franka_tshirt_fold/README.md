# M06 Franka T 恤从左往右单次折叠

surface-fast 对比口径：在统一场景、网格、物理材料、控制轨迹、时间步、子步及CUDA Graph配置下，对比DexUni surface-fast与普通VBD的求解策略，属于求解器级优化比较。基础迭代数、自适应回退、Jacobi/Chebyshev、多层校正和缓存均属于求解策略的组成部分；不同策略的一次迭代不等价。性能与任务效果分别按实测指标说明，不能仅因基础迭代次数不同而将结果解释为单纯减少迭代。

用户要求：新增任务，参考原生 DexSim `demo_dexuni_franka_fold_tshirt.py` 和 Newton `example_cloth_franka.py`，使用一台运动学 Franka，从左往右折入一次，不要求完全对折。先交付 DexSim 场景预览，由用户观看后决定后续平台实现。

## 场景与动作

- FR3 + Franka Hand，固定基座，`kinematic`；不评估机器人受载反馈。
- 使用原示例的 6436 节点 T 恤及其三角形拓扑，缩小到原尺寸的 65%，以自由下落方式铺在工作台上；不固定节点、不附着到夹爪。
- 衣服先落稳。夹爪抓住左侧袖口，抬起、从左往右越过折线、缓慢放下、释放；最后退开观察。
- 仅做从左往右单次折叠，不做底边折叠。边缘对齐不作为要求；观察袖口实际位移、抬升及释放后的覆盖形态。
- 开启布料弯曲、自碰撞和 CUDA Graph；材料和接触参数在本次完整预览内恒定。

## 所有权与交付

平台脚本：`physics/platforms/dexuni/dexsim/m06.py`。小型场景参数：`assets/physics/dexuni/m06/m06_config.json`。唯一有效输出目录：`outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/`，保存冻结输入、IK 轨迹、实际状态、源代码和视频。试作诊断放在此目录的 `history/` 中，不建立多个并列任务版本。

执行记录以该输出目录的 `README.md` 为准。用户已接受 DexUni surface-fast 方案并授权正式性能数据：EmbodiChain 8 次基础迭代、自适应回退，Isaac Lab 15 次原生 VBD；两边 10 子步和 CUDA Graph，每平台保留五份独立运行；原矩阵交替执行，EC 第 5 次按用户要求重测替换并移除原本地记录。正式汇总 16.323 ± 0.354 ms 对 36.257 ± 0.951 ms；当前保存样本均通过检查，不作为未经替换的首轮成功率估计。当前 EmbodiChain 和对照视频已更新到 fast 配置；原始计时、状态、日志与协议位于同一任务的 measurement/，正式报告见 [REPORT.md](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/REPORT.md)。

当前修订：用户要求减轻柔软和皱褶，松爪无需贴桌。提高弯曲刚度、设置平面静息折角；在桌面上方 95 mm 松爪并垂直退开。DexSim 修正版双视角预览已完成且用户已确认，验证见 [VALIDATION.json](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/VALIDATION.json)。

EmbodiChain 使用同一冻结输入，全部场景资源经过 SimulationManager；新增可选 attrs.edge_rest_angle 保留平面静息折角。平台代码：physics/platforms/dexuni/embodichain/m06.py。视频：[EmbodiChain 预览](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/preview/embodichain/embodichain.mp4)；验证：[VALIDATION.json](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/preview/embodichain/VALIDATION.json)。

Isaac Lab 实现：physics/platforms/dexuni/isaaclab/m06.py。使用标准 Newton VBD 管理器扩展及子步运动学重放，不使用动态机器人求解。平台视频：[Isaac Lab](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/preview/isaaclab/isaaclab.mp4)；当前主预览：[EmbodiChain / Isaac Lab 对照](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m06_franka_tshirt_fold/preview/comparison.mp4)。上排侧视、下排俯视，同一实际时间。
