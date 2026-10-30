# mjfit 迁移实现与验收记录

本文件保留首次迁移阶段的记录。后续学习结果分别见
[nominal 模型实测](validation/rl_attempt/RESULTS.md)与
[默认机器人资产重训练](../../learning/blind_assembly/validation/default_retrain_20261026/RESULTS.md)；
后者独立评测为 88.38%，未达到 95% 目标，其
[实验复核](../../learning/blind_assembly/validation/default_retrain_20261026/EXPERIMENT_REVIEW.md)
记录了资产、控制、夹持及可重复性差异。

本次已实现可运行的几何、仿真与原生 RL 通路，并交付测试、实验工具和 smoke 模型。**完整论文复现尚未完成**：任意碰撞拓扑的异构训练、源 Robotiq 闭链动力学、完整训练/泛化/消融仍未通过资格验收。

源码基线为 EmbodiChain `19c254577`；生成器固定为 mjfit
`5833f465896e1e06bc4be4f003be847f5183a4db`；公开机器人资源固定为 Menagerie
`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`。SDK 补丁基线为 Dexverse
`39c817973111a1dea0e716cf7efb45594c134099`。验证日期为 2026-10-25。

## 已实现的行为

- `CompoundShapeCfg`、`ShapePartCfg`、`CylinderCfg`：一刚体、多凸碰撞块、独立视觉、父级质量惯量；既有单形状配置保持兼容。
- `AssemblyAsset`：mjfit 公共接口生成、MuJoCo 公共接口提取、源网格重定位与惯量坐标转换、校验和 manifest、直接配置、USD 和真实铰链导出。读取缓存资产不依赖生成器。
- `env_variants`：准备阶段按环境声明不同几何/质量/初态。SDK 补丁验证了相同拓扑的 MuJoCo Warp 变体，以及 semi-implicit 的不同凸块数量；不会丢弃或隐藏碰撞块。
- 原始 `solref/solimp/solmix`、可配置/关闭的场景地面、连续关节修复；三份 SDK 补丁在隔离 overlay 中验证，没有改写安装的 SDK。
- CPU、原生 Warp、Newton、Default 的物理探针；S0 静态接触扫描和 S2/S3/S4 真实滑动/转动约束。动态验收同时检查稳态最坏误差、漂移、初始重叠和可用的溢出标志。
- 三个原生任务、Rizon4S preset、版本化机器人资源、study/fullrange 配置和显式 nominal 开发配置。
- 原生时序 actor-critic、独立 critic 观测、训练用接触 decoder、PPO 辅助损失、对齐 rollout 字段、正确的终态 bootstrap、Wilson 区间、actor 导出。
- 54 trial 的训练/消融矩阵准备器；验证选优必须产生真实 best checkpoint，不能静默回退 latest。
- checkpoint 固化物理组件、实际机器人/装配资产、控制参数与许可证，支持搬移目录后离线重放。

## 实际验证证据

运行环境分别记录在 [validation](validation/index.json) 中。数值测量的开发环境使用
Python 3.11、MuJoCo 3.12、Newton 1.6、DexSim 0.5.1rc1 与三份补丁；独立历史参考环境为
Python 3.12.14、MuJoCo/Warp 3.14.0、Warp 1.17.0。

### 几何与资源

圆孔、USB-C、M10、齿轮的 USD 回读检查保留碰撞块归属、局部/世界几何、基础形状尺寸、质量、COM 与惯量，均满足新增误差 1 μm 和质量/惯量相对误差 1e-5 的门槛。该门槛衡量**转换新增误差**，不消除源圆弧/螺旋离散化误差；曲线分辨率仍须小于目标间隙。

原生两环境测试验证了孔不会被视觉网格封闭、质量只作用一次、选择性 reset、删除重建、变体质量和初态、连续铰链超过整圈旋转。

公开机器人 100 个随机关节状态对照源 MuJoCo，最大 FK 平移差为 **0.163 μm**；这验证运动学，不验证作者辨识动力学或源夹爪闭链。[原始记录](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/rizon_fk.json)

### 接触精度

以下为精确方孔、100 μm 间隙、0.15 s、每类一个施力方向的 smoke 测量；它不代表完整方向/尺寸/负载矩阵。表中为稳态误差中位值，保存的窗口最坏误差也通过同一门槛。

| 后端 | S2 侧向 | S3 倾斜 | S4 偏航 | 资格状态 |
|---|---:|---:|---:|---|
| MuJoCo CPU | 0.1020 μm | 0.1322 μm | 0.0500 μm | 参考样本通过 |
| 原生 Warp | 0.1027 μm | 0.1316 μm | 0.0581 μm | 样本通过，累计溢出为 0 |
| Newton/Warp | 0.0683 μm | 0.1296 μm | 0.0579 μm | 行为通过；native 溢出诊断尚缺 |
| Default | 0.0010 μm | −0.0954 μm | 0.0034 μm | 行为通过；参数映射仍不等价 |

Default 记录到接触 offset 从 0 钳制到 100 μm 的提示，因此不能将其结果解释为原始 MuJoCo 参数等价。原生引擎未暴露的健康计数保留为 `unverified`。[CPU/Warp/Newton 数据](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/constrained_cpu_warp_newton.json)，[Default 数据](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/constrained_default.json)

独立 worker 只导入 MuJoCo/NumPy，对两版引擎加载同一份 MJCF；MULTICCD 显式启用、步长 1 ms、作用力 0.1 N、时长 0.3 s、四个方向。修正后的圆孔最坏稳态误差如下：

| 间隙 | MuJoCo 3.12 | MuJoCo 3.14 | 结果 |
|---|---:|---:|---|
| 20 μm | 0.494 μm | 0.210 μm | 通过 |
| 50 μm | 0.632 μm | 0.530 μm | 通过 |
| 100 μm | 0.867 μm | 1.360 μm | 通过 |
| 200 μm | 1.331 μm | 1.748 μm | 通过 |

验收容差没有改变。此前解析圆柱未继承 nested defaults 的结果已作废；回归测试检查两侧的实际 `solref/solimp/solmix/priority`。这里仅包含修正后的 [独立参考 XML、spec 与数据](validation/independent_reference/)。

匹配源地面接触参数后，圆销在 CPU、原生 Warp、Newton 直接构建和 Newton USD 路线的下降量均约 15.000 mm；native 路线保留溢出诊断未验证状态。[数据](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/passive_matched_floor.json)

### 原生任务与模型

三任务均通过构建、reset 与 25 步实际仿真。双侧接触比例为 peg 1.0、gear 1.0、nut 0.76；螺母仍有接触间歇，未认定为稳定抓持。[数据](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/task_smoke.json)

nominal 配置使用显式双滑动夹爪、源手指质量与接触几何，以及编码器驱动的模型重力前馈。它不保留 Robotiq 四连杆/肌腱动力学。当前 reset 是局部接触与固定 ready pose，没有声称完成论文的全工作空间初态分布。

真实 PPO smoke 完成 2 环境 × 32 步 × 1 update，即 **64 transitions**。最终 checkpoint 的 SHA-256 为
`205585e4e31e1c93e5fd69cd91d5eecae92246a3072a1b72ea9142761f778f72`。

搬移完整 run 后，在禁止 mjfit 和原机器人缓存访问的条件下完成 2 个 250 步 episode：**成功 0/2**、抓持指标 1.0、平均夹具穿透 1.296 μm，Wilson 95% 区间为 [0, 0.6576]。这是训练与重放通路验证，尚未学会装配。[评估记录](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/native_smoke_eval.json)

actor-only TorchScript 已导出并与 eager 推理多 batch 对照，最大差异为 0。它不含 critic 或 contact decoder。[导出描述](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/actor_smoke.json)

独立合成 tensor 容量测量支持 12288 环境：rollout payload 2261 MiB、一次 PPO 更新的 warmed Torch 峰值约 6360 MiB。该测量不包含真实场景、渲染或非 Torch 分配。[完整范围说明](../../learning/blind_assembly/CAPACITY.md)

## 尚未完成的验收

| 项目 | 当前限制 |
|---|---|
| 任意凸块数量的 Warp 异构环境 | Newton/MuJoCo Warp 当前要求相同拓扑；需要分组执行或进一步引擎支持 |
| 源机器人动力学 | 作者辨识模型不可用；源 Robotiq 闭链/tendon 未等价迁移 |
| 完整物理资格 | native 溢出诊断、USD 齿轮原始接触覆盖、完整方向/负载/稳定性和独立校准集矩阵未完成 |
| 论文训练分布 | nominal 为每 run 一个几何；全工作空间初态、全几何与接触随机化未完成 |
| 完整学习效果 | 54 trials、完整收敛、多种子统计、8192-episode ID 与各轴 OOD、消融尚未执行 |
| 达标模型 | 当前只有未完成训练的 smoke 模型，没有达到论文或计划成功率的模型 |

正式 profile 明确拒绝把 nominal、同构单几何或未资格动力学认作论文复现；不能通过收窄评测分布或放宽误差门槛代替这些交付。

## 验证与使用

- 集成 CPU 回归 **1086 通过、2 跳过**；独立 native GPU 回归 **9 通过**；历史上游测试 **197 通过**。分开记录，不将重叠测试次数相加。
- API 覆盖检查、项目 context 检查与局部 API Sphinx dummy 构建通过。
- 完整 Sphinx 构建受到现有 architecture 检查的“相关源码必须已提交”条件限制；没有为绕过它创建提交。
- wheel 验证包含机器人来源清单、任务与训练配置、examples、SDK 补丁；额外的 4 项打包测试通过。完整数据与代码入口参见 [README](README.md) 和 [RL README](../../learning/blind_assembly/README.md)。
- Context 已修订 simulation、robot、RL、GenSim 与 data-assets 的新边界；其余受影响 topic 的通用 manager、Task Program 和场景契约保持准确，因此未为变更记录重复扩写。
