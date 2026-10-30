# 齿轮精度对齐报告

原生几何、质量属性和接触参数检查通过；36 个原生 profile 运行均通过行为诊断。独立源 Warp 参考已明确分离初始化协议：旧 `cpu_compiled` 为 **0/3**，受控 A/B 的 GPU 常量重算为 **3/3**，显式 `warp_recomputed` 公共运行器也为 **3/3**。同 CG/GPU 重算协议下，原生与参考的驱动端点差距不超过0.15%；完整准静态、长时间、批量及 CPU-PGS 动力学等价仍未完成验收。

固定生成器 `5833f465896e1e06bc4be4f003be847f5183a4db`。硬件：NVIDIA GeForce RTX 5090, 32607 MiB, 595.58.03；原生 MuJoCo/MuJoCo Warp 3.12.0、Newton1.6.0、DexSim0.5.1rc1。历史参考单独使用 MuJoCo3.14.0。所有原生运行 B=1；测量汇总和源码依赖摘要见[归档数据](https://github.com/DexForce/EmbodiChain/blob/f8f5dfcc81b03e0fe4a9698d46bfcf406d9a89a0/examples/sim/assembly/validation/20261026/gears.json)。更细的运行文件保留在本地 `outputs/assembly_many_gears/`；本文末尾列出其相对路径，未将它们作为仓库文件发布。

## Time & Memory

| 数据集 | 工作量 | 记录时间范围 ms | CPU / GPU 内存 | 计时含义 |
| --- | --- | --- | --- | --- |
| 原生 MC-off | 30 | 10803.851–93175.797 | 未测量 / 未测量 | 30场景；冷启动、逐步诊断、清理 |
| 原生 MC-on 请求 | 6 | 9391.503–15181.468 | 未测量 / 未测量 | 6场景；冷启动、逐步诊断、清理 |
| 匹配 CPU3.12 | 81 | 19.368–138.209 | 进程RSS high-water1193.43MiB / n/a | 81场景；每场景2次轨迹计时中位数 |
| 历史 CPU3.14 | 81 | 19.947–134.158 | 进程RSS high-water1193.43MiB / n/a | 81场景；每场景2次轨迹计时中位数 |
| 完整 live 属性捕获 | 6初始化、各1ms一步 | 7513.584–7877.588 | 未测量 / 未测量 | 冷启动、传输与检查 |
| 旧源 WarpCG / cpu_compiled | 3 ×1.5s | 5016.968–5122.093 | 未测量 / 未测量 | 准备及 endpoint 参考 |
| 受控常量 A/B | 6 ×1.5s | 4412.164–5013.497 | 未测量 / 未测量 | 逐步状态与健康读取 |
| 公共源 WarpCG / warp_recomputed | 3 ×1.5s | 4513.985–4730.782 | 未测量 / 未测量 | 步进循环；首步编译可能计入 |

这些时间的工作负载和计时边界不同，不能用于吞吐排名；CPU RSS是整个参考进程的累计高水位，不是单格峰值。性能、峰值显存和批量容量没有据此验收。

## Success & Other Metrics

| 证据 | 实际结果 | 结论边界 |
| --- | --- | --- |
| 原生 MC-off 30格 | 30/30；3齿轮 ×2后端profile；恒转1/0.5/0.25ms，负载/反转1ms；3s | 稳态尾段比例、行程、健康与静态回差诊断通过 |
| 追加 MC-on 请求6格 | 6/6；3齿轮 ×CPU PGS/Warp CG；1ms、1.5s恒转 | 参数请求部分对齐；有效 Warp容差仍被夹至1e-6 |
| 独立属性复核 | 30/30 +6/6；质量相对误差≤4.15e-08；完整host惯量误差≤1.73e-06；world COM≤1.520nm | 初末实际MJB；按几何匹配碰撞块后检查raw contacts/condim/过滤语义 |
| 几何与回差 | 碰撞几何误差≤1.954nm；静态回差差值≤1.689nm | 实际模型与源模型之差；静态回差为CPU副本查询，非Warp运行时oracle |
| 完整live属性捕获 | 6/6；完整live惯量相对误差≤1.73e-06 | MC-off初始化与一步；不追溯观测30条轨迹或新MC-on运行的所有缺失字段 |
| 旧源 WarpCG / cpu_compiled | **0/3**；三任务 follower均保持0；endpoint齿比误差100%；finite且overflow=0 | 原始初始化协议结果永久保留 |
| 受控源 WarpCG 常量 A/B | cpu_compiled **0/3** → warp_recomputed **3/3**；重算齿比误差0.02380% /0.09082% /0.02091% | 同源模型、控制、初态、CG、MC-on、dt1ms；唯一干预为公开set_const_0调用 |
| 公共运行器 / warp_recomputed | **3/3**；齿比误差0.05299% /0.08534% /0.02091%；finite且overflow=0 | 显式参考协议；旧默认cpu_compiled没有被改写 |
| spur_pair / cpu_pgs / MC-on请求 | 尾段齿比误差 0.00995%；驱动转角相对同solver CPU源参考 -3.176% | 齿比通过不等于动力学轨迹等价 |
| spur_pair / warp_cg / MC-on请求 | 尾段齿比误差 0.04733%；驱动转角相对同solver CPU源参考 +23.097% | 齿比通过不等于动力学轨迹等价 |
| reduction_12_36 / cpu_pgs / MC-on请求 | 尾段齿比误差 0.31347%；驱动转角相对同solver CPU源参考 +0.836% | 齿比通过不等于动力学轨迹等价 |
| reduction_12_36 / warp_cg / MC-on请求 | 尾段齿比误差 0.03022%；驱动转角相对同solver CPU源参考 +1.189% | 齿比通过不等于动力学轨迹等价 |
| fine_pitch_32_48 / cpu_pgs / MC-on请求 | 尾段齿比误差 0.05979%；驱动转角相对同solver CPU源参考 +1.016% | 齿比通过不等于动力学轨迹等价 |
| fine_pitch_32_48 / warp_cg / MC-on请求 | 尾段齿比误差 0.01727%；驱动转角相对同solver CPU源参考 +19.800% | 齿比通过不等于动力学轨迹等价 |
| spur_pair / 原生CG vs 公共重算Warp | 驱动端点差 +0.13076%；从动端点差 +0.15527%；phase差 -0.00294018rad | 同1.5s、1ms、MC-on、有效容差1e-6；不是逐浮点或整段轨迹等价 |
| reduction_12_36 / 原生CG vs 公共重算Warp | 驱动端点差 -0.14017%；从动端点差 -0.14980%；phase差 +0.000797272rad | 同1.5s、1ms、MC-on、有效容差1e-6；不是逐浮点或整段轨迹等价 |
| fine_pitch_32_48 / 原生CG vs 公共重算Warp | 驱动端点差 +0.01032%；从动端点差 +0.01364%；phase差 -0.000418345rad | 同1.5s、1ms、MC-on、有效容差1e-6；不是逐浮点或整段轨迹等价 |

比例验收使用末四分之一轨迹，驱动行程至少π/2，齿比相对误差≤0.5%。原生诊断另外要求完整有限轨迹、无已观测 warning/overflow/time reset、速度上限、属性误差阈值及源到实际模型静态回差差值≤1µm。原生36格均无已观测warning、overflow或time reset。独立Warp参考采用公开endpoint/overflow判据，与尾段判据分别保留。

全局设置：源`disableflags=0`。旧CPU profile为540672（禁用midphase和MultiCCD），旧Warp为524288（禁用MultiCCD）；新CPU为16384（仍禁用midphase），新Warp为0（MultiCCD开启）。新设置请求`tolerance=1e-8`，CPU有效值为1e-8，Warp有效值约1e-6；该下限见测量时 MuJoCo Warp3.12 的 `_src/io.py:429`。因此目录名`source_matched_requested`描述请求意图，不表示有效设置全部相同。

常量初始化 A/B 使用相同源 MJCF 和输入，只切换公开 `mujoco_warp.set_const_0`。所记录质量、惯量对角、COM、几何变换、solref/solimp、armature和被动阻尼差值均为0；派生 `body_invweight0` 的平动项由约1e-36–1e-33变为约950–23608。该API包含退化刚体权重回退，因此必须作为会影响接触正则化的**显式物理/参考协议**，不能称为无影响的初始化或CPU等价常量。原生CG与受控重算A/B的驱动端点差约−0.1253%/−0.1487%/+0.0103%；与公共重算运行器的差约+0.1308%/−0.1402%/+0.0103%。A/B重算记录与公共运行器的端点、phase不逐浮点相同；未证明整段轨迹等价。归档数据保留 `controlled_initialization_ab` 和 `public_warp_recomputed_references`；本地原始文件为 `constant_diagnostics/ab_results/raw.json` 和 `source_warp_recomputed/assembly_precision_20261025_164036.json`。

源模型数值检查：归档数据的 `source_local_numerics` 字段记录在驱动超过首触点角1e-6rad、从动角为0的固定构型执行CPU `mj_forward`。三任务均2contacts、8约束行，J为8×2且有限，条件数约8.75/13.69/50.53；接触正则化efc_R约9.8e-16–1.28e-15，平动body_invweight0约1e-36–1e-33。J并非零矩阵。固定上游已提示铰接齿轮接触正则化很小并指定PGS；这些测量**不证明独立Warp失败的具体原因**。

## Leaderboard

| 排序 | 算法 / 工作负载组 | 判据通过率 | 通过 / 完成 | 判据 |
| --- | --- | --- | --- | --- |
| 1 | controlled_AB_Warp_CG_warp_recomputed_1p5s | 1.000000 | 3/3 | controlled endpoint ratio/rotation/finite/overflow |
| 2 | native_MCoff_3s/cpu_pgs | 1.000000 | 15/15 | steady-tail ratio + native integration diagnostics |
| 3 | native_MCoff_3s/warp_cg | 1.000000 | 15/15 | steady-tail ratio + native integration diagnostics |
| 4 | native_MCon_requested_1p5s/cpu_pgs | 1.000000 | 3/3 | steady-tail ratio + native integration diagnostics |
| 5 | native_MCon_requested_1p5s/warp_cg | 1.000000 | 3/3 | steady-tail ratio + native integration diagnostics |
| 6 | public_source_Warp_CG_warp_recomputed_1p5s | 1.000000 | 3/3 | public endpoint ratio/rotation/finite/overflow |
| 7 | source_CPU_3.12.0_3s/cg | 1.000000 | 27/27 | steady-tail ratio/travel/finite/warnings |
| 8 | source_CPU_3.12.0_3s/pgs | 1.000000 | 27/27 | steady-tail ratio/travel/finite/warnings |
| 9 | source_CPU_3.12.0_MCon_paired_1p5s/cg | 1.000000 | 3/3 | steady-tail ratio/travel/finite/warnings |
| 10 | source_CPU_3.12.0_MCon_paired_1p5s/pgs | 1.000000 | 3/3 | steady-tail ratio/travel/finite/warnings |
| 11 | source_CPU_3.14.0_3s/pgs | 1.000000 | 27/27 | steady-tail ratio/travel/finite/warnings |
| 12 | source_CPU_3.14.0_3s/cg | 0.962963 | 26/27 | steady-tail ratio/travel/finite/warnings |
| 13 | source_CPU_3.14.0_3s/newton | 0.037037 | 1/27 | steady-tail ratio/travel/finite/warnings |
| 14 | controlled_AB_Warp_CG_cpu_compiled_1p5s | 0.000000 | 0/3 | controlled endpoint ratio/rotation/finite/overflow |
| 15 | source_CPU_3.12.0_3s/newton | 0.000000 | 0/27 | steady-tail ratio/travel/finite/warnings |
| 16 | source_Warp_3.12.0_CG_cpu_compiled_legacy_1p5s | 0.000000 | 0/3 | endpoint ratio/rotation/finite/overflow |

此表包含全部实际评估的算法组；按各自判据通过率排序，工作负载与验收层级不同，不能解释为公平性能排名。历史CPU3.14中CG为26/27、Newton为1/27；匹配CPU3.12中CG为27/27、Newton为0/27。异常结果与版本差异均保留。

本地证据路径（相对于 `outputs/assembly_many_gears/`）：原生30格 `native_alignment/report.md`；独立复核30格 `native_alignment_post_validation/report.md`；完整live6格 `native_alignment_live_properties/report.md`；追加请求设置6格 `source_matched_requested/report.md`；追加6格属性复核 `source_matched_requested_post_validation/report.md`；匹配CPU源参考 `gear_reference_matched/report.md`；历史CPU源参考 `gear_reference_historical/report.md`。

后续仍需：冻结已明确区分的常量初始化、有效接触/solver profile及标定/验收种子；比较负载、反转、长时间和驱动响应的任务量或分布；补齐Default、USD原生接触及批量隔离/容量。当前不声称完整物理等价、完整参数标定、公开Scene正式验收或实机复现。
