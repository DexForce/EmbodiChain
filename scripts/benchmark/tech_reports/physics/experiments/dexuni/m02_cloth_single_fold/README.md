# M02 双臂方布单次对折：两层接触｜中等｜P0

surface-fast 对比口径：在统一场景、网格、物理材料、控制轨迹、时间步、子步及CUDA Graph配置下，对比DexUni surface-fast与普通VBD的求解策略，属于求解器级优化比较。基础迭代数、自适应回退、Jacobi/Chebyshev、多层校正和缓存均属于求解策略的组成部分；不同策略的一次迭代不等价。性能与任务效果分别按实测指标说明，不能仅因基础迭代次数不同而将结果解释为单纯减少迭代。

**耦合配置**：单向 `one_way` + 运动学关节 `kinematic`（用户确认由原计划的双向改为单向）；机器人轨迹回放，不报告机器人受载反馈；细网格/并行扩展另评估。

**目的**：测试双点夹持、布料自碰撞和第一层间接触。

**场景**：方布 0.40 × 0.48 m，中心 `(0.40,0)`；两臂在布中线（折线）`y=±0.17` 处夹取。

**动作**：中线夹起→抬高使两半自然下垂→向前下方平铺成两层→松爪并观察 2 s；16.5 s。启用自碰撞，保持材料和摩擦恒定。（原“抓近侧两角翻折”方案经用户评审后替换，见下方状态。）

**测量**：固定材料角点/边中点的 XY 对齐误差、第二层是否落到第一层目标区域、动作完成率。对齐误差由创建时标记的材料节点组计算。

**建议成功线**：释放 2 s 后，移动侧两个角点组与目标侧对应组的 XY 距离均≤15 mm，折叠过程完整。两层来自单折流程；布面贯穿另列几何检查，不能由角点对齐推断。

## 执行状态与公共约定

状态（2026-10-22）：按用户要求改为单向 `one_way` + 运动学关节 `kinematic`（不报告机器人受载反馈），并按用户建议改为“从中间夹起、抬高、平铺到桌上”的对折方式（第一版抓两个近角翻折，用户反馈手姿不好、夹起后下滑像只夹了一个点）。DexSim 原生场景已获用户确认；EmbodiChain（surface-fast）与 Isaac Lab（自带 VBD）已实现，两平台各五次正式测量完成，见 [REPORT](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m02_cloth_single_fold/REPORT.md)。

- 预览：[dexsim.mp4](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m02_cloth_single_fold/preview/dexsim.mp4)；入口 [m02.py](../../../platforms/dexuni/dexsim/m02.py)；共用 [m02_common.py](../m02_common.py)。
- 关节轨迹离线 IK 一次烘焙为 `inputs/joint_commands.npz`（24 个命名关节 × 991 帧，16.5 s），DexSim 用 `KinematicJointTrajectoryControl` 回放；其他平台将回放同一文件。全程烘焙 TCP 误差约 0，两手相机最近 140 mm。
- 布：0.40 × 0.48 m（X × Y），中心 (0.40, 0)，S03 的 10 mm 节点间距与材料、自碰撞参数（半径/边距 2 mm），三角剖分关于两条中线镜像对称；41 × 49 = 2009 节点；10 子步、CUDA Graph。
- 求解器：`contact_mode="soft"` + `vbd_preset="surface-fast"`（8 次迭代 + Chebyshev/批量 Jacobi/多级校正/表面与截断缓存，自碰撞检测每子步一次）。原 `full` + 普通 VBD 10 次迭代时，下垂的两半布在夹爪静止时仍逐帧抖动（帧间二阶差分 p95 3.0 mm）；20 次迭代降到 1.1 mm，surface-fast 降到 0.6 mm，夹爪附近 0.09 mm。关闭自碰撞会更抖（p95 32 mm，两半互相穿过）。surface-fast 仅在 `contact_mode` 为 auto/soft 时生效（full 下静默回退），soft 模式只接受 `soft_contact_margin` 一个碰撞选项。Isaac Lab 原生 Newton VBD 没有此预设。
- 动作：两夹爪竖直向下，在布中线（折线 x = 0.40）y = ±0.17 处夹取，手指沿 X 闭合，57 mm 宽的指面沿折线夹住一条布脊（指尖离台 3 mm，预设开口 q = 0.002）；抬到 TCP 离台 300 mm，两半自然下垂；后移到 x = 0.32、TCP 离台 250 mm，再向前下方斜向移动到 x = 0.48 把布平铺成两层；指尖离台 25 mm 处松开，上抬 80 mm，观察 2 s。机器人–布摩擦 1.0。
- 预览观测（单次，非正式统计）：释放后两层自由边左/右角组 XY 误差 44.3 / 74.2 mm（同参数试跑 55–66 mm 间波动），布最高点离台 22 mm（无挂布）。收敛后布的下垂长度与普通 10 次迭代时不同，旧铺放参数会让下垂边在后移时拖地；夹取的布脊居中（夹住节点的原始 x 均值 0.401），偏差主要来自抬起后两半下垂长度/前后位置不同，铺放时转为两层错位。比普通 VBD 版本（31–35 mm）对齐更差，未为对齐阈值再调参。
- 已知限制：两臂 J7 在抬高段约 2.5 s 处于腕关节限位（运动学回放不受影响）；换成相机朝外的对称偏航会使 J2/J7 更长时间在限位且抬高段偏离 83 mm，故未采用。

```bash
/home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m02.py
```
以上物理目标与阈值是候选值，正式矩阵开始前冻结，不能根据失败平台临时放宽。

- 遵循 [PLAN](../PLAN.md) 的 G0–G6、[PARITY_CHECKLIST](../PARITY_CHECKLIST.md) 和 [SCRIPT_CONTRACT](../SCRIPT_CONTRACT.md)。
- 本目录只拥有任务定义、矩阵和编排；平台代码归 `physics/platforms/dexuni/<platform>/`，小型输入定义归 `assets/physics/dexuni/`。
- 生成二进制输入使用外部缓存，原始运行/报告/视频归仓库 `outputs/benchmarks/dexuni/`，不再建本任务 `raw_data/videos/results` 源目录。
- 正式主组使用 matched_replay 与已验证的相同驱动；单向组明确 kinematic，不报告机器人受载反馈。matched_policy 另组，不称同输入。
- 留存运行错误/超时/OOM/缺失与所有诊断尝试；不开“最新成功覆盖失败”的正式分析。
- 旧 S01 的参数/成功率不能代替本任务的材料、接触、稳定性和受载验证。

## Platform ports (2026-10-22, preview only, not measured)

Approved by the user: "可以，移植到 EmbodiChain（同样使用 surface-fast）和 Isaac Lab（使用自带 VBD）".
`tools/export_inputs.py` froze `inputs/m02_config.json` and copied the W1 USD from M01 (same URDF).
Both ports replay `inputs/joint_commands.npz` kinematically on `inputs/cloth_mesh.npz`.

| Platform | Solver | Final corner XY L / R | Hang lowest node | Hang jitter p95 |
| --- | --- | --- | --- | --- |
| DexSim (approved) | DexUni soft + surface-fast (8 it.) | 44 / 74 mm | 98 mm | 0.56 mm |
| EmbodiChain | DexUni soft + surface-fast (8 it.) | 42 / 71 mm | 101 mm | 0.70 mm |
| Isaac Lab | Newton 1.6.1 stock VBD, 10 it., VBD manager with in-graph kinematic replay (as M06) | 9 / 100 mm | 118 mm | 0.58 mm |

- EmbodiChain matches DexSim within GPU run-to-run variation (mean node distance ≤ 8 mm).
- Isaac Lab matches until the fingers close (≤ 0.4 mm at 3.0 s); it diverges at the pinch and hangs about 20 mm higher (plain 10-sweep VBD converges less, as DexSim's plain 10-sweep runs did). No tuning toward DexSim was done.
- Comparison video: `outputs/benchmarks/tech_reports/dexuni/m02_cloth_single_fold/videos/m02_comparison.mp4` (`reporting/dexuni/build_m02_media.py`).
