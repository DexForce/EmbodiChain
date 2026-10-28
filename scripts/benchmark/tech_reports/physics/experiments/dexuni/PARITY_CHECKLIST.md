# 历史严格比较准入参考

本轮执行精简 [PLAN.md](PLAN.md)：固定配置五次重复，不强制本清单中的完整矩阵、等价驱动或材料台架。
完整 gate 仅用于后续严格求解器研究；当前记录实际配置、计时边界、质量与失败日志即可。

# DexUni 新实验准入清单（dexuni-replan-v2）

本清单落实 [PLAN](PLAN.md) 的 G0–G6。当前所有 gate 均为 **pending**，不能把迁移、
AST 相同、配置相同或一次成功当作 gate 通过。每项证据必须关联代码 revision、输入 revision、
平台版本、命令、run/attempt ID 和原始文件。人工签核有作用域；不能由一行 `passed: true` 代替证据。

## G0：资产、场景与有效求解配置

- [ ] 机器人身份明确。已有 S01/S02 是 FR3 + Franka Hand，不是只改名后的 Panda。
- [ ] 两侧 URDF/USD 转换来源、所有 payload/mesh hash、固定关节合并、碰撞几何、质量/惯量、COM、重力一致。
- [ ] 核对 mimic：Isaac 转换器转普通关节、DexSim 保留 equality 的实际差异；选定相同约束语义后再跑。仅目标相同不代表动力学约束相同。
- [ ] 核对关节顺序、TCP、米/弧度、xyzw 边界、每指坐标与总开口区别。
- [ ] 节点 ID、四面体/表面拓扑、体积/总质量、初始数组与 rest 数组分别校验。
- [ ] 材料、本构、阻尼、粒子半径、碰撞 margin/gap、接触混合公式与实际接触对一致。
- [ ] 两侧实际求解器路径满足固定规则：EC/DexUni 对 Isaac/Newton MuJoCo Warp + VBD；缺能力标 N/A，不换后端。
- [ ] 按下表读回实际配置，区分 requested/effective，不能只记录共享 JSON。

| 项目 | S01 旧 pilot 值（不是全任务默认） | 验证要求 |
|---|---|---|
| coupling / joint_mode | two_way / dynamic | 分别记录；不视为同一开关 |
| proxy mode / links | staggered / all | 核对实际代理集合与反馈路径 |
| proxy mass / relaxation / iterations | 4 / 0.5 / 1 | 对齐质量/惯量缩放与更新顺序 |
| VBD iterations / friction_epsilon / hard | 40 / 1e-4 / False | 不依赖平台默认值 |
| shape margin / gap / soft margin | 0 / 0.002 / 0.003 m | 核对碰撞形状，排除仅可见形状 |
| MuJoCo cone / integrator / iterations | pyramidal / implicitfast / 100 | 连同 ls_iterations=50、驱动 mode、armature 读回 |
| rigid-soft contact | per-vertex | `contact_mode=full` 不等于启用了全表面接触 |
| gravity compensation | 1.0 | Isaac 的 PhysX `disable_gravity` 不代替 MuJoCo gravcomp |
| finger pad friction (S01-T) | shape 1.4，混合后 1.0 | 其它接触 0.6；记录实际混合，不在 episode 内变更 |

## G1：驱动对齐（S01 review 的首要阻断项）

- [ ] 新主协议选用 `zoh_position_zero_velocity`：外层 60 Hz 更新 q 目标；帧内所有子步保持 q，qd 目标显式为 0，两侧同一 PD 公式/增益/effort/速度上限。
- [ ] 旧 EC/DexSim `RealtimeJointTargetControl` 是子步插值 + 非零 qd；旧 Isaac 是位置保持 + 零 qd。当前脚本还没修正，不能直接开正式矩阵。
- [ ] 无物体接触的夹爪开合/手臂轨迹探针：保存每子步实际 q_target、qd_target、时间、关节 ID；逐样本按冻结容差比较。
- [ ] 动态 qpos/qvel 跟踪误差单独记录，不要求不同引擎实际状态逐位相同；实际输入必须相同。
- [ ] 若另选插值+速度前馈，在两侧实现并验证同一公式；新建 control_contract_id，不能与主协议混算。
- [ ] `matched_replay` 输入文件逐字节相同；`matched_policy` 仅保证策略相同，实际轨迹另存，不能称同输入。

## G2：材料与受载预算

- [ ] 独立台架先核对载荷、无载沉降、接触边界与理论适用条件；有摩擦端面不能直接套无摩擦解析真值。
- [ ] 预先登记材料曲线/参考区间及容差，开展迭代/时间步收敛；不能在看到正式结果后取消不通过的材料门槛。
- [ ] S01 旧台架 40 迭代压缩约 0.37/0.41 mm，对照估算 2.87 mm：刚度尚未验证，不是两侧都通过。
- [ ] 未通过材料验证只能保留 `geometry_demo`，不进入材料准确性或等物理质量的主加速表。
- [ ] 夹紧、压缩/拉伸所需载荷与机器人限制相容；失败不靠临时提高摩擦、驱动或 pin 补齐。
- [ ] 压板卡口/导向壁、已夹持初态、夹具预压属于场景设计，明确分组，不冒充自然摩擦抓取全过程。

## G3：计时边界

- [ ] 输入下载/生成/hash 在初始化和计时之前；初始化、编译、图捕获分别记录。
- [ ] 固定预热条件并排除全部预热帧；恢复已冻结初态和控制器状态，验证恢复正确且没有在首测帧重新捕获。
- [ ] 每个计时窗口前后同步拥有计算的设备；记录实际 Graph 状态与 recapture，不能只计异步提交。
- [ ] 分别记录 completed_step、control_readback、capture、serialization、end_to_end；旧 step_seconds 不含粒子读回。
- [ ] 新性能 run 有独立质量见证；纯计时不逐帧采全量状态，但必须验证轨迹/终态/控制事件和质量组一致。
- [ ] 同机平台交错，至少 3 个独立进程作 pilot，主图 10 个；报告每次结果、median 与离散度。
- [ ] 显存独立测量，renderer、观测配置、GPU 型号/进程采样范围一致；不可用记 unavailable。

## G4：冻结矩阵、失败与配对

- [ ] 启动前写 matrix_id、所有计划 cell、attempt ID、初态数组和重复策略，不以目录中现存文件推导矩阵。
- [ ] 每格可区分 planned/running/completed/task_failed/runtime_error/timeout/missing/unsupported/preparation_failed。
- [ ] 支持条件下运行失败进入运行成功率分母；准备失败/N/A 单列覆盖率和原因，不伪装为算法输出，也不静默删除计划格。
- [ ] 重跑新建 attempt ID，保留原失败；首轮结果与诊断复跑分开，不自动“选最新一次”。
- [ ] 新进程启动失败、OOM、NaN、超时、杀进程、零成功，都有离线故障夹具测试。
- [ ] 冻结 config/input/control/coupling/joint_mode/measurement scope；不兼容 pair 拒绝 speedup 并保留原因。
- [ ] 同一初态重复实验与初态扰动分开；10/10 对 9/10 不足以推出平台可靠性排名。

## G5：视频与几何证据

- [ ] 图像一帧一条实际 simulation_time/physics_frame，不根据编码序号推算物理时刻。
- [ ] 额外阶段关键帧单独存或按时间戳重采样；固定输出 30 Hz 不等于采集恰好 30 Hz。
- [ ] 两列按同一模拟时间选帧，标签/阶段/帧率取该实际时刻；短片结束后注明 ended，不能仍显示为实时物理状态。
- [ ] 合成测试覆盖奇数物理帧阶段切换、丢帧、不等时长；回选误差不超过已冻结采样容差。
- [ ] 几何检测公开对象、采样层级与容差；节点未下穿不等于全表面/连续无穿透。
- [ ] 一次开相机的冒烟测试，核对 RGB/depth/seg/K/外参/坐标/分辨率，排除仅可见形状对材质映射的干扰。

## G6：归档、重建与签核

- [ ] 保存实际源码快照或 commit + 完整 patch（含未跟踪源文件）；只有 diff SHA 无法还原源码。
- [ ] 依赖版本/源码提交、wheel 身份、硬件、转换参数、所有资产与输入 hash 可重建。
- [ ] reporting 仅消费冻结 run IDs 和对应输入/源码，不读取当前工作树来证明历史控制器相同。
- [ ] 原始数据只读；重建输出新建 comparison_id，保留 plotting data、SVG/PDF/视频时间映射。
- [ ] 新报告严格三表，未准入平台仍保留行但 rank/speedup 为 N/A；不把本次目录迁移写成实测成功。
- [ ] G0–G6 的证据经人工核验后，冻结新 matrix revision；任一输入/控制/判据变更使相关签核失效。
