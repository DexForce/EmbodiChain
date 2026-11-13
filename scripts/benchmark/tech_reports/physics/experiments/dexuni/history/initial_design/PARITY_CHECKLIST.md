> 历史设计快照，非当前执行计划。当前安排见[PLAN](../../PLAN.md)。旧 M04 棱台方案已被折衣任务替换。

# DexSim / Isaac Lab 对比实验检查清单（来自 S01-C，2026-10-21）

每个任务开跑前逐项核对，结果写进该任务的 `results/*_DECISIONS.md`。S01-C 的教训：共享配置一致不等于求解器一致；求解器选项的影响比平台差异还大（见 S01-C 决策记录（历史路径：`./s01_soft_compress_stretch/s01c_compress/results/S01C_DECISIONS.md`））。

## 1. 先对齐“配置文件没写到”的求解器选项

两侧的默认值不同，必须在任务配置里显式写出，两个脚本读同一个值：

| 选项 | DexSim（DexUni） | Isaac Lab | S01-C 取值 |
|---|---|---|---|
| 代理耦合模式 | staggered（内置，不可配） | `CouplerProxyMappingCfg.mode` 默认 lagged | staggered |
| VBD `friction_epsilon` | 由 `vbd_options` 给 | Newton 默认 0.01，`VBDSolverCfg` 未暴露（用子类加字段传入） | 1e-4 |
| VBD `rigid_contact_hard` | 由 `vbd_options` 给 | Newton 默认 True，同上 | False |
| 形状 margin | `NewtonCollisionDesc` 默认 0.001 | `NewtonShapeCfg` 默认 0 | 0 |
| 软接触 margin | `collision_options.soft_contact_margin` | `NewtonCollisionPipelineCfg` 默认 0.01（代理与 `collision_cfg` 都要设） | 0.003 |
| MuJoCo 摩擦锥 | 默认 pyramidal | 脚本里写什么就是什么 | pyramidal |
| 代理进 VBD 的连杆 | 全部 MuJoCo 连杆（内置） | 代理正则决定 | 全部 |
| 刚—软全表面接触 | `contact_mode="full"` 不等于全表面，默认关 | `enable_rigid_soft_full_surface_contact` | 关（逐顶点） |
| 重力补偿 | `newton_gravity_compensation` | `MujocoRigidBodyPropertiesCfg(gravcomp=)`（`disable_gravity` 只对 PhysX 有效） | 1.0 |

运行后从模型读回核对（写进 manifest）：每个形状的 ke/kd/mu/margin/gap/flags、粒子接触材料、实际子步数、CUDA Graph 状态。注意 DexUni 的粒子接触来自逐物体材料表，模型级 `soft_contact_ke` 只是未使用的默认值；Isaac Lab 里 USD 的视觉网格会以“仅可见”形状出现，不要误判为碰撞参数不一致。

## 2. 发现两边表现差异时

先怀疑配置，再怀疑平台：读回两侧实际参数逐项比对；不一致的逐项单独切换做对照（一次只改一项），把对照表写进决策记录，然后删除旧数据、对齐后全量重跑。

## 3. 求解与验证

- VBD 迭代次数用任务本身做小扫描（S01-C：10/20/40 次，选两侧都通过的最小值 40）；不要盲目加到 160。
- 自由刚体压在软体上的材料台架（质量比约 7000:1）在可用迭代次数下两侧都不收敛，只作参考，不作准入；需要材料刚度结论时另想办法。
- 子步 5/10/20 与 Graph/direct 都做成对比较，记录实际生效值而非请求值。

## 4. 数据与视频

- 任务目录只保留最终有效数据；试跑和诊断写到会话 scratchpad，结论数字写进决策记录。
- 每个任务只导出一个对比视频：2×2（左右平台、上下两个视角），相机按 1280×960 渲染，叠加两边实时物理帧率；深度/分割按计划的 640×480 存（隔点取样，K 同步换算）。
- 正上方相机常被手掌挡住，先抽帧检查视角能看到被测物体。
- Isaac Lab 相机用 OVRTX：相机放在 `{ENV_REGEX_NS}` 下；Kitless 模式不应用 `UsdFileCfg.visual_material`，软块颜色写进 USD（UsdPreviewSurface 绑在根 prim）；穹顶光 600。第一次运行编译着色器约 90 s。
- DexSim 软体材质要通过 `RenderDesc.from_geometry(..., material_ref=...)` 包一层才生效；DexSim 的 USD 解析只读绑定的 UsdPreviewSurface，不读 displayColor。

## 5. 其它坑

- Isaac Lab 由 VBD 拥有的刚体，位姿要从 `NewtonManager.backend.state_0.body_q` 读，RigidObject 视图读到的是不更新的自由关节坐标。
- DexSim 无窗口时要在采图前手动 `sync_to_dexsim`。
- 压缩包 `npz` 的数组每次按键读取都会重新解压，分析时先 `dict(np.load(...))`。
- 配置文件在矩阵运行中途不要改（会让部分 run 的配置哈希不同）。

## 6. EmbodiChain 移植（第二阶段：EmbodiChain 对 Isaac Lab）

每个任务先在本地 DexSim 上做，再写 `*_embodichain.py`，挑一两个状态与 DexSim 对比一致后再跑 EmbodiChain 对 Isaac Lab 的全量矩阵。

- **逐数组比较最终 Newton 模型。** 两个脚本 full run 都写 `newton_model.npz`，用 `tools/compare_models.py` 比较。封装层可能静默改写显式配置：S01-C 中 EmbodiChain 把 URDF mimic 从动关节（`fr3_finger_joint2`）的驱动改成主动关节的 1 % 并改写 MuJoCo mimic 约束，手指多闭合 3 mm。现已默认关闭（`root_props.newton_mimic_stabilizer=False`，与 DexSim 一致），任务脚本仍显式写出。
- **控制方式要等价。** 用 `sim.add_runtime_control()` 注册与 DexSim 相同的 `RealtimeJointTargetControl`；`set_qpos(target=True)` 整帧保持目标、速度目标为 0，驱动效果不同。
- **一致性的尺度**用 DexSim 自己两次运行之差（S01-C s00：节点 RMS 0.053 mm），而不是零。
- EmbodiChain 缺的功能先补到 EmbodiChain（带测试和 agent_context），再写任务脚本；不在任务脚本里绕过 EmbodiChain 去调 DexSim 内部接口（读 Newton 模型做诊断除外）。
- EmbodiChain 自带不可配置的 1000 m 地面（μ 0.5）；任务若有物体接触地面，需要先在 EmbodiChain 里加配置。
- 组合刚体（多个基本体）目前通过运行时写 USD + `UsdSourceCfg` 加载。
- EmbodiChain 的 GPU 测试要加 `--run-gpu`；仓库根目录若有过期的 `embodichain.egg-info` 会遮住机器人入口点。
- **有偶发事件的任务要比分布，不比单次。** S01-T 的远端粘滑是偶发的：单次 s00 对比指标一致，但重复 11 次后 DexSim 0 次、EmbodiChain 5 次。先做 1 µm 扰动测敏感度（判断第一步差异是否只是数值噪声），再各跑约 10 次同一状态比较事件频率。
- **随机事件的频率要用足够多的重复判断。** S01-T 第一阶段 DexSim 0/11 次滑移看起来比 Isaac 稳，改动无关布局后同一 DexSim 就出现 1–2/8 次；样本少时的“从不失败”不能当平台优势。DexSim 参照仍建议用与 EmbodiChain 相同的 arena 布局。
- **单连杆接触材质的三平台写法（S01-T 指垫）。** DexSim：`ContactMaterialSchedule` 运行时控制（没有公开的 finalize 前单连杆接口）；EmbodiChain：`ArticulationCfg.link_attrs`；Isaac Lab：`sim.reset()` 后原地写 `shape_material_mu` + `add_model_change(SHAPE_PROPERTIES)`（USD 物理材质不会被 Isaac Lab 3.0 的 Newton 导入）。每次都从 manifest 形状表核对实际值。
- **冒烟测试要带相机跑一次。** Isaac Lab 开相机时，连杆会多出“仅可见”的视觉网格形状（`shape_flags` 不含碰撞位）；按连杆选形状时只选带 `COLLIDE_SHAPES`/`COLLIDE_PARTICLES` 的。S01-T 指垫摩擦第一次全量矩阵因此在 Isaac 全量 run 上全部报错（无相机冒烟测试没发现）。
