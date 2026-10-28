# M01 双臂软梁弯曲、搬运与恢复｜中等｜P0

**耦合配置**：基准组使用双向 `two_way` + 动态关节 `dynamic`；属于预计低风险配置，尚未实测确认，细网格/并行扩展另评估。

**目的**：在一个物体上测试双臂协同、非均匀大变形和受载响应。

**场景**：软梁 B 沿 Y 放置，中心 `(0.48,0,0.025)`；左右夹爪分别抓取距两端约 30 mm 的端部区域。

**动作**：两臂同步夹持→共同抬高 100 mm→两端保持近似等高，将抓取点间距从约 240 mm 缩至 200–220 mm，两端各向内倾转约 10°，形成向下弯曲→保持 2 s→恢复端距与姿态→搬运到另一标记区→释放；约 16 s。先用无载路径检查与软梁试跑冻结端距和姿态，避免强拉伸。不能仅靠两端高度不同制造整体倾斜，再把倾斜称为弯曲。

**测量**：梁中部标记节点组相对两端标记组连线的下垂距离、两端实际间距、最低节点离台高度、释放后的中部下垂残差。

**建议成功线**：两端抓取保持，梁中部相对“刚抬起且尚未缩短端距”的基准帧产生额外下垂，并完成搬运；具体幅度在 pilot 冻结。无 NaN。端点连线相对距离消除整体平移和倾斜，不报告尚未实现的曲率或截面应力。

## 执行状态与公共约定

状态（2026-10-22）：DexSim 原生场景已获用户确认（“ok，继续做embodichain和issacLab的场景吧，做完我看没问题再测指标”）。EmbodiChain / Isaac Lab 场景已实现并录制预览：[并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/videos/m01_comparison.mp4) · [曲线](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/videos/m01_response.png)。用户确认后两平台各五次正式测量完成（EC 14.275 ± 0.433 / IL 14.852 ± 0.045 ms/外步），见 [REPORT](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/REPORT.md)。

- 两平台回放冻结的 DexSim 关节指令（`inputs/joint_commands.npz`，24 个命名关节 × 961 帧），使用同一 1225 节点四面体网格（`inputs/soft_mesh.npz`）、同一材料和接触参数（`inputs/m01_config.json`）。Isaac Lab 的 W1 USD 复制自 S04（同一 URDF，哈希核对）；桌面/桌腿/标记摩擦 0.8、机器人 1.0，与 DexSim 一致（已从 Newton 模型读回核对）。两平台 CUDA Graph 均 captured，10 子步 × 12 迭代，实际 16.000 s。
- 预览一致性（单次）：保持段中部下垂 DexSim 83.6 / EC 83.6 / Isaac 83.7 mm；刚抬起 27.1 / 27.0 / 26.9 mm；搬运 29.9 / 30.0 / 29.5 mm；释放后 4.1 / 4.4 / 3.8 mm；保持段端部间距三者均 346.5 mm。
- 入口：[EmbodiChain m01.py](../../../platforms/dexuni/embodichain/m01.py)、[Isaac Lab m01.py](../../../platforms/dexuni/isaaclab/m01.py)、[输入导出](tools/export_inputs.py)、[并排视频生成](../../../../reporting/dexuni/build_m01_media.py)。

- 预览视频：[dexsim.mp4](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/preview/dexsim.mp4)；入口 [m01.py](../../../platforms/dexuni/dexsim/m01.py)；输入与来源记录在 `outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/manifest.json`。
- 场景：W1 V030 双平行夹爪、固定根部、动态关节、two_way；泡沫软梁 460×40×40 mm（Y×X×Z），密度 40，k_mu 2e4、k_lambda 5e4、k_damp 1e-3，1225 节点 / 3840 四面体；初始底面离台 5 mm（静置约 4.5 mm，避免初始碰撞弹起）；60 Hz、10 子步 × 12 VBD 迭代、CUDA Graph。
- 抓取：两夹爪从上方夹持距两端 30 mm 处（间距 400 mm），手指沿 X 闭合；夹爪向机器人方向后仰 15°，左夹爪绕竖直轴转 180°。纯 IK 扫描：竖直向下的姿态在抓取高度碰到腕关节限位导致抖动；300 mm 梁时两手相机重合穿模，加长到 400 mm 抓取间距后两相机最近 131 mm，关节距限位 ≥ 0.42 rad。
- 动作（16 s）：抓取 → 抬高 150 mm → 两端各内收 25 mm（缩短 12.5%）、内倾 10° → 保持 2 s → 恢复 → 向机器人方向搬运 100 mm → 放下松开。
- 预览观测（单次，非正式统计）：刚抬起时中部下垂 27.1 mm，保持段 83.6 mm（额外 56.5 mm）；端部节点组间距 392.6 → 346.5 mm；保持段最低节点离台 66.5 mm；搬运时 29.9 mm；释放后 4.1 mm（台面摩擦保持的残余弯曲）；TCP 跟踪误差最大 2.8 mm。刚抬起时的下垂与密度无关（25/40 kg/m³ 相同），推测来自离台时夹持处的小转动并由摩擦保持（未单独验证）；k_mu 2e5 时手指压入更少，下垂反而更大。

```bash
/home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/m01.py
```
以上物理目标与阈值是候选值，正式矩阵开始前冻结，不能根据失败平台临时放宽。

- 遵循 [PLAN](../PLAN.md) 的 G0–G6、[PARITY_CHECKLIST](../PARITY_CHECKLIST.md) 和 [SCRIPT_CONTRACT](../SCRIPT_CONTRACT.md)。
- 本目录只拥有任务定义、矩阵和编排；平台代码归 `physics/platforms/dexuni/<platform>/`，小型输入定义归 `assets/physics/dexuni/`。
- 生成二进制输入使用外部缓存，原始运行/报告/视频归仓库 `outputs/benchmarks/dexuni/`，不再建本任务 `raw_data/videos/results` 源目录。
- 正式主组使用 matched_replay 与已验证的相同驱动；单向组明确 kinematic，不报告机器人受载反馈。matched_policy 另组，不称同输入。
- 留存运行错误/超时/OOM/缺失与所有诊断尝试；不开“最新成功覆盖失败”的正式分析。
- 旧 S01 的参数/成功率不能代替本任务的材料、接触、稳定性和受载验证。
