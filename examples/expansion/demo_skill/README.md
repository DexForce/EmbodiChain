# DemoSkill：从专家示教构建任务专属技能

DemoSkill 将连续子任务或完整任务保存为一个技能资产，通过 registered Semantic Call
降低为 `ExecuteDemoSkill`，复用现有 Task Program、AtomicSkillEngine 和 Expansion Engine。
内部阶段用于约束、来源映射和扩增授权，无需逐段映射到标准技能。

## 原生搬运与持续持物

生成使用简写、共享目标和 verified effects 的独立配置：

```bash
python -m embodichain run-example expansion/demo_skill/present.py -- \
  --output /tmp/carton-present --duration 2
python -m embodichain run-task --gym_config /tmp/carton-present/task.yaml \
  --num_envs 1 --device cpu --headless --renderer no-render \
  --disable_sensor --filter_dataset_saving --max_episodes 1
```

输出目录必须不存在。准备器复用官方 Panda 场景与标定，在写出前执行普通部署 preflight。
实际执行按控制时钟逐步验证物体附着与位姿，未满足完整保持窗口不会判定成功。

## 真实数据示例

本例使用已有 carton real2sim 示例的同一条 DROID 真机遥操记录、Panda + Robotiq、物理
场景和任务验收条件。当前实测范围与回执见 [qualification.json](qualification.json)。原始 HDF5、导入的实测关节/速度/TCP、原始提交命令，以及经过
原链路修正的 nominal reference 分开保存。原始 HDF5 被校验并随资产归档。DROID 真实输入沿用原示例的 CC BY 4.0 许可。

```bash
python -m embodichain run-example expansion/demo_skill -- \
  --output /tmp/demo-skill-carton
```

首先通过官方 `carton_stocking` 配置生成一条经过验证的参考记录，再构建并实际执行：

- `mixed`：标准 Pick → DemoSkill 运输 → 标准 Place，保留两个 prototype 的验证器。
- `whole`：一次 DemoSkill 调用封装完整任务，包括连接、操作和等待区间。

可直接复用已有 real2sim 示例的已验证数据集；提供包含 `meta/real2sim/collection-report.json`
的 dataset 目录：

```bash
python -m embodichain run-example expansion/demo_skill -- \
  --output /tmp/demo-skill-carton \
  --baseline /path/to/verified/carton-dataset --prepare-only

python -m embodichain expansion run --job /tmp/demo-skill-carton/mixed/job.yaml
python -m embodichain expansion run --job /tmp/demo-skill-carton/whole/job.yaml
```

`*_asset/` 包含不可变资产和来源文件；`mixed/`、`whole/` 包含普通的 program、integration、
部署和扩增 job。执行器来自生产包，示例不实现自己的物理推进或提交调度。
已有输出使用 `expansion resume`；生成的来源与数据集不提交到仓库。最终稳定等待由
Task Program 的 `wait_stable` post-policy 执行。准备器根据执行 trace 区分动作与确认等待，
按关节名称恢复 reference，并保存 mimic/保持目标修正前后的数组和说明。

这是迁移已有 prepared motion 的兼容示例。新的遥操任务可以直接创建 RobotDemonstration
和自己的 DemoSkillSpec，不需要先执行标准 Pick/Place。此例在已有刚体代理中的成功
不等于对真实动力学或实机执行的认证。

## 构建自己的资产

使用 `save_robot_demonstration()` 保存 RobotDemonstration，然后运行：

```bash
python -m embodichain real2sim build-demo-skill recording/demonstration.json \
  --spec skill.yaml --joints joints.json --control-dt 0.03333333333333333 \
  --output prepared-skill
python -m embodichain real2sim validate-demo-skill prepared-skill/demo_skill.json
```

`DemoSkillSpec` 声明逻辑场景角色、任意命名的连续阶段、硬事件、入场状态、实测约束和
扩增权限。必须配置终止成功条件。接触阶段必须有持续接触约束；缺失或不完整的接触
观测既不能证明接触，也不能证明无接触。默认要求空手入场；运输已有工具时显式声明
`initial_held_role`，并要求独占的已验证抓持状态及终止相对位姿证据。

只有 qpos 的资产支持固定场景复现；XY/yaw 迁移需要显式 TCP 标定、参考位姿和同一
物理几何。入场误差与允许的场景迁移是不同配置。接触入场不自动添加接近动作。
显式授权的自由入场连接需要宿主安装独立的完整计划碰撞检查；已准备的连接也可以
作为 DemoSkillReference 的生成区间保存。

使用 DemoSkillReference 时，原始状态仍保存在 `demonstration`，原始规范化结果保存在
`qpos` / `world_tcp`。执行使用单独的 `execution_*` 视图，保留变换后的场景参考、准备
证据、原始帧映射和 `source_mask`；生成的连接/等待帧不被标为真实观测。准备不能产生
物理成功计数。

## 配置接入与扩增

新配置在 `runtime_services.demo_skills` 中命名资产并固定 manifest、内容摘要及所需接触绑定；
program 使用 `demo_skill: {skill, roles}`，部署自动为展开后的每次调用生成来源与指纹。
本示例的 `mixed`、`whole` 生成器使用这条路径。旧配置仍可声明
`kind: demo_skill` 的 registered lowerer，并在 `nominal_motion_sources`
中使用 `provider_id: demo_skill`、`revision: '1'`、`call_id`、`call_index`、
`program_fingerprint`、`asset_manifest` 和 `asset_sha256` 固定来源。同一资产在循环中的
不同出现位置使用不同 call index。没有来源绑定的普通调用继续使用 native planning。
manifest 可使用部署相对路径或现有 `{asset, path}` 引用。这里的资产摘要是
`DemoSkillArtifact.content_sha256`，不使用随文件位置改变的路径作为身份。

原生观测适配器支持刚体相对位姿和显式 ContactSensor 证据。接触声明同时配置
`contact_sensor_uid`；使用 `tcp` 接触角色时另配 `tcp_contact_links`，它们是资产定义的
原始 link 名。任务专属 scalar metric 通过可信的 DemoSkillEvidenceProvider 提供；
配置文件不接受任意 Python 回调或 import 路径。

自由区间按资产、部署/Expansion Profile 和 job 的权限交集扩增。内部事件及其导数邻域、
夹爪、接触和保持区间受到保护。接触局部路径只随已授权参考系整体迁移。改变几何、
跨机器人迁移、力控和在线策略学习不在首期支持范围。

空间变体使用既有 `motion_profile`。独立的 mapped timing job 设置
`demo_skill_duration_scale`、`demo_skill_velocity_limits` 和
`demo_skill_acceleration_limits`；倍率必须不小于 1，且不能超过资产 `max_duration_scale`，仅 `retime`
授权的自由区间改变时长。空间和时间变体分别运行与验收。

`prototype_validation_profile` 为混合程序中选定的标准 prototype 保留原有 tabletop
准备和计划验证。整个 episode 的碰撞及任务结果独立由 `collision_profile` 验收。
示例使用 `sampled_rigid_v1`，覆盖全部物理 box/plane 与显式允许接触对；它是离散采样
检查，不构成连续碰撞保证。更广场景继续使用既有 feasibility collision profile。

记录保留 T 个实际提交动作、T+1 个实测状态、调用来源、事件与验收结果。失败 attempt
保留诊断和可用的部分轨迹；成功计数只来自确认回执。视觉子样本仍沿用 #918 的父子
关系与计数规则，不增加物理成功次数。


接触传感器在 `capture_media: false` 时仍会采集；摄像头和渲染保持关闭。
`sampled_rigid_v1` 检查器在每次场景准备后绑定各物理行最新的机器人基座，避免
scene atlas 中的安装位姿变化沿用旧检查坐标。带有符号状态变化的 DemoSkill 必须
同时获得宿主效果批准；只传入 `runner.step()` 会继续等待，不会由来源批准自动提交。

资产时间映射按完整字段和数组字节计算摘要，不依赖 NumPy 的打印精度或省略设置。
本次修正前构建的 DemoSkill 资产需要重新构建并更新 integration 中固定的内容摘要；
原始示教文件不需要修改，旧摘要不会被自动视为新身份的别名。
