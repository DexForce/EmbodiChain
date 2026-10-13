# Franka 单臂马克杯倒水 demo

## Sim 公共接口接入

当前实现用 `SimulationManager.add_fluid_object(FluidObjectCfg(...))` 声明水体；
求解器仍直接使用 DexSim 的 `SPHSolverCfg`。杯壁、台面和碗通过刚体配置声明，
碰撞网格与显示网格可分离。原生 SDF 和液面重建由底层适配负责。
离屏相机采集前使用 `sim.sync_render_state()` 发布状态，裁剪仅影响显示副本。
多子步时杯壁轨迹通过 `register_kinematic_rigid_trajectory()` 在 solver 子步采样。

当前支持 Newton CUDA、固定粒子数和单向耦合。场景准备后物理拓扑冻结。
多环境要求 DexSim SPH 提供 world 隔离及选择性重置能力；旧运行时仍限制为单环境。
`IMPLEMENTATION_PLAN.md` 记录分阶段验证结果。

## Task Program → 流体 Gym

```bash
python -m embodichain.lab.scripts.run_fluid_task_program --num-envs 2
```

默认先在 MuJoCo 中执行现有 Task Program，验证真实杯子抬升、倾倒和放回，
再启动独立 SPH Gym 回放其关节与杯子实测状态。输出包含源轨迹、源验收报告、
流体接水量统计和 `fluid_gym.mp4`。复用本工具生成的已验收轨迹可传
`--trajectory outputs/fluid-task-program/task_program_trajectory.pt`。
`--renderer no-render --seconds 0.125 --particle-spacing 0.006 --partial-reset-step 1`
用于低成本双环境局部重置检查；粗粒度检查不代表完整液体效果。

`env.fluid.yaml` 拥有 SPH 参数、刚体边界、初始粒子和流体观测；
`task.franka.fluid.yaml` 选择环境和机器人组件。粒子文件相对环境配置解析，
杯子、边界网格和初始粒子随任务包发布。求解器配置直接使用 DexSim SPHSolverCfg。

`FluidReplayEnv` 支持标准 reset/step、按行时钟与流体重置、summary 和区域体积观测。
它从单行合格源轨迹广播到多个环境，要求固定机器人根位姿和单个 Newton 子步。
机器人动作由源状态决定，因此拒绝 action manager 和策略 action 数据录制。
流体不会反向影响机械臂；Task Program 源验收与流体接水效果分别报告。
本次实测依赖隔离的 DexSim JIT 构建、Newton 1.6 / Warp 1.17；发布 wheel 尚未验收。



使用真实 Franka Panda 机型的仿真模型，从侧面夹住正常大小的马克杯杯把，
抬杯、向碗里倒水、回正并放回。这里包含两个入口：

- **空杯 Task Program**：只用 YAML 组合现有 `Pick → simulation.pour → Place`，
  分别提供 Default 和 Newton + MuJoCo-Warp 部署。
- **DFSPH 带水演示**：教程脚本回放已验证的 MuJoCo 机械臂和杯子轨迹，
  同时通过 EmbodiChain 的 Newton 后端模拟液体。

原始教程入口保留用于效果对照；生产接入位于 `embodichain/` 生产代码。液体采用单向耦合：移动杯壁
影响水，水不反作用于机械臂；DFSPH 场景不重新求解夹持接触。

## 直接运行带水演示

从源码仓库根目录，在已安装 EmbodiChain 和兼容 DexSim 的环境中运行：

```bash
python scripts/tutorials/sim/franka_dfsph_pour_demo.py --headless
```

脚本默认读取随本分支提交的
[`trajectory.pt`](../../../../../../scripts/tutorials/sim/assets/franka_dfsph_pour/trajectory.pt)，
不需要先生成轨迹。去掉 `--headless` 可打开原生窗口。
短时检查或指定自行生成的轨迹：

```bash
python scripts/tutorials/sim/franka_dfsph_pour_demo.py \
  --headless --seconds 2 --output outputs/franka_pour_cup/smoke

python scripts/tutorials/sim/franka_dfsph_pour_demo.py --headless \
  --trajectory outputs/franka_pour_cup/newton_smooth_mug_trajectories/traj_env0_000000.pt
```

本机验证使用 Python 3.10、DexSim 0.5.1 本地 wheel、Newton 1.6.0、
MuJoCo / MuJoCo-Warp 3.12.0、Warp 1.17.0 和 NVIDIA RTX 5090 D v2。
还需要 Torch、TensorDict、NumPy、SciPy、Trimesh、OpenCV、ImageIO 和
ImageIO-FFmpeg。完整版本记录见
[`environment.json`](../../../../../../scripts/tutorials/sim/assets/franka_dfsph_pour/environment.json)。
DexSim 必须包含原生 SPH、volume map、液体网格重建和 OptiX 支持；
本地 wheel 和虚拟环境不随分支提交，也未更改项目的全局依赖。
本演示针对源码 checkout，不承诺随 wheel 打包二进制资产。

Franka 模型沿用 EC 现有资产解析；杯子网格随任务提交，接水碗由脚本生成。
默认输出目录是 `outputs/franka_pour_cup/dfsph_stable_bowl/`，包括全景/近景
MP4、每秒关键帧、粒子位置和速度 NPZ、参数及统计 `validation.json`。
已录制的最终结果见
[演示资料目录](../../../../../../scripts/tutorials/sim/assets/franka_dfsph_pour/README.md)，
包含三段视频、原始轨迹、最终粒子快照和验证报告。

## 场景和求解器

| 项目 | 配置 |
| --- | --- |
| 马克杯 | 杯口外径 8.5 cm、高 10 cm、壁厚 4 mm、底厚 6 mm |
| 杯子容量和重量 | 几何满口容量约 362 mL，空杯 300 g |
| 杯把 | 平侧面截面 12 × 12 mm，总宽约 11.33 cm |
| 接水碗 | 弧形内壁、平底，口径约 18.9 cm、高 6.6 cm |
| 碗中心 | `[-0.755, -0.137, 0.50]` m |
| 液体 | 76,538 粒、间距 1.5 mm、初始名义水量约 258 mL |
| 边界 | volume map，最大网格分辨率 192，额外 SDF 惩罚力关闭 |
| 物理更新 | 1440 Hz，每步 1 个子步 |
| DFSPH | 散度迭代上限 10、密度迭代上限 20、压力松弛 0.2 |
| 液体参数 | 密度 1000 kg/m³、黏性系数 0.0006、离散表面张力 0.002 |
| 录制 | 24 fps，全景 1280 × 960，近景 960 × 720 |

体积按粒子初始体积估算。液体参数为演示调参，未做材料物性标定。
`--spacing`、`--volume-map-resolution`、`--viscosity` 和 `--surface-tension`
可调整对应参数。默认轨迹两倍速，首尾各静置 1 秒，共 30.975 秒仿真。

杯子位姿、线速度和角速度每个物理步更新。机器人关节与杯子位置用 PCHIP
插值，杯子姿态用 SLERP。volume map 从容器网格 SDF 烘焙，保留表面投影；
`volume_map_boundary_enabled=True`，`sdf_boundary_enabled=False`。
为适配 Newton 1.6，使用显式预烘焙 `force_sdf=False` 和 `approximation=SDF`。

脚本参考 DexSim 的
`examples/python/physics/particle_fluid/demo_newton_sph_dam_break_optix.py`，
通过 `SimulationManager` / `NewtonPhysicsCfg` 运行
`SPHSolverCfg(sph_method="dfsph")`。临时接入直接访问
`sim._spawn_scene.builder` 和 Newton 后端 `render_sync.sync()`；
这不是正式的公共液体 API。

## 碗内稳定性处理

旧版在碗内有持续的高频运动，散落粒子还会把重建范围拉大，使水面变粗或消失。
本版增加黏性耗散、减小离散表面张力、提高 volume map 分辨率，并把碗朝水流
落点平移 3.5 cm。重建使用 `aabb_trim_fraction=0.0`；每帧单独复制渲染位置，
把 `z < 0.49 m` 的点标为非有限值，由原生重建器过滤。

此过滤只作用于渲染副本，不回写求解器。全部粒子仍参与物理、保存和统计，
没有冻结水面或删除散落粒子。水流和碗内液面使用 OptiX 光追及表面网格重建。

最终完整运行结果：

| 最终位置 | 粒子数 | 名义水量 |
| --- | ---: | ---: |
| 碗内 | 41,594 | 140.38 mL |
| 杯内 | 34,698 | 117.11 mL |
| 容器外 | 246 | 0.83 mL |

散落比例为 0.32%，不是无泄漏仿真。第 19 秒至结束的每秒及最终快照中，
碗内速度均方根平均 **5.05 mm/s**、最大 5.87 mm/s；99 分位水面高度极差
约 **0.074 mm**。静水对照在相同旧版快照、相同零初速度及固定容器下各运行
5 秒，速度均方根从 0.13404 m/s 降到 0.004749 m/s（降低 96.46%）；
这是特定对照结果，不是完整倒水全过程的统一降幅。

两份完整视频各 744 帧、31 秒，18 秒静置近景片段为 432 帧，均逐帧解码通过。
31 份粒子快照的数量均为 76,538，位置和速度均有限。
这些是一次完整运行和离散快照的结果，不代表批量成功率。
已提交最终快照和全部统计，其他每秒快照可由运行脚本重新生成。

## 空杯任务及轨迹生成

首次运行 YAML 部署前，把杯子放入本机资产缓存；若设置了
`EMBODICHAIN_DATA_ROOT`，改用该目录：

```bash
mkdir -p ~/.cache/embodichain_data/PourCupDemo
cp embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/assets/cup.glb \
  ~/.cache/embodichain_data/PourCupDemo/cup.glb
```

Newton + MuJoCo-Warp：

```bash
embodichain run-env \
  --gym-config embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.newton.yaml \
  --headless --renderer hybrid --record_trajectory \
  --trajectory_save_dir outputs/franka_pour_cup/newton_smooth_mug_trajectories
```

Default：

```bash
embodichain run-env \
  --gym-config embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.yaml \
  --headless --renderer hybrid --record_trajectory \
  --trajectory_save_dir outputs/franka_pour_cup/mug_trajectories
```

YAML 部署均为空杯任务，不直接加载 DFSPH 或碗。
Newton 版杯心 `[-0.75, 0, 0.552]` m，机械臂基座在原点、朝向 180°。
50 cm 高的支撑台与侧抓预备姿态减少了腕部折叠；夹爪沿世界 -X 方向接近，
两指沿 Y 方向夹住杯把，TCP 距杯心朝机械臂方向 6.5 cm，抬升约 12 cm。
Default 版保留地面布局和约 20 cm 抬升。

Newton 配置使用 `mujoco_warp`、`solver: newton`、`implicitfast`，
5 ms 物理步分成 50 个子步；杯子和指腹使用 `condim: 6` 的抗扭/滚动摩擦。
运动学继续使用 FEP IK 和 `ik_interp`，没有新增 IK 或动作实现。
两种执行策略均为 `projected`，程序完成须结合实际杯子位姿和录像判断。

随分支提交的 MuJoCo 轨迹来自固定种子 0 的一次完整成功运行，1160 帧、20 Hz，
包含侧抓、抬升、倾倒、回正、放回和松爪。相对优化前布局，累计关节转动量
从 19.23 rad 降到 6.03 rad（减少 68.7%），最大关节速度从 0.473 rad/s
降到 0.163 rad/s。杯子最大抬升 11.98 cm、实际最大倾角 66.09°，
最终倾斜 0.28°、放回误差 2.09 mm。实际杯子倾角小于 80° 指令，
夹持中仍有相对转动。详细记录见资料目录的 `newton_smooth_validation.json`。

## YAML 分工

两种后端各使用 6 份 YAML，共 12 份：

| 配置 | 内容 |
| --- | --- |
| `task.franka.yaml` / `task.franka.newton.yaml` | 运行入口、种子、环境/机器人/程序/策略选择、初始姿态 |
| `env.yaml` / `env.newton.yaml` | 物理后端、杯子和台面、接触参数、灯光、录制和复位 |
| `task_program/program*.yaml` | Pick → Pour → Place、放回目标、稳定等待与终点验证 |
| `task_program/integration*.yaml` | 杯子语义绑定、侧抓位姿、抬升距离、80° 倾倒指令 |
| `components/embodiments/franka_panda_firm_grip*.yaml` | 原 Franka 模型、传感器、夹爪力度、接触参数和技能声明 |
| `components/execution_policies/trajectory_open_loop_pour_cup.yaml` / `trajectory_open_loop_gentle.yaml` | IK 插值，分别 240 / 360 点，动作后稳定 0.4 秒 |

组件路径以 `embodichain_tasks/configs/` 为根。为适配当前主分支，夹爪控制组
使用 `eef`；Default 策略单独命名，避免覆盖现有共享慢速策略。
机器人 link / joint 资产名保持原值。

## 验证和范围

静态检查两个部署：

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.yaml \
  embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.newton.yaml
```

原始 PR #771 新增任务配置、教程、资产和演示证据；当前接入分支另增加了 EC 流体公共 API。
没有新建教程单元测试；使用部署检查、现有配置/资源测试、语法检查和 GPU
短测验证。完整演示最初在提交 `3bb2a03c5` 上录制，分支基于 `453675d8a` 整理；
当前分支的检查结果单独记录在 PR，不能把原录像当作新基线的完整复跑。
现有上下文已覆盖配置和模拟器接入边界，无需增加架构文档。
