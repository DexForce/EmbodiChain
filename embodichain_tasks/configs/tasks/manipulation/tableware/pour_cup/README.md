# Franka 单臂马克杯倒水

使用现有 `Pick → simulation.pour → Place` Task Program 生成并验证空杯轨迹，
再在独立的 Newton SPH 环境中回放机械臂关节和杯子实测状态，观察接水效果。
流体接入使用 `SimulationManager` 的公共接口，求解器参数直接复用 DexSim `SPHSolverCfg`。

## 运行要求与资产

需要支持原生 SPH、volume map、液面重建和 OptiX 的 DexSim / Newton CUDA 运行时。
多环境还要求 SPH 的 world 隔离和选择性重置能力；旧运行时保留单环境门禁。
当前批量实现依赖尚未发布的 DexSim 扩展，发布 wheel / AOT 验收状态见 PR。

运行时网格和可选演示资料通过标准资产缓存下载：

```bash
embodichain data download --name PourCupAssets
# 仅复现原始 Sim 演示时需要：
embodichain data download --name PourCupDemoData
```

| 资产包 | 内容 |
| --- | --- |
| `PourCupAssets` | 杯子显示网格、杯壁、接水碗、台面和地面碰撞网格 |
| `PourCupDemoData` | 原始已验证轨迹、参考视频、最终粒子快照和验证报告 |

未预下载时，`get_data_path("PourCupAssets/cup.glb")` 等路径会自动下载并缓存。
`EMBODICHAIN_DATA_ROOT` 可覆盖缓存根目录；数据包托管于
[Hugging Face task_assets](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/task_assets)。
代码仓库和 wheel 仅携带下载器与配置，默认初始水体直接在内存生成。

## Task Program → 流体 Gym

从源码仓库根目录运行：

```bash
python -m embodichain.lab.scripts.run_fluid_task_program --num-envs 2 --renderer fast-rt
```

脚本先在 MuJoCo 中执行并验收抬杯、倾倒、放回，再启动独立 SPH Gym。
默认输出到 `outputs/fluid-task-program/`，包含源轨迹、源验收报告、接水量统计，
以及 `videos/episode_0_fluid_gym_env0.mp4` 等每环境独立视频。
录制通过 `record_camera` 事件调用 `record_camera_data`，使用 `per_env: true`、
`fps: null`，按实际控制频率保存（本配置为 24 fps）。

复用本工具生成的合格轨迹，或进行粗粒度局部重置检查：

```bash
python -m embodichain.lab.scripts.run_fluid_task_program \
  --trajectory outputs/fluid-task-program/task_program_trajectory.pt --num-envs 2

python -m embodichain.lab.scripts.run_fluid_task_program \
  --trajectory outputs/fluid-task-program/task_program_trajectory.pt \
  --num-envs 2 --renderer no-render --seconds 0.125 \
  --particle-spacing 0.006 --partial-reset-step 1
```

流体阶段默认使用 FastRT。当前液面几何只发布到 OptiX 场景，Hybrid 的 Vulkan
主表面无法显示，因此流体对象会拒绝 Hybrid；空杯轨迹生成仍可使用 Hybrid。
`no-render` 仅用于物理检查。粗分辨率短测不能替代完整倒水效果验证。

`FluidReplayEnv` 提供标准 `reset` / `step`、按环境独立时钟、流体选择性重置、
summary 和区域体积观测。它将单行合格轨迹广播到多环境，要求固定机器人根位姿
和单个 Newton 子步。回放由源状态驱动，不接受 action manager 或策略 action 数据录制。
耦合为单向：杯壁影响水，水不反作用于机械臂，SPH 阶段不重新求解抓持接触。
Task Program 验收和流体接水结果分别报告。

## 初始水体配置

`env.fluid.yaml` 声明 SPH 参数、刚体边界、流体和观测。
初始水体由杯子局部坐标系下的腔体截面描述，单位为米：

```yaml
spawn:
  kind: revolved_volume
  reference_uid: cup
  profile: [[-0.044, 0.031], [0.05, 0.0385]]  # [局部 Z, 腔体内半径]
  fill_level: 0.91
  clearance: 0.003
```

截面绕局部 Z 轴旋转并封闭两端。`fill_level` 是完整腔体高度的比例，
液面裁剪与壁面间隙分别处理；它不表示稳定后的体积百分比。
默认 1.5 mm 规则网格采样得到 76,607 个粒子，名义体积约 258.55 mL。
局部点集只生成一次并保存在内存；reset 根据杯子位姿和 arena 偏移恢复选中环境，
回放起始的静置阶段用于液体稳定化。

修改配置后重新构建会重新采样。`--particle-spacing` 同时按间距比例缩放显示半径、
液面网格间距和重建核半径。外部点集 / NPZ 导入仍受支持，但其采样分辨率固定，
不接受脚本的间距覆盖。

## 独立 Sim 演示

```bash
python scripts/tutorials/sim/franka_dfsph_pour_demo.py --headless
```

默认从 `PourCupDemoData/trajectory.pt` 获取原始已验证轨迹，使用 OfflineRT 渲染；
运行时杯子网格来自 `PourCupAssets/cup.glb`，接水碗由脚本生成。
去掉 `--headless` 可打开原生窗口；`--renderer fast-rt` 可切换到 FastRT。
`--trajectory` 可指定其他轨迹，`--seconds 2` 可限制运行时长。
默认结果保存到 `outputs/franka_pour_cup/dfsph_stable_bowl/`，包括视频、关键帧、
粒子状态和 `validation.json`。原始演示的环境版本与统计见下载包中的报告。

## 空杯 Task Program 部署

以下两个部署只生成空杯任务，不包含 SPH 或接水碗。
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

| 配置 | 职责 |
| --- | --- |
| `task.franka*.yaml` | 运行入口及环境、机器人、程序和执行策略选择 |
| `env.yaml` / `env.newton.yaml` | 空杯场景、物理后端、接触参数、录制和复位 |
| `env.fluid.yaml` | SPH 场景、初始水体、流体观测和每环境相机录制 |
| `task_program/program*.yaml` | Pick → Pour → Place、稳定等待和终点验证 |
| `task_program/integration*.yaml` | 语义绑定、侧抓位姿、抬升距离和倾倒指令 |
| `components/embodiments/franka_panda_*.yaml` | 机器人、传感器、接触参数和技能声明 |
| `components/execution_policies/trajectory_open_loop_*.yaml` | 轨迹插值和动作后稳定策略 |

组件路径以 `embodichain_tasks/configs/` 为根。静态检查空杯部署：

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.yaml \
  embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.newton.yaml
```
