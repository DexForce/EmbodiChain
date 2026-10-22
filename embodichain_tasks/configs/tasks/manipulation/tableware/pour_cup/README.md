# Franka 在线倒水

Sim demo 和带水 Gym 部署均在同一个 World 中在线推进：MuJoCo-Warp 求解机械臂驱动、
重力和杯把抓持接触，DFSPH 读取当前刚体位姿和速度。当前是单向耦合，水没有反作用力。
初始填充由参数采样，不需要预录关节/杯子轨迹或粒子文件。

## 依赖与 PR 分层

基础流体能力和不依赖任务资产的 `scripts/tutorials/sim/create_fluid.py` 属于
[PR #777](https://github.com/DexForce/EmbodiChain/pull/777)，支持
`dexsim_engine==0.5.1rc1` 的单环境 SPH/DFSPH。
本任务、机器人倒水示例、Gym/Task Program 和诊断回放统一在
[PR #790](https://github.com/DexForce/EmbodiChain/pull/790) 中维护。

本页的在线 SPH + MuJoCo 路径不能使用原版 rc1：它需要包含
[DexSim MR !1473](http://192.168.3.16/Engine/dexsim/-/merge_requests/1473)
（`codex/fluid-sph-worlds`）的构建，提供 `rigid_solver_cfg`、
`rigid_solver_substeps`、粒子边界质量排除和多环境隔离/选择性重置。
不能仅凭版本名为 `dev` 判断这些能力已存在。

另一个 DexUni + DFSPH 原型需要
[DexSim MR !1475](http://192.168.3.16/Engine/dexsim/-/merge_requests/1475)
（`feat/dexuni-dfsph-prototype`）中的 `DexUniSolverCfg.sph_cfg` 等原型选项。
该路径目前仅支持单环境，仍为单向耦合：

```bash
python scripts/tutorials/sim/franka_dexuni_pour_demo.py --headless
```

下列多环境命令针对包含 MR !1473 的 SPH + MuJoCo 构建，不适用于 DexUni 原型。

## 运行

Gym 使用普通 `EmbodiedEnv`，通过 Task Program 的 `Pick → MoveHeldObject → Pour → Place`
生成动作目标，和其它任务一样由 `env.step(action)` 实时执行：

```bash
python -m embodichain.lab.scripts.run_fluid_task_program \
  --num-envs 2 --renderer fast-rt --output outputs/fluid-task-program-online
```

也可使用普通任务入口：

```bash
embodichain run-env \
  --gym-config embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml \
  --headless --renderer fast-rt
```

独立 Sim demo 直接创建 `SimulationManager`，从当前场景规划 Atomic Action 关节目标，
每帧设置驱动目标并推进真实动力学：

```bash
python scripts/tutorials/sim/franka_dfsph_pour_demo.py --headless --renderer fast-rt
```

Gym 的 `--particle-spacing 0.003 --renderer no-render` 可用于较快的物理检查；
`--seconds 2` 只验证起始片段，不能作为完整倒水验收。Sim 对应参数是 `--spacing`。
两种入口默认均为 1.5 mm；粗分辨率会改变水量和边界精度。

`env.fluid.yaml` 拥有求解器、场景、初始水体、观测和录制；`task.franka.fluid.yaml`
选择 Franka 真实接触配置、Task Program、语义绑定和执行策略。
带水程序先验证实际抬杯，再将杯子移动到相对接水碗定义的倾倒位置，最后倾倒并放回。
`env.newton.yaml` 与 `env.yaml` 仍提供独立的空杯任务。

## 求解与碰撞几何

直接复用 DexSim 的 `SPHSolverCfg`，通过 `rigid_solver_cfg` 选择现有
`MJWarpSolverCfg`，没有另建 EmbodiChain SPH 配置类。外层物理步为 1/240 s，
含 6 个 SPH 子步；每个 SPH 子步含 7 个 MuJoCo 微步，控制周期为 0.025 s。
两种求解器共享模型、状态和仿真时钟，微步不会重复增加时间。

杯子是一个动态刚体，质量 0.30 kg：`shape` 的凸分解网格提供杯身/杯把的刚体接触，
`particle_collision_shape` 的杯内 SDF 提供流体边界。后者挂在同一个刚体上，
不增加质量、不参与刚体接触，也不通过位姿跟随器或固定关节伪造抓持。
机器人使用已有 Franka firm-grip 配置，保留关节驱动和指腹的抗扭/滚动摩擦。

当前要求兼容的 DexSim Newton CUDA 实现，包含 SPH 与 MuJoCo 的在线组合、
world 隔离及选择性重置支持。实测运行时使用隔离 JIT 构建；发布 wheel/AOT 尚需单独验收。
场景准备后粒子数量和物理拓扑固定。

## 初始填充

```yaml
spawn:
  kind: revolved_volume
  reference_uid: cup
  profile: [[-0.044, 0.031], [0.05, 0.0385]]
  fill_level: 0.91
  clearance: 0.003
```

截面绕杯子局部 Z 轴旋转。`fill_level` 是腔体高度比例；壁面间隙不再裁剪液面。
1.5 mm 网格生成 76,607 粒，名义体积约 258.55 mL。点集只在声明时采样一次并保存在内存，
reset 恢复选中环境的初始状态，不额外推进其它行的物理时间。机械臂接近期间
容器和液体自然稳定。需要在移动后的容器中重新填充时使用
`reset_fluid_in_container` 事件。NPZ 仍可用于外部点集导入，默认路径不读取或生成 NPZ。

## 录制与验收

当前原生液面位于 OptiX 场景，使用 FastRT；Hybrid 的 Vulkan 主表面无法显示液体，
因此流体门禁会拒绝 Hybrid，包括复用已初始化 Hybrid 的进程。纯物理检查使用 NoRender。

Gym 的 `record_camera` 事件使用现有 `record_camera_data`，配置 `per_env: true`、
`fps: null`，每个环境以控制频率 40 fps、960 × 720 分辨率录制杯口与接水碗的独立视角，输出
`videos/episode_0_fluid_gym_env0.mp4` 等文件。Sim 使用相同视角的 record camera，
输出 `fluid_sim_env0.mp4` 等文件。离屏读取前显式发布最新渲染状态。

Gym runner 在结束时保存最后一段视频，并写入 `fluid_gym_result.json`，分别记录
Task Program 结果、实测抬升/倾角/放回误差、接水量及墙钟耗时。
执行策略的 `projected` 完成状态不能单独证明物理成功；完整验收还要求实际杯子
抬升至少 5 cm、倾角至少 45°、放回误差不超过 3 cm，且碗内最终接水量至少 50 mL。
`--record-trajectory` 仅用于可选的实测状态诊断，不作为下一阶段输入。
Sim 报告为 `validation.json`。旧回放证据归档在 `PourCupDemoData`，不能用来证明在线效果。

任务网格由标准资产缓存中的 `PourCupAssets` 提供，托管于
[Hugging Face task_assets](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/task_assets)。
本 PR 不重新提交网格、预录轨迹、粒子快照或视频文件。
