# Franka 在线倒水

倒水任务在同一个 World 中通过 DexUni 推进 Franka、动态杯子和 DFSPH 水体。机器人
使用关节驱动和真实接触抓起杯子，将水倒入接水碗，再放回杯子。耦合方向为刚体到
流体；水不会反作用于机器人或杯子。初始水体按杯内体积采样，不依赖预录粒子或轨迹。

## 依赖

基础流体接口来自 [PR #777](https://github.com/DexForce/EmbodiChain/pull/777)。
运行此任务需要提供 `DexUniSolverCfg.sph_cfg`、SPH world 隔离和选择性 reset 的
DexSim 开发构建；`dexsim_engine==0.5.1rc1` 不支持该组合。当前验证使用
`0.5.1+0b8422cc6f6717058bd15371e61bd35cdbbe3c0c`、Newton 1.7 和 CUDA。
任务网格由资产缓存中的 `PourCupAssets` 提供。

## 运行

Gym Task Program 在线生成 `Pick → MoveHeldObject → Pour → Place` 动作，
并记录实测流体与杯子状态：

```bash
python -m embodichain.lab.scripts.run_fluid_task_program \
  --num-envs 1 --renderer fast-rt --output outputs/fluid-task-program-online
```

任务的执行节奏由
[`trajectory_open_loop_fluid_pour.yaml`](../../../../components/execution_policies/trajectory_open_loop_fluid_pour.yaml)
配置。轨迹采样数是下限；规划器会根据关节速度和加速度约束增加采样点，并保留
抓取及放回时的稳定等待。

也可使用普通任务入口：

```bash
embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml \
  --headless --renderer fast-rt
```

独立 Sim 示例从当前场景规划 Atomic Action 目标，通过相同的 DexUni/DFSPH 物理场景
执行并录制视频：

```bash
python scripts/tutorials/sim/franka_dexuni_pour_demo.py --headless --renderer fast-rt
```

增加 `--num-envs 4` 可在同一个 World 中并行运行四个环境，分别输出
`fluid_sim_env0.mp4` 至 `fluid_sim_env3.mp4`。
场景间距按录制相机的视野设置，以避免相邻环境的倒水工作区出现在预览中。
各环境仍共享渲染场景；其他相机视角不自动隔离环境。

默认粒子间距为 1.5 mm，初始填充 76,607 粒。Gym 的 `--particle-spacing 0.003`
或 Sim 的 `--spacing 0.003` 可用于较快的物理检查，但会改变流体分辨率。
`--seconds 2` 只运行起始片段，不代表完整任务验收。NoRender 仅用于物理检查。

## 求解与碰撞

`env.fluid.yaml` 声明 DexUni 的 `sph_cfg`、机器人和场景。外层物理步为 1/240 s，
每步 2 个 DFSPH 子步，每个 DFSPH 子步包含 15 个 MuJoCo 刚体子步；控制周期为
0.025 s。未指定关节所有权时，流体准备阶段将完整的机器人和杯子关节拓扑分配给
MuJoCo，以满足当前 DexUni/DFSPH 原型的纯 MuJoCo 刚体要求。任务使用完整机器人
关节目标，使主从手指都收到夹持目标。

杯子质量为 0.30 kg，凸分解形状提供机器人抓取接触，独立的
`particle_collision_shape` SDF 提供流体边界。当前 DexSim 的碰撞描述符没有单独
排除边界质量的字段，因此杯子的质心和惯量显式使用仅由刚体接触形状计算的值，
避免额外 SDF 改变质量属性。使用支持 `contributes_to_mass` 的新构建时，边界也会
显式标记为不参与质量计算。

## 录制与验收

FastRT 使用蓝色液面材质，并为每个环境生成独立的 960 × 720、40 fps 视频。Gym
正常录制保存到 `videos/env_000/episode_000000_fluid_gym.mp4`；未完成或不合格的
片段保存到 `videos/diagnostics/`。Sim 示例输出 `fluid_sim_env0.mp4`。

Gym 的 `fluid_gym_result.json` 和 Sim 的 `validation.json` 分别记录任务完成情况、
杯子抬升、最大倾角、放回误差和接水量。完整验收要求实际抬升至少 5 cm、倾角
至少 45°、放回误差不超过 3 cm，且最终接水至少 50 mL；规划成功或部分视频不能
替代这些物理测量。`--record-trajectory` 仅保存可选的在线状态诊断。

`FluidReplayEnv` 是独立的历史记录诊断工具，不是在线任务的默认输入。它使用
运动学边界和固定根记录，不能替代动态抓取验证。
