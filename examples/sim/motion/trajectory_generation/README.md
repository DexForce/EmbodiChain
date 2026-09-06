# 固定场景轨迹生成与增强示例

## 并行抓取固定方块：姿态和路径增强

`cube_grasp_parallel.py` 在四个同步仿真环境中，让 UR5 + PGI 夹爪抓取同样的 5 cm 方块。机器人和方块的初始位姿相同，方块受重力和接触作用，夹起后能够移动。

在已安装 EmbodiChain 仿真依赖、UR5/PGI 资产及 TOPPRA 的环境中运行。视频使用离屏渲染，无需桌面窗口；此例使用 UR IK 和 `ik_interp`，不需要 cuRobo 后端。

```bash
python -m pip install 'imageio[ffmpeg]'
python examples/sim/motion/trajectory_generation/cube_grasp_parallel.py \
  --output /tmp/cube-grasp-parallel
```

输出目录必须不存在或为空。`--seed` 默认为 `13`，控制路径残差的局部随机种子；`--cuda-device` 默认为 `0`，选择渲染显卡。

| 四宫格位置 | 抓取朝向 | 自由运动路径 |
|---|---|---|
| 左上 | 0° 参考抓取 | 参考路径 |
| 右上 | 0° 参考抓取 | 增强路径 |
| 左下 | 绕方块局部 Z 轴旋转 90° | 为新抓取朝向重新规划的参考路径 |
| 右下 | 绕方块局部 Z 轴旋转 90° | 增强路径 |

姿态增强调用 `rotate_grasp_about_object_axis`，利用方块的四分之一转对称性生成新的 TCP 目标；每个目标都重新求解 IK。路径增强调用 `joint_residual`，只改变到预抓取位置之前的自由运动段，保留起终点、夹爪命令及后续接近、闭合和抬升段。现有 `MoveEndEffector` 与 `PickUp` 技能负责动作规划。

四路在同一个 `SimulationManager`、同一个物理时钟下执行。每个控制时刻一起采集四个相机画面，再拼成四宫格；彩色线是实测 TCP 轨迹，标题显示抓取角度、运动阶段和方块抬升高度。

输出文件：

- `preview.mp4`：1280×1056、20 fps 的同步对比视频，包含初始与终态画面。
- `rollout.npz`：四路参考/指令/实测关节轨迹、实测 TCP 和方块位姿、抓取目标与统一时间戳。轨迹数组以 `(环境, 时间, ...)` 排列；参考轨迹不含末尾额外保持段。
- `report.json`：增强参数、阶段边界、路径差异、各路抓取验收结果。末尾保持段要求抬升至少 12 cm、方块相对 TCP 的位置漂移不超过 1 cm，且始终靠近 TCP。

默认参数的实测结果为四路全部成功，保持期间抬升约 17.4–17.7 cm，两组增强/参考路径的最大 TCP 间距约 17 cm。物理运动为 13.55 秒，加上终态画面的播放间隔，272 帧视频时长为 13.60 秒。改变种子后仍会重新验收，未全部成功时返回退出码 1。

这是抓取增强的物理可视化示例，不写入专家 LeRobot 数据。需要碰撞、真实接触检查和多轮采集时，使用下方 `cube_pickup_collection.py`。

## 并行抓取的专家数据采集

`cube_pickup_collection.py` 复用上面的四路 cube 场景和原子动作规划，接入 `FixedSceneHost → GenerationRunner/GenerationSession → LeRobotEpisodeSink`。四路各自生成自由段残差，使用 0°/90° 抓取目标；每轮结束恢复整批初态，再继续生成，直到已确认提交数量达到目标或预算耗尽。

```bash
python -m pip install -e '.[trajectory-generation]' 'imageio[ffmpeg]'
python examples/sim/motion/trajectory_generation/cube_pickup_collection.py \
  --output /tmp/cube-experts --episodes 8 --record-video
```

`--episodes` 默认 8，范围 1–64；`--seed` 默认 13，`--cuda-device` 默认 0。输出目录必须不存在或为空。CPU 物理 200 Hz，控制及视频 20 Hz。采集版提高机械臂刚度到 200000，并将夹爪打开目标设在关节下限内侧 1 mm，以满足既定跟踪门槛并避免自由段触及限位；重力保持开启。

验收包括：

- 规划使用 URDF 碰撞形状的保守凸包，按完整关节状态检查手指和 mimic 几何，并检查持物路径与桌面、地面、其他物体的碰撞。非相邻自碰撞也参与检查，结构上相距两条关节边以内的 link 对沿用现有 cuRobo 排除规则。
- 实测关节及 cube 位姿再次检查几何，每个 5 ms 物理子步检查原生接触。只有声明的指尖、方块、支撑和固定安装部位接触被允许；接近段指尖接触限于 TCP 距方块 6 cm 的进入区域。未知物体、跨行接触、超过 2 mm 的穿透或接触数据超限都会拒绝。
- 闭合后抬升并保持 1.5 秒，要求方块至少抬高 12 cm、两个手指各有至少 95% 的保持采样出现真实接触力，且从抬升到保持的 TCP 相对漂移不超过 1 cm / 0.15 rad。原有速度、加速度、轨迹质量和末端跟踪检查继续生效。
- 只有验收通过、写入封存且成功回读的 episode 才进入 manifest。视频也会显示未通过的尝试，不能用视频代替数据验收。

输出 `preview.mp4`（开启视频时）、`generation_report.json`、`pickup_report.json`、`manifest.json` 和逐 episode 的 LeRobot 分片。每条数据含 T 个真实控制目标及 T+1 个实测观测/时间戳。cube/TCP 位姿在 LeRobot 中按行优先展开为 16 维，`episode.json` 的 `observation_shapes` 保留 `[4, 4]`；终态 `terminal.npz` 保持矩阵形状。`commanded_joint_indices` 标识完整关节向量中真正下发的关节列。

范围：单个固定基座 URDF 机器人、盒状刚体、CPU 物理。碰撞验证是有界采样，未提供连续碰撞保证。默认执行离线编译模板的 qpos 回放，不提交编译时预测的符号效果。加上 `--runtime` 可切换到下方的原子运行时执行；Gym 接触采集与通用 YAML 启动器仍是后续工作。

## 通过 Atomic Runtime 采集

```bash
python examples/sim/motion/trajectory_generation/cube_pickup_collection.py \
  --output /tmp/cube-runtime-experts --episodes 8 --runtime --record-video
```

场景、增强因子和物理验收与上面的采集示例相同。`PickUpRuntimeSource` 在每次全批恢复后创建新的 invocation 和 `ExecutionSession`，通过 `initial_plan_provider` 将选中的五阶段候选物化为一个 PickUp 计划。执行直接使用候选关节分支；运行时抓取目标取模板闭合前的 URDF FK 姿态，避免将分析 IK 目标与资产几何之间的腕部偏移带入持物效果。

`ExecutionRunner` 通过 `SimulationExecutionAdapter` 下发命令并检查机械臂反馈，阈值为运行中 0.08 rad、末端 0.05 rad。手指依靠真实双侧接触及持物稳定性验收。全部命令执行后，使用当前观测关节的 FK 和实测物体位姿，在计划使用的 motion endpoint TCP 坐标系中验证持物效果；原生接触 TCP 仍单独检查，成功后才提交新的 held-object 状态。

运行时与采集器共用 20 Hz 控制时钟；每条数据仍是 271 个实际命令和 272 个观测。计划重试/重规划、命令或时钟不匹配、额外等待及效果验证失败都会停止整批活跃行、保持当前关节状态并拒收数据。恢复命令不进入专家轨迹。尾批可保留空闲行，单行运行时失败会保守拒收同批其他活跃行。

每条 episode 的 `atomic_runtime` 元数据记录命令数、计划尝试数、事件计数、完成状态、效果误差及计划/接触 TCP 定义；同名强制验收项必须通过。`pickup_report.json` 标识本次执行来源。视频与 LeRobot 文件的保存格式保持一致，完整的手写/原子 × sim/Gym 四组合验收仍待后续完成。

## 自由运动采集

`free_motion.py` 使用普通重力下的纯机械臂 UR5，在固定场景中执行关节轨迹，经过规划、实测碰撞、任务和运动质量验收后保存一条 LeRobot episode。仿真物理在 CPU 上运行，cuRobo 和可选离屏相机使用 CUDA，无需桌面窗口。当前示例不包含抓取或持物。

在已安装 EmbodiChain 仿真依赖、CUDA、cuRobo 和 LeRobot 的环境中，从仓库根目录运行。保存视频还需要 `imageio-ffmpeg`：

```bash
python -m pip install imageio-ffmpeg
python examples/sim/motion/trajectory_generation/free_motion.py --output /tmp/ur5-video --record-video --duration 5 --joint-displacement 0.4
```

输出目录必须不存在或为空。上述命令记录 5 秒物理运动，使用 640×480 离屏相机流式写入 20 fps 的 H.264 视频。包含初始帧和终态帧，共 101 帧，因此视频时长为 5.05 秒。

不需要视频时可直接运行默认的 1 秒轨迹：

```bash
python examples/sim/motion/trajectory_generation/free_motion.py --output /tmp/ur5-free-motion
```

可用参数：

| 参数 | 含义 |
|---|---|
| `--record-video` | 在输出目录保存 `preview.mp4` |
| `--duration` | 运动时长，默认 1 秒；范围 0.05–30 秒，必须是 0.05 的整数倍 |
| `--joint-displacement` | 第一个机械臂关节的正向位移，默认 0.08 rad；范围 `(0, 0.5]` rad |
| `--cuda-device` | CUDA 设备编号，默认 0 |
| `--robot panda` | 运行 Panda 负例；当前锁定夹爪模型会拒绝普通重力下的实测手指漂移 |

输出文件：

- `preview.mp4`：启用视频时的实际执行记录，独立于数值 LeRobot 观测。
- `generation_report.json`：计数、验收证据、失败原因及 `target_reached`。
- `manifest.json`：已确认提交的 episode 分片目录；被拒绝的轨迹不会作为已提交数据列入。
- 每个提交分片中的 `dataset/`、`terminal.npz` 和 `episode.json`：LeRobot 训练帧、终态/T+1 时间证据和谱系/验证信息。

视频存在不代表任务或专家数据验收通过，应查看报告和 manifest。未达到采集目标时脚本返回退出码 1；改变时长或位移后也会执行全部验收检查。
