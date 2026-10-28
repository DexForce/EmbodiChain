> 历史设计快照，非当前执行计划。当前安排见[PLAN](../../PLAN.md)。旧 M04 棱台方案已被折衣任务替换。

# DexUni 机器人操作与多材料耦合实验

统一存放十一项任务的脚本、配置、原始测量、对比视频、分析结果与说明。2026-10-21 由 DexSim 仓库 `benchmark/dexuni_tasks/` 整体移入 EmbodiChain 项目（本目录）；所有命令在 EmbodiChain 仓库根目录 `/home/oem/code/repos/EmbodiChain` 执行。

- [总计划与测量/耦合约定](./PLAN.md) 是共用契约。
- [对比检查清单](./PARITY_CHECKLIST.md)：每个任务开跑前逐项核对求解器选项对齐，以及 EmbodiChain 移植的检查方法。
- **最终对比对象：EmbodiChain（DexUni）对 Isaac Lab（Newton MuJoCo Warp + VBD）。** 每个任务先在本地 DexSim 上实现，再写 EmbodiChain 脚本，挑一两个状态与 DexSim 对比一致（差异不超过 DexSim 自身运行间差异）后，才跑 EmbodiChain 对 Isaac Lab 的全量矩阵。缺少能力标记 N/A，不更换求解器凑对照。
- **数据保留**：S02 起，DexSim 只用于一致性检查，数据不保留（结论数字写进该任务的决策记录）；任务目录只保留 EmbodiChain 与 Isaac Lab 的最终有效数据。S01 保留了第一阶段（DexSim 对 Isaac Lab）的数据作为记录。
- 解释器：DexSim `/home/oem/code/engine/dexsim/.env-3.12/bin/python`；EmbodiChain `.venv/bin/python`；Isaac Lab `~/app/IsaacLab/.venv/bin/python`（相机用 OVRTX）。`_lib/benchlib/` 是 DexSim 显存采样工具的副本。
- 每个任务的 `README.md` 是该任务场景、动作和判据的唯一说明来源。
- 九项预计低风险基准任务以双向为初始主组，M03/C01/C02 先用单向。模式仅为设计选择，性能未测；单向组不沿用接触限力闭爪或机器人受载反馈结论。

## 任务索引

| 任务 | 内容 | 机器人 | 初始主组 | 状态 |
|---|---|---|---|---|
| [S01](./s01_soft_compress_stretch/README.md) | 软体小变形压缩／拉伸与恢复 | Franka Panda + Panda 夹爪 | 双向 / `dynamic` | S01-C、S01-T：EmbodiChain 对 Isaac Lab 完成（5 %，待评审） |
| [S02](./s02_soft_pick_place/README.md) | 软块大变形抓取、提起与定点放置 | Franka Panda + Panda 夹爪 | 双向 / `dynamic` | 未实现／未运行 |
| [S03](./s03_cloth_corner_lift/README.md) | 薄布单角提起、悬停与放回 | Franka Panda + Panda 夹爪 | 双向 / `dynamic` | 未实现／未运行 |
| [S04](./s04_rigid_soft_squeeze_grasp/README.md) | 双侧分别夹取刚块和软块：小变形夹持 | W1 V030 + 双 Pika | 双向 / `dynamic` | 未实现／未运行 |
| [M01](./m01_soft_beam_bend/README.md) | 双臂软梁弯曲、搬运与恢复 | W1 V030 + 双 Pika | 双向 / `dynamic` | 未实现／未运行 |
| [M02](./m02_cloth_single_fold/README.md) | 双臂方布单次对折：两层接触 | W1 V030 + 双 Pika | 双向 / `dynamic` | 未实现／未运行 |
| [M03](./m03_four_layer_cloth_pick_place/README.md) | 双臂直接夹取四层布料并定点放置 | W1 V030 + 双 Pika | 单向 / `kinematic` | 未实现／未运行 |
| [M04](./m04_low_stretch_cloth_frustum/README.md) | 双臂持低延展布下压棱台平顶 | W1 V030 + 双 Pika | 双向 / `dynamic` | 未实现／未运行 |
| [M05](./m05_inflatable_pouch_load/README.md) | 夹取充气软包放到桌面，再放刚体块压载 | W1 V030 + 双 Pika | 双向 / `dynamic` | 未实现／未运行 |
| [C01](./c01_supermarket_packing/README.md) | 超市混合装袋：刚体、软体和充气软包 | W1 V030 + 双 Pika | 单向 / `kinematic` | 未实现／未运行 |
| [C02](./c02_bag_frame_packing/README.md) | 左臂持袋口框、右臂装填软块与刚块 | W1 V030 + 双 Pika | 单向 / `kinematic` | 未实现／未运行 |

## 目录结构

```text
dexuni_benchmark/
├── README.md                    # 总入口和运行产物约定
├── PLAN.md                      # 共用实验设计、测量来源和模式政策
├── PARITY_CHECKLIST.md          # 两平台求解器选项对齐清单
├── assets/README.md             # 共享资产引用与 hash 约定
├── report/README.md             # 汇总图表、对比结论和数据索引
├── s01_soft_compress_stretch/
│   ├── README.md                # 本任务场景、动作、测量与验收
│   ├── scripts/README.md        # 计划脚本入口和当前实现状态
│   ├── configs/README.md        # 场景、初态、轨迹与判据配置要求
│   ├── raw_data/                # 按平台/run_id 保留原始测量
│   ├── videos/                  # 原始视频和配对对比视频
│   └── results/                 # 指标、曲线与分析结论
└── ...                          # 其余十一项使用相同结构
```

## 原始数据与视频放置

每次运行使用独立 `run_id`，推荐包含任务、子项、平台、coupling、joint_mode、初态 ID 和重复序号；时间戳用 UTC。每个 manifest 保存这些字段，不能只靠文件名区分。不同模式、冷/热、纯计时/全量采集分别运行、分别编号。

```text
<task>/raw_data/<platform>/<run_id>/
  manifest.json                 # 资产/源码/环境/配置 hash，实际模式，完成状态
  config.json                   # 本次完整配置快照；也可使用 YAML
  states.npz                    # 复制后的实际关节、物体/节点状态与时间戳
  timing.csv                    # 完成计时和计数；采集开销另列
  run.log                       # 原始标准输出/错误及失败阶段
  images/                       # 开启图像采集时保存原始图像/标定
<task>/videos/<run_id>/
  front.mp4                     # 视角名举例，与 manifest 对应
  top.mp4
<task>/videos/comparisons/<comparison_id>/
  comparison.mp4                # 同初态、模拟时间对齐的并排对比
  comparison.json               # 两侧 run_id、视角、时间映射、制作命令
<task>/results/<comparison_id>/
  metrics.csv                   # 可回溯 run_id 的计算结果
  plots/
  analysis.md                   # 解释、失败、缺项及引用
```

以上是产物格式约定，不代表文件已经存在。S01 的压缩/拉伸使用不同子项 ID。借用的粒子视图须在下次 update 前复制；不能把最终状态引用存十一次冒充轨迹。失败、NaN、OOM 和超时也保留 manifest/log，不与成功结果混算或覆盖。原始数据只读留存，重算结果写入 `results/`。

双向和单向，以及动态和运动学组分别标注；对比视频不能把不同耦合语义剪成同一性能比较。没有对等平台结果时保留单平台视频并注明缺项，不制作假的并排对照。

## 当前工作边界

本次只建立目录、说明与模式分配。之后逐任务实现脚本、验证公开读数、冻结配置并采集；复用仓库已有 SceneBuilder/Scene、控制器与 benchlib。不得将 TODO 或空数据集标为实验通过。
