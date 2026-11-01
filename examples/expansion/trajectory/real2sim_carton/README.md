# Real2Sim：真实示范驱动的包装盒定向理货

本示例从固定 DROID 真实机器人示范构建 Pick、MoveHeldObject、Place 运动原型，
完成 affordance 修正、空间重定向、轨迹与时间扩增，再执行仿真并记录 LeRobot 数据。
所有正式候选均来自真实示范。

官方任务定义位于 `embodichain_tasks/configs/tasks/manipulation/carton_stocking/`。
场景、Task Program、来源绑定、执行策略和扩增配方均由官方任务维护；本目录提供来源
重建、复现实验和展示入口。正式任务执行不依赖 example Python 模块。

## 直接执行官方任务

```bash
embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/carton_stocking/task.panda.yaml \
  --dataset-dir /tmp/carton-dataset
```

默认直接读取已发布、固定哈希的三个真实原型，在基准布局执行六种配方。
无需先运行 example。`--max_episodes 1` 可执行名义轨迹，`--viser` 可查看实时执行。
首次使用会通过公共资源缓存获取并校验 `Real2SimCartonV1`；也可预先运行：

```bash
embodichain data download --name Real2SimCarton
```

## 从真实输入重建并采集

```bash
embodichain run-example expansion/trajectory/real2sim_carton -- \
  --output /tmp/carton-example
```

入口依次完成原始 HDF5、视频和标注校验，当前机器人／场景资格检查，三个原型构建，
基准布局六种配方的物理执行与独立审计。默认采集 **6 条完整 episode**。
原型和派生部署写入输出目录，不修改官方配置或已发布数据。

```bash
# 仅准备来源及派生部署
embodichain run-example expansion/trajectory/real2sim_carton -- \
  --output /tmp/carton-inputs --prepare-only

# 运行刚构建的真实来源
embodichain run-task --gym-config /tmp/carton-inputs/deployment/task.yaml \
  --dataset-dir /tmp/carton-rebuilt-dataset

# 重用原型，改变空间布局和扩增配方
embodichain run-example expansion/trajectory/real2sim_carton -- \
  --prepared /tmp/carton-inputs --output /tmp/carton-variants \
  --layouts baseline object_x_plus combined_a --recipes nominal joint_7 time_130
```

可选 `--source-dir` 复用经过固定哈希校验的原始记录。保留准备目录及外部来源目录，
派生部署引用其中的原型、标定和 typed evidence。新增连接段、refinement、空间／时间
变形及实际执行分别保留血缘；原始状态、提交命令和 FK 派生 TCP 不互换字段。

## 完整矩阵与展示

```bash
embodichain run-example expansion/trajectory/real2sim_carton -- \
  --prepared /tmp/carton-inputs --output /tmp/carton-full \
  --full-matrix --jobs 3

python -m http.server 8000 --bind 127.0.0.1 --directory /tmp/carton-full
# 打开 http://127.0.0.1:8000/replay/
```

矩阵固定为 **13 布局 × 6 配方＝78 个候选**。布局覆盖物体 X/Y ±3 cm、yaw ±15°、
目标 X ±3 cm、yaw ±15°及两个组合。配方为 nominal、joint_7、joint_11、task_space、
time_115、time_130。`--jobs` 运行至多三个独立 B=1 进程；正式宿主仍为 B=1。
每个候选保留通过或失败结果；`report_complete` 表示完整报告，`all_passed` 表示全通过。
即使有拒绝，也先保存完整报告并导出回放，最终返回非零状态。

`--no-video` 可跳过讲解视频。三种修正对照为原样目标、刚性锚点目标及完整 refinement；
前两者标记为未执行，物理回放仅使用已记录的实际状态。失败部分轨迹不计为成功。
回放无需重新执行物理仿真，视频按显式源／执行时间映射同步。

| 产物 | 内容 |
| --- | --- |
| `prepared/qualification.json`, `scene.lock.json` | 来源、标定、FK、当前几何和时钟证据 |
| `prepared/measured/prototypes/` | 从原始记录重新构建的三个 JSON/NPZ 原型 |
| `deployment/task.yaml`, `profile.yaml` | 可独立交给 run-task 的派生部署 |
| `dataset/`, `collection-report.json` | 实际动作、T+1 观测、物体状态、附件和回执 |
| `matrix/acceptance-report.json` | 全部 78 个候选及失败原因 |
| `replay/`, `video/` | 只读交互回放、轨迹图和约 150 秒讲解视频 |

## 素材、依赖与边界

真实来源为 `RAIL+d027f2ae+2023-11-20-11h-50m-49s`。包装盒采用多视角与夹爪开口
约束下估计的刚性代理，约 96.6 × 47.2 × 36.6 mm。质量、摩擦和阻尼是显式仿真
假设。机器人复用官方原始 Panda＋Robotiq embodiment；安装和 TCP 必须通过 FK 与
物理几何身份检查。人工阶段、坐标及相机证据见 `annotations/` 和 `source.lock.json`。

素材在 [Hugging Face](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/demo/real2sim_carton)
版本化发布；`resources.lock.json` 固定分发 revision、SHA-256 和 manifest 哈希。
DROID 原始和派生真实输入保留 CC BY 4.0。已发布 v1 中的历史合成内容保持字节不变，
不参与正式任务、候选统计或来源重建；`inputs/manifest.json` 保留原分发清单。
内部 `_package_v1_resources.py` 仅用于维护该历史分发，不是任务执行入口。

需要兼容的 DexSim、Panda/Pink、同步 LeRobot、h5py、OpenCV、trimesh、pycollada、
Matplotlib 和 Pillow。采集使用 NoRender，无 RGB 记录。视频另需 imageio/ffmpeg、
Hybrid/Vulkan 和 Noto Sans CJK 字体。所有实际状态与回执均来自本次运行。

本案例限定为真实运动输入在声明物理代理中的仿真迁移；不宣称真实动力学或实机验证。
保持 B=1、physics: default、30 Hz 控制／240 Hz 物理步进，不包含 RL 或场景重建。
