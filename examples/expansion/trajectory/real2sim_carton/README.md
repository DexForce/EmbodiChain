# Real2Sim 双来源演示：包装盒定向理货

同一原子技能链路处理两种来源：DROID 公开真实运动，以及在同一机器人、盒体模型中
独立执行生成的合成参考。机器人将闭合包装盒抓起、搬运，并按指定位置与朝向放到
标记区域。原来的 [UR5 立方体示例](../real2sim/README.md) 保留为回归基准。

```text
原始 HDF5 + 视频 / 独立合成参考
  → 明确标定、人工阶段、来源资格检查
  → Pick / MoveHeldObject / Place 固定 revision 原型
  → 对象参考系重定向 → affordance 修正 → IK / 运动 / 采样碰撞检查
  → 同网格与映射感知时间扩增
  → run-task / Gym step → 实测验收 → 同步 LeRobot + 附件 + confirmed receipt
  → 离线交互回放 / 150 秒字幕讲解视频
```

## 冻结的真实来源与代理边界

选定 episode 为 `RAIL+d027f2ae+2023-11-20-11h-50m-49s`，来自
[DROID 官方原始发布](https://droid-dataset.github.io/droid/the-droid-dataset.html)
`droid_raw/1.0.1/RAIL/success/2023-11-20/`。只获取该记录的 HDF5、metadata 和三个
单视角 MP4。原始五个文件的来源、字节数、SHA-256 保留在
[source.lock.json](source.lock.json)。已核验的副本与标注、原型、物理代理一起发布于
[Hugging Face demo/real2sim_carton](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/demo/real2sim_carton)，
版本化 ZIP 约 14.5 MB，分发 revision 与校验值见 [resources.lock.json](resources.lock.json)。
DROID 原始数据及其派生真实输入保留 CC BY 4.0，独立合成数据保留 Apache-2.0；
资源包中的 `ATTRIBUTION.md` 和逐文件清单说明具体来源与修改。

腕部视角可确认白蓝色 STAPLES 5000 COUNT 闭合盒体。没有确认 SKU，也未使用 YCB
模型代替。盒体是两幅已标定外部视角和观测夹爪开口约束下估计的刚性长方体：
约 **96.6 × 47.2 × 36.6 mm**。质量 50 g、摩擦与阻尼是显式仿真假设。

使用原始 Panda 七关节运动链和 Robotiq 2F-85；18 mm 同轴安装件和 150 mm TCP
偏移是条件标定参数。公共 `FrankaCfg` 统一选择
`arm_model: panda` 与 `gripper: robotiq_2f85`；embodiment 提供安装与 TCP，
来源资格检查依据型号、工具、实际关节布局、固定几何身份和 FK，不依赖类名。
Panda 状态到源 flange FK 最大位置偏差约 0.00018 mm；
相机一的人工夹指投影拟合 RMS 约 4.58 px，独立留出点约 8.78 px；盒体双视角
边界框拟合 RMS 约 3.64 px。这些投影残差不能解释为真实物体毫米级测量认证。
具体依据、源索引和条件假设在 [annotations](annotations/) 及生成的
`qualification.json` 中。

相机标定辅助数据来自 DROID 作者的
[补充标注库](https://huggingface.co/KarlP/droid/tree/bcb840c3b496533e0adf548a54b51f2f00057837)，
固定 revision `bcb840c3b496533e0adf548a54b51f2f00057837`。外部相机序号和原始 metadata
的视角字段命名不能混用。原始视频以 60 fps 编码，而采集时钟约为 14 Hz；回放显式
组合源样本、规范化、计划、实际执行与视频时间映射。最后一条机器人状态没有对应
视频帧，采用的源片段止于更早的样本。

| 原型 | 原始区间，左闭右开 | 阶段 |
| --- | --- | --- |
| PickUp | `[25, 66)` | approach `[25,52)` / close `[52,60)` / lift `[60,66)` |
| MoveHeldObject | `[65, 75)` | transport |
| Place | `[74, 94)` | approach `[74,83)` / release `[83,91)` / retract `[91,94)` |

真实原型保持 `measured` 来源；源 Cartesian pose 标记 `fk_derived`。原始实测 arm
qvel 保留在原始与 typed evidence 中，不填充成完整关节观测。动作提交和状态分别
保留时钟；夹爪实测开口经联动机构映射，不把全闭命令当作实测手指位置。

结论限定为：**公开真实运动输入在已声明、估计的物理代理中的仿真迁移**。
场景没有验证真实动力学或实机部署。两条来源分别统计。

## 一键运行

需要兼容的 DexSim、原有 IK/轨迹依赖、同步 LeRobot、`h5py`、`trimesh`、
`pycollada`、Matplotlib、Pillow、imageio/ffmpeg，以及生成视频所用的 Hybrid/Vulkan 渲染能力。
字幕字体使用 Noto Sans CJK。采集不要求 RGB 或异步录像。

```bash
embodichain run-example expansion/trajectory/real2sim_carton/run.py -- \
  --output /tmp/carton-real2sim --jobs 3

# 回放页完全使用预计算产物；绑定本机地址。
python -m http.server 8000 --bind 127.0.0.1 --directory /tmp/carton-real2sim
# 打开 http://127.0.0.1:8000/replay/
```

也可以先通过统一资产入口下载，再运行示例：

```bash
embodichain data download --name Real2SimCarton
```

缓存默认位于 `~/.cache/embodichain_data/extract/Real2SimCartonV1/`，可通过
`EMBODICHAIN_DATA_ROOT` 在启动前指定。下载复用公共锁、ZIP 校验与解压机制，另核验
SHA-256 和每个文件的哈希。`get_data_path("Real2SimCarton/raw")`、
`get_data_path("Real2SimCarton/inputs")` 和 `get_data_path("Real2SimCarton/scene")`
分别解析原始记录、冻结输入和场景资源；运行产物仍写入 `--output`，不写入资源缓存。

输出目录必须不存在，已有产物不会被替换。`--source-dir` 可复用经过五个固定哈希
校验的原始文件工作目录；不存在的文件从公共 `Real2SimCarton` 缓存复制，已有文件
必须匹配原始哈希。`--no-video` 显式跳过视频，不能算视频阶段完成。若矩阵有失败，
入口仍导出完整报告、回放与视频，最终返回非零状态。资格或名义执行失败会停止后续
阶段并保留日志，不能用合成结果代替真实分支。

也可分阶段执行；每个入口均支持 `--help`：

```bash
python -m examples.expansion.trajectory.real2sim_carton.prepare acquire \
  --source-dir /tmp/carton-source
python -m examples.expansion.trajectory.real2sim_carton.prepare prepare \
  --source-dir /tmp/carton-source --output /tmp/carton-prepared
python -m examples.expansion.trajectory.real2sim_carton.capture \
  --gym-config /tmp/carton-prepared/task.native.yaml --output-dir /tmp/carton-native
python -m examples.expansion.trajectory.real2sim_carton.prepare_synthetic \
  --prepared /tmp/carton-prepared --capture /tmp/carton-native --output /tmp/carton-synthetic
python -m examples.expansion.trajectory.real2sim_carton.matrix \
  --prepared /tmp/carton-prepared --synthetic /tmp/carton-synthetic \
  --output /tmp/carton-matrix --jobs 3
python -m examples.expansion.trajectory.real2sim_carton.export \
  --prepared /tmp/carton-prepared --synthetic /tmp/carton-synthetic \
  --matrix /tmp/carton-matrix/acceptance-report.json --source-dir /tmp/carton-source \
  --output /tmp/carton-replay
python -m examples.expansion.trajectory.real2sim_carton.video \
  --prepared /tmp/carton-prepared --replay /tmp/carton-replay --output /tmp/carton-video
```

源码运行需从仓库根目录启动，或将 examples 的父目录加入 PYTHONPATH。
`run-example` 与一键入口同时支持安装后的 examples 树和非仓库工作目录。

## 配置、原型与修正

生成配置分别拥有物理环境、embodiment、执行策略、语义集成和 program；来源绑定只在
runtime services 中以 program fingerprint + compiled occurrence 定位。`program.yaml`
不包含来源路径、机器人信息或可执行导入。场景只有一个动态盒体、一个静态盒状
工作台与固定地面，B=1，`physics: default`，30 Hz 控制 / 240 Hz 物理步进。

原型在扩增前冻结。六个已验收输入 bundle 发布在 `Real2SimCarton/inputs`；仓库
[inputs](inputs/) 只保留检查清单和获取说明，避免再次存储轨迹数组。资源包还包含
原始 HDF5、三个视频、匹配 typed evidence，以及干净合成参考与其实际 T+1 观测。
冻结副本用于检查来源 revision；正式运行仍重新规范化和核验实际场景。
扩增不调用生成器替代重定向。

合成参考在同一物理模型中独立执行；原生采样先通过关节运动检查，再保存设计参考、
实际动作、T+1 观测与物体姿态。capture 使用独立的慢速名义生成策略；后续来源执行
继续用声明的 Panda 每关节速度、加速度限值。注入误差为平移 `[+6,-4,+3] mm`、
世界 yaw `+3°` 和自由阶段最多 3 mm 的缓慢漂移。干净轨迹不提供给修正器。

盒体的合法对向抓面是对象 ±Y 面，长边保留 10 mm 边界，抓取高度相对中心 −5 mm。
选择离输入抓取锚点最近的合法侧面姿态，再由已有开口、标定、IK 与几何检查接纳。
真实来源修正预算为 70 mm / 60°，合成来源为 20 mm / 10°；预算包含最终 FK 误差。
真实输入中较大的姿态修正用于使斜向示范适应声明的规则侧面代理，并非毫米级标定
误差认证。每次调用保留实际修正量与原始目标，回放页可直接查看。

三种对照来自同一输入：原样目标、仅刚性锚点对齐、完整 refinement。前两种是标明
**未执行**的目标路径诊断；只有完整 refinement 的已记录结果被展示为物理执行。
回放不将未执行对照或失败候选渲染为成功。失败中止前的部分实测保留为部分轨迹，
没有完整回执或逐步源视频映射时明确标注。

## 固定验收矩阵

每个来源 13 布局 × 6 配方，共 **156 个候选**，每个候选独立 B=1、一条尝试。
`--jobs` 仅运行最多三个相互独立的 B=1 进程，不是 B>1 宿主。布局包括基准、
对象 X/Y ±3 cm、对象 yaw ±15°、目标 X ±3 cm、目标 yaw ±15°，以及两个固定组合。
配方为名义轨迹、关节种子 7/11（尺度 0.002，去重容差 0.0005）、任务空间 5 mm、
自由阶段时长 ×1.15/×1.30。

接触、持有、夹爪与必要导数邻域受保护。真实 Place 的五样本短撤离没有可编辑时间
区间，profile 仅允许其 approach 参与时间扩增；该撤离段保留原网格。合成来源较长
的 retract 仍允许时间扩增。原型字节保持不变，权限参与候选身份。

验收同时要求 IK/FK 位置 ≤2 mm、角度 ≤2°，实测抬升 ≥3 cm，搬运持有关系有效，
最终中心 ≤5 mm、有向 yaw ≤3°、完整支撑足迹以及稳定窗口漂移 ≤3 mm。位置和方向
相对语义目标计算，包含求解器误差。每条成功还必须通过实际碰撞检查、T+1 与动作
附件完整性、源/时间映射校验和已有 recording journal / confirmed receipt。

`acceptance-report.json` 分别报告两个来源的全部候选、失败阶段和原因。
`report_complete` 只表示完整报告 156 项，`all_passed` 才表示全部通过。
固定矩形采样域不是整体可行域保证；每个布局须通过独立检查。失败保留，不删除布局。

本次固定矩阵实测结果：

| 来源 | 已报告候选 | 通过并确认提交 | 失败 | 通过条目的最大中心 / yaw 误差 |
| --- | ---: | ---: | ---: | --- |
| DROID 真实运动 | 78 | 72 | 6 | 1.193 mm / 0.733° |
| 独立合成参考 | 78 | 78 | 0 | 0.982 mm / 1.530° |

真实分支在目标 yaw +15° 的六种配方均在 transport 准备阶段被拒绝：Joint2
采样加速度约 8.09 rad/s²，超过声明的 7.5 rad/s²。失败发生在时间扩增前的名义
计划检查，因此两个慢速配方也不绕过该检查。保存的部分实测只覆盖中止前的 Pick。
本次是 **156 项完整报告、150 条确认成功**，不是矩阵全通过。机器可读摘要在
`acceptance_report.json`，完整逐条报告随运行产物保存。

## 输出与展示

| 产物 | 内容 |
| --- | --- |
| `prepared/qualification.json`, `scene.lock.json` | 原始数据、运动链、标定、几何与时钟资格证据 |
| `native-reference/` | 独立合成参考、设计输入、原生执行结果与实际 T+1 数组 |
| `synthetic/source.lock.json` | 合成父 revision、误差配方与未复制真实数组的血缘 |
| `matrix/acceptance-report.json` | 156 个候选的独立实测与持久化结论 |
| `matrix/<origin>/<layout>/<recipe>/dataset/` | 原有 LeRobot 身份、附件、journal、回执或失败证据 |
| `replay/index.html`, `recordings.json` | 只读 3D 回放、输入录像、时间轴、轨迹、失败分析和报告下载 |
| `replay/trajectory-comparison.png`, `.svg` | 两条来源按技能拆分的独立轨迹对比图 |
| `video/real2sim-carton-explained.mp4` | 150 秒字幕讲解，无音轨；只回放已保存状态 |
| `video/explanation.zh.srt`, `video-evidence.json` | 字幕、来源哈希、分镜与无新采集身份说明 |

回放依赖已固定的本地 Three.js 文件，不需要物理引擎重新运行。它支持来源、布局、
配方、修正方式、三相机切换，时间拖动、暂停、倍速、抓取/释放放大与 JSON 下载。
程序化的白蓝包装面、标记与 +X 箭头用于说明几何方向，不是真实扫描纹理。
[assets.lock.json](assets.lock.json) 和 [机器人模型说明](../../../../extensions/embodichain_robots/franka/README.md)
记录资产、尺度、惯量来源与许可。未使用 YCB 或 Poly Haven 素材。

`Real2SimCarton/scene` 中的 GLB 是与盒体尺寸和物体局部坐标一致的显示代理；
物理参数由同目录 `environment.yaml` 与 `physical.json` 声明。原始 Panda 与
Robotiq 模型继续引用已有模型资产，不在 demo 包中重复发布。`preview.mp4` 和轨迹
对比图位于 Hugging Face 资源目录，完整生成的数据集和确认回执仍属于各次运行。

维护者可在源码 checkout 中使用 `package_resources.py`，从经过核验的原始记录、
prepared、native capture 和冻结 inputs 构建排序、固定时间戳的 ZIP。该脚本只生成
资源包与哈希索引，不上传、不执行仿真。发布新内容时使用新资源版本与缓存前缀，
不得覆盖已经固定哈希的 v1。

生产测试覆盖来源映射/缺失/损坏、独立 FK、联动首次接触、合法抓取区、方向验收、
修正预算、源无回退、完整计划/时间映射、持有状态与持久化边界。示例通过实际执行、
产物审计和浏览器检查验证，不将示例脚本作为生产测试 API。
