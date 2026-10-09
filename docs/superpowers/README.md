# 数据多样性分析系统：总体设计与实现现状

更新日期：2026-09-17。本文是本项目设计与交付状态的总入口，汇总既有决策，并按当前代码核对实现边界。

**当前定位：仿真扩增优先、兼容历史数据的本地分析工作台。** 已交付可体验的分析与回放原型，完整 P0/P1 尚未全部完成。用户已决定暂不实现覆盖缺口驱动补采及模型归因等下一阶段功能；本次只整理文档。

## 1. 文档导航

| 文档 | 用途 |
|---|---|
| [TODO 与推进顺序](plans/2026-09-17-data-diversity-todo.md) | 待办、依赖、验收方式及暂缓状态 |
| [P0/P1 验收标准](specs/2026-09-17-data-diversity-p0-p1-acceptance.md) | 完整目标基线；不是完成声明 |
| [首版实施计划](plans/2026-09-17-data-diversity-preview.md) | 首次可体验版本的实现范围 |
| [第二轮实施计划](plans/2026-09-17-data-diversity-analysis-v2.md) | 六类维度与阶段轨迹比较的实现范围 |
| [首版验收报告](reports/2026-09-17-data-diversity-preview.md) | 真实任务、账本、离线回放与首版性能证据 |
| [第二轮验收报告](reports/2026-09-17-data-diversity-analysis-v2.md) | 联合分布下钻、阶段曲线及回归证据 |

历史报告保留当时的测量结果；最新完成边界看本文与 TODO，详细完整标准仍以验收标准为准。

## 2. 目标与使用流程

系统需要回答三层问题：

1. **数据层**：生成了什么，哪些条件真正执行并入库，哪些维度未知，分布是否集中或重复？
2. **仿真验证层**：某个数据切片对应哪些场景与轨迹，任务执行是否满足物理检查，失败发生在哪个阶段？
3. **模型层（后续）**：模型在哪些条件下失败，这些条件在训练数据中如何分布，补充数据后性能是否改善？

当前操作流程是：接入/采集 → 查看完整性与来源 → 筛选分布 → 联合单元格下钻 → 查看场景和阶段轨迹 → 导出切片或比较快照。

未来才扩展到：定义覆盖目标 → 发现缺口 → 生成并执行扩增配置 → 比较有效新增数据 → 关联模型评估。数据相关性只能形成归因假设；因果结论需要受控补采、训练与评估实验。

## 3. 技术方案与模块边界

采用 **Gradio + EmbodiChain Viser + Matplotlib + Polars + SQLite**，复用已有生态。独立分析入口使用最小 `analysis` extra，当前声明 Gradio `>=6.17.3,<6.18` 和 Matplotlib；NumPy、Polars、Viser 复用项目依赖。

| 层次 | 职责与当前实现 |
|---|---|
| 接入层 | 任务专用仿真采集器、分析 JSONL 导入、历史 LeRobot 元数据适配；真机在线接入尚未实现 |
| 数据契约层 | episode 身份、维度值及来源、状态、父记录引用、阶段、指标、产物与 provenance |
| 目录层 | SQLite WAL 保存当前记录、事件账本、不可变元数据快照；媒体和数组留在文件系统 |
| 分析层 | 维度定义与完整度、筛选、分布、精确单元格成员、阶段曲线与几何距离 |
| 工作台 | Gradio 管理查询和会话状态，Matplotlib 绘图，Dataframe 承载列表与联合单元格选择 |
| 回放层 | 复用 Viser 场景协议与后端，离线载入场景、轨迹和 RGB；录制时间同步状态与相机 |
| 后续服务 | 覆盖目标、扩增配置执行、模型评估关联，均未作为当前服务交付 |

GenSim Gradio 页面提供分析入口；分析领域代码独立位于 `embodichain/data_analysis/`。独立统计与录制回放不要求启动 GenSim、Blender、DexSim 或物理仿真。`embodichain.lab` 的惰性导入用于维持这条边界。

当前 SQLite 存储 JSON 记录，分析函数加载相应记录后做 Python/Polars 聚合。原始方案中的 Parquet 惰性分析、分页和增量特征缓存属于后续规模化设计，不能视为已经落地。当前也没有独立作业调度服务、外部数据库或新建前端工程。

主要代码责任如下，路径均相对仓库根目录：

| 模块 | 责任 |
|---|---|
| `embodichain/data_analysis/schema.py` | schema v1、身份与测量结构、记录和产物校验 |
| `catalog.py` | 状态转换、幂等写入、事件、JSONL 导入和元数据快照 |
| `dimensions.py` / `statistics.py` | 维度语义、完整度、筛选、分箱、聚合和显式目标网格覆盖函数 |
| `importers.py` | 历史 LeRobot 元数据只读适配 |
| `collection.py` / `recording.py` | 既有任务采集、状态记录、数值产物、物理检查和目录发布 |
| `trajectory.py` | 同名阶段匹配、时间/几何分离、比较曲线 |
| `replay.py` | Viser 会话、录制时间轴、场景/RGB/TCP 展示 |
| `ui.py` / `_i18n.py` | 工作台、切片导出、会话级中英文显示；当前默认英文 |
| `cli.py` | `embodichain analyze-data` 命令入口 |

## 4. 数据模型与统计口径

### 4.1 记录与产物

当前逻辑主键为 `episode_id`，同时关联 `run_id`、`candidate_id`、`attempt_id`、`task_id` 和 `robot_id`。记录还包含 `seed`、`parents`、`segments`、`dimensions`、`metrics`、`artifacts` 和 `provenance`。

稳定 scene case、完整 commit 身份、多父片段变换血缘及依赖内容哈希是完整目标的一部分；当前字段和记录不足以证明这些全部闭环。

- SQLite 的 `records` 保存当前记录，`events` 保存变化事件，`snapshots` / `snapshot_records` 保存不可变元数据版本。
- 相同记录重复写入不改变结果；身份冲突、同状态不同内容和不允许的终态变更被拒绝。
- `scene.json` 保存可视化场景；`trajectory.npz` 保存时间戳、关节、TCP 和场景状态；可选 `camera.npz` 保存 RGB 与相机时间戳。数值 NPZ 不使用 pickle。
- 当前快照冻结元数据，没有冻结外部媒体文件内容；导出当前切片也不等于自动生成了完整、带内容哈希的数据发布版本。

### 4.2 来源和可用性

每个测量值保留 `value / source / unit / frame / scope`，未知值附带原因。来源可为 measured、configured、annotated、derived、estimated、unknown。

分析区分 known、unknown、missing、invalid：显式未知、字段缺失、类型或单位不兼容不合并为数值 0。结构/版本错误由记录校验处理，部分逐维语义不兼容由分析层标记；尚未做到完整语义契约都在入库时强制校验。

配置范围、实际采样值、规划轨迹、执行轨迹和最终入库记录不能互相替代。程序报告成功与物理测量通过分开保存。

### 4.3 六类维度

| 类别 | 当前字段与单位 | 当前边界 |
|---|---|---|
| 物体资产 | `asset`，类别 | 当前验收为两种立方体尺寸条件，不能代表广泛语义类别 |
| 姿态 | `pose_x/y/z`，m；`yaw`，rad | 初始位置与偏航角；不是完整 SO(3) 姿态覆盖 |
| Affordance | `affordance`、`approach`，类别 | 任务标注；接近方式标签不等于测得的物理方向 |
| 轨迹 | `trajectory_family`；`duration_s`，s | 已有阶段曲线/几何距离；尚未自动分族或去重 |
| 材质 | `material`，类别 | 配置条件，尚无统一物理材质属性向量 |
| 光照 | `light`，renderer_intensity | 当前渲染配置强度，不是 lux 等物理测量 |

六类共 11 个标量字段，维度定义版本为 1。完整度为兼容已知记录数 / 当前切片记录数；空集合不产生虚假的完整率。

### 4.4 分布、覆盖与轨迹

- 页面默认分析 committed 记录，可切换状态。生命周期总览统计整个目录；当前切片的计数、图表、列表和导出使用相同已应用记录集合，两者口径明确区分。
- 联合视图是资产 × 初始 X 分箱、Affordance × approach、材质 × 光照。每格保留精确 episode ID；下钻作用于 episode，尚无独立 segment 切片浏览。
- 数值区间筛选上下界包含端点；联合分箱左闭右开，最后一箱右闭。分箱边界保存在导出中。当前位置自动分箱基于同状态参考集合，数据集变化后可能重新计算，并非版本化覆盖目标。
- **观测分布不等于覆盖率**。只有明确合法目标单元集合时才有覆盖分母；底层已有覆盖函数，目标配置 UI 与缺口闭环尚未实现。
- 阶段匹配使用名称和出现次序。曲线展示归一化阶段时间，速度和时长仍从原始时间戳计算。弧长重采样后的平均空间距离描述世界坐标下的路径差异，不是旋转/平移不变距离、跟踪误差或聚类结果。
- 关节曲线使用原始列索引，拒绝不同机器人 ID 或不同维数；仍缺少关节名称、单位和顺序签名的完整兼容性检查。
- Viser 内部的状态与相机遵循录制时间；分析阶段曲线尚未与 Viser seek 联动。

## 5. 已实现与验证范围

| 能力 | 当前状态 |
|---|---|
| 数据事实基础 | 已有版本化结构、来源、未知值、状态账本、冲突拒绝、JSONL 幂等导入与快照 |
| 六类分析 | 已有完整度/来源表、类别多选、数值范围、可用性筛选、单维与三类联合分布 |
| 下钻与导出 | 已有单元格选择、精确 episode 清单、分箱与筛选历史导出；刷新恢复切片尚未实现 |
| 轨迹与回放 | 已有场景/RGB/TCP 回放、双轨迹叠加、阶段 TCP/速度/关节曲线 |
| 版本比较 | 已有 committed 数量、ID 增减和材质分布比较；全维度/覆盖/近重复比较尚未实现 |
| 历史数据 | 已有 LeRobot 元数据适配；不能据此声称历史媒体或真机回放已验收 |
| 中英文显示 | 当前代码提供会话级语言切换，显示语言不改变稳定筛选值及导出记录 |

已有真实仿真验收使用 `repeated_pick_place` 的 Franka 部署：12 条完整 episode、36 个任务阶段、2,892 个记录样本。全部 12 条通过声明的抬升和最终落点检查；这不等于每个中间放置、接触稳定性都已验证。

第二轮报告记录 295 项相关测试通过及浏览器下钻/导出/阶段曲线验收。这是该次交付的历史证据，本次文档整理没有重新运行功能测试，也不把它作为此后所有代码改动的测试计数。

首版元数据基准为 10,000 episode / 100,000 segment，30 次查询 p95 0.379 s、建库和查询峰值 RSS 464.5 MiB。它不包括第二轮新增分析、页面延迟和媒体解码；完整目标的分页、增量提取及全部查询组性能仍未验收。

## 6. 后续模型归因的设计接口（未实现）

未来将评估结果关联到模型版本、评估协议、场景/任务条件和评估 episode；训练数据通过不可变快照及训练清单建立关联。评估 episode 不需要与训练 episode 一一相同，应以可比维度定义构造训练与评估切片。

首先展示每个切片的成功率、失败阶段、样本量和不确定性，并关联训练数据覆盖。随后通过固定评估集和受控训练对照验证数据干预效果。训练曝光量、混合采样权重、模型与评估版本都需要记录，否则不能将失败直接归因于数据缺口。

上述内容只保留为演进方向；本次不新增评估调度、训练调度、补数作业、多用户权限或集群服务。

## 7. 运行与交付位置

分支：`codex/data-diversity-p0-p1`，开发基线为 main `3224ac1`；worktree：`/home/dex/workspace/sources/EmbodiChain/.worktrees/data-diversity-p0-p1`。当前按 Draft PR 交付评审，尚未合并。

已有验收数据位于 `/home/dex/.cache/embodichain/data_analysis/repeated-pick-place-preview/`。本机预览地址为 `http://127.0.0.1:7865/`，可用以下命令重新启动：

```bash
cd /home/dex/workspace/sources/EmbodiChain/.worktrees/data-diversity-p0-p1
/tmp/embodichain-analysis-env/bin/python -m embodichain analyze-data serve \
  --catalog /home/dex/.cache/embodichain/data_analysis/repeated-pick-place-preview/catalog.sqlite \
  --port 7865
```

通用环境安装项目 `analysis` extra 后，可用 `python -m embodichain analyze-data`，支持 `serve`、`summary`、`import --format jsonl|lerobot` 和任务专用 `collect-preview`。临时 venv 和本机缓存路径不是部署要求；复现实测仍需对应数据产物。
