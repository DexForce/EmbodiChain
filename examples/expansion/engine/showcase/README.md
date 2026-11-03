# Expansion Engine · 全能力验收与交互展示

同一个入口展示两类专家运动来源：**UR5 方块抓放的规划生成专家轨迹**，以及
**Franka 纸盒搬运的真实示范驱动专家轨迹**。两者都通过 Task Program、Atomic Skills
和 Expansion Engine 执行、验收及提交；Real2Sim 不是第二套任务执行系统。

```bash
embodichain run-example expansion/engine/showcase -- \
  --output outputs/expansion_showcase --open
```

默认 `--mode full`。首次运行通过已有资产缓存获取锁定的 DROID 输入，重新执行来源
资格检查，生成场景／安装位置联合 atlas，再顺序执行各个独立 World 作业。
在已安装兼容 DexSim、Newton、LeRobot、Panda/Pink、imageio/ffmpeg 等现有依赖的机器上运行。
无需新增服务或前端依赖。实际耗时由规划、原生渲染、验证和存储速度决定。

## 运行与恢复

```bash
# 少量样本，未执行能力仍明确标记
embodichain run-example expansion/engine/showcase -- \
  --output outputs/expansion_showcase_quick --mode quick --open

# 恢复原作业；已确认数据不会重新执行
embodichain run-example expansion/engine/showcase -- \
  --output outputs/expansion_showcase --resume

# 仅从冻结的展示产物重建页面，不加载仿真
embodichain run-example expansion/engine/showcase -- \
  --output outputs/expansion_showcase --reuse --open

# 调试指定案例；其余案例不会被当成已验收
embodichain run-example expansion/engine/showcase -- \
  --output outputs/expansion_showcase --resume --case ur5-scene
```

输出目录必须为空，或者显式使用 `--resume`／`--reuse`。恢复会保留已冻结的配置和预算修订；
需要更多尝试时，显式指定 `--resume --case CASE_ID --extra-attempts 8`，不会改变扩增分布。`index.html` 可直接在本地浏览器
打开；页面和依赖均在输出目录，不使用 CDN。保留整个目录即可分享交互回放。
原始训练数据位于每个作业自己的导出目录；预览 MP4 和三维采样不是训练数据替代品。

单次运行总预算 Full 默认 4 小时、Quick 30 分钟。物理采集预算为单项 10 分钟、
组合及批量矩阵 20 分钟；有重渲染／导出的作业额外预留最多 15 分钟后处理，
其他作业留 5 分钟用于清理。所有阶段仍受剩余总预算约束。常规容量 B=8，Quick B=2；单项目标 2 个确认提交，
联合场景目标 4 个，组合目标 8 个。纯状态批量矩阵 B=1、8、32 分别采集 B+1 个，覆盖尾批。
每个 World 独立进程顺序运行。作业配方、预算、来源与结果均保存在输出目录中。

## 能力与证据

| 领域 | 展示内容 | 证据 |
|---|---|---|
| 来源 | 原生规划、固定 Real2Sim 原型 | Task Program 调用、来源附件、保护阶段、时间映射 |
| 场景 | 方块布局与 UR5 安装位置的联合候选 | atlas 身份、实际状态、重新规划和实测验收 |
| 原生运动 | 抓取分支、joint residual、via points | 扩增记录、实测关节曲线及 FK 推导 TCP |
| 原型运动 | nominal、joint_7、joint_11、task_space、time_115、time_130 | 原型与实际执行的映射、保护阶段与接触验证 |
| 物理 | 质量／惯量、摩擦、CoM、关节刚度和阻尼 | 采样值与读回值、参数漂移检查、恢复及独立动力学探针 |
| 视觉 | 材质颜色／PBR、灯光、相机位置／焦距 | 原生 RGB-D、相机标定、视觉 profile |
| 重渲染 | 材质、灯光、相机、组合视觉子样本 | 完整物理父轨迹、零物理步进、相同动作与时间 |
| 并行与恢复 | B=1/8/32、尾批、中断、丢失及重复回执、重渲染故障 | journal、冻结产物、实际确认回执 |
| 数据 | LeRobot、相对路径附件、来源分组 | 数据集校验、迁移后数值／媒体核对、分组划分清单 |

Default 覆盖双来源完整展示；Newton 覆盖原生基准、物理、视觉和组合案例。
Newton 原型来源作为资格诊断，不放宽成功标准。未支持的组合、独立 World 资产拓扑
边界和未接入算法单独列出，不用其他来源或算法静默替代。

所有条件固定后执行；不会为了通过而自动收窄参数分布、替换来源或降低验收阈值。
未通过项保留实际原因。质量／摩擦受控响应探针和故障作业独立保存，不给主数据集增加
专家成功次数。诊断故障仅在该子进程内装饰 writer／renderer，不改变正常公共作业配置。

## 页面如何阅读

- **对照实验室**：选择能力及 episode，旋转三维视图，拖动时间轴，对照 RGB、深度和参数。
- **能力矩阵**：明确区分已通过、未资格验证、失败和未运行。
- **来源与配方**：显示来源、recipe、attempt、物理 episode、视觉子样本及 commit。
  Real2Sim 原始视频通过执行→原型→采集→编码时钟映射同步；入口连接没有虚构源帧。
- **验收与性能**：查看检查结果、阶段耗时和全部尝试。嵌套阶段耗时不能直接求和。显存和进程树 RSS 每秒采样，记录为采样峰值；
  原生 host 的 OS RSS 峰值单独保留。未取得的测量不填充为零。

三维机器人动画来自实际 qpos，动态实体来自实际场景状态，TCP 是由实测关节计算的 FK。
原生／FK 坐标需一致才导出模型。回放最多保留 241 个实测采样点，不插值伪造执行结果；
原始数值和媒体完整保留。材质／灯光最终效果以原生 RGB-D 为准，浏览器三维用于空间解释。

视觉 profile 相对于规范 World 基线定义，不隐式继承父采集 profile。
“规范基线 · 视觉单项对照”专门固定父样本为规范外观，只改变一个视觉领域；
“原生视觉采集与观测重渲染”另行展示采集外观及多种再观察配置。

视觉父子共用物理时钟；独立物理执行可以按真实时间或 segment 阶段归一化对比。
归一化只影响展示，不修改原数据。深度预览采用固定 0–2 m 色标，原始浮点深度未修改。

## 产物与完成判定

- `cases.json`、`configs/`：完整案例清单和冻结的实际作业设置。
- `jobs/`：Engine 原始证据、journal、回执、媒体和数据集。
- `results/`、`logs/`：逐操作状态和原始日志。
- `replay/`、`media/`：按需加载的三维与媒体展示资源。
- `diagnostics/`：受控实验、故障恢复和数据迁移／来源划分证据。
- `acceptance.json`：机器可读检查结果；`report_complete` 与 `acceptance_passed` 分开。
- `showcase.json`、`index.html`：可脱离仿真环境查看的展示索引和页面。

成功采集不自动证明扩增差异；验收会核对实际参数和数值／媒体契约。迁移检查移动真实
导出数据集、临时隐藏原作业，再验证数据、媒体、源附件和标定，逐 episode 对比全部
动作／状态及首、中、末 RGB 帧，最后恢复目录。来源分组不得把同源后代拆到不同集合。

完整模式只有全部必验项通过才返回 0；未完成／未通过返回 2。资格诊断结果仍完整保留，
不会用“报告完整”宣称所有组合成功。Quick 和指定案例运行不能作为完整能力资格证据。

## 已记录的验收结果

[qualification.json](qualification.json) 保存本次实现的验收摘要：Full 共 33 个案例、
40 项检查，所有必验项通过，确认 120 条物理轨迹并生成 102 个视觉子样本。
Newton Real2Sim 诊断在 30 Hz 控制／240 Hz 物理时钟下实际执行两次，均未通过
物理效果／分段验收，因此不产生成功回放，也不代表已资格验证。未放宽原有阈值。

B=1、8、32 分别确认 2、9、33 条纯状态轨迹，33 个采样配方身份一致。
故障检查覆盖同配方中断恢复、B=1→8、写入后丢失回执、重复回执、渲染失败隔离，
以及一行拒绝、另一行继续的实际执行。迁移后的三个导出数据集均通过数值、媒体、
标定、来源附件和来源分组检查。Quick 的四个选中案例通过，生成 20 个预览条目；
其他必验项仍标记未运行，因此 Quick 不返回完整资格通过。

本次保留原始帧、冻结副本、导出包和诊断历史的 Full 目录约 106 GiB，Quick 约 14 GiB。
这是实际运行的存储量，不是普遍上限。计时和资源测量范围见摘要，不将单次测量外推为
加速倍数或总体成功率。
