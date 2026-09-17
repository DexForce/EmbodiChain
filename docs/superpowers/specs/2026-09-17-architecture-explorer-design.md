# EmbodiChain Architecture Explorer — 第一版设计

状态：范围、数据契约和源码样例已整理；可交互前端预览已实现，自动提取器及 Sphinx
集成尚未实现。按用户“先实现一个版本到可以前端展示”的要求，本轮先使用固定提交
快照完成可视化；运行方式见 [前端 README](../../architecture/web/README.md)。

源码基线：`main@3224ac1ee28b6730b245d5cb69dc25a8d2d8dd94`，包版本 `0.2.4`。

## 目标与验收问题

借鉴参考视频的分层卡片、关系筛选、关联高亮和详情面板，在文档中解释
EmbodiChain 的模块职责及交互边界。读者应能够回答：

1. 这个模块负责什么，哪些职责由其他模块承担？
2. 谁构造或调用它，它持有哪些对象？
3. 关系适用于哪个分支、集成或阶段，证据在哪里？
4. 对应版本的源码和 API 文档在哪里？

页面展示的是经过整理的源码静态关系。连线不表示调用次数、时延、运行时轨迹，
也不自动表示严格的执行先后。邻接节点用于解释直接关系；未来的多跳分析只能
标为“潜在影响”，不能宣称完整影响范围。

最初交付为范围、数据契约、样例和实施计划。本轮增加独立静态前端预览；
不启动仿真、不修改运行时代码、不发布网站。

## 第一版功能边界

- 两个视图：全局架构、Task Program 集成。全局默认展示下面的 24 个节点。
- 工具栏：视图切换、搜索、关系类型筛选、缩放、适应画布、重置选择。
- 图中选中节点后，高亮当前视图和筛选条件下的直接关系，其他节点淡化。
- 详情面板：职责、边界、关系方向和条件、源码片段、提交、文档链接。
- 同一对节点可有多种关系；合并显示时保留各条关系的类型、条件和证据。
- 独立全屏入口和文档内嵌入口共用同一构建产物。
- 分享链接保存视图、选中节点、关系筛选和搜索词；不保存编辑后的图数据。
- 使用键盘可操作的节点列表作为画布的等价入口；选中状态不只依赖颜色。
- 文档入口包含可检索的文字摘要和模块列表，JS 不可用时仍可跳转说明页面。

暂缓：完整类图、全仓库调用图、运行时追踪、在线编辑、AI 问答、跨版本差异、
协同编辑，以及动态注册的完全自动解析。

## 全局视图的 24 个节点

下面是展示投影，不是新的主题索引。主题 ID 仍以 `agent_context/MAP.yaml` 为准；
无对应主题的节点可以使用空 `topic_ids`。展示分组不暗示严格依赖方向。

| 分组 | 节点 | 源码归属 |
|---|---|---|
| 入口与部署 | CLI | `embodichain/cli/main.py` |
| 入口与部署 | Official tasks | `embodichain_tasks/embodichain_tasks/__init__.py` |
| 入口与部署 | Component composition | `embodichain/lab/gym/utils/_component_composition.py` |
| 环境 | EmbodiedEnv | `embodichain/lab/gym/envs/embodied_env.py` |
| 环境 | Managers | `embodichain/lab/gym/envs/managers/__init__.py` |
| Task Program | TaskProgramCompiler | `embodichain/lab/task_program/compiler/program.py` |
| Task Program | SemanticCallCompiler | `embodichain/lab/task_program/compiler/lowering.py` |
| Task Program | SemanticCallExecutor | `embodichain/lab/task_program/runtime/executor.py` |
| Task Program | TaskProgramEnvironmentAdapter | `embodichain/lab/task_program/integrations/environment.py` |
| Task Program | TaskProgramDemoBridge | `embodichain/lab/gym/envs/task_program/bridge.py` |
| 运动能力 | AtomicActionEngine | `embodichain/lab/sim/atomic_actions/engine.py` |
| 运动能力 | MotionGenerator | `embodichain/lab/sim/motion/motion_generator.py` |
| 运动能力 | Planners | `embodichain/lab/sim/motion/planners/__init__.py` |
| 运动能力 | Solvers | `embodichain/lab/sim/motion/solvers/__init__.py` |
| 仿真 | SimulationManager | `embodichain/lab/sim/sim_manager.py` |
| 仿真 | Robot | `embodichain/lab/sim/objects/robot.py` |
| 仿真 | Sensors | `embodichain/lab/sim/sensors/__init__.py` |
| 仿真 | Physics backends | `embodichain/lab/sim/physics/__init__.py` |
| 仿真 | Browser visualization | `embodichain/lab/visualization/runtime.py` |
| 计算与应用 | Compute | `embodichain/compute/__init__.py` |
| 计算与应用 | RL training | `embodichain/learning/rl/utils/trainer.py` |
| 计算与应用 | Data pipeline | `embodichain/data_pipeline/engine/data.py` |
| 计算与应用 | Scene generation | `embodichain/gen_sim/scene_engine/pipeline/generate.py` |
| 计算与应用 | Devices | `embodichain/lab/devices/device.py` |

资产下载、Utils、各类具体机器人等可在详情中链接；第一版不声称总览穷尽所有包。
预览已覆盖上述节点，并收录有源码证据的语义关系及选定模块的静态导入关系。
关系数量表示本视图收录数量，不从目录邻接推导依赖，也不声称完整覆盖。

## Task Program 样例与已核对的边界

机器可读样例位于
[task-program.sample.json](../../architecture/task-program.sample.json)，
包含 16 个节点、25 条关系，分为程序定义、运行时组装、语义与原子执行、Gym 集成。

- `TaskProgramCompiler` 接收 `TaskProgramCfg`，经私有 helper 产生
  `CompiledTaskProgram`；它与执行时使用的 `SemanticCallCompiler` 是不同角色。
- `TaskProgramEnvironmentAdapter.compile()` 的直接编译路径只适用于未安装
  integration catalog 的情况；有 catalog 时使用 preflight 路径。
- `SimulationTaskProgramFactory` 构造引擎与运动生成器；运动生成器允许注入工厂。
- `SemanticCallExecutor` 从 integration 获得 engine，每个调用建立执行 session
  和 `ExecutionRunner`，并使用共享的 sink 与 clock。
- `TaskProgramDemoBridge` 接收 protocol；样例中的具体对象关系只描述生产 Gym
  组装。样例没有穷尽并行分支、恢复、验证器和记录行为。
- `BufferedGymCommandSink` 接受命令只表示复制到缓冲区，不能当成物理执行成功。
- `EmbodiedEnv` 继承 `BaseEnv`，后者在场景初始化时创建 `SimulationManager`。

样例故意保留构造、持有和调用之间的区别。例如 engine 持有运动生成器，
并不能据此自动增加一条“每次执行直接调用生成器”的边。

## 数据契约

[architecture.schema.json](../../architecture/architecture.schema.json)
使用 JSON Schema Draft 2020-12，`schema_version` 固定为 `1`。

| 对象 | 必备字段与语义 |
|---|---|
| 快照 | `kind`、`coverage`、`repository`、40 位 `revision`、`source_ref`、`package_version`、`topic_index`、`limitations` |
| 节点 | 稳定 `id`、`label`、`kind`、`topic_ids`、`summary`、`boundaries`、`evidence`、`documentation` |
| 关系 | 稳定 `id`、`source`、`target`、`relation`、`description`、`provenance`、`scope`、`evidence` |
| 证据 | 仓库相对 `path`、词法限定 `symbol`、1 起始闭区间行号、精确 `excerpt` |
| 文档 | 不含扩展名的 Sphinx `docname` 和展示 `label`；v1 不生成未经验证的符号锚点 |
| 视图 | `id`、`label`、`description`、显式 `node_ids`／`edge_ids`、展示 `groups` |

`source_ref` 仅用于说明，源码链接必须绑定 `revision`。所有证据均针对该提交，
不能混合多个提交的行号。`coverage` 区分本次样例与未来正式整理的发布快照；
两者都不代表完整调用图。`provenance` 区分源码核对和静态提取，不表示物理验证。
Python 模块级证据使用保留 symbol `<module>`，范围为整个文件；类和函数使用
词法限定名称，例如 `TaskProgramEnvironmentAdapter.create_bridge`。配置文件的
证据使用 `<document>` 表示整份文本，不能按 Python AST 解析。

关系方向规定如下：

| `relation` | source → target |
|---|---|
| `imports` | 导入方 → 被导入模块或符号；纯类型、条件导入写入 scope |
| `inherits` / `implements` | 子类或实现 → 基类或接口 |
| `constructs` | 构造方 → 被构造对象类型；分支条件写入 scope |
| `holds` | 持有引用的对象 → 被引用对象；不自动表示独占所有权 |
| `calls` | 调用方 → 被调用方；间接绑定必须说明具体集成 |
| `reads` / `writes` | 读写方 → 数据或缓冲对象；箭头表示访问关系而非时间顺序 |
| `produces` | 生产方 → 产物类型 |
| `configures` | 配置来源 → 被配置对象 |

JSON Schema 只负责结构约束。图校验还必须检查唯一 ID、端点存在、视图边的端点
包含在该视图内、分组无遗漏或重复、主题 ID 有效、证据行号及符号范围、精确源码
片段、文档源文件存在。任何未知字段或失效引用都应给出可定位错误。

## 数据生成与维护

1. 从现有 MAP 读取主题和源码归属，禁止把 `related_topics` 作为依赖边。
2. 对显式选中的源码使用 AST，解析符号定义及少量明确的导入／继承事实。
   不 import `embodichain`，不要求 GPU、DexSim 或运行中的环境。
3. 人工维护职责、边界、关键组装／执行关系及唯一源码定位片段。重复片段必须
   结合词法 symbol 定位；找不到或匹配多处时失败，不能默默选择第一处。
4. 构建时解析当前检出提交的行号、精确片段及文档位置，输出排序稳定的 JSON。
   相同源码和输入产生相同输出，不写入时间戳或机器绝对路径。
5. 样例保持当前历史提交，作为契约参考。正式发布快照每次从对应 checkout
   重新生成，禁止直接把历史样例复制成最新版架构数据。

职责说明继续以现有项目上下文和代码为依据，展示文件只保存必要摘要。生成器不会
覆盖项目上下文，也不另建主题清单。默认只提取选中节点之间的关系，避免全仓库
导入图淹没解释性视图。

## 前端与布局

前端预览使用 React、TypeScript、React Flow 和 Vite，依赖固定在独立目录的
lockfile 中。字体本地打包，不影响 Python 包的运行时依赖。

全局视图采用稳定的分组顺序和响应式列布局，默认按 100% 字号阅读，提供适应全图
按钮。专题视图沿用同样布局；直接关联模式隐藏无关节点，仅展示当前节点的一跳关系。
该模式随 URL 保存，空关系筛选保留当前节点。未收录关系的节点明确显示 Not mapped。
总览已补充启动器、任务发现、配置加载和 BaseEnv，覆盖 28 个节点、40 条关系。节点展开等
需求出现后再引入自动布局依赖。画布位置由视图层决定，不写进核心语义数据。
第一版禁止连接编辑，节点拖动只改变当前浏览位置。

URL fragment 格式：`#view=task-program&node=atomic-engine&relations=calls,holds&q=engine`。
未知视图回退默认视图并提示；已移除节点取消选择并提示；过滤器不能抹掉选中节点
的详情。无匹配结果显示明确空状态，数据失败显示错误和文档入口。

## Sphinx 与版本发布

开发源码建议放在 `docs/architecture/web/`，构建产物放在
`docs/source/_static/architecture/`，生成产物不提交。Sphinx 页面建议为
`docs/source/overview/architecture/index.md`，并加入 Overview 导航。

页面内嵌同源 iframe，使用有意义的 title，提供独立全屏链接。JS/CSS 使用相对
资源路径，主题通过显式参数同步。iframe 中的文档跳转使用父页面上下文，避免把
整份文档嵌套进小画布。构建配置提供 iframe 相对于当前版本文档根的路径，不硬编码
域名、`/main/` 或发布标签。

源码链接格式为 `https://github.com/DexForce/EmbodiChain/blob/<revision>/<path>#L<start>-L<end>`。
文档链接由同版本 docname 生成，不能沿用当前 `conf.py` 中固定为 `main` 的值构造
版本化源码链接。用 fragment 保存状态，避免静态托管需要服务器路由回退。

统一构建命令必须接入以下所有现有入口，不能只修改发布 workflow：

- `docs/Makefile` 的本地构建入口；
- `.github/workflows/main.yml` 的文档 artifact 构建；
- `.github/workflows/docs-pages.yml` 的手动／发布构建。

`docs/scripts/build_versions.py` 目前只筛选需保留的版本列表，不调用 Sphinx，
不应为本功能向它添加构建职责。

版本测试同时覆盖本地根目录、`main/`、`v0.2.4/` 和仓库名前缀部署。
HTML 版本在构建阶段生成快照和文字摘要。非 HTML 输出使用摘要及模块链接。
现有 `lab/visualization` 继续只承载仿真可视化，本工具由 docs 工具链拥有。

## 验证与里程碑

| 阶段 | 可审核产物 | 验收 |
|---|---|---|
| 本次 | 设计、Schema、16 节点样例、后续计划 | Schema 通过；图完整性、每条证据、主题及文档文件可核对 |
| 数据生成 | 受控 overlay、静态生成器、全局与专题快照 | 不导入仿真包；旧证据报错；输出确定；24 个全局节点 |
| 交互原型 | 使用真实快照的静态页面 | 搜索、关系筛选、详情、键盘列表、URL 恢复可用 |
| 文档集成 | 内嵌／全屏页面及构建接线 | 多版本路径正确；文字可检索；主题及小窗口可用 |

后续实施顺序和文件边界见
[implementation plan](../plans/2026-09-17-architecture-explorer.md)。
