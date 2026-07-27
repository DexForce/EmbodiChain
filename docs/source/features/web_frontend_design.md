# EmbodiChain Web 前端概念设计

> 状态：概念设计
>
> 更新日期：2026-07-27
>
> 目标读者：EmbodiChain 维护者、仿真与训练平台开发者、前端开发者

## 1. 背景与结论

EmbodiChain 当前已经覆盖仿真、环境、传感器、运动学、运动规划、数据生成、
强化学习和资产工具等能力，但主要通过 Python API、配置文件和命令行入口使用。
本设计希望增加一个类似 Gradio 的 Web 使用界面，使用户能够发现能力、编辑配置、
启动任务、观察运行状态并管理产物，同时让本地工作站和远程服务器复用同一套运行逻辑。

结论是：**基于 Web 的前端方案合理，且适合成为 EmbodiChain 的统一控制面；
但 Web 进程不应直接承载仿真运行时。** 每个仿真、数据生成或训练任务应运行在独立
子进程或容器中，Web 服务只负责任务编排、状态管理、日志和可视化数据转发。

这一定位与 Isaac Sim / Isaac Lab 的原生仿真窗口不同：

- EmbodiChain 原生窗口继续承担高保真渲染、软体/布料调试和本地交互。
- Web 界面承担跨平台配置、运行管理、监控、结果回放和轻量三维检查。
- 本地与服务器共享任务规范和 Worker，不要求共享完全相同的显示后端。

最终应达到“**功能语义一致、显示方式按部署环境选择**”，而不是强求浏览器和原生窗口
逐像素一致。

## 2. 设计目标与非目标

### 2.1 目标

1. 使用同一份 `RunSpec` 在本地工作站或远程服务器启动任务。
2. 复用现有 EmbodiChain 环境、配置、数据和训练逻辑，不在 Web 层复制业务实现。
3. 支持能力发现、配置编辑、运行生命周期管理、实时监控和产物管理。
4. 为服务器提供不依赖桌面窗口的三维场景检查能力。
5. 隔离仿真进程、GPU 资源和用户任务，避免单个任务影响控制面。
6. 为后续数据集、RL、运动规划、资产工具和协作功能保留扩展边界。

### 2.2 非目标

- 第一阶段不在浏览器中重新实现 DexSim 渲染器或物理引擎。
- 第一阶段不追求原生窗口的全部 Gizmo、键盘快捷键和调试能力。
- 第一阶段不允许用户通过 Web 任意执行 Python 代码或导入未授权模块。
- 第一阶段不以公网低延迟遥操作为主要目标。
- Web 控制面不直接持有 `SimulationManager` 或 Gym 环境实例。

## 3. 当前能力盘点

以下能力以本设计编写时的仓库状态为准。Web 前端不应硬编码此清单，运行时应通过
能力探测接口获得可用项和依赖状态。

| 领域 | 当前能力 | Web 入口建议 |
| --- | --- | --- |
| 仿真 | CPU/GPU 仿真、多环境、刚体、刚体组、关节体、机器人、软体、布料、约束、灯光、材质、Gizmo、窗口录制 | 场景与运行工作台 |
| 传感器 | RGB、深度、分割/掩码、法线、位置、立体相机、接触传感器 | 传感器面板与流选择 |
| 环境 | Gymnasium `reset`/`step`、`BaseEnv`、`EmbodiedEnv`、Wrapper | 任务浏览器与运行面板 |
| 管理器 | Observation、Reward、Event、Action、Dataset、Randomization | 结构化配置编辑器 |
| 机器人 | DexForce W1、CobotMagic、Franka Panda、UR、Dual Arm | 机器人和控制部件选择器 |
| 运动学 | PyTorch、Pinocchio、Differential、Pink、OPW、SRS、Neural IK、UR 求解器 | IK/FK 工作台 |
| 规划与动作 | TOPPRA、实验性神经规划器、Motion Generator、Atomic Action Engine | 轨迹与动作工作台 |
| 数据 | 同步/异步 LeRobot 记录、在线共享内存数据引擎、数据管理器 | 数据生成与数据集面板 |
| 学习 | PPO、GRPO、采集器、Rollout Buffer、分布式训练、TensorBoard、Weights & Biases 集成 | 训练任务和指标面板 |
| 生成式仿真 | Prompt-to-Scene、SimReady Pipeline、Action Agent Pipeline | 流程式生成任务与阶段产物 |
| 工具 | 资产管理、SimReady、抓取采样/标注、URDF Assembly、Workspace Analyzer、Benchmark | 独立工具入口 |
| Agent 任务 | 任务中保留 Agent 环境变体，但当前源码树中的层级 Agent 实现不完整 | 能力探测失败时禁用并解释原因 |
| 设备 | 真实设备控制抽象仍处于基础阶段 | 暂不作为首期核心入口 |

当前官方任务包中可发现的任务包括：

- `CartPoleRL`
- `PushCubeRL`
- `StayStillSave-v1`
- `SimpleTask-v1`
- `Rearrangement-v3`
- `RearrangementAgent-v3`
- `StackBlocksTwo-v1`
- `MatchObjectContainer-v1`
- `StackCups-v1`
- `PlaceObjectDrawer-v1`
- `BlocksRankingRGB-v1`
- `PourWater-v3`
- `PourWaterAgent-v3`
- `BlocksRankingSize-v1`
- `ScoopIce-v1`

此外，核心包注册了通用的 `EmbodiedEnv-v1`。任务列表、机器人、求解器和传感器是否可用
可能受可选依赖、资产和插件影响，因此 Web 界面需要同时显示：

- 已注册；
- 可实例化；
- 缺少依赖或资产；
- 仅本地可用；
- 仅特定 GPU / 渲染能力可用。

当前还存在一个需要显式暴露的能力缺口：`BaseAgentEnv` 会导入
`embodichain.agents.hierarchy` 下的层级 Agent，但当前源码树只保留了相关缓存目录，
没有对应的可分发 Python 源文件。`RearrangementAgent-v3` 和
`PourWaterAgent-v3` 可以被注册，不代表它们在干净安装中一定可实例化。Web 控制面必须
将“已注册”和“已通过实例化探测”区分开，避免给用户错误承诺。

## 4. 当前运行机制及对 Web 的影响

### 4.1 典型环境运行流程

现有 `embodichain run-env` 入口的核心流程是：

```text
发现任务包 entry points
        ↓
执行初始化 hooks
        ↓
读取 YAML / JSON 和命令行参数
        ↓
构造 EmbodiedEnvCfg
        ↓
gymnasium.make(task_id, cfg=...)
        ↓
env.reset() / env.step(action)
        ↓
Observation / Reward / Event / Action / Dataset Managers
        ↓
数据记录、训练、日志或产物
        ↓
env.close() → SimulationManager.destroy()
```

Web Worker 应沿用这条路径，避免建立第二套不兼容的环境构造方式。Web 层只生成规范化
输入，并通过受控 Worker 调用现有入口或等价的内部 API。

### 4.2 进程隔离是硬性要求

`SimulationManager.destroy()` 默认读取
`EMBODICHAIN_SIM_EXIT_PROCESS=1`，并可能调用 `os._exit(0)`。这意味着：

- 仿真环境不能与 Web API 服务运行在同一进程；
- 每次运行必须是独立子进程，服务器部署时优先使用独立容器；
- 日志、状态和产物必须在 Worker 退出前持续发送或落盘；
- 控制面必须根据进程退出码、心跳和最后事件恢复最终状态；
- 停止任务时需要终止整个进程组，并处理子 Worker 和共享内存。

即使未来支持不退出进程的清理模式，首版仍应保持一任务一进程，降低 C++ 资源、
GPU context 和全局注册状态相互污染的风险。

### 4.3 配置必须以不可变快照传递

当前 `config_to_cfg()` 在解析 `robot_type` 时会修改传入字典。因此控制面需要：

1. 保存用户提交的原始配置；
2. 深拷贝并规范化为运行快照；
3. 对快照计算摘要并禁止就地修改；
4. 将原始配置、规范化配置、代码版本和依赖信息一起写入运行元数据。

这既避免多次运行之间相互影响，也为复现和差异比较提供基础。

### 4.4 能力探测优于静态假设

任务注册依赖包发现、初始化 hook、可选依赖和资产状态。控制面启动后应通过短生命周期
探测进程生成 `CapabilityManifest`，而不是直接扫描 Python 文件或在 API 进程导入所有
仿真模块。

建议探测内容：

- EmbodiChain 版本、Git commit、Python/CUDA/GPU/驱动信息；
- 已发现的任务包、任务 ID 和最大 episode 步数；
- 机器人、传感器、求解器、规划器、记录器和训练算法；
- 每项能力的 schema、默认值、说明和依赖检查结果；
- 可用资产根目录和存储配额；
- 本地窗口、headless、Web 3D、视频流等显示模式；
- 已安装但导入失败的能力及可操作的错误说明。

## 5. 总体架构

```text
┌──────────────────────────── Browser ────────────────────────────┐
│ React/TypeScript UI                                             │
│ REST：查询与命令   WebSocket：状态/日志/遥测   WebRTC：视频     │
│ Viser：结构化 3D 场景                                           │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌────────────────────── Web Control Plane ────────────────────────┐
│ API + Auth + Capability Service + Run Service + Artifact Service│
│ 持久化：RunSpec / 状态 / 用户 / 产物索引 / 审计                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │ 受控命令与事件协议
┌────────────────────────── Run Orchestrator ─────────────────────┐
│ 本地：Supervisor + subprocess                                   │
│ 服务器：Scheduler + isolated container + GPU lease              │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────── EmbodiChain Worker ─────────────────────┐
│ Runtime Adapter                                                 │
│   ├─ task discovery / config build / gym.make                    │
│   ├─ reset / step / rollout / record / train                     │
│   ├─ structured event + scene telemetry                          │
│   └─ artifact finalization + env.close                           │
│ Existing EmbodiChain runtime                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 5.1 分层职责

| 层 | 负责 | 不负责 |
| --- | --- | --- |
| Browser | 表单、布局、交互、图表、3D/视频展示 | 加载仿真引擎、直接访问文件系统 |
| Control Plane | 鉴权、校验、任务记录、状态机、事件订阅、产物索引 | 持有仿真环境、执行用户 Python |
| Orchestrator | 资源分配、进程/容器生命周期、心跳、超时、清理 | 解释任务内部业务逻辑 |
| Worker Adapter | 将 `RunSpec` 映射到现有 EmbodiChain 入口并发出标准事件 | 用户管理、跨任务调度 |
| EmbodiChain | 仿真、环境、数据、训练和工具的实际执行 | Web 会话和 HTTP 生命周期 |

### 5.2 共享运行规范

本地和服务器必须消费相同版本的 `RunSpec`。概念结构如下：

```yaml
schema_version: 1
kind: environment
task_id: SimpleTask-v1
source:
  git_commit: "<commit>"
config:
  original: {}
  normalized: {}
execution:
  seed: 42
  device: cuda:0
  num_envs: 1
  max_episodes: 1
visualization:
  mode: web
  scene_fps: 15
  video_streams: [camera]
recording:
  dataset: false
  window_video: false
limits:
  wall_time_seconds: 3600
  gpu_memory_mb: 12000
artifacts:
  output_uri: "<managed-run-directory>"
```

`RunSpec` 应有版本号和 JSON Schema。服务端只接受受支持字段，并在运行前完成：

- schema 校验；
- 路径和资产引用解析；
- 能力与依赖校验；
- 资源上限校验；
- 规范化和内容摘要；
- 用户权限检查。

### 5.3 Worker 命令与事件协议

控制面与 Worker 之间应使用稳定协议，而不是解析控制台文本。

命令建议：

- `START`
- `PAUSE` / `RESUME`（仅能力声明支持时开放）
- `STOP`
- `RESET`
- `SET_RUNTIME_PARAMETER`（仅白名单参数）
- `REQUEST_SNAPSHOT`

事件建议：

- `RUN_STATE_CHANGED`
- `HEARTBEAT`
- `LOG`
- `METRIC`
- `EPISODE_PROGRESS`
- `SENSOR_FRAME`
- `SCENE_MANIFEST`
- `SCENE_FRAME`
- `ARTIFACT_CREATED`
- `WARNING`
- `ERROR`

所有事件至少包含 `run_id`、单调递增序号、Worker 时间戳、事件类型和 schema 版本。
大体积图像、视频、点云和产物应使用对象存储地址或专用媒体通道，不直接塞入普通事件。

### 5.4 运行状态机

```text
CREATED → VALIDATING → QUEUED → STARTING → RUNNING
                                      │        ├→ PAUSED → RUNNING
                                      │        ├→ STOPPING → KILLED
                                      │        ├→ SUCCEEDED
                                      │        └→ FAILED
                                      └──────────────→ FAILED
```

状态转换由控制面持久化。Worker 断连不应立刻等同于失败；Orchestrator 需要结合进程状态、
心跳超时和最终元数据进行判定。

## 6. 技术方案评估

### 6.1 推荐的长期方案

- **后端：FastAPI**
  适合定义类型化 REST API、WebSocket 通道和异步服务，并能与现有 Python 生态集成。
  它只作为控制面，不运行仿真主循环。

- **前端：React + TypeScript**
  更适合复杂配置表单、多面板工作台、运行状态管理、时间序列图表和可扩展插件入口。

- **轻量三维：Viser**
  EmbodiChain 已有 Viser 依赖和抓取标注使用经验。它适合展示机器人、刚体、坐标系、
  轨迹、点云和交互控件，可降低首版自研 Three.js 场景协议和控件的成本。

- **原生画面或高帧率相机：WebRTC**
  用于服务器端渲染画面、相机 RGB 流或需要接近实时的视频。日志和低频遥测仍走
  WebSocket，避免媒体数据阻塞控制消息。

- **回放与时间同步：先自有事件/产物协议，可选接入 Rerun**
  Rerun 适合作为调试和回放工具，但不应成为任务生命周期和权限模型的唯一基础。

### 6.2 Gradio 的定位

Gradio 适合快速验证以下内容：

- 任务和配置选择；
- 单用户启动/停止；
- 日志、图像和少量参数展示；
- 演示级原型。

它不适合作为长期控制面的主要原因是：

- 多任务状态机、资源调度和故障恢复需要独立后端模型；
- 复杂多面板工作台和大型配置编辑体验会受到限制；
- `Model3D` 更适合文件结果展示，不等同于高频场景状态流；
- 多用户权限、产物 ACL、审计和服务器资源治理不应绑定 UI 框架的队列模型。

因此可以使用 Gradio 做一次性技术验证，但正式架构仍应让 API、Orchestrator 和 Worker
协议与前端框架解耦。

### 6.3 NiceGUI 的备选定位

如果团队在 MVP 阶段没有前端工程资源，NiceGUI 可以用于比 Gradio 更自由的 Python
驱动界面。但只要目标包含复杂配置编辑、长期多用户运行管理和可扩展工作台，
React + TypeScript 仍是更稳妥的长期选择。

## 7. 可视化设计

### 7.1 为什么服务器需要 Web 3D

服务器通常运行在 headless 环境，没有可直接操作的原生窗口。只有日志和相机画面时，
用户难以快速判断：

- 机器人关节或物体位姿是否正确；
- 场景层级和资产是否加载；
- 轨迹、目标位姿和碰撞点是否合理；
- 多环境布局是否符合预期。

因此建议首版包含简单 Web 3D，并将 Viser 作为优先验证方案。

### 7.2 场景协议

Worker 不应每帧发送完整网格。建议拆分为：

**`SceneManifest`：初始化或拓扑变化时发送**

- 节点 ID、父子关系和语义类型；
- 网格/URDF/GLB 引用及内容摘要；
- 材质和默认可见性；
- 关节、坐标系、相机和传感器定义；
- 环境实例和批次关系。

**`SceneFrame`：按显示帧率发送**

- 节点位姿；
- 关节位置；
- 可见性变化；
- 轨迹、目标点、接触点和调试线；
- 轻量指标；
- 仿真步、episode、时间戳。

传输原则：

- 网格和静态资源缓存，只传一次或使用内容寻址；
- 位姿尽量批量编码；
- 图像、深度和点云独立限频；
- 出现积压时采用“最新帧优先”，不阻塞仿真；
- `num_envs` 很大时只显示抽样环境或聚合统计；
- 遥测采集应可关闭，并有明确的性能预算。

### 7.3 三种显示模式

| 模式 | 用途 | 本地 | 服务器 |
| --- | --- | --- | --- |
| `native` | 最高保真渲染、软体/布料、原生 Gizmo 调试 | 支持 | 通常不支持桌面交互 |
| `web` | 结构化场景、机器人位姿、轨迹、点云和传感器 | 支持 | 支持 |
| `hybrid` | 原生窗口与 Web 控制面同时使用 | 推荐高级调试 | 可用视频代替原生窗口 |

### 7.4 一致性边界

| 能力 | 本地与服务器能否一致 | 说明 |
| --- | --- | --- |
| 配置、任务、种子、运行命令 | 可以 | 同一 `RunSpec` 和 Worker |
| 日志、指标、状态、产物 | 可以 | 同一事件和产物协议 |
| 数据生成与训练结果 | 原则上可以 | 仍受 GPU、驱动、数值后端影响 |
| RGB/深度等传感器 | 接近一致 | 需要相同渲染配置、资产和 GPU 环境 |
| 刚体、机器人、轨迹 3D 检查 | 功能一致 | Web 3D 不保证像素一致 |
| 光线追踪、软体、布料效果 | 不能仅靠结构化 3D 等价 | 使用原生渲染视频或离线录制 |
| Gizmo、键盘和窗口快捷键 | 不能直接复用 | 需在 Web 中重新定义交互协议 |
| 广域网遥操作 | 不保证 | 受延迟、抖动和安全策略影响 |

### 7.5 Viser 使用边界

适合：

- 机器人关节和 link 位姿；
- 刚体、坐标系、目标位姿和轨迹；
- 抓取候选、接触点、点云和少量交互控件；
- 服务器 headless 运行的场景检查。

不适合单独承担：

- DexSim 原生渲染的逐像素复现；
- 大规模软体/布料粒子的高频可视化；
- 高分辨率多相机视频；
- 依赖原生窗口的高级 Gizmo 操作。

部署时应将 Viser 放在统一鉴权和反向代理之后，避免为每个任务暴露动态公网端口，
也不应依赖临时公网分享隧道作为正式方案。

## 8. 前端功能模块

### 8.1 P0：能力与系统状态

- EmbodiChain、任务包和代码版本；
- GPU、CUDA、驱动、显存和 Worker 健康状态；
- 任务、机器人、传感器、求解器和算法能力清单；
- 依赖/资产缺失原因；
- 本地或服务器部署模式；
- 存储位置和配额。

### 8.2 P0：任务与配置工作室

- 按类别搜索任务；
- 从任务默认配置创建草稿；
- schema 驱动表单和 YAML/JSON 双向编辑；
- 字段说明、默认值、类型和约束；
- 配置校验、差异比较和只读运行快照；
- 机器人、传感器、对象、灯光、管理器和随机化分组；
- 种子、设备、`num_envs`、episode 和记录策略；
- 保存命名模板，但不允许直接执行任意 Python。

### 8.3 P0：运行管理

- 创建、排队、启动、停止和可选暂停/恢复；
- 当前状态、步骤、episode、耗时和资源占用；
- 实时结构化日志与级别筛选；
- 超时、取消、失败原因和重试；
- GPU 选择或自动调度；
- 运行历史、配置快照和复现；
- 僵尸任务检测和管理员强制清理。

### 8.4 P0：实时工作台

- Viser 3D 场景；
- 相机、深度、分割和其他传感器标签页；
- episode、reward、性能、GPU 和遥测带宽图表；
- 环境实例选择和多环境抽样；
- 轨迹、目标位姿、坐标系、接触点显示开关；
- 快照、录制和调试标注；
- 显示模式、帧率、分辨率和质量控制。

### 8.5 P0：产物中心

- 配置、日志、视频、图像、数据集、checkpoint 和分析报告；
- 产物类型、大小、摘要、创建时间和来源运行；
- 下载、预览、保留策略和清理状态；
- 运行间对比；
- LeRobot 数据记录进度和 finalize 状态；
- 不暴露服务器任意绝对路径。

### 8.6 P0：资源与安全

- 用户、项目和运行级权限；
- GPU/CPU/内存/运行时长/并发数限制；
- 工作目录和资产白名单；
- 审计日志；
- Token、密钥和外部服务凭据托管；
- Worker 网络访问策略；
- 上传文件类型、大小和解压安全检查。

### 8.7 后续模块

- IK/FK、运动规划和 Atomic Action 工作台；
- 抓取采样与标注、SimReady、URDF Assembly；
- 数据集浏览、清洗、转换和回放；
- PPO/GRPO 训练配置、指标和 checkpoint 对比；
- Workspace Analyzer；
- 任务模板和可控插件；
- 多用户协作、评论和审批；
- 在明确实时性与安全约束后的遥操作。

## 9. 本地与服务器部署

### 9.1 本地模式

```text
Browser → localhost Control Plane → Local Supervisor → Worker subprocess
                                              ├→ native window
                                              ├→ web visualization
                                              └→ hybrid
```

本地模式可以提供桌面安装器或 CLI 启动器。浏览器和 Worker 仍通过标准协议通信，
以便本地运行可以原样迁移到服务器。

### 9.2 服务器模式

```text
Browser → HTTPS Gateway → Control Plane → Scheduler
                                         └→ isolated Worker container + GPU
                                                ├→ scene telemetry
                                                ├→ WebRTC media
                                                └→ managed artifacts
```

服务器模式需要增加调度、租户隔离、共享存储、鉴权、反向代理和资源回收。
这些属于部署差异，不应改变任务配置和 Worker 业务逻辑。

### 9.3 体验一致性的判定标准

“一样的使用体验”应定义为：

- 同一任务和配置可以创建、验证、运行、停止和复现；
- 状态、日志、指标和产物组织一致；
- 用户可在两端判断场景、机器人、传感器和任务是否正常；
- 运行规范可以直接从本地导出并在服务器提交；
- 显示能力差异被明确标注，并允许选择合适模式。

它不意味着：

- 浏览器复刻所有原生窗口像素；
- 网络环境下保持本地键鼠级延迟；
- 不同 GPU/驱动产生完全相同的浮点和渲染结果。

## 10. 主要风险与缓解措施

| 风险 | 等级 | 影响 | 缓解措施 |
| --- | --- | --- | --- |
| 仿真销毁可能结束进程 | 严重 | API 服务被连带结束、状态丢失 | 一任务一进程/容器；事件持续落盘；Supervisor 归档最终状态 |
| C++/GPU/全局注册状态泄漏 | 严重 | 后续任务异常或资源无法释放 | 禁止复用长生命周期仿真 Worker；进程组清理；周期性泄漏测试 |
| GPU OOM 和任务争抢 | 高 | 任务互相影响、节点不稳定 | GPU lease、显存预算、并发上限、预检、OOM 分类与回收 |
| 任意配置变成代码执行 | 严重 | 服务器被入侵 | schema 白名单、禁止任意 import/eval、容器隔离、最小权限和网络策略 |
| 路径穿越和恶意压缩包 | 严重 | 文件泄漏或覆盖 | 托管工作目录、规范化路径、类型/大小限制、安全解压 |
| 产物和数据集泄漏 | 高 | 用户数据越权 | 项目级 ACL、签名下载、审计、默认私有和保留策略 |
| 日志/图像/场景遥测拖慢仿真 | 高 | 数据生成和训练吞吐下降 | 异步采集、限频、批量、最新帧优先、可关闭遥测、性能预算 |
| Web 3D 与原生渲染差异 | 中 | 用户误判视觉结果 | 明确“结构模式/像素模式”；关键场景提供原生视频或录制 |
| 浏览器内存和带宽过高 | 高 | 页面卡死、多人使用成本高 | 静态资源缓存、LOD、点云抽样、多环境抽样、自适应码率 |
| 僵尸进程和共享内存残留 | 高 | GPU/内存长期占用 | 心跳、超时、进程组、容器 runtime 回收、启动时残留扫描 |
| 能力注册与实际可用性不一致 | 中 | UI 可选但运行失败 | 短生命周期能力探测；显示缺失依赖；提交前再次校验 |
| 配置被解析过程修改 | 中 | 复现失败或多次运行不一致 | 深拷贝、不可变快照、摘要、原始/规范化配置同时保存 |
| 版本和数值环境漂移 | 高 | 本地与服务器结果不同 | 保存 commit、镜像、依赖锁、资产摘要、种子、GPU/驱动和渲染配置 |
| WebSocket 断线或事件乱序 | 中 | UI 状态错误 | 事件序号、断点续传、状态快照、幂等命令和服务端权威状态 |
| 遥操作延迟与误操作 | 高 | 设备或场景安全风险 | 首版不承诺 WAN 遥操作；后续加入死手开关、限速、急停和本地安全控制 |

## 11. 分阶段实施建议

### Phase 0：技术验证

目标是验证架构边界，不追求完整产品 UI。

1. 定义 `RunSpec`、`CapabilityManifest` 和事件 envelope。
2. 实现本地 Supervisor 和一个 `run-env` Worker Adapter。
3. 用 WebSocket 展示状态、日志、episode 和基础指标。
4. 用 Viser 展示机器人、刚体、坐标系和轨迹。
5. 用低帧率图像通道展示一个相机；验证 WebRTC 方案但可暂不产品化。
6. 以托管运行目录保存配置、日志、截图和最终状态。

验证用例：

- `SimpleTask-v1`：环境、相机、数据记录和正常关闭；
- `CartPoleRL`：高频 step、reward/episode 指标；
- 一个软体或布料示例：验证结构化 3D 的边界和原生视频回退。

验收建议：

- 连续启动/停止至少 20 次，无残留 Worker、GPU context 或共享内存；
- 关闭浏览器不终止任务，重新连接后能恢复状态；
- 遥测开启时的仿真吞吐下降目标不超过 10%–15%；
- 同一 `RunSpec` 能在本地和服务器 Worker 执行；
- Worker 异常退出后控制面能给出稳定最终状态和错误信息。

### Phase 1：可用 MVP

- React 配置工作室和运行工作台；
- 服务器任务队列与单节点 GPU 调度；
- Viser 3D、图像流、日志和指标；
- 运行历史、产物中心和复现；
- 基础用户鉴权、项目隔离和配额；
- 原生、本地 Web、服务器 Web 三种显示路径。

### Phase 2：平台化

- 多节点调度和镜像/依赖环境管理；
- 数据集与 RL 工作台；
- 运动规划、Atomic Action 和资产工具；
- Rerun 或自有时间轴回放；
- 插件化任务面板；
- 更完整的审计、配额、成本和团队协作能力。

## 12. 关键决策摘要

1. Web 是统一控制面和观测面，不是新的仿真运行时。
2. 每个 EmbodiChain 任务必须运行在独立进程或容器中。
3. 本地和服务器共享 `RunSpec`、Worker Adapter、命令协议和事件协议。
4. FastAPI + React/TypeScript 作为长期架构；Gradio 只适合快速验证。
5. 服务器需要轻量 Web 3D；首选验证 Viser，并以视频补足高保真显示。
6. 一致性目标是功能语义、配置、状态和产物一致，不要求显示逐像素一致。
7. 首期优先完成能力探测、配置、运行、监控、3D/传感器和产物闭环。
8. 安全、GPU 调度、进程回收和可复现性必须从首个版本进入架构。

## 13. 相关实现位置与外部参考

仓库中的关键实现位置：

- `embodichain/lab/scripts/run_env.py`
- `embodichain/lab/gym/utils/gym_utils.py`
- `embodichain/lab/gym/utils/registration.py`
- `embodichain/lab/gym/envs/`
- `embodichain/lab/sim/sim_manager.py`
- `embodichain/lab/sim/sensors/`
- `embodichain/lab/sim/solvers/`
- `embodichain/lab/sim/planners/`
- `embodichain/lab/sim/atomic_actions/`
- `embodichain/data_pipeline/`
- `embodichain/learning/rl/`
- `embodichain_tasks/`

技术调研入口：

- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [Viser](https://viser.studio/main/)
- [Gradio Model3D](https://www.gradio.app/docs/gradio/model3d)
- [Rerun Web Viewer](https://rerun.io/docs/getting-started/integrations/web-viewer)
- [WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
