# Gradio 可视化系统架构

本文档以当前代码为准，描述 Gradio 中的三个引擎，以及它们与 EmbodiChain、SimReady、Articraft 和 DexSim 的边界。`gradio_app.py` 只负责启动；界面、资产工作流、场景工作流和进程管理分散在专用模块中。

## 架构总览

```text
gradio_app.py
    │ 启动、队列、allowed_paths
    ▼
app_services.py（兼容门面）
    ▼
app_ui.py ───────────► app_asset_engine.py ───► SimReady CLI
    │ 布局、引擎选择和事件绑定 │    │
    │                     │       └──────────► app_articraft.py ───► Articraft CLI + Codex CLI
    ▼
app_workflows.py ────► EmbodiChain Scene Engine CLI + Viser
    │                  └► action-agent `run_agent` + DexSim
    ├──────────────► app_commands.py   命令构造
    ├──────────────► app_processes.py  子进程、环境、日志和阶段检测
    ├──────────────► app_state.py      共享 RuntimeState、锁和计时
    ├──────────────► app_media.py      视频、数据集预览和日志归档
    └──────────────► app_config.py     UI 常量、路径推导和命令定义
    └──────────────► app_env.py        部署配置读取
                         └──────────► ../.env  部署路径、端口和服务凭据
```

| 模块 | 职责 |
| --- | --- |
| `gradio_app.py` | 唯一启动入口；校验 `EMBODICHAIN_ROOT`，创建 Blocks，设置队列和本地文件访问路径。 |
| `app_ui.py` | 顶部图标、引擎面板切换和回调绑定；不实现 pipeline。 |
| `app_asset_engine.py` | SimReady 上传适配、输入/输出 GLB 预览、处理日志，以及 Asset engine 的 Articraft 标签页。 |
| `app_articraft.py` | Articraft checkout/环境检查、外部记录创建、Codex 生成与校验、URDF bundle 和 Viser 关节预览。 |
| `app_workflows.py` | Scene Engine 工作流、Action Engine 的会话状态、Viser 预览和 DexSim。 |
| `app_processes.py` | 子进程环境、进程组终止、stdout 读取和 pipeline 阶段检测。 |
| `app_state.py` | `RuntimeState`、互斥锁、进度阶段、运行 token 和耗时统计。 |
| `app_commands.py` | Action engine 的 `run_agent` 参数构造。 |
| `app_media.py` | DexSim 观众视频发现和 Articraft Viser CLI 适配。 |
| `app_config.py` | UI 文案、引擎模式、路径推导和 CLI 固定参数。 |
| `app_env.py` | 从 `.env` 读取 Gradio、Articraft 和 SimReady 的部署值，并保留未配置时的默认值。 |
| `../.env` | Gradio 与 Scene Engine 共用的路径、端口、LLM 和服务端点配置；不提交凭据。 |

## 启动、路径和网络环境

从本项目目录启动：

```bash
conda run -n embodichain python gradio_app.py
```

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| EmbodiChain root | 自动从 `embodichain/gen_sim/env.py` 的源码位置推导 | EmbodiChain 根目录；不再从 `.env` 配置。 |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Gradio 监听地址；非回环地址必须启用认证。 |
| `GRADIO_SERVER_PORT` | `7860` | Gradio 监听端口。 |
| `GRADIO_AUTH_USERNAME` | 空 | 非本机部署的 Gradio 用户名。 |
| `GRADIO_AUTH_PASSWORD` | 空 | 非本机部署的 Gradio 密码。 |
| `SCENE_ENGINE_VISER_PORT` | `8080` | Scene Engine 的首选 Viser 端口；占用时为会话分配其他可用端口。 |
| `ARTICRAFT_VISER_PORT` | `8081` | Articraft 关节预览的首选 Viser 端口；占用时为会话分配其他可用端口。 |
| `ACTION_ENGINE_VISER_PORT` | `8082` | Action Engine 已保存场景预览的首选 Viser 端口；占用时为会话分配其他可用端口。 |
| `ARTICRAFT_ROOT` | `<项目>/.articraft` | Articraft checkout。 |
| `ARTICRAFT_CONDA_ENV` | `articraft` | 运行 Articraft CLI 的 Conda 环境。 |
| `ARTICRAFT_OUTPUT_ROOT` | `<项目>/.gen_sim/articraft` | Articraft 记录、运行日志和导出 bundle。 |

`app.launch()` 仅开放 UI 静态资源、`.gen_sim/` 生成物和配置的 Articraft 输出目录，并显式禁止 `.env`、`.git/` 等敏感路径。pipeline 子进程由 `build_pipeline_env()` 从共享 `.env` 创建环境：它清除代理变量、设置 `NO_PROXY=no_proxy=*` 并关闭 Gradio analytics。只有 SimReady 子进程会额外把非空的 `SIMREADY_OPENAI_*` 映射为其上游 CLI 需要的 `OPENAI_*`；Scene Engine、DexSim、Viser 和 Articraft 直接继承 `.env` 中的原始配置。Codex 作为用户指令驱动的子进程，使用独立登录状态和最小化环境，不继承 GenSim 服务凭据。

## 页面与引擎

页面顶部保留 DexForce 图标，并直接显示 `Asset_engine`、`Scene_engine`、`Action_engine` 三个入口，不再提供模式切换。它们的实际输入和产物并不完全相同：

| Engine | 输入 | 预览/下载 | 实际产物 | 是否启动 DexSim |
| --- | --- | --- | --- | --- |
| Asset engine / SimReady | 一个网格、可选材质附件、类别 | 输入 GLB、SimReady GLB、原始输出下载 | `.gen_sim/assets/runs/<token>/` | 否 |
| Asset engine / Articulation | 文字、可选参考图 | URDF articulation 的 Viser、zip 下载 | `.gen_sim/articraft/` | 否 |
| Scene engine | 一张图片 | Scene Engine 的 Viser | `.gen_sim/scenes/<image-sha256-前16位>/` | 否 |
| Action engine | 已生成场景列表、任务、机器人 | 选中场景的 Viser 和 DexSim 视频 | 场景预览来自 `.gen_sim/scenes/`；DexSim 暂沿用现有命令 | 是 |

因此，Scene engine 是独立的图像条件场景生成器；它不会提升、复制或转换输出到 `gym_project/current`。Action engine 只消费已有的 `current` Gym 场景。界面中的 “Scene engine” 文案表达的是所需场景类型，并不意味着独立 Scene Engine 输出已自动连到 Action engine。

Scene/Action 与 Asset engine 使用同一会话边界：Gradio 回调从 `request.session_hash` 取得会话 ID。每个 ID 拥有独立的 `RuntimeState`，Scene 生成/Viser、Action DexSim 和 Action Viser 分别由以该 ID 为键的进程 registry 管理。Reset、Stop、同会话新任务替换及页面卸载只会终止该会话的进程；其他浏览器会话的进程和 UI 状态不受影响。

## Asset engine

### SimReady：单资产目录适配

SimReady CLI 接收目录，而 Gradio 接收上传文件。上传文件会复制到隔离目录，文件名只保留 basename，重名追加序号，避免上传路径或重名影响处理：

```text
mesh + sidecar files
  → .gen_sim/assets/runs/<token>/input/
  → trimesh 导出 input_preview.glb
  → SimReady CLI
  → output/**/asset_simready.glb（优先）或 asset_simready.obj
  → GLB 预览 + 原始文件下载
```

主网格支持 `.glb`、`.gltf`、`.obj`、`.ply`、`.stl`；可一并上传 `.mtl`、纹理和 `.bin` 等附件。执行命令为：

```bash
python -m embodichain.gen_sim.simready_pipeline.cli.start \
  --input_dir <isolated-input-dir> \
  --output_root <isolated-output-dir> \
  --category <category>
```

处理函数以 generator 持续返回最近的 stdout；完成时优先预览 `asset_simready.glb`，只有 OBJ 时再转为 GLB。此路径不依赖 DexSim。

`Reset SimReady` 会清空当前浏览器会话的上传、类别、预览、下载项和日志，并仅按进程组终止该会话正在运行的 SimReady CLI 及其子进程。

### Articulation：Articraft + Codex

Articulation 标签页根据文本和可选参考图生成一个可下载的 articulated asset。先点击环境检查：若 `ARTICRAFT_ROOT` 不存在，应用会 clone `ARTICRAFT_REPOSITORY_URL`；随后检查 Conda、指定的 Articraft 环境和 Codex CLI。该操作会创建 checkout 和 `.gen_sim/articraft/` 中的输出目录，现有的非 Articraft 目录不会被覆盖。

生成流程：

```text
description + optional image
  → Articraft external init（创建 rec_ui_articraft_* 记录）
  → 启动 Codex CLI，仅授权编辑该记录的 active model.py
  → Articraft external check
      └─ 旧版 CLI 无 check 时：compile --validate --strict-geom-qc + compile_report
  → Articraft external finalize
  → materialized model.urdf + meshes
  → exports/<record-id>.zip + Viser articulation preview
```

产物、记录和参考图均在 `ARTICRAFT_OUTPUT_ROOT` 下，不能直接当作 Action engine 的 Gym 场景或 SimReady 资产；若要进入后续仿真，需要另行定义并实现转换/导入流程。Articraft Viser 按 Gradio 会话管理预览进程：首先尝试 `ARTICRAFT_VISER_PORT`，若已被占用则分配其他可用端口，不会查找或发送信号给占用端口的外部进程。

`Reset Articulation` 会清空当前会话的描述、参考图、记录与下载结果，终止该会话的 Articraft/Codex 命令进程组，并关闭该会话启动的 Viser。

## 独立 Scene engine 和 Viser

Scene engine 只接收图像。上传图像会先进行 EXIF 归正并转为 RGB PNG，以 PNG 字节的 SHA-256 前 16 位作为目录名；相同图像会复用同一目录：

```text
image
  → .gen_sim/scenes/<hash>/input.png
  → python -m embodichain scene-engine
       --image <input.png>
       --output_root <hash-dir>
  → <hash-dir>/scene_export/scene_config.json
  → preview.py <hash-dir> --viser --viser-host 0.0.0.0 --viser-port <session-port>
  → Gradio iframe
```

当 `scene_export/scene_config.json` 存在且生成进程返回成功时，应用才启动 Viser。iframe 使用 Gradio 页面当前的协议和主机名转向 Viser 端口，因此从其他设备访问时，浏览器必须能访问该端口。每个会话优先使用 `SCENE_ENGINE_VISER_PORT`，占用时选择其他可用端口；同会话的新 Scene Engine 任务会终止该会话旧的 Scene Viser，不会终止其他会话的预览。输出目录会显示在 UI 中，便于检查 hash 命名的场景导出。

`Reset Scene Engine` 会清空当前会话的图像、进度、输出目录和 iframe，并终止该会话当前生成命令与 Scene Viser 的进程组；registry 运行 token 会使已经失效的生成器停止回写界面。

## Action engine：Gym 场景契约

Action engine 不接收裸 GLB。普通 GLB 只有渲染数据，而 DexSim 还需要碰撞、物理参数、初始位姿、资源相对路径和 action 配置。当前实现的前置条件是：

```text
gym_project/current/gym_export/
gym_project/action_agent_pipeline/configs/current/fast_gym_config.json
gym_project/action_agent_pipeline/configs/current/agent_config.json
```

进入 Action engine 或点击 `Refresh scenes` 会扫描 `.gen_sim/scenes/`，只列出包含 `scene_export/scene_config.json` 的完整场景。列表不会自动选中场景；用户显式选择后，右侧通过 Viser 展示该场景。当前场景选择只负责可视化，尚未传递给 DexSim 命令。

点击 `Run DexSim` 仍会检查任务、现有 `current` Gym/action 配置、运行占用和可导入的 `embodichain.gen_sim.action_agent_pipeline.cli.run_agent`，再以当前配置调用 `run_agent`。

运行命令的核心参数为：

```bash
python -m embodichain.gen_sim.action_agent_pipeline.cli.run_agent \
  --task_name current \
  --gym_config <.../fast_gym_config.json> \
  --agent_config <.../agent_config.json> \
  --regenerate --renderer fast-rt --num_envs 1
```

`--robot-profile` 仅在通过 `run_agent --help` 探测到该参数时加入。DexSim 完成后会寻找本次运行产生的 audience 视频并显示在 Action engine 中。

`Stop Action Engine` 仅重置当前会话的 Action 进程 registry，终止该会话的 DexSim 进程组和已保存场景 Viser。它不终止同会话的 Scene Engine，也不影响其他浏览器会话。

## 会话状态、并发和进度

Scene engine 和 Action engine 的 UI 状态存放在 `SessionRuntimeRegistry`，键为 `request.session_hash`。`RuntimeState` 只包含当前会话的输入、预览、日志和阶段，不再保存服务器级全局进程引用。Scene 生成/Viser、Action DexSim 和 Action Viser 有独立的 `SessionProcessRegistry`；每个 registry 的 token 用于丢弃同会话内过期线程的更新。SimReady 和 Articraft 使用相同的会话键。Reset、Stop 和页面卸载只清理所属会话，并直接向该会话所属进程组发送 `SIGKILL`，不等待交互式任务优雅退出。Gradio 应用正常关闭仍统一清理全部已注册子进程，并保留 `SIGTERM` 宽限期后再升级为 `SIGKILL` 的原有逻辑。

`app.queue(default_concurrency_limit=1)` 将队列中的高成本回调串行化。Action engine 的 `Timer(2.0)` 通过当前请求的 `session_hash` 只读取该会话状态。Asset/Articraft 使用各自的会话状态，但仍会受 Gradio 队列限制。

共享阶段如下；独立 Scene Engine 将其日志映射到相同的进度条：

```text
idle → received → started → scene_intake → relations
→ asset_generation → gym_export → config → preview → complete
                                      └──────────────→ failed
```

## 环境前置条件与验证

SimReady 需要 Blender、trimesh、LLM 配置以及可导入的：

```text
embodichain.gen_sim.simready_pipeline.cli.start
```

SimReady 的 OpenAI-compatible 设置来自环境变量，且不应写入 Git：

```bash
export SIMREADY_OPENAI_API_KEY='<key>'
export SIMREADY_OPENAI_MODEL='<model>'
export SIMREADY_OPENAI_BASE_URL='<base-url>'
```

Action engine 只需要 action-agent 的运行模块：

```text
embodichain.gen_sim.action_agent_pipeline.cli.run_agent
```

独立 Scene engine 还需要：

```text
python -m embodichain scene-engine
embodichain/gen_sim/scene_engine/cli/preview.py
.env
```

Articulation 还需要 Git（首次 clone）、Conda、`ARTICRAFT_CONDA_ENV` 和已通过独立凭据存储完成登录的 Codex CLI。Codex 子进程不继承 `.env` 中的 API key、token 或密码，输出在返回浏览器前还会按已知敏感环境值脱敏。生成请求会交给本机 Codex CLI 执行，因此默认只在本机可信工作台中使用。

每次修改后至少执行：

```bash
python -m py_compile \
  gradio_app.py app_config.py app_env.py app_state.py app_commands.py \
  app_processes.py app_media.py app_workflows.py app_ui.py \
  app_asset_engine.py app_articraft.py app_services.py

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  conda run -n embodichain python -c \
  "from app_ui import build_app; assert build_app() is not None"
```

手动检查：

1. SimReady 上传简单网格后能显示输入预览；执行后显示 SimReady 输出或明确错误。
2. Articulation 环境检查能报告 checkout、Conda 和 Codex 状态；成功生成后有 zip、记录目录和 Viser 或明确的预览错误。
3. Scene engine 从图像生成 `scene_export/scene_config.json`，并以 `8080` 为首选端口显示 Viser；它不应改写 `gym_project/current`。
4. Action engine 在没有 `current` Gym/action 配置或缺少 CLI 时给出预检错误。
