# Gradio 可视化系统架构

本文档以当前代码为准，描述 Gradio Demo、Debug 下的三个引擎，以及它们与 EmbodiChain、SimReady、Articraft 和 DexSim 的边界。`gradio_app.py` 只负责启动；界面、资产工作流、场景工作流和进程管理分散在专用模块中。

## 架构总览

```text
gradio_app.py
    │ 启动、队列、allowed_paths
    ▼
app_services.py（兼容门面）
    ▼
app_ui.py ───────────► app_asset_engine.py ───► SimReady CLI
    │ 布局、模式和事件绑定 │       │
    │                     │       └──────────► app_articraft.py ───► Articraft CLI + Codex CLI
    ▼
app_workflows.py ────► EmbodiChain Scene Engine CLI + Viser
    │                  └► prompt2scene / action-agent pipeline + DexSim
    ├──────────────► app_commands.py   命令构造
    ├──────────────► app_processes.py  子进程、环境、日志和阶段检测
    ├──────────────► app_state.py      共享 RuntimeState、锁和计时
    ├──────────────► app_media.py      视频、数据集预览和日志归档
    └──────────────► app_config.py     路径、端口、文案和固定参数
```

| 模块 | 职责 |
| --- | --- |
| `gradio_app.py` | 唯一启动入口；校验 `EMBODICHAIN_ROOT`，创建 Blocks，设置队列和本地文件访问路径。 |
| `app_ui.py` | Demo/Debug 布局、引擎面板切换和回调绑定；不实现 pipeline。 |
| `app_asset_engine.py` | SimReady 上传适配、输入/输出 GLB 预览、处理日志，以及 Asset engine 的 Articraft 标签页。 |
| `app_articraft.py` | Articraft checkout/环境检查、外部记录创建、Codex 生成与校验、URDF bundle 和 Viser 关节预览。 |
| `app_workflows.py` | Demo 的 prompt2scene/action-agent 工作流、独立 Scene Engine 工作流、GLB 预览、场景提升和 DexSim。 |
| `app_processes.py` | 子进程环境、进程组终止、stdout 读取、Demo pipeline 阶段检测。 |
| `app_state.py` | `RuntimeState`、互斥锁、进度阶段、运行 token 和耗时统计。 |
| `app_commands.py` | prompt2scene、动作配置和 `run_agent` 的参数构造。 |
| `app_media.py` | 观众视频、LeRobot 数据预览、组合视频和运行日志归档。 |
| `app_config.py` | 路径、环境变量、端口、UI 文案、引擎模式和 CLI 固定参数。 |

## 启动、路径和网络环境

从本项目目录启动：

```bash
conda run -n embodichain python gradio_app.py
```

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `EMBODICHAIN_ROOT` | `/home/dex/桌面/EmbodiChain` | EmbodiChain 根目录。 |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio 监听地址。 |
| `GRADIO_SERVER_PORT` | `7860` | Gradio 监听端口。 |
| `SCENE_ENGINE_VISER_PORT` | `8080` | 独立 Scene Engine 的 Viser 端口。 |
| `ARTICRAFT_VISER_PORT` | `8081` | Articraft 关节预览的 Viser 端口。 |
| `ARTICRAFT_ROOT` | `<项目>/.articraft` | Articraft checkout。 |
| `ARTICRAFT_CONDA_ENV` | `articraft` | 运行 Articraft CLI 的 Conda 环境。 |
| `ARTICRAFT_OUTPUT_ROOT` | `<项目>/.debug_engine/articraft` | Articraft 记录、运行日志和导出 bundle。 |

`demo.launch()` 仅开放 EmbodiChain 根目录、`assets/` 和 `.debug_engine/` 给浏览器读取。pipeline 子进程由 `build_pipeline_env()` 创建环境：它清除代理变量、设置 `NO_PROXY=no_proxy=*`、关闭 Gradio analytics，并把非空的 SimReady 配置映射为 `OPENAI_*`。这不会改写启动 Gradio 的父进程环境。

## 页面与引擎

顶部的 `Demo` / `Debug` 只切换可见面板，不会启动任务；切换后共享运行状态保留。Debug 有三个按钮：`Asset_engine`、`Scene_engine`、`Action_engine`。它们的实际输入和产物并不完全相同：

| Engine | 输入 | 预览/下载 | 实际产物 | 是否启动 DexSim |
| --- | --- | --- | --- | --- |
| Asset engine / SimReady | 一个网格、可选材质附件、类别 | 输入 GLB、SimReady GLB、原始输出下载 | `.debug_engine/assets/runs/<token>/` | 否 |
| Asset engine / Articulation | 文字、可选参考图 | URDF articulation 的 Viser、zip 下载 | `.debug_engine/articraft/` | 否 |
| Scene engine | 一张图片 | Scene Engine 的 Viser | `.debug_engine/scenes/<image-sha256-前16位>/` | 否 |
| Action engine | `current` Gym 场景、任务、机器人 | `current` 的 GLB 和 DexSim 视频 | EmbodiChain `gym_project/current` 与 `outputs/` | 是 |

因此，Debug 的 Scene engine 是独立的图像条件场景生成器；它不会提升、复制或转换输出到 `gym_project/current`。Action engine 只消费 Demo/prompt2scene 工作流已经生成的 `current` Gym 场景。界面中的 “Scene engine” 文案表达的是所需场景类型，并不意味着独立 Scene Engine 输出已自动连到 Action engine。

## Demo：端到端 Gym 场景和 DexSim

Demo 提供 `Auto`、`Interact`、`Parallel Simulation` 三种运行状态，以及图像、任务、场景描述、生成模式、机器人、随机输入、视频和 GLB 预览。它们与顶部的 Demo/Debug 模式无关。

`run_generate()` 是 Demo 的主入口。初始生成会在 staging 场景中运行 prompt2scene/action-agent pipeline，成功后才 promote 为固定的 `current`；随后默认启动 DexSim。编辑和仅改任务复用已有 `current`：

```text
Initial generation
  image + task
  → _gradio_pending_<token>
  → run_agent_pipeline --skip-run-agent
  → fast_gym_config / agent_config / GLB previews
  → promote 到 current
  → run_agent（DexSim）

Edit current scene
  current + task + scene description
  → 编辑 pipeline
  → current
  → run_agent（DexSim）

Change task only
  current + task
  → generate_action_agent_config
  → current
  → run_agent（DexSim）
```

场景生成期间，工作流会从 `fast_gym_config.json` 构建场景 GLB，并将生成的对象 GLB 合并为对象预览。`launch_simulation=False` 是可用的工作流参数，但当前 Debug Scene panel 不调用这条 Demo 工作流；它调用独立的 `run_scene_engine()`。

正式场景固定在：

```text
gym_project/current/
gym_project/current/gym_export/
gym_project/action_agent_pipeline/images/current.png
gym_project/action_agent_pipeline/configs/current/
    fast_gym_config.json
    agent_config.json
    gradio_scene/
        scene_current.glb
        initial_scene.glb
        object_preview.glb
```

初始生成使用 `_gradio_pending_<token>` 路径。提升失败或 pipeline 失败时，已有 `current` 保持不变；成功提升后会重写 staging 中的路径引用。`Reset` 会清理当前场景和 staging 产物；`Stop` 通过进程组终止正在运行的 pipeline 或 DexSim。

## Asset engine

### SimReady：单资产目录适配

SimReady CLI 接收目录，而 Gradio 接收上传文件。上传文件会复制到隔离目录，文件名只保留 basename，重名追加序号，避免上传路径或重名影响处理：

```text
mesh + sidecar files
  → .debug_engine/assets/runs/<token>/input/
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

### Articulation：Articraft + Codex

Articulation 标签页根据文本和可选参考图生成一个可下载的 articulated asset。先点击环境检查：若 `ARTICRAFT_ROOT` 不存在，应用会 clone `ARTICRAFT_REPOSITORY_URL`；随后检查 Conda、指定的 Articraft 环境和 Codex CLI。该操作会创建 checkout 和 `.debug_engine/articraft/` 中的输出目录，现有的非 Articraft 目录不会被覆盖。

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

产物、记录和参考图均在 `ARTICRAFT_OUTPUT_ROOT` 下，不能直接当作 Demo 的 Gym 场景或 SimReady 资产；若要进入后续仿真，需要另行定义并实现转换/导入流程。Articraft Viser 每次成功预览会终止旧的 Articraft 预览进程，再以 `0.0.0.0:<ARTICRAFT_VISER_PORT>` 启动新进程。

## 独立 Scene engine 和 Viser

Scene engine 只接收图像。上传图像会先进行 EXIF 归正并转为 RGB PNG，以 PNG 字节的 SHA-256 前 16 位作为目录名；相同图像会复用同一目录：

```text
image
  → .debug_engine/scenes/<hash>/input.png
  → python -m embodichain scene-engine
       --image <input.png>
       --output_root <hash-dir>
       --config <EMBODICHAIN_ROOT>/embodichain/gen_sim/scene_engine_config.json
  → <hash-dir>/scene_export/scene_config.json
  → preview.py <hash-dir> --viser --viser-host 0.0.0.0 --viser-port 8080
  → Gradio iframe
```

当 `scene_export/scene_config.json` 存在且生成进程返回成功时，应用才启动 Viser。iframe 使用 Gradio 页面当前的协议和主机名转向 Viser 端口，因此从其他设备访问时，浏览器必须能访问该端口。每次新 Scene Engine 任务开始前会终止旧的 Scene Viser 进程。输出目录会显示在 UI 中，便于检查 hash 命名的场景导出。

## Action engine：Gym 场景契约

Action engine 不接收裸 GLB。普通 GLB 只有渲染数据，而 DexSim 还需要碰撞、物理参数、初始位姿、资源相对路径和 action 配置。当前实现的前置条件是：

```text
gym_project/current/gym_export/
gym_project/action_agent_pipeline/configs/current/fast_gym_config.json
gym_project/action_agent_pipeline/configs/current/agent_config.json
```

点击 `Load current scene` 只读取共享状态快照。点击 `Run DexSim` 会先检查任务、`current` 的 Gym/action 配置、运行占用和可导入的 `embodichain.gen_sim.action_agent_pipeline.cli.run_agent`，再以当前配置调用 `run_agent`。它不会因为新的任务文本重建动作图；任务改变时应在 Demo 里使用 `Change task only`，或者实现显式的配置再生成步骤。

运行命令的核心参数为：

```bash
python -m embodichain.gen_sim.action_agent_pipeline.cli.run_agent \
  --task_name current \
  --gym_config <.../fast_gym_config.json> \
  --agent_config <.../agent_config.json> \
  --regenerate --renderer fast-rt --num_envs <1|9>
```

并行模式额外传入 arena 和数据保存过滤参数。`--robot-profile` 仅在通过 `run_agent --help` 探测到该参数时加入。DexSim 完成后会寻找 audience 视频和 LeRobot 数据集；单环境可组合两种预览视频。

## 共享状态、并发和进度

Demo、独立 Scene engine 和 Action engine 共享 `RuntimeState` 与 `runtime_lock`，其中包含运行 token、pipeline/DexSim/Scene Viser 进程、输入、预览、日志、阶段和计时。运行 token 用于丢弃过期线程的更新。Articraft Viser 使用单独的锁和进程引用；SimReady 使用自己的同步 generator。

`demo.queue(default_concurrency_limit=1)` 将队列中的高成本回调串行化。Demo 的 `Timer(2.0)` 与 Action engine 的独立 `Timer(2.0)` 都读取同一共享状态。Scene Engine 和 Demo pipeline 因共享 `is_busy` 互斥；Asset/Articraft 面板不写入这一状态，但仍会受 Gradio 队列限制。

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

Demo/Action 需要 action-agent 模块，特别是：

```text
embodichain.gen_sim.action_agent_pipeline.cli.run_agent_pipeline
embodichain.gen_sim.action_agent_pipeline.cli.generate_action_agent_config
embodichain.gen_sim.action_agent_pipeline.cli.run_agent
```

独立 Scene engine 还需要：

```text
python -m embodichain scene-engine
embodichain/gen_sim/scene_engine/cli/preview.py
embodichain/gen_sim/scene_engine_config.json
```

Articulation 还需要 Git（首次 clone）、Conda、`ARTICRAFT_CONDA_ENV` 和 Codex CLI。生成请求会交给本机 Codex CLI 执行，因此只应提交可信请求。

每次修改后至少执行：

```bash
python -m py_compile \
  gradio_app.py app_config.py app_state.py app_commands.py \
  app_processes.py app_media.py app_workflows.py app_ui.py \
  app_asset_engine.py app_articraft.py app_services.py

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  conda run -n embodichain python -c \
  "from app_ui import build_demo; assert build_demo() is not None"
```

手动检查：

1. SimReady 上传简单网格后能显示输入预览；执行后显示 SimReady 输出或明确错误。
2. Articulation 环境检查能报告 checkout、Conda 和 Codex 状态；成功生成后有 zip、记录目录和 Viser 或明确的预览错误。
3. Scene engine 从图像生成 `scene_export/scene_config.json`，并在 `8080` 显示 Viser；它不应改写 `gym_project/current`。
4. Demo 初始生成成功后才替换 `current`；失败时旧场景仍可用。
5. Action engine 在没有 `current` Gym/action 配置或缺少 CLI 时给出预检错误；任务更新后通过 Demo 的 `Change task only` 重建配置。
6. Demo 的 Auto/Interact/Parallel Simulation 行为不因 Debug 面板切换而改变；Reset/Stop 能终止其对应的进程组。
