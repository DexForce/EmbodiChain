# GenSim 模型服务部署与接入

本文说明如何为 GenSim 准备模型服务、将服务绑定到
`GenSimServices` 角色，并用现有 CLI 验证连接。文档分开两个运行边界：

- EmbodiChain 运行时负责任务、场景和资产流程，以及服务配置、调用和记录；
- 模型服务运行时负责 LLM/VLM、图像生成、分割、3D 重建和可选的
  关节资产生成。

EmbodiChain 的容器和 wheel 不包含这些模型权重。本文不将服务的 HTTP
可达性视为生成质量或仿真任务成功的证明。

## 1. 服务能力和使用场景

`GenSimServices` 按角色懒加载服务；一次运行不会自动要求所有角色。

| 能力 | 角色 | 默认 adapter | 常见使用场景 |
|---|---|---|---|
| 文本/视觉对话 | `task_interpretation`、`task_image_grounding`、`task_semantic_planning`、`graph_localization`、`scene_*`、`asset_*` | `openai_compatible` | 任务理解、图像定位、场景理解/编辑、SimReady 属性推理 |
| 文本生成图像 | `image_generation` | `image_http` | 需要新图像素材的生成流程 |
| 文本提示分割 | `image_segmentation` | `segmentation_http` | 从场景图像提取对象 mask |
| 几何重建 | `geometry_reconstruction` | `geometry_http` | 从图像和 mask 生成 GLB 及 Y-up 位姿 |
| 关节资产生成 | `articulated_generation` | `articulation_http` | 生成并验证可运动 USDC |
| 关节任务控制 | `articulation_jobs` | `articulation_jobs_http` | Gradio 远程 submit/status/cancel/download |

只需配置实际会使用的角色。如果运行时请求了未绑定角色，GenSim 会显式
报告 `No profile bound to role ...`，不会自动切换到其他模型。

## 2. 推荐拓扑

模型、推理框架、调度系统和容器形式由部署者选择。GenSim 只消费稳定的
能力协议：

```text
任意模型 / 任意部署框架
        │
        ├── 直接实现 GenSim service protocol v1
        │
        └── 安装 adapter plugin 适配现有 API
                    │
              GenSimServices
                    │
          Task / Scene / Asset Engine
```

内置 HTTP 契约见 [GenSim model-service protocol v1](spec/v1/README.md)。用户可以用
Docker Compose、Kubernetes、Ray Serve、Triton、普通 Python 进程或托管 API 实现它；
这些选择不进入 GenSim 配置契约。

可以在同一 GPU 服务器或多台服务器上部署端点。如果 EmbodiChain 运行在
另一个容器中，配置中的 `127.0.0.1` 指向 EmbodiChain 容器自身，而不是
宿主机或模型服务器。此时应使用可达的私网 DNS/IP，或为容器显式配置
host gateway。

无认证服务只能运行在受信网络或防火墙后。公网端点应通过 HTTPS 和
认证入口暴露。

## 3. 准备 EmbodiChain 运行时

安装 GenSim 可选依赖：

```bash
python -m pip install -e '.[gensim]'
```

如果使用项目容器，参考 [`docker/README.md`](../../../docker/README.md)。项目容器包含
EmbodiChain 和 GenSim 业务依赖，但不包含下文的模型权重或服务。

## 4. 部署用户自己的服务

对每个需要的能力，选择下列一种接入方式：

1. 让服务直接实现 [v1 协议](spec/v1/README.md)，使用内置 adapter；
2. 在现有服务前部署一个薄网关，将其 API 映射到 v1；
3. 安装 adapter plugin，在客户端直接转换现有 API。

内置 adapter 不检查模型名称或推理框架。例如，`segmentation_http` 后面可以是
SAM 系列、自研分割模型或一个远程平台，只要它返回契约化的 RLE mask。

对不同线协议的服务，adapter plugin 通过
`embodichain.gensim_service_adapters` entry-point 注册。配置中的 `adapter` 值等于
entry-point 名称；只会导入实际被角色引用的插件。详细注册例子见
[v1 协议的 plugin 章节](spec/v1/README.md#installed-adapter-plugins)。

## 5. 准备 OpenAI-compatible LLM/VLM

GenSim 不绑定具体的 LLM/VLM 服务端。可以使用托管 API，也可以使用自建的
OpenAI-compatible 服务。当前 adapter 要求：

- `POST <base_url>/chat/completions`；如果 `base_url` 本身以 `/chat/completions`
  结尾，则直接使用它；
- 如果 profile 配置了 `api_key`，发送 `Authorization: Bearer <api_key>`；
  无认证的本地服务可以省略该字段；
- OpenAI Chat Completions 格式的 `choices[0].message.content`；
- 对需要图像的角色支持 `image_url` data URL；
- 对结构化请求支持非流式 JSON object 模式。

如果只有文本模型，它可以只绑定 `task_interpretation` 等纯文本角色；
`task_image_grounding`、`task_semantic_planning` 和多个 `scene_*`/`asset_*` 角色需要
能处理图像的端点。

使用提供方自己的 health/model 命令检查模型是否就绪。Chat adapter 没有单独的
`check_health()` 协议；最终需要用第 7 节的真实 GenSim 请求验证多模态和 JSON
兼容性。

## 6. 配置 GenSim 服务角色

下面的最小可用配置将一个多模态 OpenAI-compatible 端点绑定到所有对话角色，
并将三个用户部署的 v1 服务绑定到生成角色。它支持刚性场景流程，不包含关节资产
生成。

将以下内容保存为一个安装外的绝对路径，例如
`/absolute/path/to/model_services.json`：

```json
{
  "model_services": {
    "profiles": {
      "chat": {
        "adapter": "openai_compatible",
        "model": {"env": "GENSIM_CHAT_MODEL"},
        "api_key_env": "GENSIM_CHAT_API_KEY",
        "base_url": {"env": "GENSIM_CHAT_BASE_URL"},
        "timeout_s": 120,
        "max_attempts": 1
      },
      "images": {
        "adapter": "image_http",
        "base_url": {"env": "GENSIM_IMAGE_BASE_URL"},
        "timeout_s": 120,
        "max_attempts": 1,
        "options": {
          "health_path": "/health",
          "generate_image_by_prompt_path": "/generate_image_by_prompt"
        }
      },
      "segmentation": {
        "adapter": "segmentation_http",
        "base_url": {"env": "GENSIM_SEGMENTATION_BASE_URL"},
        "timeout_s": 120,
        "max_attempts": 1,
        "options": {
          "health_path": "/health",
          "segment_by_prompt_path": "/segment_by_prompt"
        }
      },
      "geometry": {
        "adapter": "geometry_http",
        "base_url": {"env": "GENSIM_GEOMETRY_BASE_URL"},
        "timeout_s": 600,
        "max_attempts": 1,
        "options": {
          "health_path": "/health",
          "generate_objects_path": "/generate_multiple_objects",
          "poll_timeout_s": 600,
          "poll_interval_s": 1
        }
      }
    },
    "roles": {
      "task_interpretation": "chat",
      "task_image_grounding": "chat",
      "task_semantic_planning": "chat",
      "graph_localization": "chat",
      "scene_understanding": "chat",
      "scene_edit": "chat",
      "scene_asset_orientation": "chat",
      "scene_asset_dimensions": "chat",
      "asset_orientation": "chat",
      "asset_dimensions": "chat",
      "asset_semantics": "chat",
      "asset_physics": "chat",
      "image_generation": "images",
      "image_segmentation": "segmentation",
      "geometry_reconstruction": "geometry"
    }
  }
}
```

将对应的连接值保存为另一个安装外文件，例如
`/absolute/path/to/gensim-services.env`：

```dotenv
GENSIM_CHAT_MODEL=your-multimodal-model
GENSIM_CHAT_API_KEY=replace-with-a-real-secret-or-local-placeholder
GENSIM_CHAT_BASE_URL=https://llm.example.com/v1

GENSIM_IMAGE_BASE_URL=http://MODEL_HOST:8687
GENSIM_SEGMENTATION_BASE_URL=http://MODEL_HOST:8685
GENSIM_GEOMETRY_BASE_URL=http://MODEL_HOST:8686
```

如果 Chat/VLM 服务不使用认证，从 profile 删除 `api_key_env`，同时不需要
定义 `GENSIM_CHAT_API_KEY`。

保护凭证文件，不要将它提交到 Git：

```bash
chmod 600 /absolute/path/to/gensim-services.env
```

选择这两个文件：

```bash
export EMBODICHAIN_MODEL_SERVICES_CONFIG=/absolute/path/to/model_services.json
export EMBODICHAIN_ENV_FILE=/absolute/path/to/gensim-services.env
```

`EMBODICHAIN_MODEL_SERVICES_CONFIG` 选择的 JSON 必须同时包含 `profiles` 和
`roles`；它不会与其他文件合并。进程环境变量优先于
`EMBODICHAIN_ENV_FILE` 中的同名值。普通 wheel 安装也可以将 JSON 放在
`$XDG_CONFIG_HOME/embodichain/model_services.json`。更多配置选项见
[`model_services.example.json`](model_services.example.json)。

如果任务和场景使用不同模型，复制 `chat` profile，为它们使用不同的
`model`/`base_url`/凭证，然后重新绑定相应角色；不需要修改 Python 代码。

## 7. 验证接入

### 7.1 验证配置和生成服务健康状态

先做离线校验。它解析 JSON、环境变量和 adapter plugin，检查每个角色的能力并
构造/关闭 adapter，但不执行健康检查或向 provider 发送请求：

```bash
embodichain gensim services validate \
  --config /absolute/path/to/model_services.json \
  --env-file /absolute/path/to/gensim-services.env
```

再构造所有已绑定 adapter，并调用它们提供的 `check_health()`：

```bash
embodichain gensim services doctor \
  --config /absolute/path/to/model_services.json \
  --env-file /absolute/path/to/gensim-services.env
```

可以重复使用 `--role image_segmentation` 只检查选中角色，或加 `--json` 获得
机器可读报告。Chat adapter 没有通用健康协议，因此该角色显示为
`initialized`；第 7.2 节才会向 LLM/VLM 发送真实对话。

### 7.2 验证 LLM/VLM 和 Task Engine

准备一张真实场景图片，运行：

```bash
embodichain gensim task plan \
  --image /absolute/path/to/table.png \
  --instruction "Put the cup on the tray" \
  --output outputs/gensim-task-plan
```

输出目录必须事先不存在。成功运行验证了实际对话请求、图像传输、JSON
解析和语义图产出；它不创建场景，也不执行机器人任务。

### 7.3 验证 Scene Engine

```bash
embodichain gensim scene generate \
  --image /absolute/path/to/table.png \
  --output outputs/gensim-scene
```

这一步验证场景理解、分割、几何生成、位姿解析和场景导出。是否请求
`image_generation` 取决于具体流程；未使用的 profile 不会连接或健康检查。

### 7.4 验证图像到执行的完整流程

```bash
embodichain gensim run \
  --config examples/gensim/run.yaml \
  --image /absolute/path/to/table.png \
  --output outputs/gensim-run-001
```

这一步还需要可用的 EmbodiChain/DexSim 仿真运行时、信任的机器人 embodiment 和
已验证执行策略。模型服务成功不保证抓取、放置或最终任务验收成功。详细配置和
输出说明见 [GenSim 命令指南](../README.md) 和 [workflow 示例](../../../examples/gensim/README.md)。

CLI 运行会在输出目录写入 `service_calls.json`，其中包含使用的角色、adapter、
模型、耗时和结果，不包含密钥或 prompt。

## 8. 可选：关节资产生成

EmbodiChain 不随 wheel 提供关节资产生成服务。如果需要
`articulated_generation`，服务端必须实现：

- `GET /health` 返回 `{"ok": true}`；
- `POST /generate_articulation` 接收 JSON prompt 或 prompt + image multipart，返回
  `request_id`；
- `GET /tasks/<request_id>` 返回 `queued`/`running`/`waiting`/`succeeded`/
  `failed`/`cancelled`；
- 成功结果在 `result.artifacts.usdc` 提供相对下载路径。

在 `profiles` 中添加：

```json
"articulation": {
  "adapter": "articulation_http",
  "base_url": {"env": "GENSIM_ARTICULATION_BASE_URL"},
  "timeout_s": 7200,
  "max_attempts": 1,
  "options": {
    "health_path": "/health",
    "generate_path": "/generate_articulation"
  }
}
```

并在 `roles` 中添加：

```json
"articulated_generation": "articulation"
```

Gradio 远程任务控制是独立的可选角色 `articulation_jobs`，还需要
`POST /tasks/<request_id>/cancel` 和 artifact download 能力。只运行 Scene/Asset pipeline
时不需要配置该 UI 角色。

## 9. 常见问题

| 现象 | 检查 |
|---|---|
| `Model service configuration not found` | 确认 `EMBODICHAIN_MODEL_SERVICES_CONFIG` 是存在的绝对路径 |
| `Missing service environment variable` | 确认 `EMBODICHAIN_ENV_FILE` 路径、变量名和非空值 |
| `No profile bound to role ...` | 将该角色加入 `roles`，或选择不需要它的流程 |
| 宿主机 `curl` 成功，容器内失败 | 不要在容器中使用指向宿主机的 `127.0.0.1`；检查 DNS/路由/防火墙 |
| HTTP 404 | 核对 [v1 协议](spec/v1/README.md) 和 profile 中的 endpoint path |
| LLM/VLM 400 | 确认端点支持 `image_url` data URL、`response_format` JSON object 和非流式返回 |
| `SubmissionUnknown` | POST 可能已被服务端接收；不要盲目重试并创建重复任务 |
| SAM3D 轮询超时 | 检查队列、GPU 显存和 `poll_timeout_s`；不要只增加单次 HTTP timeout |
| 服务健康但生成失败 | 查看服务日志、`service_calls.json` 和失败阶段产物；`/health` 不是真实推理测试 |

## 10. 生产化前的必要补齐

将参考服务用于长期或多用户环境前，至少需要：

1. 可从公开或组织内可审计源重建的 Dockerfile，以及固定的镜像 digest、模型版本和权重校验值；
2. 先为当前未声明根目录许可证的参考服务仓库确认代码来源和可分发权，再审查
   SAM License、Z-Image 和权重分发条款，并保留所有上游许可和 notice；
3. 可配置的 GPU/端口/存储，认证或私网隔离，以及 TLS 入口；
4. SAM3D 任务/产物的持久化、过期清理、背压和重启恢复；
5. 每个服务的真实推理 smoke test，以及与 EmbodiChain adapter 共享的 HTTP 契约测试；
6. 队列长度、耗时、失败率和 GPU 显存监控，且日志不记录凭证或原始输入。

任何参考实现的 `/health` 和空闲显存回收都只是运维基础，不代替上述产品化边界。

## 11. 历史参考：`gen_sim_server`

[`MuziWong/gen_sim_server`](https://github.com/MuziWong/gen_sim_server) 中的 SAM3、SAM3D 和
Z-Image 入口符合当前 v1 契约，可用于兼容回归或已有内部环境。它不是
GenSim 的运行时依赖。

该仓库当前依赖未公开的 `gensim-engine-5000pro:wrc` 镜像，且没有提供从
空机器重建镜像所需的 Dockerfile、完整依赖和模型权重。无法获得该镜像的
用户应按第 4 节自行部署兼容服务，不需要 clone 该仓库。

对仍拥有该镜像的环境，旧启动方式保持不变：

```bash
git clone https://github.com/MuziWong/gen_sim_server.git
cd gen_sim_server
bash bash_scripts/start_all.sh
```

默认映射是 SAM3 `:8685`、SAM3D `:8686` 和 Z-Image `:8687`。使用第 6 节的
profile 路径与第 7.1 节的 `validate/doctor` 命令进行接入；不再由 GenSim 文档
要求固定 GPU 编号或部署形式。
