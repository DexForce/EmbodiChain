# Codex mesh 部件分割与抓取区域评分

输入 mesh、物体描述和任务描述，输出原始顶点顺序的 `[0, 1]` 接触适宜度。
通过 `--target-part '把手'` 指定部件后，同时输出独立的把手分割分数、掩码与子网格。
把手根部即使抓取分数低，也可以属于把手；可抓取的杯身不会因此被归入把手。
默认使用 OpenAI 的 `gpt-6-astra`，也支持切换 DeepSeek 的视觉模型。
两个提供方都通过 Codex CLI 运行，保留图片输入、只读工具和 JSON Schema 输出。
OpenAI 模式复用 Codex 登录，不需要单独配置 OpenAI API key；DeepSeek 使用本地配置文件中的密钥。
相关接口见 [Codex 非交互模式官方文档](https://learn.chatgpt.com/docs/non-interactive-mode)。

## 运行

环境需要 `bpy >= 3.6`、NumPy、SciPy、Trimesh、Pillow；这些依赖也列在
`mesh-affordance` 可选依赖组中。Blender 和 Codex 分别在独立子进程中执行。

```bash
conda activate embodichain2
codex login
python -m embodichain.toolkits.mesh_affordance run \
  --mesh CoffeeCup/cup.obj \
  --object '马克杯' --task '倒水' --target-part '把手' \
  --patch-count 96 --output outputs/mesh_affordance/cup_handle_new
```

需要支持 Astra 的新版 Codex CLI；本机已升级到 `0.155.1` 并保留 ChatGPT 登录。
可用 `--codex-executable /path/to/codex` 显式指定安装路径。程序不会自动替换模型。

```python
from embodichain.data import get_data_path
from embodichain.toolkits.mesh_affordance import (
    MeshAffordanceCfg,
    analyze_mesh_affordance,
)

result = analyze_mesh_affordance(
    get_data_path("CoffeeCup/cup.obj"),
    object_description="马克杯",
    task_description="倒水",
    output_dir="outputs/mesh_affordance/cup_handle_python",
    cfg=MeshAffordanceCfg(
        model="gpt-6-astra",
        target_part="把手",
        patch_count=96,
    ),
)
scores = result.scores          # float32 [N]
selected = result.graspable_mask  # bool [N]
handle_scores = result.part_scores  # float32 [N]：属于把手的分数
handle_mask = result.part_mask      # bool [N]：把手分割
```

## 切换 DeepSeek

DeepSeek 默认使用 `deepseek-flash`，支持图片输入；仍由 Codex harness 调用其
OpenAI 兼容的 Responses API。参考 [视觉文档](https://api-docs.deepseek.com/zh-cn/guides/vision/)
和 [Responses API 文档](https://api-docs.deepseek.com/zh-cn/guides/responses_api/)。

仓库根目录的 `.mesh_affordance.local.json` 保存本地连接信息，并已加入 `.gitignore`。
本机已写入用户提供的密钥，文件权限为 `0600`。其他机器可复制
`deepseek.example.json` 模板并填写自己的密钥：

```json
{
  "base_url": "https://api.deepseek.com",
  "api_key": "REPLACE_WITH_YOUR_DEEPSEEK_API_KEY",
  "model": "deepseek-flash"
}
```

```bash
conda activate embodichain2
python -m embodichain.toolkits.mesh_affordance run \
  --provider deepseek \
  --provider-config .mesh_affordance.local.json \
  --mesh CoffeeCup/cup.obj \
  --object '马克杯' --task '将杯中的水倒出' --target-part '把手' \
  --patch-count 96 --output outputs/mesh_affordance/cup_handle_deepseek_new
```

模型选择优先级：显式 `--model` > 本地文件中的 `model` > `deepseek-flash`。
省略 `--provider` 时继续使用 OpenAI；Python 中 `model=None` 表示选择提供方默认模型。
已知不支持视觉的 `deepseek-v4-pro`、`deepseek-chat` 和 `deepseek-reasoner` 会提前报错。

Python 配置方式：

```python
cfg = MeshAffordanceCfg(
    provider="deepseek",
    provider_config=".mesh_affordance.local.json",
    target_part="把手",
    patch_count=96,
)
```

密钥只传入 Codex 子进程环境，不写入命令行、提示词、运行配置或结果文件，
也不更改全局 Codex 配置。模型的 shell 工具不继承该密钥变量。
输出 `report.json` 会记录 `provider`、实际 `model` 和 `harness="codex"`。
如果 Codex 提示缺少 DeepSeek 模型元数据，这是其模型目录兼容性提示；
实际的图片请求仍使用指定的 DeepSeek 模型，不会退回 OpenAI。
DeepSeek 偶尔会在 JSON 外添加 Markdown 围栏；解析器只允许剥离单个外层围栏，
仍严格校验所有字段、patch ID 覆盖和分数范围，原文保存在 `response.raw.txt`。

已有**仅准备、尚未完成评分**的证据也可以切换后端：

```bash
python -m embodichain.toolkits.mesh_affordance score \
  --output outputs/mesh_affordance/cup_review \
  --provider deepseek --provider-config .mesh_affordance.local.json
```

切换提供方时，如果未显式传 `--model`，会重新按上述优先级选择模型，
避免把旧证据配置中的 Astra 模型名发送给 DeepSeek。已完成的结果不会被覆盖。

## 输出

| 文件 | 内容 |
| --- | --- |
| `affordance.npz` | `vertices[N,3]`, `faces[F,3]`, `vertex_ids[N]`, `scores[N]`, `confidence[N]`, `valid_mask[N]`, `graspable_mask[N]`, `face_patch_ids[F]`, `face_scores[F]`, `patch_scores[K]`, `patch_confidence[K]` |
| `affordance.ply` | 原坐标和顶点顺序的彩色 mesh，蓝色低分、红色高分 |
| `heatmap.png` | 8 个视角的固定 0–1 色标热力图 |
| `segmentation.png`, `segmentation.ply` | 指定目标部件时，红色为目标部件、灰色为其他区域 |
| `target_part.ply`, `target_part.npz` | 目标部件子网格；NPZ 的 `source_vertex_ids`、`source_face_ids` 映射回原 mesh |
| `report.json` | 任务解释、语义区域、分数、理由、假设和阈值 |
| `evidence.json`, `geometry.npz`, `evidence_*.png` | 输入索引、归一化变换、表面块及模型看到的多视角证据 |
| `response.json`, `response.raw.txt`, `prompt.txt`, `codex_command.json`, `*.log` | 规范化 JSON、原始回答及运行记录 |

```python
import numpy as np

data = np.load("outputs/mesh_affordance/cup_pour/affordance.npz")
indices = np.flatnonzero(data["graspable_mask"])
positions = data["vertices"][indices]
# 或自行调整阈值，confidence 与 score 是两个独立量：
mask = data["valid_mask"] & (data["scores"] >= 0.75) & (data["confidence"] >= 0.6)
```

指定目标部件时，`affordance.npz` 还含 `part_scores[N]`、`part_confidence[N]`、
`part_mask[N]`、`part_face_mask[F]`、`patch_part_scores[K]`、`patch_part_confidence[K]`。
`part_mask` 默认选取 `part_scores >= 0.5` 且 `part_confidence >= 0.5` 的有效顶点；
`--part-threshold` 可调整部件阈值。`target_part.ply` 根据 `part_face_mask` 提取整三角形，
其边界顶点集合可能与按面积加权的 `part_mask` 略有不同，NPZ 保留精确映射。
如果目标部件不存在，部件掩码可全为 False；输出空 `target_part.npz`，不生成 PLY。

OBJ 的 `scores[i]` 对应原文件第 `i+1` 个 `v` 条目，不对应某些带材质导入器拆分后的
顶点。示例杯子原有 2,451 个顶点；常规材质加载可能产生更多顶点。其他格式以输出的
`vertices` 和 `faces` 为准。只接受三角形 OBJ；多边形 OBJ 应先三角化。
未参与任何面的顶点分数为 0、`valid_mask=False`。

## 方法和边界

1. 在表面邻接图上按测地距离和弯折代价分块，默认 64 块；相邻薄壁的两侧不会因
   欧氏距离接近而直接连通。精确重合的 seam 顶点只在邻接计算时合并，输出索引不变。
2. Blender 生成 8 个正交视角，每个视角有灰模和彩色编号块。标签位置经过射线遮挡检查。
3. Codex 结合物体、任务、图像和几何表，为每块给出适宜度、主观置信度和理由。
   指定 `target_part` 后，另行判断该块属于目标部件的程度，完整分割部件而非只选最佳抓取点。
   服务端认证、Astra 权限或格式校验失败会报错，不会回退到手写“杯柄高分”规则。
4. 面分数按面积加权到顶点，默认选中 `score >= 0.7` 且 `confidence >= 0.5` 的有效顶点。

这是任务条件下的接触偏好，不是经过校准的抓取成功概率。默认没有指定夹爪开口、
摩擦、质量、重力方向和场景障碍，也不做力闭合、碰撞或运动学验证。分块边界和面面积
加权会影响局部精度，边界顶点可能取到中间分数。可用 `--patch-count 128` 增加空间
分辨率，同时检查编号是否仍清楚。模型输出具有非确定性，置信度也是主观估计。

“倒水”有倒出/倒入的歧义，`report.json` 会记录模型采用的解释；需要更明确时使用
“握住马克杯，将杯中的水倒入另一个容器”。

## 分阶段运行

```bash
# 仅准备证据，不调用模型。
python -m embodichain.toolkits.mesh_affordance prepare \
  --mesh CoffeeCup/cup.obj --object '马克杯' --task '倒水' --target-part '把手' \
  --output outputs/mesh_affordance/cup_review

# 检查证据后调用 Codex，也可用于失败后的重试。
python -m embodichain.toolkits.mesh_affordance score \
  --output outputs/mesh_affordance/cup_review
```

`--model`、`--threshold`、`--min-confidence`、`--timeout` 可覆盖设置。
渲染可指定 `--resolution`、`--blender-python`。每次新任务使用新输出目录；已有完整
结果不覆盖。推理前会检查几何哈希，修改 mesh 后需要重新准备证据。
