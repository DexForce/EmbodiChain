# 技术报告 Benchmark 核心与纯渲染实验

正式纯渲染入口是 `rendering-formal`：公共产物 schema 为 1，实验定义版本为 `2.0`。
它声明 R-01/R-02/R-03/R-04/R-05/R-09 和独立模态消融共 **82 个 cell**，复用公共的
进程隔离、预算、计时、来源追踪和离线统计。声明矩阵不保证全部规模在每个后端都受支持；
未选、未运行、OOM、unsupported、失败和中断都必须保留。

`rendering-preview` 提供自定义共享场景和相机布局的审阅；布局批准不等于画质合格。
`camera-pilot` 与 `rendering-suite` 保留旧实验含义，其中旧 R-05 是模态测试，正式 R-05
是三相机扩展。完整复跑、运行环境、参考图像、质量评估及诊断说明见[渲染操作指南](rendering/README.md)。
当前没有统计和质量均合格的 publication 性能结论。

## 结构和职责

| Owner | 职责 |
|---|---|
| `scripts/benchmark/__main__.py` | 统一 CLI，保留其他 benchmark 入口 |
| `core/contracts.py`, `records.py` | Definition、RunSpec、状态与版本 |
| `core/planning.py`, `execution.py` | 矩阵、预算、独立进程、超时/中断 |
| `core/measurement.py`, `artifacts.py`, `provenance.py` | 计时、标准产物、来源追踪 |
| `reporting/aggregation.py`, `comparison.py` | 分母、缺失值、不确定性及比较资格 |
| `rendering/formal.py`, `formal_runner.py`, `formal_worker.py` | v2 矩阵、profile、批准、选择和运行 |
| `rendering/scenes.py`, `scene_cases.py`, `preview.py`, `replay.py` | 共同资产、固定视角、审阅与 CPU 冻结运动 |
| `rendering/capture.py`, `resources.py`, `diagnostics.py` | 完成交付、资源、trace 与 overhead |
| `rendering/backends/scene_*.py` | 两个平台的原生场景和渲染 API |
| `rendering/quality_protocol.py`, `reference.py`, `quality.py`, `evaluate.py` | 独立质量证据与追加评估事件 |
| `rendering/calibration.py` | 实测 photometric receipt、独立 held-out 验证 |
| `rendering/publication_report.py` | 标准库离线 Markdown/CSV/SVG/PDF |
| `rendering/suite*.py`, `run_benchmark.py`, `report.py` | 兼容 R-series pilot 与 camera-pilot |
| `expert_generation/` | G-03 attempt/receipt、fixture 和生产 session adapter |

表中短路径相对于 `scripts/benchmark/`。依赖方向为领域 launcher/worker → `core`，
领域报告 → `reporting` + 冻结文件协议。Core、shared reporting 和正式离线报告不导入
NumPy、Torch 或模拟器。机器人配置、物理推进和数据质量仍由对应领域 owner 负责。
其他 motion-generation、Atomic Action、RL benchmark 保持原有 owner，可逐步复用公共能力。

## 最小数据流

1. 领域配置解析并冻结共同 workload；通过 `create_experiment_directory` 保存参数。
2. `repeat_schedule` 固定平台交错顺序；launcher 将每个运行解析为 `RunSpec`。
3. `run_experiment` 先保存完整计划及 `not_run` 行，再逐个启动独立 worker。
4. worker 调用 `measure_loop`，保存领域结果、质量状态、样图及 provenance。
5. core 保存退出状态与原始结果索引；报告只消费已经写入的文件。

每次执行包含 `definition.json`、`manifest.json`、`config.json`、
`effective_config.yaml`、`assets_manifest.json`、`runs.json`、`raw.jsonl`、
`metrics.json`、`quality.json` 和 `artifact_index.json`；每个 worker 包含
`result.json` 和 `worker.log`，以及领域附加产物。`manifest.json` 记录实验、
case/run/repeat/attempt 身份、命令、预算和超时；完整环境变量不会写入计划。
运行目录不可复用覆盖，JSON 先验证再原子替换；坏的 worker JSON 保留为
`invalid-worker-result.json`，该运行记为失败。

`effective_config.yaml` 使用 JSON 兼容的 YAML 语法，因此标准库即可离线读取。
`ArtifactStore` 对 `raw.jsonl` 和证据索引采用幂等身份：同一记录重复提交不会增加
分母，不同内容复用同一身份会报错。

## 核心合同

- `status` 只表示运行状态，和领域的 `quality_status`、任务结果及持久化证据独立。
- 超时、SIGINT、SIGTERM 会回收隔离进程组。中断时保留当前结果和其余 `not_run` 行。
- 原始时间使用秒、内存使用字节；报告显示单位时换算。预热不进入测量窗口。
- 操作延迟不包含校验，完整循环吞吐包含校验；GPU 完成语义由 backend 保证。
- 聚合只从已完成运行取有效数值，失败仍保留在计划数量中，缺失值不补 0。
- `MetricDefinition` 明确单位、统计 population、定义版本和分母；公共聚合保留
  median、MAD 不确定性、有效/缺失计数、缺失原因和 source run IDs。零分母输出
  `null` 与显式计数，不转换为零。
- 不同 workload hash 分组，不能把不同分辨率或场景混算一个平台均值。
- 比较检查接收领域声明的不变量；质量、配置、硬件或计时边界缺失/不一致会给出原因。
  核心不会把程序完成或图像非空升级为质量合格，也不会自行认证渲染质量。
- 统计输出记录 metric 的单位、计数对象、有效/缺失运行数及 `median_of_runs` 规则。

`RunSpec` 是已解析的不可变进程记录，不包含仿真对象；用户编写的领域配置仍使用
项目的 `@configclass`。阶段、连续循环和 attempts/episodes 使用各自的测量接口，
而不是用一个 success-rate 字段混合三种 population。核心当前不负责集群、候选生成、
Gym reset 或数据集 commit。

## 正式纯渲染 v2

| 实验 | Cell 数 | 正式含义 |
|---|---:|---|
| R-01 | 8 | calibration/workshop，20 固定视角，N=1/64，两个 profile，RGB/depth/seg 质量采集 |
| R-02 | 4 | Franka 的 100 帧、30 Hz 运动序列；两个三相机刚体 cell 和两个外部形变占位 cell |
| R-03 | 48 | 场景、静态/移动、256²/512²/1920×1080、RGB/完整模态、两个 profile 的性能矩阵 |
| R-04 | 5 | 128² RGB、一个相机，N=1/16/64/256/1024 |
| R-05 | 5 | 相同 N、三个相机；不是旧 pilot 的模态实验 |
| R-09 | 8 | 四种交付边界 × 普通/诊断处理，N=16、三个相机 |
| modality-ablation | 4 | RGB、RGB+depth、RGB+normals、RGB+depth+seg |

只读列出或选择矩阵；dry-run 不读场景资产、不准备 replay、不创建输出目录：

```bash
python -S -m scripts.benchmark rendering-formal --list --experiment R04
python -S -m scripts.benchmark rendering-formal --list --json --show-excluded
python -S -m scripts.benchmark rendering-formal --experiment R04 --num-envs 1 --dry-run
```

在已经审阅并保存 `approval.json` 的场景根目录上运行一个小规模 smoke：

```bash
python -m scripts.benchmark rendering-formal \
  --scene-root /path/to/approved-review \
  --experiment R04 --num-envs 1 --smoke --gpu-id 1 \
  --embodichain-python /path/to/embodichain-python \
  --isaaclab-root /path/to/frozen/IsaacLab \
  --isaaclab-python /path/to/matching/isaaclab-python \
  --output outputs/benchmarks/rendering-formal
```

性能默认三个新进程重复、30 次预热、300 次测量；`--publication` 把选定性能 cell
的重复数提高到至少十个。R-01/R-02 质量采集默认每平台一次。`--smoke` 仅改为
一次重复、两次预热、四次性能测量，不偷偷降低分辨率、N、相机数或质量采集覆盖范围。
过滤、后端选择和 `--max-cases` 限制均写入完整清单；全计划、选定、可运行、已启动与
已完成的数量分别报告。形变输入缺失为 unsupported，不生成替代形变来通过。

每个新 invocation 有 `selection.json`、`index.json`、`acceptance.json`；实验子目录
保留标准产物、批准的 canonical 资产、replay、独立 worker 结果和质量证据。
RTX PRO 5000 上的工作不算 RTX 5090 验收；外部 5090 要求仍需实际执行。
Isaac Lab 是固定版本的外部依赖；本仓库的 Isaac 专用比较代码只放在
`scripts/benchmark/`，不改生产 `embodichain` API 或外部 Isaac 源码来实现 benchmark。

计时和计数由 `capture.py` 定义。设完整窗口为 T、batch 数为 B、环境数为 N、
每环境相机数为 C、输出大小为 W×H：

- `batch_steps_per_s = B / T`；`camera_frames_per_s = B × N × C / T`。
- `gpixel_per_s = B × N × C × W × H / (T × 10^9)`，只计输出像素，不乘模态数，
  也不声称是渲染器内部像素工作量。
- 边界包含需要的状态发布、渲染、选定交付和 GPU completion，不推进物理。
  Latency 不含验证；吞吐窗口包含验证/记录 bookkeeping。初始化、预热、落盘和质量评估在窗外。
- `host_ready` 是一次真实 D2H；`duplicate_d2h` 从原 device buffer 读回两次；
  `host_copy` 是一次 D2H 加额外 CPU copy，两类字节计数分开。
- CPU timer 和 Torch allocator 不是 native kernel timer 或全部 VRAM。NVML PID/device
  字节、CPU RSS 与 Torch allocation 分列，两套运行环境均需可用的 NVML Python 绑定。

R-09 诊断另存 CPU/CUDA trace 和 overhead 窗口，不混入 publication throughput。
CUPTI 不能证明 Vulkan/RTX 全部可见；没有事件不等于没有传输。
高质量 Isaac profile 使用 PT16 且每曝光重置累积；realtime DLAA 在低分辨率可能扩大内部
工作尺寸。读取实际 profile/renderer evidence，不能从最终输出或 atlas 推断内部原生分辨率。

报告和质量评估是独立操作：

```bash
python -S -m scripts.benchmark rendering-formal \
  --report-only /path/to/invocation/R-04/experiment-timestamp

/path/to/reference-python -m scripts.benchmark.rendering.evaluate \
  --experiment-root /path/to/invocation/R-04/experiment-timestamp \
  --spp 256 --threads 4
```

报告重建只依赖标准库。Mitsuba reference、学得的 LPIPS/SqueezeNet 权重和显式
photometric calibration 属于额外 CPU 评估环境；数值完整也不自动合格。
Reference 噪声、光照/颜色标定、声明 metric/check 和时序覆盖必须通过且证据 hash 匹配。
评估追加质量事件并保留原始执行快照，不重写计时值或 `raw.jsonl`。
具体安装、缓存、输出与诊断用法见[操作指南](rendering/README.md)。

### 兼容 pilot

`camera-pilot` 保留桌面/三个方块/单 RGB 相机的最小示例。
`rendering-suite` 保留旧 R-03 分辨率、R-04 小批量、R-05 模态、R-06 静态/移动和 R-09 交付实验。
这些旧入口、配置及报告不升级为 v2，不能用旧 R-05 结果代替正式三相机扩展。

```bash
python -m scripts.benchmark camera-pilot --help
python -S -m scripts.benchmark rendering-suite --list
python -S -m scripts.benchmark rendering-suite --report-only /path/to/legacy-experiment
```

## G-03 最小纵向切片

公共框架的第一条 generation 消费者是无仿真依赖的 G-03 fixture：

```bash
python -m scripts.benchmark expert-generation \
  --fixture --attempts 4 \
  --output outputs/benchmarks/expert-generation
```

它固定一个 Atomic Action 来源 case，演示执行成功、测量验证失败、持久化
receipt 失败和确认写入四类结果。只有 `execution=completed`、
`validation=passed`、`task=passed` 且 receipt `confirmed=true` 的 attempt
进入 accepted yield。重复 `commit_id` 不增加 accepted 数量，预算耗尽的
attempt 保留为 `not_run`。

`session_adapter.py` 已把 #670 的 `GenerationSession` 生命周期接到同一
executor 边界：proposal、rollout evidence、`accept_episode` 和最终 receipt
都会经过生产 session。真实 host 仍需提供 Candidate Coordinator、Physical
Executor、Measured Validator 和 EpisodeSink；fixture 不创建第二套候选或
persistence 状态机。

## 如何接入下一项实验

增加一个领域配置、一个 worker、平台适配器及领域报告模板即可。定义使用
`ExperimentDefinition` / `Budget`，launcher 使用 `build_run_plan` / `RunSpec` /
`run_experiment`，worker 使用 `measure_stages`、`measure_loop` 或
`measure_attempts` 以及 `ArtifactStore`，报告使用 `MetricDefinition` /
`aggregate_runs` / `compare_metric` / `write_technical_report`。旧结果可以先经
`convert_legacy_rows` 转入公共协议。
新的物理任务成功规则放在领域 evaluator，不加入 core；新的数据生成接入生产
协调器及最终回执，不在 benchmark 中重建候选/提交状态机。

## 验证

最小的 CPU/离线合同验证：

```bash
python -m pytest -q -c /dev/null --noconftest \
  tests/benchmark/core tests/benchmark/reporting \
  tests/benchmark/rendering/test_formal_runner.py \
  tests/benchmark/rendering/test_rendering_capture.py \
  tests/benchmark/rendering/test_publication_report.py
```

匹配测试还覆盖 worker 生命周期、资源、trace/overhead、replay、独立 reference 和质量评估。
可选 learned-metric/reference 测试需要额外 CPU 依赖；真实 GPU 验证必须串行。
`python -S` 检查 list/dry-run/report-only 边界。已完成 smoke、声明矩阵和实现测试是不同证据；
目前没有等质量的 publication 性能结果。
