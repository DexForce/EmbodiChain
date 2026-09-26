# 技术报告 Benchmark 核心与首个实验

当前交付是技术报告 benchmark 的公共框架 v1，以及使用它的 `camera-pilot` 相机实验。
它覆盖实验定义、矩阵与预算、隔离运行、三类测量、标准产物、来源追踪、离线统计、
图表导出和比较条件检查。
相机实验同时提供 EmbodiChain/DexSim 和 Isaac Lab 的实现与复跑命令。

## 结构和职责

```text
scripts/benchmark/
  __main__.py                  统一 CLI；保留已有 benchmark 命令
  core/
    contracts.py               ExperimentDefinition、Budget、RunRecord、RawObservation
                                 和 AggregateMetric
    records.py                 RunSpec、执行状态与结果版本检查
    planning.py                参数矩阵展开、重复计划和预算记账
    execution.py               固定重复顺序、独立进程、超时/中断回收
    measurement.py             阶段、连续循环、尝试和资源采样接口
    artifacts.py               标准产物、JSONL 幂等账本和原子 JSON
    provenance.py              软件版本、源文件哈希与设备快照
  reporting/
    aggregation.py             按平台和工作负载聚合；单位/分母/不确定性保留
    comparison.py              检查质量、硬件、配置、计时边界并计算成对比较
    compat.py                  旧 benchmark 结果转为公共记录
    tables.py                  稳定 Markdown 表格
    figures.py                 无绘图库依赖的 CSV/SVG 及共享样式
    report.py                  通用离线 summary/report 重建
  rendering/
    workload.py                相机配置、共同几何场景、RGB 校验及指标
    run_benchmark.py           选择本机两个 Python 环境，构建 RunSpec
    worker.py                  单个实验进程的场景/采集/验证生命周期
    backends/
      embodichain.py           EmbodiChain/DexSim 渲染与主机回读
      isaaclab.py               Isaac Lab 3 对照实现
    report.py                  相机专用报告模板，调用 reporting 公共逻辑
    camera_pilot.json           首个样例实验的参数
  expert_generation/
    contracts.py                G-03 case、attempt、persistence receipt
    runner.py                   有预算的 attempt runner 和 accepted yield
    fixtures.py                 无仿真依赖的协议 fixture
    report.py                   attempt/stage/confirmed-yield 报告
    session_adapter.py          #670 GenerationSession/receipt adapter
    common.py                  v0.1 导入兼容层
```

依赖方向为：领域 launcher/worker → `core`；领域报告 → `reporting` + 文件协议。
`core` 和 `reporting` 只依赖标准库，导入及离线分析均不启动 GPU 或仿真器。
机器人/场景配置、物理推进、图像格式与质量判定仍由领域代码及平台 API 拥有。

已有 motion-generation、Atomic Action、RL 等 benchmark 有各自稳定 owner；
相机是这一核心的首个使用者，后续实验可以按需要迁移共用能力。

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

## 首个实验与 Isaac Lab 复现

实验是程序化桌面与三个方块、一个 RGB 相机。两个 backend 使用同一几何、
相机内参/视角和计时终点：发起渲染到 CPU RGB 可读。具体设置和安装路径参数见
[相机实验说明](rendering/README.md)。

```bash
python -m scripts.benchmark camera-pilot --help
python -m scripts.benchmark camera-pilot \
  --embodichain-python /home/dex/miniconda3/envs/open/bin/python \
  --isaaclab-root /home/dex/workspace/sources/IsaacLab \
  --isaaclab-python /home/dex/workspace/sources/IsaacLab/env_isaaclab/bin/python \
  --repeats 3
```

Isaac adapter 在独立解释器里启动 Kit，使用本机固定 Isaac Lab 3 提交的
`ProxyArray.torch` 取得图像，并推进 render generation，确保静态物理状态下
每次仍产生新曝光。相机位置变化检查会拦截重复读取旧缓冲的错误。

首个样例尚未完成独立画质标定，保持 `not_qualified`，报告不计算等画质加速比。
旧版相机 `runs.json` 可以继续重建，旧导入 `rendering.common` 也保留兼容。

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

```bash
python -m pytest tests/benchmark/core tests/benchmark/reporting \
  tests/benchmark/rendering/test_camera_pilot.py tests/test_main.py -q
```

这些检查覆盖纯标量操作计时、JSON 原子写入、错误 worker 输出、计划顺序与失败保留、
分组/缺失指标、比较资格，以及相机的配置和命令兼容。离线重建可用 `python -S`
验证完全不依赖 site-packages。真实 GPU 测试使用相机实验命令，串行运行两个平台。
