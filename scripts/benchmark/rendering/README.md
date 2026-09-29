# Pure-rendering suite and camera pilot

第一阶段收缩为一个可复跑的小实验：同一程序化桌面/三个方块、一个 RGB
相机，分别在 EmbodiChain/DexSim 与已安装的 Isaac Lab 中运行。

实验使用[公共 benchmark 核心](../README.md)的进程、计时、产物和统计能力。
领域配置/场景位于 `workload.py`，两套实现位于
`backends/embodichain.py` 和 `backends/isaaclab.py`，相机报告模板位于 `report.py`。
完整 R 系列矩阵由 `suite.py` 和 `suite_runner.py` 承载；`camera-pilot` 是保留的最小兼容样例。

## R-series suite

`python -m scripts.benchmark rendering-suite --list` 显示冻结的 case：

| ID | Cell 参数 | 观测边界 |
|---|---|---|
| R-03 | `resolution=128x128, 256x256, 512x512` | RGB host readback |
| R-04 | `num_envs=1, 4, 16` | RGB host readback |
| R-05 | `modalities=rgb, rgb_depth, rgb_normals` | RGB/depth/normal host readback |
| R-06 | `temporal_mode=static, moving` | RGB host readback 和 freshness |
| R-09 | `delivery=render_only, host_readback, duplicate_readback` | device sync / host copy |

运行全部 R-series 的 smoke 矩阵：

```bash
python -m scripts.benchmark rendering-suite \
  --experiment all --smoke \
  --embodichain-python /path/to/embodichain-python \
  --isaaclab-root /path/to/IsaacLab \
  --isaaclab-python /path/to/IsaacLab/env_isaaclab/bin/python \
  --output outputs/benchmarks/rendering-suite-smoke
```

`--smoke` 使用 1 个 warm-up、3 个测量帧；正式矩阵去掉该参数，并按显存预算增加
`--repeats` 和 `--measured-frames`。`--experiment R-04` 可单独运行一项。
每个结果目录包含 `definition.json`、`config.json`、`plan.json`、`manifest.json`、
`runs.json`、`raw.jsonl`、`metrics.json`、`quality.json`、`summary.json`、`metrics.csv`、
`observations_per_s.svg` 和 `report.md`；每个 worker 另有 `result.json`、`worker.log`、
样本或模态数组。

所有 R-series 的两个后端使用同一程序化桌面/三个方块和相机内参。R-04 在 EmbodiChain
使用 arena camera groups，在 Isaac Lab 使用独立 USD camera views 聚合为一个 batch；
该差异被记录在 provenance 中。R-05 只比较两个平台共同支持的 RGB、depth 和 normals。
R-09 明确记录 render calls、GPU sync、host readback 次数与字节数。

离线重建任意 R-series 结果：

```bash
python -S -m scripts.benchmark rendering-suite \
  --report-only outputs/benchmarks/rendering-suite-smoke/R-04/<timestamp>
```

报告只读取保存的 JSON，生成 Markdown、CSV 和 SVG，不启动 Torch、DexSim 或 Isaac Lab。

## 测量合同

- 默认 256×256、一个环境、一个相机；每进程预热 30 次、测量 300 次。
- 两平台各 3 个新进程，按 E/I、I/E、E/I 交错，串行使用同一 GPU。
- 计时从发起相机渲染开始，到规范化的 HWC uint8 RGB 已完成 CPU 回读为止。
  这是同步观测获取延迟，包含读回成本；物理不步进，初始化和图片写盘不计入。
- P50/P95 来自每次 capture；camera-frames/s 使用包括循环检查在内的完整窗口，
  不是把 P50 简单取倒数。没有把 warm-up 加进分母。
- 渲染前用相机位置变化检查图像是否更新，测量后检查样图非空。
  这是可用性检查，不等同正式图像质量认证。
- 规范几何、视角、FOV、材质色值相同；两个渲染器的光照单位、环境光和着色实现
  尚未标定，结果保留 `quality_status=not_qualified`，不自动生成等质量加速比。
- CPU RSS 峰值包含初始化。GPU device memory 是前后快照，包含其他进程；
  Torch allocator 峰值单列，不能当作渲染器全部显存。

## Isaac Lab 对照环境

相机对照固定使用下面这组版本。`Isaac Lab` 源码版本和 Python distribution
版本同时记录，是因为该开发分支的源码 `VERSION` 与已安装包的 metadata 版本号并不相同。

| 项目 | 对照值 |
|---|---|
| Isaac Lab release line | `release/3.0.0-beta2.patch1` |
| Isaac Lab tag | `v3.0.0-beta2.patch1` |
| Isaac Lab commit | `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Isaac Lab source `VERSION` | `3.0.0` |
| installed `isaaclab` package | `6.1.14` |
| Isaac Sim | `6.0.1.0` |
| Python | `3.12.14` |
| PyTorch / CUDA runtime | `2.10.0+cu128` / `12.8` |

该版本组合面向 Linux x86_64，并需要能启动 Isaac Sim 的 NVIDIA 驱动和 GPU。
Isaac Sim 是 Isaac Lab 的运行时前提；请先按 [Isaac Lab 3.0.0-beta2 的安装文档](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html#local-installation)
完成 Isaac Sim 安装，再安装 Isaac Lab。两个项目使用独立 Python 环境，不能把 Isaac
Lab 的依赖装进 EmbodiChain/DexSim 环境。

### 新机器安装

下面的命令展示版本固定和核心依赖安装；`ISAACLAB_ROOT` 应指向 Isaac Lab
源码目录，Isaac Sim 本体按上面的官方安装文档准备：

```bash
ISAACLAB_ROOT=/path/to/IsaacLab

git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_ROOT"
cd "$ISAACLAB_ROOT"
git checkout v3.0.0-beta2.patch1
./isaaclab.sh -i core
./isaaclab.sh -p -c 'import sys; print(sys.version)'
```

如果已存在工作树，先执行下面的检查，确保没有使用其他分支或未记录的修改：

```bash
cd "$ISAACLAB_ROOT"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = \
  ffff603eafc6b74264a5261cc0183d6a65390d78
cat VERSION
"$ISAACLAB_ROOT/env_isaaclab/bin/python" -c \
  'import sys; from importlib.metadata import version; print(sys.version.split()[0]); print(version("isaaclab")); print(version("isaacsim"))'
```

若使用预装环境，只需要把 `--isaaclab-root` 和 `--isaaclab-python` 指向该工作树
及其 `env_isaaclab/bin/python`；benchmark launcher 会通过 `isaaclab.sh -p`
启动 worker，并清除继承的 Conda 环境变量，避免跨环境导入。

## 执行

在 EmbodiChain 工作树根目录，用能够运行 DexSim 的 Python：

```bash
EMBODICHAIN_PYTHON=/path/to/embodichain-python
ISAACLAB_ROOT=/path/to/IsaacLab
ISAACLAB_PYTHON="$ISAACLAB_ROOT/env_isaaclab/bin/python"

"$EMBODICHAIN_PYTHON" -m scripts.benchmark camera-pilot \
  --embodichain-python "$EMBODICHAIN_PYTHON" \
  --isaaclab-root "$ISAACLAB_ROOT" \
  --isaaclab-python "$ISAACLAB_PYTHON" \
  --repeats 3 \
  --output outputs/benchmarks/camera-pilot
```

其中 `EMBODICHAIN_PYTHON` 是能导入 DexSim 的 EmbodiChain 环境。

先做单次 smoke run：

```bash
"$EMBODICHAIN_PYTHON" -m scripts.benchmark camera-pilot \
  --backend isaaclab \
  --isaaclab-root "$ISAACLAB_ROOT" \
  --isaaclab-python "$ISAACLAB_PYTHON" \
  --repeats 1 \
  --output outputs/benchmarks/camera-pilot-smoke
```

通过后再运行完整的交错对照。Isaac adapter 使用该提交中的相机 API
（包括 `ProxyArray.torch`），实际源码提交、包版本和硬件信息会写入每个
`result.json`，方便复核。

只跑一个平台时使用 `--backend embodichain` 或 `--backend isaaclab`。
`--config` 可以指定同字段的新 JSON；上述限制只适用于保留的 `camera-pilot` 入口，
R-series suite 通过矩阵显式覆盖批量相机、模态和交付边界。
`--timeout-s` 限制每个进程的总墙钟时间，默认 300 秒，包含首次启动。

每次命令创建新的时间戳目录。输出包含：

```text
config.json
manifest.json                # 固定计划、case/run/repeat 身份与执行命令
runs.json                    # 每个计划运行都有状态；失败不被静默丢弃
summary.json
report.md
r00_embodichain/
  result.json                # 原始 capture_latency_s、配置/源码哈希、版本和硬件
  worker.log
  sample.png
  probe_original.png
  probe_moved.png
r00_isaaclab/
  ...
```

worker 超时/崩溃会留下日志和失败 result，父命令继续其余运行，最后以非零退出码提示。
SIGINT/SIGTERM 会停止并回收当前进程组，保存中断与后续未运行状态。
原始失败日志保留，修复后运行产生新目录，不覆盖旧记录。

## 离线重建

将下面路径换为命令输出的实际时间戳目录：

```bash
python -m scripts.benchmark camera-pilot \
  --report-only outputs/benchmarks/camera-pilot/实际运行目录
```

报告重建只使用标准库和已保存 JSON，不导入 Torch、DexSim 或 Isaac Lab。
报告按平台及 workload hash 分组，区分原始计时、失败/检查证据和独立重复的中位数。
`summary.json` 同时保存指标的原始单位、计数对象、有效和缺失运行数。

## 验证

CPU 测试覆盖预热排除、RGB 形状、无效配置、失败/超时保留，以及无仿真依赖的报告入口：

```bash
python -m pytest tests/benchmark/core tests/benchmark/reporting \
  tests/benchmark/rendering/test_camera_pilot.py tests/test_main.py -q
```

真实 GPU 验证使用上面的 pilot 命令，并检查样图、相机变化探针和每个 result.json。
短 pilot 的数值用于初步观察和框架验收；完整技术报告仍需更长测量、质量标定与正式冻结配置。
