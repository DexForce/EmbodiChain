# Camera pilot v0.2

第一阶段收缩为一个可复跑的小实验：同一程序化桌面/三个方块、一个 RGB
相机，分别在 EmbodiChain/DexSim 与已安装的 Isaac Lab 中运行。

实验使用[公共 benchmark 核心](../README.md)的进程、计时、产物和统计能力。
领域配置/场景位于 `workload.py`，两套实现位于
`backends/embodichain.py` 和 `backends/isaaclab.py`，相机报告模板位于 `report.py`。
生成实验和完整 R 系列矩阵保留为后续工作。

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

## 执行

在 EmbodiChain 工作树根目录，用能够运行 DexSim 的 Python：

```bash
/home/dex/miniconda3/envs/open/bin/python -m scripts.benchmark camera-pilot \
  --embodichain-python /home/dex/miniconda3/envs/open/bin/python \
  --isaaclab-root /home/dex/workspace/sources/IsaacLab \
  --isaaclab-python /home/dex/workspace/sources/IsaacLab/env_isaaclab/bin/python \
  --repeats 3 \
  --output outputs/benchmarks/camera-pilot
```

这些路径是本机示例，可用对应参数替换。Isaac worker 经安装目录的
`isaaclab.sh -p` 启动，使用明确的虚拟环境；两个仿真器不装进同一个环境。
Isaac adapter 针对本机提交 `ffff603eafc6b74264a5261cc0183d6a65390d78`
的 Isaac Lab 3 相机 API（`ProxyArray.torch`）。实际包版本和提交另存于结果。

只跑一个平台时使用 `--backend embodichain` 或 `--backend isaaclab`。
`--config` 可以指定同字段的新 JSON；当前第一版只支持单相机/单环境。
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
