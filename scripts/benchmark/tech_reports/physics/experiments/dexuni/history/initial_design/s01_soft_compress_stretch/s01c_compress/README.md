> 历史设计快照，非当前执行计划。当前安排见[PLAN](../../../../PLAN.md)。旧 M04 棱台方案已被折衣任务替换。

# S01-C 软块压缩与恢复（第 2 轮试跑）

状态：试跑（pilot round 2），配置未冻结；只做 5 % 目标档（10 % 档按用户 2026-10-20 决定暂缓）。

- 第一阶段 DexSim 对 Isaac Lab：数据在 `raw_data/`，结果见 results/S01C_REPORT.md（历史路径：`./results/S01C_REPORT.md`）。
- 第二阶段 EmbodiChain 对 Isaac Lab（当前对比）：数据在 `embodichain_vs_isaaclab/`，结果见 embodichain_vs_isaaclab/results/S01C_EC_REPORT.md（历史路径：`./embodichain_vs_isaaclab/results/S01C_EC_REPORT.md`）；EmbodiChain 与 DexSim 的一致性检查在 `parity_embodichain/`。设置取舍见 results/S01C_DECISIONS.md（历史路径：`./results/S01C_DECISIONS.md`）。

## 目录

| 路径 | 内容 |
|---|---|
| `s01c_dexsim.py` | DexSim 独立脚本（DexUni two-way + dynamic），只读 `inputs/` |
| `s01c_isaaclab.py` | Isaac Lab 独立脚本（Newton MuJoCo-Warp + VBD CouplerProxy，OVRTX 渲染），只读 `inputs/` 与转换好的 Franka USD |
| `s01c_embodichain.py` | EmbodiChain 独立脚本（EmbodiChain `SimulationManager` + DexUni two-way），只读 `inputs/` |
| `inputs/` | 两平台共用的数据：`s01c_config.json`（唯一参数来源）、`s01c_soft_block_mesh.npz`、`s01c_plate.json`、`s01c_initial_states.npz`（10 份初态及实际节点坐标）、`s01c_ik_table.npz`、`inputs_manifest.json`（SHA256） |
| `tools/prepare_inputs.py` | 由配置生成 `inputs/` 下的数据文件（DexSim 解释器；IK 用 DexSim IK） |
| `tools/run_matrix.sh` | 按组顺序运行全部 run，平台交错，不并发；`PLATFORMS`（默认 `embodichain isaaclab`）、`OUT_ROOT`（默认 `embodichain_vs_isaaclab`） |
| `tools/analyze.py` | 只读 `<root>/raw_data/`，生成 `<root>/results/` 和唯一的导出视频 `<root>/videos/s01c_comparison.mp4`；`--root`、`--platforms` |
| `tools/compare_models.py` | 逐数组比较两个 run 的最终 Newton 模型（`newton_model.npz`） |
| `tools/parity_embodichain.py` | DexSim 与 EmbodiChain 一致性检查，读 `parity_embodichain/raw_data/` |
| `embodichain_vs_isaaclab/` | 第二阶段的 `raw_data/`、`results/`、`videos/` |
| `parity_embodichain/` | 一致性检查的 DexSim / EmbodiChain run 和 `parity_summary.json` |
| `raw_data/{dexsim,isaaclab}/<run_id>/` | 每个 run 的原始数据，见下文“输出” |
| `videos/s01c_comparison.mp4` | 唯一导出视频：2×2 对比（见“输出”） |
| `raw_data/_logs/` | `run_matrix.sh` 每个 run 的终端输出（`analyze.py` 不读取） |

三个平台脚本互不导入。闭环下压规则 `CompressionController` 在三个文件中逐字相同，`tools/analyze.py` 用 AST 哈希检查（结果写入 `results/summary.json` 的 `controller_source`）。

## 运行环境

| 平台 | 解释器 | 版本（2026-10-21 实测） |
|---|---|---|
| DexSim | `/home/oem/code/engine/dexsim/.env-3.12/bin/python` | DexSim 0.5.1（本仓库 `install-fast` 安装）、Newton 1.7.0、Warp 1.17.0 |
| EmbodiChain | `/home/oem/code/repos/EmbodiChain/.venv/bin/python`（Python 3.11） | EmbodiChain 分支 `codex/adapt-latest-dexsim`（含本任务需要的未提交修改，见 EmbodiChain 仓库 `DEXSIM_INTEGRATION_REPORT.md` §7）、DexSim 0.5.1+dfceb00eb（`dexsimforembodychain` 源码编译的 wheel，与本仓库源码一致）、Newton 1.7.0、Warp 1.17.0 |
| Isaac Lab | `~/app/IsaacLab/.venv/bin/python` | Isaac Lab v3.0.0（`~/app/IsaacLab/source`，commit e3f7d9f；包版本 isaaclab 28.0.0、isaaclab-newton 9.1.4、isaaclab-contrib 3.2.0）、Newton 1.6.1、mujoco-warp 3.12.0、Warp 1.17.0、torch 2.11.0+cu130；已装官方 extras `importers`、`video` |

GPU：RTX 5090 D v2（cuda:0）。两平台都无头运行，不开窗口。

所有命令都在仓库根目录执行：

```bash
cd /home/oem/code/repos/EmbodiChain
P=dexuni_benchmark/s01_soft_compress_stretch/s01c_compress
DEX=/home/oem/code/engine/dexsim/.env-3.12/bin/python
ISAAC=~/app/IsaacLab/.venv/bin/python
EC=/home/oem/code/repos/EmbodiChain/.venv/bin/python
```

Isaac Lab 相机用 OVRTX，需要一次 `uv pip install --python ~/app/IsaacLab/.venv/bin/python "ovrtx==0.5.1.385782" --index-url https://pypi.org/simple`（即官方 extra `ov[ovrtx]`）。

## 一次性准备

只在机器人 URDF 或 `inputs/s01c_config.json` 改动后需要重做。

```bash
# 1. Isaac Lab 用的 Franka USD（URDF 改动后；输出 dexuni_benchmark/assets/generated/fr3_franka_hand/）
$ISAAC dexuni_benchmark/assets/convert_franka_usd.py

# 2. 共用输入数据（配置改动后；重写 inputs/ 下的 npz/json 与 inputs_manifest.json）
$DEX $P/tools/prepare_inputs.py
```

`prepare_inputs.py` 会打印网格规模、10 份初态的偏移/转角和 IK 误差（当前：281 行，最大 TCP 误差 2.0e-6 m）。

## 复现全部结果

第二阶段（EmbodiChain 对 Isaac Lab，约 35 分钟）：

```bash
bash $P/tools/run_matrix.sh            # 全部组：bench nominal states variants timing
$DEX $P/tools/analyze.py --root embodichain_vs_isaaclab --platforms embodichain,isaaclab
```

EmbodiChain 与 DexSim 一致性检查（每次改 EmbodiChain 或 DexSim 后重做）：

```bash
for a in "--state s00 --images full" "--state s05 --images keyframes"; do
  $DEX $P/s01c_dexsim.py $a --out-root $P/parity_embodichain
  $EC  $P/s01c_embodichain.py $a --out-root $P/parity_embodichain
done
$DEX $P/tools/parity_embodichain.py
```

第一阶段（DexSim 对 Isaac Lab）：

```bash
PLATFORMS="dexsim isaaclab" OUT_ROOT=. bash $P/tools/run_matrix.sh
$DEX $P/tools/analyze.py
```

也可以只跑某些组，例如 `bash $P/tools/run_matrix.sh states timing`。每次运行的日志在 `raw_data/_logs/<tag>_<platform>.log`。各组内容：

| 组 | 每个平台的运行 | 对应命令参数 |
|---|---|---|
| `bench` | 无机器人材料台架，VBD 10 / 40 次迭代 | `--scenario bench --bench-iterations {10,40} --images keyframes` |
| `nominal` | 标称初态 s00，完整图像 | `--state s00 --images full` |
| `states` | 扰动初态 s01–s09 | `--state s0N --images keyframes` |
| `variants` | s00 的 5 / 20 子步与 CUDA Graph 关闭 | `--state s00 --variant {substeps5,substeps20,direct} --images keyframes` |
| `timing` | s00 纯计时，3 个独立进程 | `--state s00 --kind timing --repeat {1,2,3}` |

同一组合重复运行时，`analyze.py` 使用最新的一次，旧的列在 `summary.json` 的 `superseded_runs`。

## 单次运行

三个脚本参数相同，下面每条把 `$DEX $P/s01c_dexsim.py` 换成 `$EC $P/s01c_embodichain.py` 或 `$ISAAC $P/s01c_isaaclab.py` 即为对应平台。第二阶段的 run 加 `--out-root $P/embodichain_vs_isaaclab`。

```bash
$DEX $P/s01c_dexsim.py                                        # s00、nominal、完整采集、完整图像
$DEX $P/s01c_dexsim.py --state s03 --images keyframes          # 扰动初态 s03
$DEX $P/s01c_dexsim.py --variant substeps20 --images keyframes # 20 子步
$DEX $P/s01c_dexsim.py --variant direct --images keyframes     # 关闭 CUDA Graph
$DEX $P/s01c_dexsim.py --kind timing --repeat 1                # 纯计时（无相机、无逐帧拷贝）
$DEX $P/s01c_dexsim.py --scenario bench --bench-iterations 40  # 材料台架

$EC $P/s01c_embodichain.py --state s03 --images keyframes     # EmbodiChain 同一 run
$ISAAC $P/s01c_isaaclab.py --state s03 --images keyframes      # Isaac Lab 同一 run
```

| 参数 | 取值 | 含义 |
|---|---|---|
| `--scenario` | `task`（默认）/ `bench` | 压缩任务 / 无机器人材料台架（机器人停在远处） |
| `--state` | `s00`（默认）–`s09` | `inputs/s01c_initial_states.npz` 中的初态 |
| `--variant` | `nominal`（默认）/ `substeps5` / `substeps20` / `direct` | 配置 `variants` 中的预设覆盖 |
| `--kind` | `full`（默认）/ `timing` | 完整记录 / 只计时 |
| `--images` | `full`（默认）/ `keyframes` / `none` | 深度与分割：每个 30 Hz 帧 / 只在阶段切换与末帧 / 不开相机；RGB 视频在 `full`、`keyframes` 下都是每帧 |
| `--repeat` | 整数，默认 0 | 写入 run_id，区分独立重复 |
| `--bench-iterations` | 整数，默认 40 | 仅 bench：VBD 迭代次数 |
| `--inputs` | 目录 | 使用另一套 inputs（网格档测试；用 `tools/prepare_inputs.py --out-dir DIR --cells N` 生成） |
| `--set a.b=<json>` | 可多次 | 仅诊断：覆盖配置项，写入该 run `config.json` 的 `_overrides` |
| `--out-root` | 目录 | 把 `raw_data/` 写到别处（诊断用） |
| `--device` | 默认 `cuda:0` | DexSim 与 EmbodiChain；Isaac Lab 用其启动器的 `--device` |

单次耗时（实测，含启动）：s00 完整图像约 50 秒，keyframes 约 35 秒，计时 run 约 33 秒，台架约 12–20 秒，20 子步约 1 分钟，direct（无 CUDA Graph）约 3 分钟；整个矩阵约 25 分钟。首次运行另有 Warp 内核编译时间。

## 输出

每个 run 目录 `raw_data/<platform>/s01c_<platform>_<state>_<variant>_r<NN>_<kind>_<UTC>_<rand>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | run 标识、输入与脚本 SHA256、git 状态、版本、GPU、请求值与实际生效的子步数/CUDA Graph 状态、相机 K 与外参、分割 id 映射、每个形状的接触参数、控制器事件、显存采样、未提供的读数（接触力、关节力矩标为 unavailable） |
| `config.json` | 实际使用的配置（含变体与 `_overrides`） |
| `timing.csv` | 每帧 `frame, sim_time, step_seconds, capture_seconds, phase`；`step_seconds` 是同步后的 `update`/`step` 墙钟时间 |
| `states.npz` | 60 Hz 逐帧：time、phase、target_q、qpos、tcp、tcp_cmd_z_offset、finger_poses、plate_pose、全部 particles；另有 rest_particles、tets、joint_names |
| `images_<cam>.npz` | frame、time、rgb（仅 keyframes 运行）、depth（到像平面距离，m）、seg（公共标签 0 背景 1 地面 2 桌 3 支架 4 机器人 5 压板 6 软块），都是 640×480（1280×960 渲染隔点取样）；K（存储图像的内参）、K_render、camera_to_world（OpenGL 约定） |
| `rgb_<cam>.mp4`（仅 `--images full`） | 1280×960、30 Hz 的 RGB 源视频，用于生成对比视频；不单独导出 |
| `run.log` | 每 0.5 s 一行的过程日志 |
| `authored_usd/`（仅 Isaac Lab） | 本 run 写出的软块 TetMesh 与压板 USD |

`tools/analyze.py` 输出：

| 文件 | 内容 |
|---|---|
| `results/task_metrics.csv` | 每个任务 run 一行：压缩率、恢复误差、桌面下穿、压板滑移、倾角、跟踪误差、计时、各项通过标记与失败项 |
| `results/bench_metrics.csv` | 台架：实测压缩量与线弹性估算之比、重物倾角、计时 |
| `results/summary.json` | 10 初态成功率与分母、子步/Graph 对照（含逐帧节点差）、计时分布、分割回投检查、控制器源码一致性 |
| `results/plots/*.png` | 压缩时间曲线（s00、10 初态、变体）、台架曲线、RGB/深度/分割样例 |
| `videos/s01c_comparison.mp4` + `.json` | 唯一导出视频：2×2，1920×1440，左 DexSim、右 Isaac Lab，上 oblique、下 front45；从 t=0 按帧对齐；每列顶部显示仿真时间、阶段和实时物理帧率（该 run 实测步进时间的 0.5 s 滑动平均：步/秒、毫秒/步、实时倍率） |
