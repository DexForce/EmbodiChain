> 历史设计快照，非当前执行计划。当前安排见[PLAN](../../../../PLAN.md)。旧 M04 棱台方案已被折衣任务替换。

# S01-T 软块拉伸与恢复（第 1 轮试跑）

状态：试跑，配置未冻结；目标拉伸 5 %（10 % 待受载预算通过后再做）。

- 第一阶段 DexSim 对 Isaac Lab：数据在 `raw_data/`，结果见 results/S01T_REPORT.md（历史路径：`./results/S01T_REPORT.md`）。
- 第二阶段 EmbodiChain 对 Isaac Lab（当前对比）：数据在 `embodichain_vs_isaaclab/`，结果见 embodichain_vs_isaaclab/results/S01T_EC_REPORT.md（历史路径：`./embodichain_vs_isaaclab/results/S01T_EC_REPORT.md`）；EmbodiChain 与 DexSim 的一致性检查（DexSim 数据不保留）记录在 results/S01T_DECISIONS.md（历史路径：`./results/S01T_DECISIONS.md`） 第二阶段。开跑前已按 [对比检查清单](../../PARITY_CHECKLIST.md) 把所有求解器选项写进配置。

## 任务

拉伸长块 T（120×40×40 mm，长轴沿 X，密度 250 kg/m³，48 g，E 30 kPa、ν 0.35）架在桌面上方 40 mm。近端放在静态夹具里（底座、上压板、两侧立柱，都是真实碰撞刚体）；开局把夹具内的节点按 15 % 预压进夹口（静止形状不变），夹具靠弹性法向力和 μ=0.6 的摩擦夹住，不用 pin 或附着。Franka 在远端闭合手指，静置到 2 s 取参考长度，再沿 +X 以 2 mm/s 拉到中心材料线伸长 5 %，保持 1 s，退回起点，张开手指，观察 2 s 恢复。

测量：材料线是自由段内固定节点 id 的折线（x 0.44–0.49，11 个节点，距夹口和指垫各 10 mm），长度 L 为相邻节点距离之和，拉伸率 L/L0-1；中心线用于闭环控制和验收，另记录上下左右四条表面线。夹持端滑移：夹口内节点相对夹具的 X 位移、指垫内节点相对指尖的位移（参考窗口 → 保持结束，上限 3 mm）。

## 目录

| 路径 | 内容 |
|---|---|
| `s01t_dexsim.py` | DexSim 独立脚本（DexUni two-way + dynamic），只读 `inputs/` |
| `s01t_isaaclab.py` | Isaac Lab 独立脚本（Newton MuJoCo-Warp + VBD CouplerProxy，OVRTX 渲染），只读 `inputs/` 与转换好的 Franka USD |
| `s01t_embodichain.py` | EmbodiChain 独立脚本（EmbodiChain `SimulationManager` + DexUni two-way），只读 `inputs/` |
| `inputs/` | 两平台共用数据：`s01t_config.json`（唯一参数来源）、`s01t_block_mesh.npz`（网格与材料线节点 id）、`s01t_clamp.json`、`s01t_initial_states.npz`（10 份初态的静止与预压节点坐标）、`s01t_ik_table.npz`（TCP 沿 X）、`inputs_manifest.json` |
| `tools/prepare_inputs.py` | 由配置生成 `inputs/`（DexSim 解释器） |
| `tools/run_matrix.sh` | 按组顺序运行全部 run，平台交错；`PLATFORMS`（默认 `embodichain isaaclab`）、`OUT_ROOT`（默认 `embodichain_vs_isaaclab`） |
| `tools/analyze.py` | 只读 `<root>/raw_data/`，生成 `<root>/results/` 和唯一导出视频 `<root>/videos/s01t_comparison.mp4`；`--root`、`--platforms` |
| `tools/parity_embodichain.py` | DexSim 与 EmbodiChain 一致性检查（`--root` 指向存放两边 run 的临时目录） |
| `embodichain_vs_isaaclab/` | 第二阶段的 `raw_data/`、`results/`、`videos/` |
| `raw_data/{dexsim,isaaclab}/<run_id>/` | 每个 run 的原始数据 |
| `raw_data/_logs/` | `run_matrix.sh` 每个 run 的终端输出 |

三个平台脚本互不导入；闭环拉伸规则 `StretchController` 在三个文件中逐字相同，`tools/analyze.py` 用 AST 哈希检查。两个脚本启动时都会核对仿真节点顺序与网格一致（材料线依赖节点 id），并核对预压位置已写入。

## 运行环境与准备

解释器、版本和 Franka USD 转换与 [S01-C](../s01c_compress/README.md#运行环境) 相同。所有命令在仓库根目录执行：

```bash
cd /home/oem/code/repos/EmbodiChain
P=dexuni_benchmark/s01_soft_compress_stretch/s01t_stretch
DEX=/home/oem/code/engine/dexsim/.env-3.12/bin/python
ISAAC=~/app/IsaacLab/.venv/bin/python
EC=.venv/bin/python

$DEX $P/tools/prepare_inputs.py        # 配置改动后重新生成 inputs/
bash $P/tools/run_matrix.sh             # 第二阶段：EmbodiChain 对 Isaac Lab，全部组（约 30 分钟）
$DEX $P/tools/analyze.py --root embodichain_vs_isaaclab --platforms embodichain,isaaclab
PLATFORMS="dexsim isaaclab" OUT_ROOT=. bash $P/tools/run_matrix.sh   # 第一阶段
$DEX $P/tools/analyze.py                # 第一阶段结果
```

| 组 | 每个平台的运行 | 参数 |
|---|---|---|
| `nominal` | s00，完整图像（RGB 源视频 + 每帧深度/分割） | `--state s00 --images full` |
| `states` | s01–s09（软块平移 ±2 mm、转 ±1.5°） | `--state s0N --images keyframes` |
| `variants` | 5 / 20 子步、关闭 CUDA Graph | `--variant {substeps5,substeps20,direct} --images keyframes` |
| `timing` | 纯计时，3 个独立进程 | `--kind timing --repeat {1,2,3}` |

## 单次运行

```bash
$DEX $P/s01t_dexsim.py                                        # s00、完整采集
$DEX $P/s01t_dexsim.py --state s03 --images keyframes
$DEX $P/s01t_dexsim.py --variant direct --images keyframes
$DEX $P/s01t_dexsim.py --kind timing --repeat 1
$EC $P/s01t_embodichain.py --state s03 --images keyframes     # EmbodiChain 同一 run
$ISAAC $P/s01t_isaaclab.py --state s03 --images keyframes      # Isaac Lab 同一 run
```

参数：`--state`、`--variant`、`--kind full|timing`、`--images full|keyframes|none`、`--repeat`、`--inputs`、`--set a.b=<json>`（仅诊断，写入 `_overrides`）、`--out-root`、`--device`。含义与 S01-C 相同；S01-T 没有材料台架。

## 输出

每个 run 目录：`manifest.json`（输入与脚本 SHA256、版本、实际子步/CUDA Graph 状态、相机 K 与外参、分割映射、形状接触参数、控制器 L0 与阶段事件）、`config.json`、`timing.csv`、`states.npz`（60 Hz：time、phase、target_q、finger_target、qpos、tcp、tcp_cmd_x_offset、finger_poses、全部 particles、五条材料线长度；另有 rest_particles、initial_particles、material_lines、tets）、`images_<cam>.npz`（640×480 深度、分割、keyframes 运行的 RGB；K、K_render、camera_to_world）、`rgb_<cam>.mp4`（仅 s00，1280×960 源视频）、`run.log`。

分割标签：0 背景、1 地面、2 桌、3 支架、4 机器人、5 夹具、6 拉伸块。相机：`oblique` 与 `side45`（-Y 方向 45° 俯视，沿块的长边）。

`tools/analyze.py` 输出：`results/task_metrics.csv`、`results/summary.json`、`results/plots/*.png`、`videos/s01t_comparison.mp4`（唯一导出视频：2×2，1920×1440，左 DexSim、右 Isaac Lab，上 oblique、下 side45，叠加两边实时物理帧率）。
