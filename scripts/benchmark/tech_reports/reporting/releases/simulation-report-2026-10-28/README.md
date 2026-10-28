# 仿真技术报告：2026-10-28 本地进度快照

本次为 **draft 源码提交**，整理已有实验、方案和离线报告工具；没有新增测量。
原始数据、报告、图片、原始帧和视频仍保留在原工作区。
Hugging Face 上传按用户要求暂缓，`dataset_revision` 为 `null`；本页不代表数据集已发布或最新引擎已经通过完整复测。

目标数据集：[DexForceAI/embodichain_reports](https://huggingface.co/datasets/DexForceAI/embodichain_reports)。
稳定报告 ID：`simulation-report-2026-10-28`；实验 ID 保留 L-02～L-05、R-01、P-01～P-07、D-01。

## 当前结果与限制

| 实验 | 状态 | 典型结果及下一步 |
| --- | --- | --- |
| P-01 自由落体／单摆 | 本轮完成 | 240 条轨迹、24 组配置。dt=1 ms，前 1 s 落体最大位置误差的跨初态均值：DexUni 7.802 mm、PGS 4/1 与 TGS 1/1 均 4.900 mm。单摆 10 s 最大相对能量漂移均值均约 0.222%。未测吞吐。 |
| P-02 摩擦／堆叠 | 本轮完成 | 10 s 堆叠搜索保留睡眠；高预算原批次最高 10/10 完整点为 DexUni 51 层、TGS 78 层。DexUni 51 层在相同初态的部分复跑中倒塌，不能称可重复上限。 |
| P-03 抓取 | 部分完成 | 两种物体、两平台共 40 条质量轨迹及原片完成；正式性能未完成。DexUni 手指最大几何穿透 1.174–1.962 mm、受力 mimic 最大差 4.620–4.825 mm；TGS 对应约 0.041–0.063 mm、0.067 mm。 |
| L-02 基础场景加载 | 部分完成 | 当前主配置 sleep=False；8000 arena 三次加载完成，原始共享 GPU 耗时约 75–80 s；16000 预载超时。公平性能和容量边界需独占资源补测，不能与 Isaac 原对照直接算速度比。 |
| L-03 机器人加载 | 本轮完成 | Cartpole／Franka／Go2：108 正式样本、18 预载、6 段原片。1000 arena、H 条件的就绪中位数（EmbodiChain/Isaac）分别为 6.989/6.153、8.451/7.919、9.852/8.634 s；EmbodiChain 为 1.07–1.14 倍。G1 不在正式矩阵。 |
| L-04 工作站 | 部分完成 | 资产预检完成一部分；旧抽屉 USD 碰撞遗漏与转换凸分块合并问题保留，完整工作站和正式矩阵未执行。优先换原生 USD 候选后重新预检。 |
| L-05、R-01、P-04～P-07、D-01 | 待开展，共 7 项 | 仅有方案，不提供未测数字或完成声明。 |

这里的“完成”指该轮实验和初稿完成，不等同于最终报告可发布。
L 系列统一用 `num_envs` 复制完整 arena；旧单环境多物体 L-02 仅保留作历史诊断。
所有当前结论采用连续误差、实际任务事件和同条件比较，不新增人为精度 pass/fail。

2026-10-28 的独立 USD parser 修复复验已恢复 G1／抽屉每环境 31／2 个碰撞，N=1、4 各运行 20 步。
它使用新构建及公开配置扩展点，不属于旧 L-03 正式矩阵：当时检查的标准客户端仍存在 `AutoSolverCfg` 兼容问题。
抽屉转换把 28 个凸分块并成 2 个整体凸包的问题仍独立存在。
本 PR 没有修改或合入引擎修复，也没有重新确认最新 main 的运行兼容性。

## 版本与原始出处

| 范围 | 测量版本／数据路径（相对原报告工作区） |
| --- | --- |
| P-01 | EmbodiChain `e8a641999cb42d66c4b84bceb5040a62a91a2999`；`results/P-01/`；当前稿 `reports/P-01.md`；逐轨迹表 `reports/P-01/comparison-descriptive/metrics.csv` |
| P-02 | EmbodiChain `f179b5558fa17df06f172f0b1e4e4f81df63be74`；`outputs/benchmarks/P-02/`，包括 `stack-limit-10s/` 和历史长窗口结果 |
| P-03、L-02、L-03 | EmbodiChain `f179b555…`，原安装 DexSim 加 MR !1501 parser 冻结副本；不是完整 MR head。原运行的 `run.json` 和依赖 manifest 为准 |
| L-03 正式对照 | Isaac Lab `ae37b028ea415c91ea2bc32609efcd759ed2b974`／Isaac Sim 6.1；`outputs/benchmarks/L-03/three-robots-v1/`；DexUni articulation-only 实际使用 SolverMuJoCo，无 VBD |
| L-02 当前稿 | `outputs/benchmarks/L-02/sleep-off-main-v1/`；容量原记录在 `sleep-off-v1/`；sleep 开启旧主测在 `multi-arena-v1/` |
| L-04 与新版 parser 诊断 | `outputs/benchmarks/L-04/`、`outputs/benchmarks/usd-import-fix/20261028-v1/`；新复验的客户端 checkout 为 `c487e87472b165e5f3bbdd94a071ffcd1486abdf` |

原硬件为 RTX 5090 D v2（约 24 GiB 显存）、Ryzen 9 9950X、约 64 GB RAM，L-03 正式记录驱动 580.159.04。
逐次命令、随机种子、初态、硬件／软件、模块来源和采样配置以每次 `run.json`、配置快照、计划／execution ledger 为准；不能用 PR 基点 `950a16c6d93ba6bbbef384fcf8350c7ca2a0aa4d` 替换历史版本。
P-01 初态种子为 20261014；P-02／P-03 使用各自冻结初态文件，不能把复制环境当成新增独立初态。

`source-map.json` 记录每个迁入源码的原路径和原 SHA-256。添加版权头、格式化、调整源码入口／数据根目录后的文件是维护版本；运行目录内的原源码快照不变。
`release.json` 记录数据上传状态与本轮离线检查范围。原始文件逐项清单由下节命令生成，不将大清单和测量产物放进 Git。

## 阅读、离线重建与执行边界

源码按 `loading/physics/.../experiments`、`platforms/<platform>`、`reporting` 归属。
方案位于各系列的 `experiments/*spec.md`；P-01 参数计划位于 `physics/experiments/p01_configs/`。
`assets/physics/P-01/` 包含四个手写输入；原有 URDF 单摆属于历史手写实验模型，不是未来通用机器人资产的转换建议。

离线工具可通过 `SIM_REPORT_ROOT` 指向原数据工作区。必须在副本中复算，输出不得覆盖冻结数据。
以下命令从 EmbodiChain checkout 根目录运行；`/path/to/report-copy` 是保留原目录结构的数据副本。

```bash
export SIM_REPORT_ROOT=/path/to/report-copy
python -m scripts.benchmark.tech_reports.reporting.p01_descriptive_comparison
python -m scripts.benchmark.tech_reports.reporting.p01_benchmark_summary \
  --metrics "$SIM_REPORT_ROOT/reports/P-01/comparison-descriptive/metrics.csv" \
  --output "$SIM_REPORT_ROOT/reports/P-01/benchmark-summary.md"
python -m scripts.benchmark.tech_reports.reporting.p02_analysis \
  --runs "$SIM_REPORT_ROOT/outputs/benchmarks/P-02/runs" \
  --output "$SIM_REPORT_ROOT/outputs/benchmarks/P-02/analysis/recomputed" \
  --phases main supplement independent independent_supplement
python -m scripts.benchmark.tech_reports.reporting.site.build \
  --output outputs/benchmarks/simulation-report-preview
```

数值分析需要 NumPy、SciPy、Matplotlib；网页依赖见 `reporting/site/requirements.txt`。
网页构建只读取报告／素材；不会启动仿真或运行 systemd 安装。
三章技术稿与恰好三张表的 benchmark 摘要分别维护。未测性能记“未测量”，无二元任务目标的成功率记“不适用”，不生成综合优胜者。

**本 draft 尚未承诺所有仿真入口可在新机器／最新 main 直接运行。**
历史批次脚本仍含原机器解释器、安装路径和冻结依赖注入；P-01 的部分旧相对路径还用于历史证据校验。
当前源码与数据工作区分离后，P-01 新测量的源码快照路径也需单独适配。
这些入口仅作原实验实现和续作起点；跨机参数化、历史 runtime 合法获取及最新版本最小试跑是后续工作，不在本次发布整理时重跑整个矩阵。
离线已验证内容和未验证内容见 `release.json`，不能从编译检查推出动力学等价。

## 延后上传的资产准备

在 EmbodiChain checkout 根目录执行：

```bash
python -m scripts.benchmark.tech_reports.reporting.prepare_release \
  --workspace /path/to/original-simulation-report \
  --output /path/to/new-hf-staging-directory
```

该命令只复制小型输入候选，逐文件记录现有结果／图／视频的来源、字节数、SHA-256 和许可审查状态。
结果保留原目录结构映射，失败尝试和历史协议不按成败筛除。
安装运行时、引擎源码补丁及通用机器人／场景资产单列排除清单，保留来源引用，不复制到公共资产库。
`NOASSERTION` 表示许可尚待核实，**不是许可授权**；派生几何和媒体需结合原资产许可检查。

拟发布目录：

```text
reports/simulation-report-2026-10-28/
  README.md
  release.json
  inventory.json
  inputs/<benchmark-id>/...
  results/<benchmark-id>/<original-relative-path>
  figures/<benchmark-id>/<original-relative-path>
  videos/<benchmark-id>/<original-relative-path>
```

`index.fragment.json` 仅含真实输入候选，不能替换远端现有 index。上传前重新核对清单和逐文件许可，将输入、报告元数据和合并后的 index 在同一数据集提交中发布；随后验证下载字节并写回完整 commit SHA。
远端已有 `dexuni-physics-2026-10-27` 是另一份报告，不能覆盖或冒用其发布状态。
本 draft 的数据尚未上传，因此无法仅凭公网 checkout 重建完整网页。

后续资产优先采用可追溯、体积适中的原生 USD。L-04 柜体候选为 ForgeCAD 的 `kitchen_base_cabinet/`，来源修订
`0573053fa2c9174ca2a4678a462ec4dfd3035d59`，约 17.7 MB，包装元数据声明 CC-BY-4.0。
来源是 [ForgeCAD/simready-assets](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/0573053fa2c9174ca2a4678a462ec4dfd3035d59/kitchen_base_cabinet)，由 NVIDIA SimReady Central 索引；不将它误写为 NVIDIA 制作。
它尚未进入共同加载预检或正式实验。下载仅选该子目录，核对依赖、许可、哈希与两端模型后再使用，避免整库下载及默认本地 URDF→USD 转换。
