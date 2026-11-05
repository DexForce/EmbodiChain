# 目录迁移记录

## 2026-10-27 最终收尾

用户已授权删除旧 `dexuni_benchmark`，此前保留旧目录和暂不归档 S02 的安排不再适用。
当前计划、任务说明、配置定义和脚本统一位于本 suite 及 tech_reports 的对应所有者目录。
缺失的历史设计已保存在 [初始设计归档](history/initial_design/INDEX.md)，不覆盖当前任务。
旧 M04 棱台方案仅作历史记录，当前 M04 是折衣。

当前正式结果、视频和已有 pilot_v1 档案不变。旧目录的早期 S02 二进制试跑产物删除，
原始源码、配置、日志及统计保存在[删除账本](../../../../../../outputs/benchmarks/tech_reports/maintenance/legacy-retirement-2026-10-27/README.md)。
旧路径继续被输出保护规则拒绝，避免重新创建废弃工作区。

以下为历史迁移经过；其中正在运行和暂不清理等表述不代表当前状态。

## 依据与范围

用户指定规范：<https://github.com/DexForce/EmbodiChain/tree/main/scripts/benchmark/tech_reports>。
核对快照：`f179b5558fa17df06f172f0b1e4e4f81df63be74`。
读取了根目录、physics/core/reporting、assets 的 README 与 DOWNLOAD_DESIGN。
上游当时是目录/协议设计，本次没有把不存在的 downloader/runner 写成现成能力。

本次迁移本地 DexUni suite 的源文件和说明，保留旧工作区，因为实施时存在运行中的 S02 进程。
**未停止进程、未改旧平台脚本/输入、未删除任何旧结果**；旧顶层入口改为指向新计划。

## 路径映射（新路径相对 tech_reports）

| 旧路径（相对 dexuni_benchmark） | 新所有者 |
|---|---|
| PLAN / PARITY_CHECKLIST / suite README | physics/experiments/dexuni/ |
| 各任务 README | physics/experiments/dexuni/<task>/ |
| S01/s01c_compress、S01/s01t_stretch | experiments/dexuni/s01c_compress、s01t_stretch（平级，S01 总目录只作索引） |
| *_embodichain.py / *_isaaclab.py / *_dexsim.py | physics/platforms/dexuni/<platform>/{s01c,s01t,s02}.py |
| tools/prepare_inputs.py、run_matrix.sh | physics/experiments/dexuni/<task>/tools/ |
| tools/analyze.py、parity_embodichain.py、compare_models.py | reporting/dexuni/<benchmark-id>/ |
| inputs/*_config.json | assets/physics/dexuni/<benchmark-id>/（候选 authored 配置） |
| NPZ、生成夹具 JSON、input manifest、转换 USD | 本机外部 local report-assets 缓存，只读输入快照 |
| assets/convert_franka_usd.py | physics/platforms/dexuni/isaaclab/convert_franka_usd.py |
| _lib/benchlib | physics/platforms/dexuni/_lib/benchlib（保留第三方版权/许可） |
| S01 raw_data/results/videos/parity 及其旧源码 | 仓库 outputs/benchmarks/dexuni/pilot_v1/s01_soft_compress_stretch/ 原样快照 |

## 档案与过渡

S01 快照共 1,318 个文件、3,586,135,697 字节，迁移时逐文件 SHA256 与旧目录一致；
清单在 `outputs/benchmarks/dexuni/pilot_v1/archive_manifest.json`。
顶层旧 PLAN/README/PARITY_CHECKLIST 也保存在 pilot_v1，以保留历史结论的原始语境。
这里的数字是文件完整性检查，不是物理通过率。历史报告含已被 review 限制的结论，
只允许引用为旧 pilot，不能自动进入新的 leaderboard/speedup。

S02 正在运行，因此没有将其当作完成的可复现档案快照；旧工作区后续数据继续归 legacy pilot。
待进程退出并确认不再写入后，再选择已完成尝试逐文件归档/校验，保留错误与日志；
确认源代码和输入都已收录后才可另行删除旧工作区。不要在运行中搬目录或清理源文件。

用户后续安排：可继续在旧目录完成当前开发和实验，结束后统一迁移；S02 暂不归档。
新协议以本目录 PLAN 为准；待办和统一迁移步骤见 [HANDOFF.md](HANDOFF.md)。
旧源目录仍需在后续提交前清理/归档，不能把其中数 GB 结果加入 Git。
输入缓存没有写成真实 HF bundle，未上传资产，未替换远程 index；源码 index.json 仍为空 bootstrap。

## 语义与验证边界

- 迁移保持旧控制器/物理算法，未宣称修复 G1/G3/G4/G5。脚本增设 pilot 确认和输出保护。
- 新默认输出不再写源码目录；分析必须指定独立的新重建目录；准备器不覆盖已有输入 revision。
- 编排默认只跑 nominal 冒烟对；子进程失败有 journal、整体退出非零，但正式矩阵分母分析待实现。
- 原会话在收尾阶段连接错误中断；测试文件已写，但没有 CPU 路径/保护/故障模拟测试、AST/语法和 Markdown 链接检查已通过的证据；未新启动 GPU 仿真。
- 公共模拟器 API 未修改，因此项目 agent_context 无需新增 DexUni 基准主题；协议归 benchmark 所有者。
- 远程飞书任务书、GitHub 分支和 HF 数据集都未自动更新。

## 当前工作区安排（后续更新）

用户已将实验 1 交由当前 agent 在 tech_reports 主导；不再在旧工作区新增本会话实验。
本目录检查时没有已有测量数据可删除，没有删除目录外快照或旧工作区文件。
新结果和视频统一写 `outputs/benchmarks/tech_reports/dexuni/`，路径保护允许 runs 子目录，继续禁止写入源码/资产和旧档案。
S02 暂不归档。历史迁移叙述保留作溯源，当前执行以 PLAN/STATUS 为准。
