# DexUni 小型输入定义

本目录只保存 authored JSON 和来源/生成说明。`s01c/`、`s01t/`、`s02/` 的配置
来自旧 pilot，**不是 dexuni-replan-v2 的已冻结参数**。S03 新建平面布网格输入和物理夹持参数；正式快照在各自 runs 批次中，详见当前 PLAN。
其余任务尚无配置，不创建假的输入包。

- 生成工具：`physics/experiments/dexuni/<task>/tools/prepare_inputs.py`（相对 tech_reports 根）。
- 输入生成必须显式 `--out-dir NEW_DIR`，默认不写源目录、不覆盖已有快照。
- 每个外部输入目录包含配置快照、实际初态、网格/拓扑、IK/控制数组与 inputs_manifest.json。
- 当前脚本用 DEXUNI_INPUT_ROOT/<benchmark-id> 或 --inputs DIR 读取本地输入。
- 旧 pilot 的本机副本在 `~/.cache/embodichain/report-assets/local/dexuni/pilot_v1/`。
  它们不是 Hub 下载 snapshot，也没有真实 bundle ID/dataset revision。
- 新候选版本另建 candidate_001 等目录，经 G0–G6 签核后冻结正式 revision；不得改写 Hub cache。
- JSON 中机器人 URDF 为旧本机位置。其它机器先用原资产 resolver 得到实际路径，修改独立候选配置后重新生成和校验；不要把绝对路径当资产身份。
- 通用机器人与许可见 [共享资产引用](../../shared/dexuni/README.md)。本目录不复制机器人网格。

发布遵循 [资产规范](../../README.md) 和 [下载设计](../../DOWNLOAD_DESIGN.md)。
已发布报告为 `dexuni-physics-2026-10-27`，真实输入 bundle 按任务目录划分，
完整数据集 commit、大小和 SHA256 记录于 [publication.json](publication.json)。
[下载命令](../../../physics/experiments/dexuni/SETUP.md) 默认只取输入，结果与视频需显式选择。
这些包保存正式运行所需的冻结输入；前述旧 pilot 缓存不等同于这些已发布包。
