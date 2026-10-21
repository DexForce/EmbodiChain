# NeRD 完整训练预算推进

阶段记录：2026-10-21（Asia/Shanghai）。**状态：Cartpole 完整预算已验收，Ant 继续训练；两机器人整体目标尚未完成。**

本轮在[小预算归因结果](neural-dynamics-numerical-attribution-20261020.md)上落地数据保真、显式数值策略和官方采样重放。Cartpole、Ant 各一个种子，从真实官方零步初值开始，以 `total_steps=4,995,000` 运行。前 5,000 步为同一正式任务的检查前缀，通过后恢复原 Adam、RNG 和采样游标继续；没有另设短训练学习率，也没有恢复旧 40 组矩阵。

## 已通过的节点

| 检查 | 结果 |
|---|---|
| 官方数据保真 | 两机器人 × train／validation／test，6 个实际 B512/H10 批次的输入、目标与支持环境逐字节一致 |
| 初始化 | 真实官方 `initial.pt` 的 45 项参数及所有 FP32 归一化 mean／var 逐位保留，实际更新数为 0 |
| 官方采样 | 原 Torch2.2 CPU DataLoader 的 999 个训练 epoch、2,997 次完整洗牌在 Torch2.7 逐一验证 SHA，一致 |
| GPU 恢复 | 连续 100 步与 50 步后新进程恢复到 100 步的模型、Adam、RNG、采样及配置逐字节一致 |
| 5,000 步门禁 | 两模型的归档 checkpoint、实际完整固定测试集、生产导出及 portable 重载推理均通过 |
| 正式续训 | 两模型实际 500,000 步 checkpoint 均经独立审计；预算未变，Adam、RMS、游标、学习率及导出一致 |

Cartpole 使用物理 GPU1，Ant 使用 GPU6。Ant 最初在 GPU3 验证；该卡随后出现其他任务，因此保留原证据并在空闲 GPU6 重做完整恢复验证，没有清理外部进程。本任务不使用 GPU7。

## 冻结的训练合同

Cartpole 为 2,713,668 参数、embedding192；Ant 为 10,891,037 参数、embedding384。两者均为 6 层、12 头、B512/H10、FP32、无 AMP，Adam `(0.9, 0.999)`／`eps=1e-8`、梯度裁剪 1。原始 1,024,000 条训练 transition 形成 931,840 个窗口，另有独立 validation／test 各 512 条轨迹。官方发布 LFS 模型的配置指向 Cartpole 5M、Ant 10M 数据，需与本轮已有 1M 数据明确区分；该 pickle 不证明实际训练的末次更新数。

原 recipe 有 1,000 个 epoch，epoch0 仅评估，所以完整实际训练预算为 `999 × 5,000 = 4,995,000` 次更新。学习率按 5,000 步间隔、offset1，从配置端点 `0.001→0.0001` 调度：首 5,000 步为 `0.0009991`，第 5,001 步为 `0.0009982`，最后一个训练 epoch 为 `0.0001009`。validation／checkpoint 每 5,000 步执行，日志每 500 步记录；test 不参与选模。

原始数据来自 Python3.8.20／Torch2.2.2 支持环境，Cartpole Warp1.8.0、Ant Warp1.5.1。新增 `preserve_legacy_velocity=True` 导入选项保留未旋转 source-world 的原始 FP32 spatial velocity，避免 Ant 经 COM 往返产生的目标舍入；默认 native 导入行为保留。新增数据指纹与旧实验不同，旧检查点不能混入此正式任务。

当前训练环境为 Python3.11.14／Torch2.7.0+cu128／Warp1.17.0／Newton1.7.0，SDK 为 DexSim MR!1483 的 b61e920 构建、版本0.5.1。选择显式 `numerical_profile=reference`：math SDPA，以及固定顺序 FP64 累加 FP32 各参数范数。该策略连同 attention backend 进入 checkpoint／portable metadata；生产默认策略和旧官方默认基线未被替换。

原始采样重放保留 DataLoader 每 epoch 丢弃未消耗尾部的行为。每机器人约 219KiB 的 seed／hash descriptor 和单次约 7.1MiB permutation 缓存替代约 20GB 全索引矩阵。原始数据驻留预算为 4GiB，包含新增保真字段。

运行使用独立只读源码快照，共 1,480 个文件。协议校验源码、数据、初值、采样、版本和 GPU UUID；训练目录绑定协议 SHA，继承的文件锁防止重复 worker。完成判定要求真实 `last.pt.global_step==4,995,000`，分别记录终点与 validation 选出的 best，提前出现 best 不算预算完成。

## 5,000 步阶段结果

两个模型均在完整 46,592 个 test 窗口、465,920 个有效位置上评估：

| 模型 | 固定 test 归一化 MSE | 5,000 步耗时，含初始化／验证／保存 | 峰值 Torch 分配显存 |
|---|---:|---:|---:|
| Cartpole | 0.001019071375 | 62.43 秒 | 1.29GiB |
| Ant | 0.008243989798 | 114.77 秒 | 2.39GiB |

短测稳态吞吐约为 Cartpole 98、Ant 50 更新／秒。全预算初估约 14、28 小时，实际耗时受后续运行资源与验证开销影响。当前进度以[运行记录](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/progress.json)和实际 checkpoint 为准。

下图快照生成于 2026-10-21 11:07:31 UTC，通过各自单个稳定文件描述符读取实际已提交 Cartpole 4,995,000、Ant 3,100,000 步。Cartpole 达到 4,995,000 步，Ant 仍以 4,995,000 为目标；总图保持 interim，`whole_budget_complete=False`。两者原 5,000 步 test 点保留，Cartpole 新增 GPU1 的终点 last（4,995,000）与验证集选定 best（4,940,000）固定 test 点，局部放大明确区分二者；这些点直接绑定 parent completion 的实际评估收据。Ant 仍只有 5,000 步 GPU test 点，未把当前未评估步数画成 test；两者 500,000 步 CPU fixed-test 在下文单列。刷新前 PNG、PDF、逐点快照及原绘图脚本已按原字节、大小和修改时间另存于 [Cartpole 4,995,000／Ant 2,915,000 归档](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/figures/archive/committed-cart4995000-ant2915000/archive-manifest.json)；[上一轮 4,545,000／2,310,000 归档](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/figures/archive/committed-cart4545000-ant2310000/archive-manifest.json)；[更早的 4,190,000／2,130,000 快照](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/figures/archive/committed-cart4190000-ant2130000/metrics-snapshot.json)和此前历史归档保留。

![正式训练的中间曲线](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/figures/training-progress.png)

[PDF 图](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/figures/training-progress.pdf)与[逐点快照](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/figures/metrics-snapshot.json)保留完整预算及实际已提交步数。

Cartpole 已于 2026-10-21 09:52 UTC 通过[完整预算正式验收](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/cartpole-full-budget-acceptance-20261021T095225Z.json)。[终点预算与导出审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/cartpole-4995000-budget-export-audit-20261021T083759Z.json)确认 last 实际 4,995,000、验证选定 best 实际 4,940,000，核对 45 项 Adam、51 项导出 tensor、冻结 RMS、999 次 validation 选模、源码／环境以及原 5,000 步前缀连续恢复证据；best 不替代实际预算。该 Cartpole 审计的 Ant 中途计数为 2,660,000。最新 [Ant 独立 CPU 审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/ant-live-budget-audit-20261021T105916Z.json)于 2026-10-21 10:59:16 UTC 保留实际 3,075,000 步稳定副本，学习率为 `0.00044650000000000007`，341 项合同检查全部通过，`findings=[]`；45 项 Adam、冻结 RMS、采样／RNG 游标及源码／环境绑定均保持。上图于 2026-10-21 11:07:31 UTC 独立读到 Ant 3,100,000，审计和图快照按各自时间、计数记录；Ant 目标仍为实际 4,995,000 步。此前 [06:33 UTC 中途审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/live-budget-audit-20261021T063357Z.json)的 Cartpole 4,515,000／Ant 2,295,000 记录继续保留；整体目标等待 Ant 原 GPU6 任务实际完成 4,995,000 步及终点验收。

同一官方支持环境、每场景固定 16 条轨迹的 100 步关节位置 qMAE：

| 模型／条件 | passive | random_effort | fixed_policy |
|---|---:|---:|---:|
| Cartpole 官方 LFS／原默认推理 | 0.021108 | 0.061312 | 0.017129 |
| Cartpole 本轮 5,000 步／显式 math | 0.588071 | 0.647311 | 0.484932 |
| Ant 官方 LFS／原默认推理 | 0.195290 | 0.204721 | 0.140644 |
| Ant 本轮 5,000 步／显式 math | 0.409015 | 0.373054 | 0.417351 |

这批长预测仍明显落后于官方发布模型。5,000 步仅为目标预算约 0.1%；公开配置的数据量及数值路径也不同，不能据此将差距归因于迁移实现或运行环境。teacher／raw action 前缀逐项哈希一致，全部失败轨迹保留；新模型明确使用 math，官方 LFS baseline 保持其原默认推理。完整 H1／10／100 指标见实验目录下的 evaluation 报告。

后续第一个 50,000 步观察阈值已复评。Cartpole 采用首次捕获的实际 60,000 步模型，Ant 为实际 50,000 步模型，未将两者误标为相同步数：

| 模型／实际步数 | passive H100 qMAE | random_effort | fixed_policy |
|---|---:|---:|---:|
| Cartpole／60,000 | 0.327882 | 0.353821 | 0.261394 |
| Ant／50,000 | 0.166831 | 0.277809 | 0.169040 |

两个模型的长预测总体继续改善。Ant 被动场景已低于官方 LFS 的同项误差，但另外两个场景仍偏高；Cartpole 差距仍较大。不能将一个场景较低的误差扩展为全部精度验收通过。完整指标及哈希见[阶段复评](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/milestones/RESULTS.md)。

Cartpole 的实际 500,000 步节点随后完成原支持环境复评，H100 qMAE 进一步降至 passive `0.0760825`、random_effort `0.1179903`、fixed_policy `0.1098365`。其 H1 qMAE 分别为 `0.00016520`／`0.00018086`／`0.00017016`；H10 为 `0.00182072`／`0.00201143`／`0.00184519`。相同固定语料上仍高于官方 LFS，完整 4,995,000 步尚未完成。

同一 500,000 步 portable export 另经当前 Python3.11／Torch2.7 **CPU** 生产离线 fixed-test 评估：46,592 窗口、465,920 位置的 normalized MSE 为 `2.041487108986725e-5`。该 CPU 评估与原支持环境 H1／10／100 分开记录；test 不参与选模，未使用额外 GPU。[离线结果及模型 SHA](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/offline-tests/cartpole-000500000-current-cpu.json)。

[500,000 步独立审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/cartpole-500000-audit.json)核对了 45 项 Adam 游标、分段学习率、冻结 RMS、官方采样及数据指纹、源码和导出位模式，并从 NPZ 复算三模式指标与失败数。观察器只记账一次，全部 16 条轨迹保留；该节点不表示完整预算结束。

Ant 的实际 500,000 步节点也已在 Python3.8.20／Torch2.2.2／Warp1.5.1 官方支持环境完成固定 16 条轨迹复评：

| Ant 500,000 步端点 qMAE | passive | random_effort | fixed_policy |
|---|---:|---:|---:|
| H1 | 0.00052551 | 0.00055877 | 0.00053560 |
| H10 | 0.0137785 | 0.0126133 | 0.0151508 |
| H100 | 0.137212 | 0.259807 | 0.123302 |
| 官方 LFS 原默认 H100 | 0.195290 | 0.204721 | 0.140644 |

三个模式的 H100 qMAE 均较 50,000 步继续下降；其中 passive、fixed_policy 的该项均值已低于官方 LFS，random_effort 仍高约 26.9%。random_effort 的 H100 qdMAE 仍为 `2.03222`，高于官方 LFS 的 `1.53084`。三个模式的神经模型物理失败数分别为 8／8／6，teacher 为 8／9／6，所有失败轨迹均保留在 16 条均值中。这些结果限于当前固定语料与指标；公开模型数据量不同，无法据此单独归因训练实现、环境或物理参数。fixed_policy 保持同一 PD 规则，各模型按自身预测状态生成反馈动作。

逐轨迹比较进一步限制了均值结论：三个模式 H100 qMAE 优于 LFS 的轨迹数为 7／16、5／16、8／16；fixed_policy 的 qMAE 中位数为 `0.10015rad`，高于 LFS 的 `0.06886rad`。random_effort 的 H100 qdMAE 从 50,000 步 `1.95453` 回升至 `2.03222`，而同一前缀平均从 `1.08781` 降至 `0.87030`；端点与前缀分别记录。其 H100 逐轨迹 qdMAE 的 P90 为 `5.03614`，LFS 为 `3.25634`，速度长尾仍待改善。qdMAE 沿用 14 维混合速度的原定义，包含基座线速度和角速度，未统一标为 rad/s。完整分位数、配对差值、离群轨迹及首次失败步数见[Ant 500,000 步分析](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/ant-500000-analysis.md)与[逐项证据](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/ant-500000-analysis.json)。

random_effort 的 16 条轨迹在 H100 端点的最大单关节速度绝对误差为 `39.37205rad/s`，对应 seed1000008 的 hip_1；teacher 与模型在此前均未触发协议的物理失败规则。该极值与“逐轨迹平均速度误差最高的轨迹”分别统计，数值有限或未触发失败规则均不能代表速度精度已经达标。

[分布统计独立复核](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/ant-500000-analysis-review.json)从原 NPZ 复算配对计数、分位数及 72 项同单位关节速度极值，29 份输入 SHA 均匹配。最差均值轨迹与全部轨迹最大分量的范围澄清已落实，旧版分析归档，原模型和评估结果保持不变。

同一 Ant 导出模型在当前 Python3.11／Torch2.7 **CPU** 的完整固定 test 上得到 normalized MSE `0.0011545916441009268`，仍为 46,592 窗口、465,920 位置。相较 5,000 步记录明显下降，但两次 test 分别运行在 CPU 与 GPU，不构成同设备逐位比较。test 不参与选模，完整预算继续。[离线 test](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/offline-tests/ant-000500000-current-cpu.json)和[Ant 500,000 步独立审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/ant-500000-audit.json)保留模型 SHA、45 项 Adam 更新数、初始 RMS、采样、1,480 份源码及三个模式的完整轨迹核对。

## Cartpole 4,995,000 步终点与验证选定 best

Cartpole 的 last 达到 `999 × 5,000 = 4,995,000` 次实际更新，best 由固定 validation 选在 4,940,000 步；test 和 rollout 均未参与重选。[Parent completion](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/training/cartpole/completion.json)绑定的当前训练环境 GPU1 生产离线 fixed-test 如下，两次均覆盖 46,592 窗口、465,920 个有效位置，使用显式 `reference`／math：

| Cartpole 权重 | 实际更新数 | 固定 test normalized MSE |
|---|---:|---:|
| 终点 last | 4,995,000 | `1.518426341927072e-07` |
| validation-selected best | 4,940,000 | `1.4898648293093266e-07` |

以下为原 Python3.8.20／Torch2.2.2 CPU／Warp1.8.0 支持环境、每模式固定 16 条 held-out 轨迹的 H100 端点均值。**q 总 MAE 是 m 与 rad 的混合数值平均，qd 总 MAE 是 m/s 与 rad/s 的混合数值平均，均没有单一物理单位。** 滑块位置／杆角／滑块速度／杆角速度的四分量单位表，以及完整 H1／10／100 端点、前缀指标，见[Cartpole 完整预算分析](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/cartpole-full-budget-analysis-20261021T083836Z.md)和[逐项 JSON](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/cartpole-full-budget-analysis-20261021T083836Z.json)。

| 模式 | LFS q 总 MAE | last q 总 MAE | best q 总 MAE | LFS qd 总 MAE | last qd 总 MAE | best qd 总 MAE |
|---|---:|---:|---:|---:|---:|---:|
| passive | 0.021108404 | 0.0028777127 | 0.0028534411 | 0.077126942 | 0.0083973707 | 0.011800601 |
| random_effort | 0.061312363 | 0.0043903147 | 0.0043696053 | 0.24482719 | 0.011709001 | 0.018219153 |
| fixed_policy | 0.017129084 | 0.0027469154 | 0.0030762274 | 0.040733926 | 0.0063099302 | 0.010844845 |

last 与 best 在三个模式的 H1／10／100 端点及前缀 q／qd 聚合均值均低于原默认官方 LFS；**该优势不覆盖每条轨迹**。[独立原始轨迹复核](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/cartpole-full-budget-analysis-review-20261021T094012Z.json)确认 81 份输入 SHA、15 份 NPZ 和 540 项分布重算，最大数值差 `3.608224830031759e-16`。每模式全部 16 条轨迹均保留，包括 H100 累计物理失败；passive／random_effort／PD 的 teacher、last、best 失败数均分别为 13／14／12。数值有限和失败数相同不表示所有轨迹稳定，逐 seed 的首次失败时间也并非全部相同。

best 的 H100 qd 总 MAE 均值比 last 高 40.5%／55.6%／71.9%（passive／random_effort／PD），速度尾部也更大：random_effort 最差逐轨迹 qd 总 MAE 为 best `0.1458285`、last `0.0447483`；PD 为 best `0.0999441`、last `0.0233717`。best 的 fixed-test nMSE 较低不能保证长 rollout 速度更好；保留验证集选择和终点 last，不用这些结果重选模型。

passive／random_effort 使用相同记录动作；PD 只保持同一反馈规则及 teacher 动作，各模型依据自身预测状态计算实际动作，动作可以分叉。seed1000006 的 t49→t50 中，last 与 LFS 位于 ±π 两侧，环绕后的物理杆角差约 `0.000608rad`，实际滑块力却相差约 `314.17N`；这是边界附近 2π×50 增益跳变，不能将其称为逐位相同输入动作，也不能将表示差当作约 6.28rad 的真实角误差。

公开 Cartpole LFS 配置指向 5M 数据，本轮约 1M，公开权重实际完成的更新数未知；本次不是同数据复现，也不能从数据、预算、数值策略和设备均不同的比较作单因素因果推断。官方默认 LFS baseline 与旧 40 组产物保留。Cartpole 满预算只证明上述训练／恢复／导出／离线推理合同通过，manifest 仍为 `unqualified`／`offline_only`，不赋予现代 Newton 部署资格；Ant 及两机器人整体目标仍未完成。

## 官方固定权重的注意力推理对照

另在同一支持环境对两份真实 LFS 权重增加六项 math-only 推理，保持 45 项参数、全部 RMS（含 count）、19 类物理字段及固定 teacher 前缀。此项只改变推理后端，没有将公开模型重新训练或标为 reference 训练；原默认 baseline 和既有观察器的 82 个受保护文件保持原 SHA。

| 官方权重 | passive H100 qMAE 差 | random_effort | fixed_policy |
|---|---:|---:|---:|
| Cartpole：math − default | −1.0682e−6 | −1.3225e−7 | −5.7835e−7 |
| Ant：math − default | +7.5128e−5 | −5.8886e−5 | +1.4384e−5 |

六项 H1 预测均为零差；18 个 H1／10／100 聚合及 288 条逐轨迹记录已独立从 NPZ 复算。这里 H100 端点平均 qMAE 的后端差值很小，将官方模型也置于 math 后，本轮模型与它的该项平均差距基本不变。该结论不涵盖所有指标或整条轨迹的一致性。

Ant random_effort 的 seed1000012 轨迹在第 80 步，原 Y-up spatial ν_y 分量差达 5.8726m/s，native `qd[8]`（hip_2 速度）差达 3.3945rad/s；同一前缀的关节角最大差约 0.03066rad。两个速度峰值并非 2π 绕回。两分支在第 64 步均已触发物理失败，失败后的轨迹仍完整保留。passive／random_effort 的实际动作相同；fixed_policy 只保持同一 PD 规则，两分支按各自预测状态计算反馈，实际动作可以分叉。

完整结果见[推理后端对照](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/evaluation/lfs-attention-control/REPORT.md)及[独立复核](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/lfs-attention-control-review.json)。

## 后续节点和证据

Cartpole 已完成 4,995,000 步及独立终点验收，Ant 沿原配置和原 GPU6 任务继续完成实际 4,995,000 步；独立 CPU 观察器按既定节点保存实际已提交检查点并复评固定轨迹。固定 test、长预测和训练预算分别记录，Ant 仍需终点 last、验证选定 best 的导出和原支持环境验收。该历史物理训练不自动取得当前 Newton 部署资格，原始物理 metadata 不会被替换。

本轮 focused CPU／上下文检查为 170 passed、6 skipped、5 deselected；Black26.3.1 全树检查、2748/2748 API 覆盖和 context-map 检查通过。另有真实 GPU 恢复和 CPU launcher 负测，拒绝错误协议／锁、分数或非有限 Adam 游标及错误 moment。项目上下文选择 revise：保真导入、采样格式和持久化数值策略改变了跨模块合同，已更新 neural-dynamics overview。

旧产物另经只读保留审计：40 个旧计划 cell 的冻结 identity／artifacts，以及 8 个已停止检查点的原始 SHA 全部匹配；共检查 175 个文件、2,211,814,225 字节。原官方 1,000 步默认基线的选模权重、portable 权重、源码和原始训练环境标识保持一致，未发现旧队列／旧 spec 训练进程。该审计只证明保留与停止状态，不将旧检查点计入本轮完整预算。

- [冻结协议](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/protocol.json)
- [正式 GPU 验证](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/validation/completion.json)
- [5,000 步及恢复后的独立审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/live-prefix-audit.json)
- [旧基线与 40 组产物保留审计](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/readiness/legacy-preservation-audit.json)
- [Cartpole 配置](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/configs/cartpole.json)／[Ant 配置](/workspace/exp_data/embodichain-pr822-accuracy-20261021/full-budget-v1/configs/ant.json)
