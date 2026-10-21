# NeRD 数值差异归因与 1,000 步复核

完成时间：2026-10-21 00:05（Asia/Shanghai）。实验目录沿用起始日 20261020。

三个子 agent 完成并交叉审查了训练实现、Torch 数值路径和 Ant 数据运输三条链路。**相同当前环境、相同 canonical 数据下，两机器人迁移训练更新均已验证到 1,000 步完全一致。Cartpole 在一套明确的共同数值策略下，也完成了原环境与真实迁移训练器的 1,000 步对齐。Ant 另外证实了速度运输舍入对训练后长预测的影响。**

本报告接续[原始入口小预算比较](neural-dynamics-small-budget-20261020.md)。旧官方默认基线、原主实验数据／权重／日志全部保留；共同策略是独立受控变体，未替换旧基线、修改生产默认数值策略或恢复 40 组队列。全部新增训练使用 CPU，模型规模、B512／H10、种子 0 和实际官方采样索引均保持原约定；没有增加 5,000 步训练或模型种子。

## 同环境训练实现

Ant 将独立复制的第 10 步 source-method 控制精确恢复到 1,000 步。全部 1,000 条 loss／学习率一致；10／250／500／750／1,000 步的 59 项模型 state、135 项 Adam 张量与参数组完全一致，最大差为 0，固定测试归一化 MSE 为 `0.025661787018179893`。Cartpole 的同环境 1,000 步参数与 Adam 一致性已在上一轮完成。

该结论针对同一 canonical 数据协议：源方法也通过 transport 后读取共同数据，不能据此把 Ant 原始 legacy 数据与 canonical 数据的差异全归到 Torch。Ant 的日志 `grad_norm` 仍有约 `1.49e-6` 差异，来自报告字段的不同范数归约；实际模型和优化器更新一致。

## Cartpole：找到两处数值分叉并跑通共同策略

默认 CPU 路径的首个前向差异位于第一层 SDPA 输出，最大约 `1.79e-7`；此前线性层、LayerNorm 和 q／k／v 均一致。完全相同梯度跨两版重放裁剪和 Adam 时，更新一致，因此首步 Adam 实现不是分叉源。接近 `eps=1e-8` 的梯度扰动会被同一 Adam 公式放大，复现了首步约 `4.61e-6` 的参数差。

共同指定 math SDPA 后，第 4 步仍出现归约差异：对同一个 45 元素 FP32 参数范数向量，Torch 2.2／2.7 的总范数分别为 `1.6617134809494019`／`1.6617133617401123`，相差 1 ULP；关闭 foreach 无效。共同将这 45 个 FP32 范数的总平方和以固定顺序 FP64 累加，再转回 FP32 裁剪，消除了已观察的第二处差异。

在这个新共同策略下，独立运行原方法 Torch 2.2、原方法 Torch 2.7，以及当前环境生产 `DynamicsTrainer.step` 三个 1,000 步控制。真实迁移训练器没有重写，策略只在独立 worker 的上下文中注入。

| 检查 | 结果 |
|---|---|
| 1,000 条 loss／裁剪总范数／LR | 逐值一致 |
| 250／500／750／1,000 步模型与 Adam 张量 | 45 参数、135 Adam 张量，原始字节均一致 |
| 冻结归一化 mean／var | 6 项张量一致 |
| 同源原始固定测试归一化 MSE | 三者均 `0.007527741757068006` |
| 生产离线评估 MSE | `0.007527743803930806` |

测试末两项相差约 `2.05e-9`。独立复核对同一组 91 批均值分别执行原 FP32 累加和生产 Python 加权累加，准确复现了该差异。Torch 2.7 参数组额外存在 `decoupled_weight_decay=False`，已单列为版本元数据；这里的 Adam 一致性指张量状态和共同有效超参数，而非跨版本字典结构完全相同。

**共同策略不复现原官方默认 flash 轨迹。** 默认官方 Cartpole 测试 MSE `0.006280350161122752` 仍保留；不能拿新的 `0.00752774` 替换它并宣称默认精度通过。本证据限于一个种子、CPU、1,000 步和已直接检查的阶段，不涉及任意设备或最终收敛。

## Ant：区分推理运输与训练运输

原始 spatial velocity 与 native COM velocity 的减后加在 FP32 下不能保证逐位恢复。首个真实训练批的目标最大差 `9.536743164e-7` 已精确复现；EC target 与往返后的原始 target 一致。固定权重时，在两模型、三场景各 16 条轨迹中，每宏步增加速度往返后，完整 101 个状态点的 q、COM qd 和控制均不变，1／10／100 步指标和失败计数不变。纯状态回写 sham 也与原轨迹完全一致。

梯度探针最初误用了原模型 forward 后已归一化的冻结快照，导致再次归一化。独立审查发现后，旧脚本／结果已完整归档并标 INVALID；更正版本从原 HDF5 重建真实输入，经一次预处理和一次模型归一化。更正后的首批目标扰动使初始化／官方 1,000 步模型／迁移 1,000 步模型的最大梯度差分别约 `9.31e-10`／`1.86e-9`／`1.40e-9`，没有执行优化器更新。原目标和 rollout 证据未改。

随后只新增一项官方支持环境 Torch 2.2 的 1,000 步运输控制 B；旧原始训练 A 和旧当前环境迁移 D 直接复用。新 loop 首先在无运输条件下精确复现旧官方第 1／10 步模型和 loss；B 再仅在原预处理前对 states／next_states 做 spatial→COM→spatial。原初值、RMS、记录索引、默认注意力／裁剪、Adam、FP32／8 线程不变。

7 个实际完整训练批的输入／目标与 canonical 逐字节相同，在这 7 批内 A／B 输入相同、差异位于目标；全部 91 个完整测试批也完成 canonical 一致性核验。训练批检查是抽查，不能表述为全部 1,000 批输入逐值证明。

三套权重统一在同一个支持环境原始 evaluator 上评估 46,592 个固定测试窗口：

| 训练条件 | 共同 raw-test MSE | 共同 roundtrip-test MSE |
|---|---:|---:|
| A：官方 2.2／原始 raw | 0.025272413925 | 0.025272413925 |
| B：官方 2.2／速度往返 | 0.025085947016 | 0.025085947016 |
| D：当前 2.7／迁移 canonical | 0.025661790764 | 0.025661790764 |

运输控制 B 的单步测试略改善，但相同 held-out 轨迹上的 100 步关节位置误差增加：

| 场景 | A qMAE | B qMAE | D qMAE | A／B／D 失败数 |
|---|---:|---:|---:|---|
| passive | 0.472263 | 0.578930 | 0.568879 | 12／14／15（各 16 条） |
| random_effort | 0.334289 | 0.420823 | 0.455908 | 11／12／12（各 16 条） |
| fixed_policy | 0.459634 | 0.488172 | 0.500939 | 12／14／11（各 16 条） |

![Ant 单步测试与长预测的不同响应](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/figures/ant-training-attribution.png)

这证明运输舍入经训练后可对本种子的长预测产生可见影响；不能把 Ant 的全部长预测差异归到 Torch，也不能把 A→B、B→D 变成全局可加性的因果百分比。这些对照不是当前 Newton 与旧 Warp 物理差异的测量，没有重新生成 teacher 或调整物理参数。所有失败轨迹均保留。

D 日志第 14／19 步的 LR 有一个 float64 ULP 差，但全部有效 FP32 LR 和带偏差修正的 Adam 步长一致；D 恢复为原模型格式时，RMS count 为默认先验而非旧估计器样本数，但 mean／var 全等，冻结 normalize 不读 count。这两项元数据差异均显式记录，未改检查点。

## 观测修复与验证

本轮只修改实验观测入口和相应测试：首批物理特征在原 loss／forward 前保存，使用独立 CPU clone，写入包含数据阶段和 tensor SHA 的 sidecar；消费者按新 physical stage 或旧的无 sidecar normalized stage 比较。未知格式／阶段和错误 SHA 会拒绝。原冻结快照未重写。

10 个 focused tests 全部通过，覆盖回调在归一化前捕获、旧／新数据兼容、真实差异检测和错误 metadata。另在 Python 3.8／Torch 2.2 实际运行官方原入口 1 步，模型参数、统计量和 loss 与冻结第 1 步完全一致；实际完整首批的新／旧消费者比较也均为 0 差。Black 26.3.1 全树检查、Python3.8 编译、whitespace 和 context-map 检查通过。生产训练／物理／codec 默认值未改，原 source tracked files 无变动。

项目上下文选择不更新：本轮改变的是独立实验观察数据的阶段标识和复现说明，生产层级、配置路由与物理契约仍准确；实验 README 已更新。全部原始比较、当前同环境控制、共同数值策略控制与运输控制均分开记录。

## 后续建议

下一步应将已验证的共同数值策略整理为显式实验配置，并为官方训练复现保留原始 spatial velocity／target 的保真通道，减少导入后的训练舍入变量。确认这两项后再考虑增加一个配对种子；本轮没有启动 5,000 步或恢复 40 组。当前证据支持“迁移更新逻辑对齐”，不支持“原默认官方训练、GPU 与当前 Newton 长预测全部精度验收通过”。

详细证据：
- [Ant 同环境 1,000 步](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/ant-same-environment/README.md)
- [Cartpole 算子诊断](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/cartpole-numerics/REPORT.md)
- [Cartpole 跨环境共同策略](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/cartpole-compatible-1000/REPORT.md)
- [真实迁移共同策略闭环](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/cartpole-compatible-migrated-1000/REPORT.md)
- [Ant 推理运输／更正说明](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/ant-transport/report.md)
- [Ant 训练运输与三模型共同评估](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/ant-transport-training-1000/REPORT.md)
- [本轮机器汇总](/workspace/exp_data/embodichain-pr822-accuracy-20261020/followup-alignment-v1/summary.json)
