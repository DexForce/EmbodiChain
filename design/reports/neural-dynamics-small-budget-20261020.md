# NeRD 官方入口小预算训练对比 — 2026-10-20

本轮按用户更新后的完成条件结束：四项主训练各完成 1,000 次优化器更新，固定测试和 1／10／100 步预测结果齐备，EmbodiChain 当前依赖环境的生成、训练产物、导出与原生推理链路已核验。**跨环境精度尚未完全对齐；本轮不宣称完整质量验收。**

40 组全预算计划及后续评估已停止。实际停止本任务的 8 个训练进程和 15 个调度／等待进程；先停止调度父进程，再停止训练子进程，未对其他任务使用批量终止。8 个已提交检查点均保留并记录 SHA256 和提交步数，日志、配置、数据和冻结代码均保留。没有将旧计划的 4,995,000 步预算改写成 1,000 步，也没有将部分结果标记成原全预算完成。

## 官方基线与统一条件

官方代码为未修改的 `NVlabs/neural-robot-dynamics@370e89d2b8ce68aa3578aa7812859c681b7502bc`。本轮实际运行原始 `train/train.py`，保留原构造器、TrajectoryDataset、DataLoader shuffle／drop_last、FP32 RunningMeanStd、损失、epoch-zero 选模、原始 rollout 评估与独立的 best-valid／best-eval 保存逻辑。原始 YAML 只调整已有数据路径和 `num_iters_per_epoch=1000`；`num_epochs=1000` 保留原学习率分母。预算守卫在首个训练 epoch 的验证、评估和保存全部完成后，阻止第二个训练 epoch 的更新。官方实际每次更新学习率均为 `0.0009991000000000002`。迁移恒定端点的插值日志有时为 `0.0009991`，与官方标量相差一个 float64 ULP（最大 `2.168404344971009e-19`）；逐步检查含一阶矩偏差修正的 FP32 Adam 更新系数全部一致。原始日志保留该差异，未改写。

| 条件 | 官方 | 迁移 |
|---|---|---|
| 环境 | Python 3.8.20、Torch 2.2.2+cpu、NumPy 1.24.4；Cartpole Warp 1.8.0、Ant Warp 1.5.1 | 当前干净依赖环境 Python 3.11.14、Torch 2.7.0+cu128、NumPy 2.4.6 |
| 主训练设备 | CPU，8 个 Torch 线程 | CPU，8 个 Torch 线程 |
| 模型 | Cartpole 192／Ant 384 维，6 层、12 头、history 10，原始完整模型 | 相同；原始初始权重逐项转换，不依赖“相同 seed”推断 |
| 数据 | 已有官方支持环境生成的原始文件，实际训练 1,024,000 transitions；验证／测试各 51,200 | 复用相同 split；Ant 保留原始 Y-up，避免额外旋转引入误差 |
| 采样与归一化 | 原始 loader 实际消耗的索引；原始 FP32 RunningMeanStd | 重放捕获的实际索引，直接使用官方初始权重和统计量 |
| 优化器 | Adam，betas 0.9／0.999、eps 1e-8，FP32，梯度范数上限 1 | 相同；无 AMP、梯度累积或模型缩小 |
| 验证选择 | epoch 0 和预算结束后分别验证，保留原始选择流程 | 使用相同验证索引和 epoch-zero／结束时的最小损失规则；四项主模型最终均选择第 1,000 步 |

迁移数值更新调用生产 `DynamicsTrainer.step`，导出调用生产工作流 API；实验外层按捕获的官方协议驱动。它与默认 CLI 的有放回采样、FP64 流式统计和验证顺序不同，不能混称为默认配置比较。

明确的执行差异：官方旧 Torch 在本机采用 CPU；原入口的无条件 CUDA-stream 请求通过 `wp.stream_to_torch(cpu) -> None` 适配，后续 `torch.cuda.stream(None)` 为原有空操作。未改模型、积分器或数值计算方法。主机 OS／CUDA 硬件并非论文完整测试平台，本轮只对已记录的实际环境作结论。Ant 原始 spatial velocity 导入 native COM 再还原会有 FP32 舍入；首批归一化输入最大差为 0，目标最大差为 `9.5367431640625e-7`。Cartpole 首批输入和目标均为 0 差。

## 固定测试集结果

下表为归一化 MSE，越低越好。每个测试集包含 512 条未参与训练或选模的轨迹、46,592 个完整滑窗／465,920 个预测位置；重叠位置不是独立样本。官方数字由原始模型在支持的 Python／Torch 环境中计算。共同 EmbodiChain 离线评估器也重新测量官方导出，差异仅为浮点舍入。

| 机器人 | 官方原始入口训练 | 迁移训练 | 相对变化 |
|---|---:|---:|---:|
| Cartpole | 0.0062803502 | 0.0107227133 | +70.73% |
| Ant | 0.0252724139 | 0.0256617870 | +1.54% |

Cartpole 存在明显的跨环境差异。Ant 单步测试变化较小，但仅一个模型种子不足以建立统计非劣结论，且长预测差异更大。

![训练与固定验证曲线](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/figures/training-validation.png)

图中保留所有训练日志，实线为 25 步滑动均值；固定验证曲线在 0／250／500／750／1,000 步使用相同的 4,096 个验证滑窗，仅用于观察，没有改变选模。完整原始值见训练 CSV。

## 差异归因

**训练实现。** Cartpole 在相同 Torch 2.7／CPU 条件下，官方数值训练方法与迁移实现经过完整 1,000 步后，所有模型参数及 Adam 状态的最大差均为 **0**。该控制重放同一初始权重、统计和源 loader 索引，属于环境归因实验，不能替代真正的官方入口基线。Ant 的同环境 10 步短检查，全部模型参数最大差也为 **0**；没有把这 10 步外推为完整 1,000 步证明。

**运行环境。** Cartpole 同一原始模型、同一真实首批数据在 Torch 2.2／2.7 下，首步梯度最大差 `2.2351741790771484e-8`，一次 Adam 更新后的参数最大差 `4.614586941897869e-6`。微小差异随后累积。相同当前环境下 1,000 步参数和优化器完全一致，表明环境变化足以解释本次 Cartpole 的最终差异。额外保留的 GPU 诊断测试 MSE 为 `0.008951334292990165`；设备变化也影响训练轨迹，故未将它作为主公平比较。

**推理与物理。** 原始官方 checkpoint 在固定输入上的 Torch 2.2 前向与当前 portable 前向，最大绝对差分别为 Cartpole `1.3709068298339844e-6`、Ant `1.0967254638671875e-5`。训练主比较的两边读取同一官方数据，物理参数没有成为训练变量。1／10／100 步比较也复用同一受支持环境生成的 teacher 前缀，并让两套权重在同一原始 Python 3.8／Torch 2.2／旧 Warp 积分器中独立演化。因此这些比较不混入当前 Newton 重新生成的真值。Ant 的跨版本反向差异与原始速度导入舍入尚未完全拆开，不应将全部长预测差异直接归咎于迁移代码或 Newton 参数。

## 固定少量轨迹的 1／10／100 步预测

每个场景固定 16 条轨迹，seed `1000000..1000015`，来自已冻结、已执行初态 holdout 审计的官方 corpus。官方与迁移两套权重逐轨迹 teacher 内容哈希一致，teacher 未重新仿真。被动、随机 effort、固定 PD 均覆盖；PD 为声明的固定基线，使用各自预测状态重新计算动作。

以下为第 h 步端点 q MAE，格式为“官方 → 迁移”。Ant q 为 8 个关节角；Cartpole 汇总滑轨位置与杆角，不能与 Ant 的数值横向比较。qd MAE、Ant 根位置／姿态误差、前缀均值以及失败计数均在 CSV／JSON 中。

| 机器人／场景 | 1 步 | 10 步 | 100 步 |
|---|---:|---:|---:|
| Cartpole／被动 | 0.003603 → 0.004517 | 0.033924 → 0.052967 | 0.929975 → 1.018315 |
| Cartpole／随机 effort | 0.004139 → 0.005739 | 0.033720 → 0.061154 | 0.737382 → 0.974834 |
| Cartpole／固定 PD | 0.003915 → 0.004543 | 0.036591 → 0.054184 | 0.770007 → 1.189118 |
| Ant／被动 | 0.005320 → 0.004744 | 0.128578 → 0.127550 | 0.472263 → 0.568879 |
| Ant／随机 effort | 0.006481 → 0.006159 | 0.098915 → 0.108031 | 0.334289 → 0.455908 |
| Ant／固定 PD | 0.005306 → 0.004745 | 0.121987 → 0.126089 | 0.459634 → 0.500939 |

Ant 的 100 步 q MAE 在被动和随机 effort 场景分别增加约 20.5% 和 36.4%；固定 PD 约增加 9.0%。所有 12 项评估均保留全部 16 条轨迹，100 步内没有非有限状态；物理失败没有被剔除。两个仅训练 1,000 步的模型都存在较高失败率，不能把这组结果解释为最终收敛精度或完整部署验收。

![固定轨迹预测曲线](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/figures/rollout-errors.png)

## 当前 SDK 的完整流程

DexSim 使用 MR !1483 的 `b61e920fbb222e50213574ad57d885e7b96148b9` CI wheel（`dexsim-engine 0.5.1`），当前依赖为 Newton 1.7.0／Warp 1.17.0／Torch 2.7.0+cu128，`pip check` 通过。两个机器人重新执行真实 CLI 数据生成、导出、原生 NeRD provider 推理均成功。生成使用 `physics_profile=warp_reference`、Cartpole 5／Ant 16 个子步，质量矩阵更新间隔对应 5／16，物理签名和重力检查实际通过。仅使用物理 GPU 0／1，未使用 GPU 7。

训练阶段复用了本任务此前已经完成的两项原生数据 1,000 步训练，检查点和训练结果哈希均已核验，未追加原生训练预算。新生成数据、重新导出的模型和原生推理结果均在 `native-workflow/{robot}`。该流程检查是独立的 native-data 模型证据：官方历史数据模型缺少当前原生物理签名，未伪造它的 metadata 来绕过部署限制。

## 本轮结论与下一步

本轮完成了“小预算是否退化”的判断和工作流核验。**当前不能宣布整体精度已对齐**：Cartpole 的跨环境训练差异明显；Ant 的单步变化小，100 步部分场景仍有较大误差。当前环境下的 Cartpole 1,000 步逐参数／优化器对齐，以及 Ant 10 步对齐，支持迁移更新实现正确，但不等价于跨版本训练轨迹相同。

下一步优先定位并固定 Torch 2.2／2.7 的 SDPA 反向、梯度裁剪和 Adam 数值路径，分别观察 FP32 梯度差和后续参数差；Ant 继续拆开速度导入舍入与 Torch 环境因素。确定可重复的数值设置后，再决定是否训练到 5,000 步或增加种子。本轮保持 1,000 步完成条件，旧的 40 组计划保持停止。

## 产物与验证

- [机器可读汇总](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/comparison-summary.json)
- [原始训练曲线 CSV](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/figures/training-loss.csv)
- [1／10／100 步指标 CSV](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/rollout-metrics.csv)
- [旧全预算进程停止回执](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/stop-full-budget-result.json)
- [保留检查点清单](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/preserved-full-budget-checkpoints.json)
- [实际执行脚本归档](/workspace/exp_data/embodichain-pr822-accuracy-20261020/small-budget-official-v1/executed-scripts/sha256.json)

四项主训练和 12 项固定轨迹评估均已执行完成；官方入口及 SDK CLI 子命令均正常退出。四个新实验 worker 均在 Python 3.8 下编译通过；`black==26.3.1` 已格式化整个 worktree，`git diff --check` 通过。已有生产 codec／trainer／reference／report 的测试证据继续保留，未用新实验脚本重复扩大无关测试范围。项目上下文选择不更新：本轮新增实验入口，未改变生产路由、层级所有权、默认采样或部署契约；具体复现说明更新在实验 README，设计文档已区分本轮完成条件与未来完整质量验收。

额外预算已显式区分：四项 1,000 步主训练之外，Cartpole GPU 1,000 步诊断、Cartpole 同环境官方方法 1,000 步控制、Ant 同环境 10 步检查和一批次梯度探针均单独归档；它们不是更多主实验种子，不参与主结果平均。没有执行 5,000 步续训。

GPU 诊断 v1 的首批输入观测曾把归一化前后值直接比较；已补充 `feature-comparison-correction.json`，使用同一批数据在 CPU 比较归一化后的两个输入特征，最大差均为 0。该观测问题不影响训练、检查点或本文主实验结果；原始文件保留。
