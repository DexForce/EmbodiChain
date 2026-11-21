# DexUni 实验计划

## 2026-10-27 当前安排

C03 已按用户要求删除；当前整理已有结果。S01-C/T、S02–S04、M01–M06、C01–C02共13项的最新状态、性能层级和视频统一见[当前对比总览](../../../../../../outputs/benchmarks/tech_reports/dexuni/COMPARISON.md)。不新增仿真或测量。C02保留EC、Isaac96×24成功版与24×12失败版；已完成单次全程和低配短段摸测。

下文保留各阶段历史执行安排，若与上述总览冲突，以总览及最新用户指示为准。旧调参文件已按用户要求精简，见[清理账本](../../../../../../outputs/benchmarks/tech_reports/maintenance/2026-10-27/README.md)。

有效结果见[运行目录索引](../../../../../../outputs/benchmarks/tech_reports/dexuni/README.md)。S03已完成10×10两平台各五次性能测量，只保留[outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift](../../../../../../outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/REPORT.md)。全部数据和已确认视频合并于同一目录，旧S03版本已按用户明确要求删除。
计划、任务说明和源码统一由 tech_reports 管理；生成结果位于 outputs/benchmarks/tech_reports。旧 dexuni_benchmark 已按用户要求删除，当前 S02 正式结果保留。

## 当前任务四的确认顺序

1. 先在原生DexSim实现W1双侧分别夹持刚块/软块场景，并录一段预览。
2. 用户看过并确认后，再在EmbodiChain和Isaac Lab实现同一场景。
3. 两平台视频再次确认后，才冻结参数并执行每平台五次性能测量。

每个任务使用一个固定输出目录；未确认调参预览替换，不添加多个版本目录。布/软体最终形态不作为性能样本筛选条件，物理质量不排名。其它严格矩阵要求见历史FULL_PROTOCOL，仅作为参考，当前用户流程优先。

## 实验 1：S01-C/T 固定配置五次重复

- 比较：EmbodiChain + DexUni 与 Isaac Lab + Newton MuJoCo Warp/VBD。
- 每子任务每平台固定 `s00 + nominal`，独立启动进程 5 次，共 20 次采样；重复间轮换平台先后。
- 两侧共用输入快照、资产与任务目标；保留各自原生控制优化。结论描述整体平台表现，不单独归因于求解器，不标 matched_replay。
- 不扫网格、材料、子步、扰动或并行规模。先用既有可工作的参数：S01-C 10 子步/40 VBD 迭代；S01-T 以冻结 JSON 为准。
- 采样不启用相机，逐帧保存状态供质量复核。每子任务每平台另录一次完整视频，共 4 次媒体运行；媒体计时不进入主表。
- 预先排除最初 30 帧（0.5 s 仿真初始稳定阶段），包含首次图捕获开销；状态连续演化，不做恢复/重置。主表是后续 episode 的同步公开 step 均值，再对五次成功样本报告均值及样本标准差。
- 两侧同一高层闭环规则可能产生不同 episode 长度，因此这是各自任务运行窗口的整体 step 耗时，不是同一状态轨迹的纯求解器吞吐。
- step 同步覆盖所属平台；目标计算、主机读回、状态记录、渲染和写盘不计入 step。另存初始化、前 30 帧、读回/记录/采集、进程总耗时。
- 保留每次尝试和日志，失败写明原因。仅成功样本汇总耗时，另列成功/尝试数量；不宣称五次成功等于鲁棒性或可靠性排名。
- 不在这轮扩展严格材料标定；报告几何响应，不声称物理精度更高。

## 质量与视频

S01-C：5% 压缩，保持末段均值容差 ±2 pp；恢复误差≤5%，全程压板相对 TCP 的 Z 漂移≤3 mm，台面节点下穿≤4 mm，粒子状态有限，目标到达。
Z 漂移不能表述为三维夹持滑移。

S01-T：5% 拉伸，保持末段均值容差 ±2 pp；恢复误差≤5%，固定端/夹持端漂移≤3 mm，夹具/台面节点检查≤4 mm，粒子状态有限，目标到达。
材料长度由冻结材料线节点计算；松爪后的恢复不用于推断塑性。

正常视频严格按 30 Hz 写入；阶段切换的额外采样仅保存在图像数组，不插入固定帧率 RGB 流。
合成以仿真时间映射源帧，标注时间/阶段，短片结束后显示 ended，不延长成仍在运行的画面。

## 目录与运行

- `physics/experiments/dexuni/`：本计划、编排、任务说明。
- `physics/platforms/dexuni/`：各平台独立进程入口。
- `assets/physics/dexuni/`：小型配置定义；机器人保留外部资产引用。
- `reporting/dexuni/`：离线指标与媒体合成。
- `outputs/benchmarks/tech_reports/dexuni/<batch>/`：输入/代码快照、日志、原始数据、报告、视频，由仓库 outputs 忽略规则覆盖。

入口见 [SCRIPT_CONTRACT.md](SCRIPT_CONTRACT.md)，执行状态见 [STATUS.md](STATUS.md)。
后续任务沿用同一精简原则；早期任务定义、严格控制/材料/矩阵方案保留在 [FULL_PROTOCOL.md](FULL_PROTOCOL.md) 作参考。

## 实验 2：S02 大变形抓取搬运

第一组和第二组均已完成；第二组 [S02](s02_soft_pick_place/README.md)，沿用固定配置五次重复、独立媒体、保留原始数据的规则。
增加原生 DexSim–EmbodiChain 的临时适配检查，检查不通过先修适配；DexSim 数据不进入报告或归档。
执行状态见 [S02 STATUS](s02_soft_pick_place/STATUS.md)。

## 实验 3：S03 布角提起与释放

[S03](s03_cloth_corner_lift/README.md)已完成，最终预算10子步/10次VBD迭代，两平台各五次无相机运行，全部CUDA Graph captured。报告、原始绘图数据、输入及已确认视频统一在outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift。状态见[S03 STATUS](s03_cloth_corner_lift/STATUS.md)。

## 实验 4：S04 刚块/软块双侧夹持

任务定义见[S04](s04_rigid_soft_squeeze_grasp/README.md)。原生W1场景已获用户确认，EmbodiChain和Isaac Lab实现及并排预览已完成。EC默认十倍外步的问题已修正。DexSim SDK 修改后两平台各五次正式测量完成：EmbodiChain 14.839 ± 0.396 ms；Isaac Lab 15.098 ± 0.365 ms，EC 外步耗时低 1.72%；全部Graph captured、实际12秒/720外步。SDK 重测前的批次归档，不作为当前结果。[报告](../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/REPORT.md)。

## M05 当前执行阶段

指定demo的弹性气袋直接平桌夹放，W1/Pika运动学one_way，30mm/4kg动态块提速压载，总35s。表压16kPa、物理rest bulge .030、气体bulk3e6；四指/块/桌面完整表面接触候选均核验，复核袋表面桌面穿入量。DexSim场景已获用户确认，EC完整等价复现、主视角视频和原生/EC并排完成，等待EC视频确认后再测量。当前Isaac Lab28.0.0/原生Newton1.6.1缺少封闭等温气压腔体，unsupported，无替代视频或性能样本。[M05报告](../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/REPORT.md)。Graph和增量体积更新开启，pneumatic排除surface-fast；原生/EC分别只有一套有效版本，未改SDK。

## C01 当前执行阶段

用户已确认正中视角、左面包后右牛奶的原生场景，EmbodiChain 完整 28.1 s 移植和并排视频已完成。两次独立完整运行均有限、Graph captured、两件留存、无倒置四面体，离散表面检查通过；冻结输入、粒子接触2000/.25/.4、原有气腔/FilmDrag及普通VBD24×16保持。EC 当前视频末秒袋速度P95约0.177mm/s，中心最大轴摆幅0.032mm。按最新授权已完成[两边各一次性能摸测](../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/pilot_performance/REPORT.md)：EC双商品147.744ms/外步，Isaac单面包92.325ms/外步，无相机、普通24×16、前30步预热排除；不同负载不计算加速比，尚未正式多次测量。Isaac Lab28.0.0/nativeNewton1.6.1仍缺少等温气腔；按用户后续明确要求，已完成无牛奶的单面包15.7s版本，独立试跑和完整录像均通过，472帧正中视角；原生/EC双商品保持原样，不作跨负载性能比较。[平台报告与视频](../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/platforms/REPORT.md)。

C01 最新执行更新：正式测量只包含EC完整双商品场景，五次独立进程132.132948 ± 1.562131ms/外步，全部完整28.1s/Graph/有限状态、无倒置、两件留存；不包含Isaac Lab正式结果。[正式报告](../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/measurement/REPORT.md)。下一任务为C02袋框装填与提袋，先验证可复现几何和原生预览。

## C02 当前执行阶段

用户否定了自建橙色硬框和小方块的旧版，现已直接基于指定的 `demo_dexuni_w1_bag_packing.py` 重做。[当前视频](../../../../../../outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/preview/dexsim.mp4)、[记录](../../../../../../outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/REPORT.md)。原纸袋1381节点/2712面、柔性提手、罐头、纸盒和桌面，原建场景函数/控制器/材料/接触/12×6求解配置不变。先完成50秒原demo扶正装填，再追加右手换抓对侧袋口、双手带载提起并放回，总64秒。一次无相机试跑和一次完整录像均3840外步/3841有限状态/Graph captured，两件保持并最终留存；录像袋体实际平均提升77.700mm，保持时袋底最低离台69.790mm。两次全部3841实际姿态的机器人非相邻链节/桌面FCL检查通过，原每步几何检查与每10步提手穿纸检查保留，不声称连续无碰撞或穷尽袋壁自交。原demo没有独立刚性框，不再增加外框、粒子跟随或外力适配器。旧版完整归档并标用户否定；当前仅一套有效视频，待用户检查，未移植平台和测C02性能。
