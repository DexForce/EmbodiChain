# 当前操作分工

S01/S02/S03/S04 已完成；有效报告从[runs索引](../../../../../../outputs/benchmarks/tech_reports/dexuni/README.md)进入。S03统一在outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift，10×10两平台各五次、Graph全部captured；SDK 集成后已按指令重测，当前有效值为 EC 12.722 ± 0.434 / IL 13.463 ± 0.316 ms，重测前记录归档在同目录 history/formal_before_sdk。其它S03版本已按用户要求删除，保留当前版本的完整测量及复现输入。

任务四：DexSim原生场景已确认，EmbodiChain/Isaac Lab实现及并排视频已完成；SDK 集成后两平台各五次正式测量完成（EC 14.839 ± 0.396 / IL 15.098 ± 0.365 ms）。重测前的修正节奏批次和默认外步数错误批次分别归档在 outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp 的 history/formal_before_sdk 与 measurement/invalid_cadence。每个任务一个固定输出目录，避免再产生版本目录。

M06 Franka T 恤单折：两平台各五次正式测量完成（EC 16.323 ± 0.354 / IL 36.257 ± 0.951 ms/外步），EC 第 5 次按用户要求重测替换；对照视频未标人工批准。

M01 软梁弯折两平台各五次正式测量完成（EC 14.275 ± 0.433 / IL 14.852 ± 0.045 ms/外步，见 outputs/benchmarks/tech_reports/dexuni/m01_soft_beam_bend/REPORT.md；测量期间 M03 原生预览曾与 EC 第 1 次重叠，该次已作废重测）；M02 布料单折已改为单向 one_way / kinematic（中线夹起、抬高、平铺对折），三平台场景完成，两平台各五次正式测量完成（EC 9.194 ± 0.169 / IL 12.488 ± 0.341 ms/外步），见 outputs/benchmarks/tech_reports/dexuni/m02_cloth_single_fold/REPORT.md。

计划和源码统一在 tech_reports，结果在 outputs/benchmarks/tech_reports；旧 dexuni_benchmark 已删除，当前 S02 正式结果保留。原生适配诊断临时运行并清理，用户明确要求的S04原生预览视频可保留。
