# C01 超市装袋：左面包、右牛奶

C01 当前正式结果只包含 EmbodiChain 完整双商品场景：五次独立进程，**132.132948 ± 1.562131 ms/外步**，全部实际28.1s/Graph/有限状态、两商品留存且无倒置四面体。Isaac Lab不参加此场景正式测量。[正式报告](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/measurement/REPORT.md)。下文早期单次摸测为历史记录。

当前 [28.1 秒原生预览](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/preview/dexsim.mp4) 已获用户确认，采用正中固定视角，为左手先放面包，左臂退回后右手再放牛奶。[EmbodiChain 完整视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/platforms/embodichain/preview/video.mp4)及[并排对照](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/preview/comparison.mp4)已完成；两次完整运行通过状态、持握、形变和离散碰撞检查。Isaac Lab 按用户后续要求改为[单面包无牛奶完整预览](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/platforms/isaaclab/preview/video.mp4)，15.7秒，独立试跑与完整录像均通过；原生和EC仍为两件商品。最新已完成[各一次无相机性能摸测](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/pilot_performance/REPORT.md)：EC双商品147.744ms/外步，Isaac单面包92.325ms/外步；不同负载，不计算加速比，尚未正式多次测量。[平台报告与复现命令](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing/platforms/REPORT.md)。

面包输入镜像到 Y+0.22 m，面包/奶包落点分别为 Y+35/−35 mm。左臂回位新增 1 秒，保留末段 4 秒静置。面包释放沿用 +25 mm；奶包释放 TCP 增加 40 mm，以避让换侧后的袋提手。奶包仍抓远侧封边，2.6 mm 夹缝，2 秒松爪。

上一版接触系数 2000/0.25/0.4、膜/四面体材料、气腔、原有空气阻力、左右手指及支架接触不变。数值子步从 8 提至 16，普通 VBD24 次、1/60 s 外步、one_way/kinematic、Graph 保持；8 子步独立完整录像曾有一个瞬时倒置四面体，已明确拒绝并保留原始记录。没有额外袋膜耗散力或状态修正。

三次 16 子步完整运行均有限、两件入袋、抓取稳定、无倒置四面体。相机调整沿用相同冻结命令和物理设置。当前录像末秒袋节点速度 P95 为 0.000274 m/s，袋中心最大单轴摆幅 0.107 mm。全部 1687 姿态的手部/台面以及放下后 481 姿态的商品间表面检查未检测相交；相同命令的机器人自身/台面/支架检查亦通过。检查为离散几何检查，不证明连续无穿透或完整包含。

使用用户确认的 dexsimforembodychain 暂存 SDK，显式 PYTHONPATH 与 conda dexsim1。输入、执行源码、状态、图表原数组、全部尝试和发布哈希见报告；仅保留一套当前媒体。

平台入口 `physics/platforms/dexuni/{dexsim,embodichain}/c01.py`，Isaac Lab 的 `c01.py` 仅审计当前版本气腔支持，`c01_bread.py` 运行用户授权的单面包变体；布局及阶段 `physics/experiments/dexuni/c01_common.py`，`GOOD_ARMS` 显式指定商品所属手臂。EC 录像末秒袋节点速度 P95 为 0.000177 m/s、中心最大轴摆幅 0.032 mm，面包无倒置四面体。移植保留冻结输入及普通 VBD24×16，EC 地面与视觉 UV 差异见平台报告。沿用上级 PLAN、PARITY_CHECKLIST 和 SCRIPT_CONTRACT 的预览确认流程。
