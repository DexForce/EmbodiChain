# M04：W1/Pika 将 T 恤对折一次

使用原始 W1 V030/Pika 夹爪和指定参考demo的完整T恤，初始全部平铺在一张固定矩形桌面上。双手夹起近端衣摆，另一端沿桌面滑动，向领口方向翻折一次；布面搭回桌面后松爪、垂直退让。无额外支撑、附加夹爪板、固定布节点或SDK改动，外观用默认FabricGray。

[当前视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/preview/dexsim.mp4) · [完整设计、参数与结果](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/REPORT.md) · [滑移和伸长曲线](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/analysis/sliding_response.png)

当前配置为one_way/kinematic/soft/surface-fast，10子步、显式16次基础迭代及自适应收敛，CUDA Graph captured。衣服质量、膜与弯曲机械参数沿用demo；按最新滑移要求，布料/桌面摩擦均改0.1，夹持时指摩擦2.3，有效布/指摩擦仍1.2，并加8/s被动空气阻力。释放TCP在桌面上方55mm。

实际32秒/1920外步、841视频帧。提起时桌面端滑动81.18mm、同阶段边长95分位最大伸长7.95%；最终下摆误差22.43mm，原夹持/支撑限值和新增滑移/伸长判据通过。目标19201、实际385状态FCL采样无所查相交，非连续保证；仍有局部弹性伸长及自然褶皱。

原生最终松爪视频已由用户确认。EmbodiChain与IsaacLab移植已完成，使用同一冻结输入与松爪节奏；[跨平台报告与并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/PORTS.md)已保留。正式性能测量完成：EC21.672±0.839ms/步、IsaacLab71.486±2.955ms/步（五个独立进程均值±样本SD），EC耗时低69.68%；两侧原六项任务判据均5/5通过。详见[正式性能报告](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/measurement/REPORT.md)，[单次预试](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/pilot_performance/REPORT.md)另存。复现命令、参考差异、曲线原始数组及源码哈希以完整报告为准。旧M04棱台任务已被本定义替换，不再保留其有效版本。

本次仅修正26秒以后的松爪与退让：2秒缓慢张开，手指摩擦1.6秒五次平滑降低；0–26秒指令逐元素不变，所有物理系数保留。[具体校验与指标](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m04_bimanual_tshirt_single_fold/analysis/release_comparison.json)。
