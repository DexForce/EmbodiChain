# M05 平放弹性充气袋与刚块压载

DexSim 更饱满气袋/提速场景已获用户确认。EmbodiChain 完整复现待视频确认；当前 Isaac Lab 28.0.0 / 原生 Newton 1.6.1 不支持封闭等温气压腔体。尚未正式测量，只保留各平台当前有效版本。

[当前报告](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/REPORT.md) · [视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/preview/dexsim.mp4) · [恢复与压陷剖面](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/analysis/surface_profiles.png)

[EmbodiChain 视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/preview/embodichain/video.mp4) · [DexSim / EC 并排](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/preview/comparison.mp4) · [Isaac Lab 支持证据](../../../../../../../outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load/analysis/isaaclab_support.md)

W1 V030 + 双 Pika，运动学具名关节回放、one_way。指定 demo 的闭合 OBJ，216 节点/428 面、scale0.30、rest bulge0.030（示例0.020），初始约108×80×49 mm。直接平桌夹袋，无塑性/pin/垫台/抓取附件；右手放袋，左手放30 mm/4 kg动态块，质量独立指定，目标(0.435,0)m。

封闭等温气体参考117325Pa、环境101325Pa（表压16kPa），bulk3e6；膜ke/ka2500、kd60，edge ke5/kd0.4。较上一版8kPa/.020 bulge参考气量增加约13.6%，未加载高度从35.316增到40.496mm。接触margin0.5mm/gap0、粒子半径0.75mm、检测包络2mm；完整表面接触包括四指、块和桌面，初始化核验实际候选。

夹块接近/下降各1.2s、闭爪0.8s、稳定0.4s、提起2.4s、横移3.4s、下降3s、松爪1s，总35s。闭爪30.1mm、指尖高5mm，垂直搬运，放下渐变5°。录像中央平均/最大下沉5.412/7.802mm，恢复RMS0.452mm；袋表面全段最大穿入桌面0.000mm。单次前后差值含继续自沉降，不是纯载荷压痕。

60Hz、8子步×24普通VBD，Graph captured、增量体积更新；pneumatic cavity排除surface-fast，不更换无气压模型拿fast标签。2101实际手部状态/16801子步机器人几何样本已查，采样不保证连续轨迹。原始状态、冻结输入/源代码和完整限制见当前报告。

EC worker 读取同一冻结输入，使用公开 SimulationManager 和对象控制 API；新增 pneumatic 配置以及 articulation SDF 请求传递，确保原生气体项和完整接触生效。运行入口：`.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m05.py`。Isaac 支持检查入口：`/home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m05.py`。它生成版本化支持证据，不运行替代布袋或产生计时。

流程：先确认 EC 视频，再安排后续测量。IL 当前 unsupported，无等价场景视频和性能样本。保留其它已完成数据；本轮未修改SDK，不stage/commit/push。生产配置修改和focused验证见当前报告与 source/ports。
