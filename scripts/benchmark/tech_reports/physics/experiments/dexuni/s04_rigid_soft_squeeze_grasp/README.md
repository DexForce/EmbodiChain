# S04：双侧分别夹持刚块和软块

按用户2026-10-22要求，先在DexSim实现和录视频，确认原生场景后才实现EmbodiChain与Isaac Lab；两平台视频再次确认后才测五次性能数据。

原生场景已获用户确认，两平台实现及预览已完成：[EmbodiChain / Isaac Lab并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/videos/s04_comparison.mp4)。开始测量后发现并修正EC默认十个外步的问题；[修正后的并排视频](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/videos/s04_comparison.mp4)实际12秒/720外步。DexSim SDK 集成后，两平台各五次正式测量完成：EmbodiChain 14.839 ± 0.396 ms；Isaac Lab 15.098 ± 0.365 ms，EC 外步耗时低 1.72%。SDK 重测前的修正节奏批次已归档，不作为当前结果。[完整报告](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/REPORT.md)。

## 场景与动作

直接参照demo_dexuni_w1_squeeze_grasp.py及其W1TwoWayScene：W1 V030双平行夹爪，固定根部、动态关节PD驱动、two_way耦合，手指URDF effort limit为10 N。蓝色刚块在机器人左侧（世界Y+0.25），橙色软块在右侧（Y-0.25）。灰色桌面和一个斜视相机。

- 桌面Z=0.86 m；刚块48×48×90 mm、密度450 kg/m³；软块60×60×80 mm、密度250 kg/m³。
- 软体k_mu=6000、k_lambda=15000、k_damp=0.001；粒子接触20000/10/1.0，粒子半径3 mm。
- 初始网格441个动态节点、1440个四面体；无pin、attachment或逐帧重写物体状态。
- 双手靠近、限力闭爪、提起150 mm、保持、下降、松爪及后退。源demo时间表约12 s。
- 60 Hz、10子步、12次VBD迭代；CUDA Graph实际状态captured。

保持源demo物理参数，预览只用于验收场景。软块局部压缩与包络形状应按实际视频/节点观察；不将单指关节位置直接当作总开口或全局压缩率，不宣称候选5%–15%变形线已达标。预览没有执行物理质量排名或性能统计。

## 文件和复现

平台入口为[原生s04.py](../../../platforms/dexuni/dexsim/s04.py)。任务输出固定为outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp，调参重录替换同一未确认预览，不创建版本目录。源demo、依赖文件、W1机器人和生成软体拓扑已冻结在此目录。

从仓库根目录运行：

```bash
/home/oem/code/engine/dexsim/.env-3.12/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/dexsim/s04.py
```

使用原生安装DexSim 0.5.1+7cfe364b67aaaa52944cb78ba87886f99aa4a50f、Newton1.7.0、Warp1.17.0；这不是其它构建的交叉对比。当前两平台复用已确认的网格、材料与命名关节目标；实际版本及控制差异见批次VALIDATION.md。

[阶段状态](STATUS.md) · [完整输入及来源记录](../../../../../../../outputs/benchmarks/tech_reports/dexuni/s04_rigid_soft_squeeze_grasp/manifest.json)。五次测量和跨平台比较在两次视频确认之后进行。

性能测量与重建：

```bash
.venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_s04.py --rebuild-report
```

有效测量已完成，原始计时和状态不可覆盖；首次新测量命令由run_s04.py编排，现有结果只使用--rebuild-report重建报告。
