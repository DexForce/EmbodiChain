# C02 W1 纸袋扶正、两件装填与提袋

用户否定了上一版自建橙色硬框与小方块场景，要求参考指定的 `demo_dexuni_w1_bag_packing.py` 重做。旧版完整原始结果与实现已归档为 [rejected_custom_scene.zip](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/analysis/rejected_custom_scene.zip)，不是当前有效预览。

本次直接复用指定 demo 的纸袋及柔性提手、罐头、纸盒、桌子、W1 初始姿态、建场景函数、原控制器与物理配置。原始 50 秒流程为：纸袋由侧躺扶正，左手换抓袋口，右手依次抓取、装入并释放罐头和纸盒。原 demo 无独立刚性袋框，因此不再用另造硬框或跟随力替代其袋口和提手连接。两件商品在原 demo 中均为动态刚体，没有将新软块混入原场景。

在原始流程完成后，追加右手换抓对侧袋口、双手带载提起 80 mm、保持 2 秒再放回的几何操作。所有袋粒子保持动态，运动来自机器人接触；不增加世界固定、隐藏支撑或直接粒子位置写入。原袋壁自碰撞开启，袋口与提手保留原资产的网格连接和材料。

入口：[原生记录脚本](../../../platforms/dexuni/dexsim/c02.py)。运行输入、冻结源码、实际状态、检查结果与唯一当前视频归 [outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing](../../../../../../../outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/README.md)。只做场景预览；未移植 EmbodiChain/Isaac Lab，也未测 C02 性能。场景和提袋效果应按实际运行检查与用户反馈确认，不引用旧版结果或外推成功率。
