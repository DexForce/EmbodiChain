> Publication in progress: data upload has not completed. Download remains disabled until the verified dataset commit is pinned.

# 已有任务对比总览（2026-10-27）

本页仅汇总已有结果，没有新增仿真或重跑测量。共13个任务项（S01-C/T分别计），保留当前正式数据、视频、原始状态、复现输入与源码。目录整理见[迁移说明](../../physics/experiments/dexuni/MIGRATION.md)。

主要口径为无相机、GPU同步完成的物理外步耗时；越小越快，不包含渲染，不是整应用FPS。±为5个进程均值的样本标准差；单次摸测不附虚假的SD。不同任务的计时边界、控制、SDK版本与求解策略以各任务原报告为准，不能跨任务合成总榜。硬件为RTX5090 D v2。

## 性能概览

| 任务 | 场景 | EmbodiChain ms/步 | Isaac Lab ms/步 | EC耗时降低 | 数据级别 |
| --- | --- | --- | --- | --- | --- |
| [S01-C](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/s01_fixed_five/REPORT.md) | 软块压缩 | 41.124 ± 1.385 | 52.770 ± 1.704 | 22.1% | 各5次正式测量 |
| [S01-T](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/s01_fixed_five/REPORT.md) | 软块拉伸 | 35.177 ± 0.751 | 37.000 ± 0.716 | 4.9% | 各5次正式测量 |
| [S02](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/s02_fixed_five/REPORT.md) | 大变形抓取搬运 | 36.476 ± 1.116 | 39.002 ± 1.297 | 6.5% | 各5次正式测量 |
| [S03](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/s03_cloth_corner_lift/REPORT.md) | 布角提起与释放 | 12.722 ± 0.434 | 13.463 ± 0.316 | 5.5% | 各5次正式测量 |
| [S04](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/s04_rigid_soft_squeeze_grasp/REPORT.md) | 刚块/软块双侧夹持 | 14.839 ± 0.396 | 15.098 ± 0.365 | 1.7% | 各5次正式测量 |
| [M01](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m01_soft_beam_bend/REPORT.md) | 双臂软梁弯曲搬运 | 14.275 ± 0.433 | 14.852 ± 0.045 | 3.9% | 各5次正式测量 |
| [M02](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m02_cloth_single_fold/REPORT.md) | 方布单次对折 | 9.194 ± 0.169 | 12.488 ± 0.341 | 26.4% | 各5次正式测量 |
| [M03](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m03_four_layer_cloth_pick_place/REPORT.md) | 四层布夹取铺放 | 14.653 ± 0.317 | 53.748 ± 1.419 | 72.7% | 各5次正式测量 |
| [M04](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m04_bimanual_tshirt_single_fold/measurement/REPORT.md) | 双臂T恤对折 | 21.672 ± 0.839 | 71.486 ± 2.955 | 69.7% | 各5次正式测量 |
| [M05](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m05_inflatable_pouch_load/REPORT.md) | 充气包与刚块压载 | 未测 | 不适用/无同场景数据 | — | 仅功能演示，未测性能 |
| [M06](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m06_franka_tshirt_fold/REPORT.md) | Franka单臂T恤折叠 | 16.323 ± 0.354 | 36.257 ± 0.951 | 55.0% | 每侧保留5次，EC有1次指定替换 |
| [C01](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/c01_supermarket_packing/measurement/REPORT.md) | 面包/奶包双商品装袋 | 132.133 ± 1.562 | 不适用/无同场景数据 | — | EC正式5次；Isaac完整场景不适用 |
| [C02](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/c02_bag_frame_packing/performance_comparison/REPORT.md) | 袋体扶正、双载荷装填提袋 | 35.667（单次） | 1408.346（单次） | 97.5% | 单次64s摸测；低配另有12s短测 |

上述百分比只描述该行已测配置的物理推进时间。M02/M03/M04/M06的surface-fast与普通VBD属于统一任务条件下的求解策略比较；基础迭代、自适应回退、Jacobi/Chebyshev/多层校正均是策略组成，一次迭代不等价。C02还存在平台机器人轨迹差异，不能把巨大耗时差单独归因于求解器。

## 效果、配置与视频

参数×号按“迭代次数×子步数”书写，fast自适应与普通VBD不能按乘积等价换算。

| 任务 | 当前配置 | 已观察到的效果及限制 | 视频 |
| --- | --- | --- | --- |
| S01-C | 普通VBD；S01两侧40×10；S02两侧10子步 | 两侧当前5次均通过既定任务检查；固定初态，不代表普遍成功率 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/s01_fixed_five/videos/s01c_comparison.mp4) |
| S01-T | 普通VBD；S01两侧40×10；S02两侧10子步 | 两侧当前5次均通过既定任务检查；固定初态，不代表普遍成功率 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/s01_fixed_five/videos/s01t_comparison.mp4) |
| S02 | 普通VBD；S01两侧40×10；S02两侧10子步 | 两侧当前5次均通过既定任务检查；固定初态，不代表普遍成功率 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/s02_fixed_five/videos/s02_comparison.mp4) |
| S03 | 两侧10×10；SDK优化后有效批次 | 两侧5次完整有限执行；不以协议完成率代替布形精度 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/s03_cloth_corner_lift/videos/s03_comparison.mp4) |
| S04 | 两侧12×10；SDK优化后有效批次 | 两侧5次完整有限执行；旧错误步长批次不计入 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/s04_rigid_soft_squeeze_grasp/videos/s04_comparison.mp4) |
| M01 | 两侧12×10，two-way动态关节；Isaac CouplerProxy | 下垂/端距响应接近；未设置物理质量通过阈值 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/m01_soft_beam_bend/videos/m01_comparison.mp4) |
| M02 | EC surface-fast8；Isaac普通10；均10子步 | 两侧完成流程；左右角对齐各有优劣，未设折叠通过线 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/m02_cloth_single_fold/videos/m02_comparison.mp4) |
| M03 | EC fast8+自适应；Isaac普通20；均10子步 | 抬升均5/5；20mm放置线EC5/5、Isaac0/5；全部样本计时均保留 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/m03_four_layer_cloth_pick_place/preview/comparison.mp4) |
| M04 | EC fast16+自适应；Isaac普通20；均10子步 | 两侧原6项检查均5/5；最终下摆XY误差EC约26mm、Isaac约46mm | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/m04_bimanual_tshirt_single_fold/preview/comparison.mp4) |
| M05 | EC普通24×8，封闭等温气腔 | EC已演示压载、恢复和搬运；本次Isaac版本缺少所需气腔，视频为DexSim/EC | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/m05_inflatable_pouch_load/preview/comparison.mp4) |
| M06 | EC fast8+自适应；Isaac普通15；均10子步 | 当前保留样本两侧折叠检查5/5；EC第5次按用户要求重测替换 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/m06_franka_tshirt_fold/preview/comparison.mp4) |
| C01 | EC普通24×16；左面包后右奶包 | EC5/5两件留存且无倒置；Isaac仅单面包，不能与双商品算速度比 | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/c01_supermarket_packing/platforms/embodichain/preview/video.mp4) |
| C02 | EC DexUni12×6；Isaac普通96×24；另保留24×12失败版 | EC与96×24录像完成载荷提袋；24×12倒袋失败。96×24独立计时终态纸盒超界0.191mm | [查看](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/c02_bag_frame_packing/budget_reduction/iteration_comparison.mp4) |

M05本次安装的Isaac原生Newton缺少封闭等温气腔，[支持证据](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/m05_inflatable_pouch_load/analysis/isaaclab_support.md)仍保留。C01另保留[Isaac单面包视频](https://huggingface.co/datasets/DexForceAI/embodichain_reports/resolve/main/reports/dexuni-physics-2026-10-27/videos/c01_supermarket_packing/platforms/isaaclab/preview/video.mp4)及[异负载摸测](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/browse/c01_supermarket_packing/pilot_performance/REPORT.md)，不冒充双商品等价比较。M06的指定重测记录和全部正式批次原始样本保留。

## C02低配与成功效果的取舍

当前录到成功装袋/载荷提袋的Isaac配置为96×24，24×12倒袋后未完成任务；尚未证明96×24是最低所需预算。两档录像与EC均为实际64秒，无冻结补帧。以下单独比较相同前12秒、剔除前30步的690个稳态样本，不能与上表64秒均值混用。

| 配置 | 前12秒物理推进均值 ms/步 | 计时来源 | 完整录像结果 |
| --- | --- | --- | --- |
| EmbodiChain 12x6 | 29.857 | 复用已有64s计时前缀 | 当前录像完成装袋和载荷提袋 |
| Isaac Lab 24x12 | 226.643 | 新测一次12s | 倒袋，装袋/提袋失败 |
| Isaac Lab 96x24 | 1295.982 | 复用已有64s计时前缀 | 当前录像完成装袋和载荷提袋 |

C02高配视频的原严格袋口弯曲限值未全过（51.285mm>30mm）；独立64秒性能轨迹在悬停阶段装载成立，但终态纸盒X边界超0.191mm。保形失败及失败计时均照实保留，不把录像时长完成当作任务成功。

## 当前结论

- 简单软体/布角任务S01–S04与M01的物理步耗时差距较小，约2%–22%；读回与端到端开销需另看。
- 布料优化组M03/M04/M06的当前配置分别约3.67×、3.30×、2.22×逆耗时比；效果按各任务检查描述，不外推通用物理精度。
- M05及C01完整含气腔场景缺少本次Isaac等价能力；不生成虚构的平台速度比。
- C02展示了明显的速度/任务效果取舍，目前仅有摸测，不能包装为五次正式统计。

完整机器可读数据与来源哈希见[comparison.json](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/comparison.json)，可分享表格见[comparison.csv](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/comparison.csv)。各任务原始协议、运行日志、状态、时序和源码快照继续在原目录，历史诊断档案按原标签保留；本地清理账本不作为正式测量数据。

## 原始数据与恢复

数据上传中，固定版本尚未生成。

[完整报告及原始数据 ZIP](https://huggingface.co/datasets/DexForceAI/embodichain_reports/blob/main/reports/dexuni-physics-2026-10-27/README.md) · [输入/结果/视频哈希](../../assets/physics/dexuni/publication.json) · [下载与本地路径说明](../../physics/experiments/dexuni/SETUP.md)。

这里是便于 GitHub 审阅的结果摘要。完整测量数组、日志、冻结源码和输入保存在上述数据集，未将大文件提交到 Git。原始报告中的相对路径保留原样；完整恢复后可按原目录复算。
