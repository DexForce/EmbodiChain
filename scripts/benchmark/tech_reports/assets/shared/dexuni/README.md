# DexUni 共享机器人资产引用

已有 S01/S02 实际使用 **FR3 + Franka Hand**，而非 Panda。旧本机源 URDF：
`/home/oem/code/assets/franka_emika_panda/fr3_franka_hand.dexsim.urdf`。
这个路径只用于定位，不是跨机器的稳定资产 ID。正式 G0 必须核对原资产来源/revision、
URDF 引用的全部 mesh、碰撞简化、许可与 hash，不复制整份用户资产进技术报告源码。

Isaac USD 转换器：`physics/platforms/dexuni/isaaclab/convert_franka_usd.py`。
转换写外部新目录，source.json 保存转换参数、URDF hash 与生成文件 hash。
设置 DEXUNI_FRANKA_USD 为转换器实际输出的文件，不能假定导出器一定使用同一目录结构。
当前转换保留固定链接但将 mimic 变普通关节；G0/G1 需要验证/统一约束，不能只比较目标。

本机迁移保留的转换结果在
`~/.cache/embodichain/report-assets/local/dexuni/pilot_v1/generated/fr3_franka_hand/`。
W1 V030 + Pika 沿用原资产所有权，型号、安装变换、TCP、限位/受载仍待验证；不复用 FR3 轨迹。

技术报告不得以代码仓库 Apache 许可自动覆盖第三方机器人/纹理/视频许可。
通用资产继续使用既有资产 resolver；报告专用输入的 HF 发布遵循
[资产规范](../../README.md)，本次没有进行上传或自动下载。
