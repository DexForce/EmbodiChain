# S01T 候选配置

`s01t_config.json` 是从旧 pilot 迁入的小型参数起点，不是新协议的已冻结配置。
运行不能只凭此 JSON 宣称 G0–G6 通过；当前 adapter 仍有 [脚本契约](../../../../physics/experiments/dexuni/SCRIPT_CONTRACT.md) 所列限制。

NPZ、生成几何 JSON、IK 和 manifest 通过 [本套输入规范](../README.md) 指定的生成工具写入
外部新目录，源码不存这些生成文件。通用机器人不复制，路径只是本机定位信息；跨机器先重新解析原资产并校验来源/许可/hash。
