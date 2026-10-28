# DexUni 操作与多材料耦合实验

surface-fast 对比口径：在统一场景、网格、物理材料、控制轨迹、时间步、子步及CUDA Graph配置下，对比DexUni surface-fast与普通VBD的求解策略，属于求解器级优化比较。基础迭代数、自适应回退、Jacobi/Chebyshev、多层校正和缓存均属于求解策略的组成部分；不同策略的一次迭代不等价。性能与任务效果分别按实测指标说明，不能仅因基础迭代次数不同而将结果解释为单纯减少迭代。

**从 [PLAN.md](PLAN.md) 的当前范围开始。常规测量采用固定配置、每平台五次独立进程；C02 只做摸测，M05 尚无跨平台正式测量。**
原完整协议为 `dexuni-replan-v2`；本轮精简范围覆盖原先的严格矩阵和控制等价要求，详见 PLAN 开头。
本轮 S01-C/T 的 20 次主采样与 4 次独立媒体运行均已完成并通过；报告和视频见 [STATUS](STATUS.md)。
S02 两侧各五次主采样和独立媒体均已完成；见 [S02 STATUS](s02_soft_pick_place/STATUS.md)。
S03 同样完成；见 [S03 STATUS](s03_cloth_corner_lift/STATUS.md)。

- [历史准入清单](PARITY_CHECKLIST.md)：后续严格研究参考，本轮以 PLAN 为准。
- [当前运行命令](SCRIPT_CONTRACT.md)：采样、报告重建和媒体工具。
- [执行记录](HANDOFF.md)：历史阶段与当前结果入口。
- [迁移记录](MIGRATION.md)：上游规范来源、新旧路径、旧进程与档案保留规则。
- 最终比较：EmbodiChain / DexUni 对 Isaac Lab / Newton MuJoCo Warp + VBD。
  DexSim 原生脚本仅作适配一致性诊断，不能代替 EmbodiChain 主结果。
- 保留 S/M/C 编号，不猜测 issue #679 的 P-xx 对应关系；suite=dexuni、family=physics。

## 目录与所有权

```text
scripts/benchmark/tech_reports/
  physics/
    experiments/dexuni/              # 本目录：计划、任务与编排
      s01_soft_compress_stretch/      # S01 总说明，链接两个子任务
      s01c_compress/tools/            # 输入生成、pilot 编排
      s01t_stretch/tools/
      s02_soft_pick_place/tools/
      s03_cloth_corner_lift/tools/    # 平布输入生成与任务定义
      s04_.../ ... c02_.../           # 其它任务定义
    platforms/dexuni/
      embodichain/{s01c,s01t,s02,s03}.py
      isaaclab/{s01c,s01t,s02,s03}.py
      isaaclab/convert_franka_usd.py
      dexsim/{s01c,s01t,s02,s03}.py
      _lib/benchlib/                  # 保留来源许可的旧显存采样器
  assets/
    physics/dexuni/{s01c,s01t,s02,s03}/    # 小型 authored JSON + 输入说明
    shared/dexuni/                   # 机器人来源和转换引用，不复制机器人
  reporting/dexuni/                  # 离线指标、build_s01_media.py、历史诊断

outputs/benchmarks/tech_reports/dexuni/<batch>/                        # 相对仓库根目录：本轮全部生成产物
  inputs/                            # 从外部缓存复制的冻结输入与 USD
  source/                            # 启动时源码快照
  measurement/{s01c,s01t}/            # 无相机五次采样
  media/{s01c,s01t}/                  # 独立媒体运行
  logs/                              # 退出码和失败细节另见 attempts.json
  videos/                            # 合成视频、时间映射、响应曲线
  REPORT.md                          # 三表报告
```

S01/S02/S03/S04 分别使用 run_s01.py/run_s02.py/run_s03.py/run_s04.py 编排和三表报告；跨实验 `core/` 公共框架尚待实现；已发布证据可通过 [下载命令](SETUP.md) 按任务取回。
二进制输入不进 source assets，运行输出仅写 outputs/benchmarks/tech_reports/dexuni/，不写进 experiments/platforms/assets。
每次准备新输入目录、每次运行新 run_id、每次重建新 comparison_id，不覆盖历史失败。

## 任务索引

| 任务 | 内容 | 本轮机器人 | 初始主组 | 当前状态 |
|---|---|---|---|---|
| [S01](s01_soft_compress_stretch/README.md) | 小变形压缩/拉伸与恢复 | FR3 + Franka Hand | two_way / dynamic | 五次主采样及独立视频完成 |
| [S02](s02_soft_pick_place/README.md) | 大变形抓取、提起、定点放置 | FR3 + Franka Hand | two_way / dynamic | 五次主采样、视频及原始数据完整 |
| [S03](s03_cloth_corner_lift/README.md) | 单角提布 | FR3 + Franka Hand | two_way / dynamic | SDK 集成后两平台各五次正式测量、视频及绘图原始数据完成 |
| [S04](s04_rigid_soft_squeeze_grasp/README.md) | 双侧刚/软块夹取 | W1 V030 + 双平行夹爪 | two_way / dynamic | SDK 集成后两平台各五次正式测量完成（14.839 / 15.098 ms） |
| [M01](m01_soft_beam_bend/README.md) | 软梁弯曲搬运 | W1 V030 + 双 Pika | two_way / dynamic | 三平台场景完成；两平台各五次正式测量完成（EC 14.275 / IL 14.852 ms） |
| [M02](m02_cloth_single_fold/README.md) | 方布单折 | W1 V030 + 双 Pika | one_way / kinematic | 三平台场景完成；两平台各五次正式测量完成（EC 9.194 / IL 12.488 ms） |
| [M03](m03_four_layer_cloth_pick_place/README.md) | 四层布夹取放置 | W1 V030 + 双 Pika | one_way / kinematic | 当前视频已确认，两平台各5次实测完成；surface-fast求解器级优化比较：EC14.653 ms、IL53.748 ms/外步，见当前REPORT |
| [M04](m04_bimanual_tshirt_single_fold/README.md) | T 恤对折一次 | W1 V030 + 双 Pika | one_way / kinematic / surface-fast | DexSim 32s低摩擦单折预览完成，桌面端滑移/伸长检查通过；待查看，未移植/计时 |
| [M05](m05_inflatable_pouch_load/README.md) | 软包抓放与压载 | W1 V030 + 双 Pika | one_way / kinematic | 原生场景已确认，EC 完整气压场景预览待确认；当前 Isaac Lab 28.0.0 / Newton 1.6.1 不支持封闭等温腔体；未测性能 |
| [M06](m06_franka_tshirt_fold/README.md) | T 恤从左往右折一次 | FR3 + Franka Hand | one_way / kinematic | 三平台场景完成；两平台各五次正式测量完成（EC 第 5 次重测替换），对照视频未标人工批准 |
| [C01](c01_supermarket_packing/README.md) | 三类商品混合装袋 | W1 V030 + 双 Pika | one_way / kinematic | 任务定义 |
| [C02](c02_bag_frame_packing/README.md) | 持框装袋 | W1 V030 + 双 Pika | one_way / kinematic | 任务定义 |

模式是候选设计，不是性能/稳定性结论。每任务必须重新验证质量，不能从 S01 推断其它任务的接触或收敛。

## 后续严格研究路线（本轮以 PLAN 开头精简范围为准）

1. 阅读 FULL_PROTOCOL §0/§3.3/§7 和历史准入清单，确定 FR3、驱动契约及 matched_replay 主组。
2. 先开发 G1 无接触驱动探针与共同目标下发；不要直接照旧 README 批跑“正式”结果。
3. 同时用 CPU 夹具实现 G4/G5 的失败分母与视频时间戳验证。
4. 完成材料台架和计时预热恢复，再冻结输入/矩阵、实施 S01-C/T。
5. 遇到能力缺口保留 N/A，不临时换后端或提高摩擦/驱动求成功。

实验 1–3 已完成；旧 `dexuni_benchmark/` 已按用户要求删除；当前 S02 正式结果保留。
新协议以本目录 PLAN 为准，后续实现与验收见 [操作清单](HANDOFF.md)。
当前结果限定为固定初态的整体平台与几何响应比较。未自动同步任何飞书文档或远程数据集。

初始计划和已替换的任务设计见[历史设计索引](history/initial_design/INDEX.md)；当前执行以[PLAN](PLAN.md)及上方任务索引为准。
