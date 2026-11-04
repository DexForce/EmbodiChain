# Observed B>1 第一阶段实现与同卡性能对照

第一阶段已实现：共享静态世界、相同被动关节锁值、无 payload 的真实环境 batch 复用现有 `BatchMotionPlanner`。同卡固定输入对照中，B=2 有效轨迹吞吐为串行的 **1.57×**，B=4 三次独立测量为 **2.73× / 2.79× / 2.73×**。这证明此小场景的 native batch 有收益，不代表所有任务的加速比例。

## 支持边界与修改

实现基于 #935 合并提交 `45d986e5f597c8c68bad092b0d0ed2be3511ab2c`，包含同一工作树已有的单球修复，分支为 `codex/observed-query-lifecycle`。结果针对本地工作树而非未修改的基线提交；关键文件散列见 `source-sha256.json`。

- `CuroboQuerySnapshot` 接受 full_qpos `(B,D_full)`、world_to_base `(B,4,4)`，拥有输入副本；env_ids 默认规范化为 `(0,...,B-1)`，显式重排/子集拒绝。
- 共用内部 batched SE(3) validator，一次小矩阵 CPU 传输完成校验；已有 B=1 Scene provider 校验包装仍保持 B=1，不扩大 Scene 写操作范围。
- `snapshot_collision_sources()` 可接收每行观测位姿及基座变换；共享 shape descriptors 声明各行几何相同。投影后的所有障碍物位姿须一致（atol=1e-5、rtol=0），否则拒绝，返回已有 B=1 共享 provider。该接口无法代表逐行不同几何，不从一个公共 descriptor 推断外部世界一定一致。
- observed backend 检查快照 B、目标 B 和 Robot 实例数，复用已有 BatchMotionPlanner dispatch；帧变换逐行消费 snapshot，保持不重读 live base pose。
- 从模型锁关节名称取所有观测行，在创建 native 模型前验证锁值精确相等。只有相等才使用共同锁值；活动关节起点和目标可以不同。
- 既有 PlanResult、native 有效样本截断、末端位置补齐和零 dt、success mask 与 checker 路径复用。未增加新的查询服务或结果协议。

B>1 payload、不同被动锁值、逐环境不同世界、子集/重排、多目标复制环境、CUDA Graph、动态世界更新仍不支持。B=1 payload 保持此前行为；任意 payload 轨迹 checker 仍 unavailable。默认无 observed snapshot 的行为不变。

通用 `validate_plan_options` 要求目标 B 与所属 Robot 的 num_instances 相同。本轮不放宽它：单环境旧调用继续使用 B=1 Robot，多环境 batch 使用对应 B 实例 Robot；`prepare_backend` 允许的尺寸不等于所有 facade 调用都允许子集。

## GPU 与合同回归

原生 GPU **3 passed**：新增 B=2/4 参数化回归，加上既有单球/27 球生命周期回归。

新增测试直接确认 native backend 类型为 BatchMotionPlanner，并覆盖 joint-space 和 EEF：不同环境的显式平移/旋转观测帧、不同目标、终点关节/位置/方向误差、起点一致、有效 dt、无 payload sampled-world checker、旧 positions/dt 所有权、机器人 qpos 不变及 GC 后原生 sphere buffer weakref 失效。

B=2 EEF 还包含同一 batch 中一行可达、一行远超工作空间的反例：结果必须为 `[True, False]`，失败行保持起点、dt 为零；不能把 hold 输出当作成功。

CPU/合同回归 **149 passed, 1 skipped, 2 deselected**，含快照所有权、非刚体变换、行数/环境顺序不符、不同锁值、B>1 payload、共享世界不一致拒绝，以及原有单球修复前的附件失败清理/alias 合同。上下文测试 **42 passed**，map check 与 API coverage gate **3650/3650** 通过。跳过项不计为已验证能力。

未执行物理抓持/任务 rollout；上述是 observed 输入、原生规划、有限采样世界碰撞检查和生命周期证据，不是完整 self-collision、CCD 或物理任务成功资格。

## 测量方法

- 同一物理 GPU 3：RTX PRO 5000 72GB Blackwell；驱动 580.126.20。Torch 2.7.0+cu128、cuRobo 0.8.0、Warp 1.17.0、DexSim 0.5.1rc1；CPU 线程 1。硬件信息见 `hardware.csv`。
- 固定 Panda 初态，第一关节目标增量在 0.04–0.12 rad；同一个静态 box，尺寸 `[0.18,0.40,0.36]`、位置 `[0.45,0,0.18]`；共享几何及基座投影后的世界；无 payload。start/target 与 corpus digest 在 JSON 中。
- 每组完成同样 B 行工作：serial 创建 B 个 fresh B=1 查询，batch 创建一个 fresh B 查询。同一进程内交替先后顺序，使用相同种子 949、max_attempts=2、warmup_iterations=0、graphs off、原生样本保留；没有降低 batch 的质量预算。
- 由于所属 Robot 实例数约束，在同一进程内预先建立两个相同模型的 Robot owner：一个 B=1、一个 B 实例。它们共同驻留同一 GPU，消费同一份冻结数值输入与世界；不修改 robot.num_instances、不绕过生产校验、不计仿真初始化为查询收益。没有并行进程执行规划。
- 计时从快照构造到结果返回、scope close、整组 GC 后同步；同一组 serial 的 B 个查询之后和一个 batch 查询之后各做一次 GC。这样包括不同构造次数及真实回收成本，未偷偷保留可复用 planner。冻结场景采集与两份 owner 初始化单列，不计入 group_ms。
- 结果校验在计时之外：每条检查 success、有限数值、起点/目标误差、非负 dt 与正 duration。CPU/GPU 原生回归另外验证冻结世界碰撞；benchmark 没有在每条计时内重复运行 checker。
- 每个配置 2 次预热/模式，排除后各 10 次稳态/模式，最后各 1 次独立 memory probe。B=4 共 3 个独立进程运行，每次内部串行对照；B=1/2 各 1 次。B=4 共 30 个稳态 group/模式，非 3×30 次。所有稳态合计 serial 150/150、batch 150/150 成功，均为 21 个样本、0.5 s 轨迹。
- 暖机器人 YAML/世界磁盘缓存，首次进程初始化排除，不宣称空缓存 cold-start 性能。主机其他 GPU 存在外部工作，CPU 非独占；本报告的样本量不足以给出稳定生产 P95 保证。单进程/单卡、group 级计时口径不能与旧双 GPU 批次墙钟直接比较。

## 稳态结果

每格是完成整个 B 行 group 的 P50 / P95（ms）。吞吐倍数按全部稳态样本总时间与成功行数计算，不用 P50 比值代替。

| B / 独立运行 | 串行 B=1 | native batch | 有效轨迹吞吐倍数 |
| --- | ---: | ---: | ---: |
| 1 | 1217 / 1280 | 1226 / 1256 | 1.004× |
| 2 | 1832 / 1907 | 1169 / 1220 | 1.573× |
| 4 / 1 | 3574 / 4321 | 1259 / 1762 | 2.730× |
| 4 / 2 | 3197 / 3606 | 1161 / 1264 | 2.786× |
| 4 / 3 | 3163 / 3244 | 1152 / 1213 | 2.727× |

B=1 基本持平，是两条对照路径没有人为收益的 sanity check。B=4 的绝对延迟有波动，但三次吞吐倍数一致；本场景适合推荐已有 native batch，而不是另建多进程调度。GC 成本在原始 JSON 的 `gc_ms` 中单列（本机常为数百 ms），这是本次明确的资源边界，不是“close 已立即释放”。

## 显存代价

独立 memory probe 的 Torch peak allocated（GiB），包含本进程两份 host 的 Torch 状态与组内存活 native 对象，不包含所有非 Torch 分配：

| B | serial | native batch |
| --- | ---: | ---: |
| 1 | 0.445 | 0.445 |
| 2 | 0.494 | 0.887 |
| 4 | 0.589 | 1.754 |

B=4 的 native batch 峰值约为该串行策略的 **2.98 倍**，收益需要额外显存；不能只报加速而忽略容量。组后 GC 的 live allocated 两种路径约 8.3 MiB，没有随重复组数线性增长。Torch reserved 包含缓存分配，不作为泄漏证据；原始 memory rows 同时保存 allocated_before、peak_reserved 等口径。

后四个独立运行另以 NVML 每 100 ms 采样整卡。整进程峰值约 B=1 9.83 GiB、B=2 10.37 GiB、B=4 11.13 GiB，包含两份仿真 owner/渲染原生资源、其他常驻占用和非 Torch 分配，不能归为单次 query 的独占显存。采样可能漏掉瞬时峰值。首个 B=4 运行未做 NVML 采样，没有补填。

## 使用与复现

生产使用继续构造标准 `CuroboPlannerCfg`，设置 `world.multi_env=False`、`use_cuda_graph=False`、`preserve_plan_samples=True`。用同一次 B 行观测构造 `CuroboQuerySnapshot` 和 `snapshot_collision_sources`；通过 `curobo_motion_generator` 调用既有 `generate`。Robot 实例数、观测和目标 B 必须一致，所有行使用同一个 control_part。

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 /root/miniconda3/envs/py311/bin/python \
  -m scripts.benchmark.motion.observed_batch --batch-size 4 --warmup 2 --repeats 10 \
  --output batch4.json

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 /root/miniconda3/envs/py311/bin/python -m pytest \
  tests/sim/motion/planners/test_curobo_integration.py::test_observed_shared_world_native_batch \
  tests/sim/motion/planners/test_curobo_integration.py::test_observed_payload_buffers_and_query_lifetimes \
  --run-gpu -q
```

将 batch-size 改为 1/2 可复现其余配置。各 JSON 保存逐组原始时间、成功行、采样长度、duration、memory probe、依赖版本和脚本散列；`series.json` 保存顺序运行的精确命令及 NVML 采样。首个 B=4 运行后仅对 benchmark 做 Black 格式化，故其脚本 hash 与后续运行不同，计时逻辑与参数相同。

本轮不扩大到逐环境独立世界、不同锁值或 payload batch，不新增缓存、实例池、并发服务。性能结论限于这组 joint-space 小场景；EEF 的功能回归通过，不代表已测得相同 EEF 加速倍数。
