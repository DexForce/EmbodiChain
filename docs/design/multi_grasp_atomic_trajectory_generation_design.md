# 多抓取候选经原子技能生成批量轨迹

状态：设计基线为 2026-09-09；首版实现更新于 2026-09-10，分支 `cj/add-affordance-trajectory-gen`。当前环境更正为 `conda activate embodichain2`。下文保留完整目标设计，“拟议”接口及后续阶段不等于全部已交付；当前边界见以下实现状态。

## 当前实现状态

| 能力 | 当前落点与边界 |
|---|---|
| 多 grasp 输入 | `toolkits/graspkit/candidates.py::GraspCandidateBatch`，公共 `get_grasp_candidates()` 保留 pose/cost/width/mask/稳定 ID；Antipodal 支持局部 RNG，兼容旧 ragged 接口 |
| Atomic 候选接口 | `atomic_actions/candidates.py`、`core.py`、`engine.py` 提供候选枚举/选择及带 `eligible_mask` 的编译；保留授权、场景绑定、证书失效检查 |
| PickUp 分支 | `PickUpCandidateBatch` 保留 grasp/roll 和 pre/grasp/lift 关节锚点；selected `ik_interp` 生成 transit/approach/close（含 settle）/lift，逐点 IK/FK/限位/连续性过滤 |
| Slide 分支 | `SlideCandidateBatch` 每个原始 grasp 对应一个候选；保留 pre/grasp/translated 锚点和把手局部轴，selected `ik_interp` 生成接近、抓取、滑动、松手及 push 返回路径 |
| 教程 N 环境扩增 | `integrations/atomic_affordance.py::plan_affordance_batch` 在当前真实环境行上选择不同 raw grasp，失败有界补位；`pickup.py`、`slide.py` 支持 `--n_affordance_multi_gen N`，创建 N 个 env 并同步回放 |
| 真实副本并行 | `trajectory_generation/replicas.py::SceneReplicaPool` 验证物理副本等价和 host epoch，将逻辑候选分轮映射到不同真实 env；单实例仍逐轮生成 |
| 批量导出 | `integrations/atomic_candidates.py::AtomicTrajectoryGenerator` 支持可选同 arm MoveJoints/MoveEndEffector 前缀＋单个末尾 PickUp，返回紧凑 `(B_out,N_max,D_full)`、有效长度、padding mask、阶段、来源和有界失败审计 |
| 有界会话 | proposal/输出数量/字节/waypoint/时间/audit 预算；无 validator 时为 planning-only。注入会话必须显式提供含 `path_collision` 的 validator，只进入 ready 队列，不更新 committed coverage |
| 示例 | `examples/sim/motion/trajectory_generation/affordance_parallel.py` 使用真实 mesh antipodal 采样与四个同场景 UR5 副本，输出 NPZ 和 JSON；不执行轨迹或写专家数据 |
| 尚未交付 | held-object 后缀、多个 PickUp 的组合分支、独立于真实 E 的 native backend、多 IK seed/retry/resampling、`GenerationRunner.run_source`、Atomic Runtime 回放及四类 source/host 物理验收 |

旧 winner 路径仅复用 grasp 姿态规范化，保留原有筛选行为；将其全部迁移到新严格 evaluator 仍是后续工作。新接口的精确参数以实现和测试为准。以下第 2 节是设计前的基线限制，不是更新后的代码状态。

2026-09-10 真实示例验证（`conda activate embodichain`、`--trajectories 8 --seed 13`）：从 mesh 采样 32 个有效 grasp，4 个物理实例分两轮规划，8 个提出的 grasp/roll 分支全部通过，输出 `qpos.shape == (8, 260, 8)`，进程正常退出。8 条路径内容不同；NPZ 的时间、有效长度、mask 和身份已检查。这是规划链路验证，不是物理抓取或专家数据验收。联调中修复公共 PK 导入对资产非标准 `link.origin` 的处理，使 FK 与解析 IK/实际末端一致；未修改资产或放宽 1 mm FK 过滤阈值。

环境更正后，已在 `embodichain2`（Python 3.10.20、Torch 2.7.0+cu128）复核：真实 GPU 烟测通过，仍输出 `(8, 260, 8)`；完整相关 CPU 组为 1261 passed、2 skipped、10 deselected、1 failed。唯一失败为原有 `test_runner.py` 无条件访问 Python 3.11 才支持的异常 `__notes__` 属性；本次未修改该测试及 runner 实现。API 文档覆盖与格式检查通过。

## 教程扩展：N 个不同 grasp → N 个真实环境

`scripts/tutorials/atomic_action/pickup.py` 与 `slide.py` 新增可选参数：

```bash
conda activate embodichain2
python scripts/tutorials/atomic_action/pickup.py --headless \
  --n_affordance_multi_gen 4 --affordance_output /tmp/pickup-affordances
python scripts/tutorials/atomic_action/slide.py --headless \
  --n_affordance_multi_gen 4 --affordance_output /tmp/slide-affordances
```

`--n-affordance-multi-gen` 是同义写法；N 必须为正整数，显式覆盖 `--num_envs`，
不是在单环境中循环回放 N 条轨迹。未指定新参数时保留原有 winner 教程路径。
可用 `--affordance_seed` 设置候选采样 seed（默认 13）。省略 `--affordance_output`
仍会规划和回放，只是不写 NPZ/JSON；指定时不能覆盖同名已有输出。

### 候选数量与失败处理

N 统计不同的**原始 grasp ID**，而非同一 grasp 的两个 roll。公共入口仍为
`get_grasp_candidates()`；旧 `get_grasp_poses`/best-pose 行为不被替换。
`affordance_utils.py::sample_affordance_grasps` 对目标 mesh 采样一次，将冻结的
object-local grasp 集变换到每个真实环境的物体位姿，保留相同的 raw 身份。
教程提供至少 `max(32, 4*N)` 的候选容量，候选有效数量仍受几何筛选限制。

`plan_affordance_batch` 调用 engine 的候选评估和编译接口，通过确定性的
raw-grasp/环境匹配避免重复计数。每轮成功行锁定，失败行从同一观测初态尝试剩余
grasp/roll；最多 32 轮，不自动重采样、不放宽 IK/FK/限位/连续性检查。
没有足够可行 grasp 或整条路径时，返回 partial/empty，并记录原因和终止条件。
失败物理行保持初始 qpos，不复制成功轨迹填满 N，也不将 hold 算作成功。

### 当前环境适配器与固定场景 source 的区别

`AtomicAffordanceBatch` 提供用于同步回放的完整 `(E=N, T, D_full)`
`trajectory`，以及仅成功行的 `compact_positions: (B_success, T, D_full)`、
`compact_valid_length`、`compact_env_ids`、完整 `success_mask` 和所选 raw grasp。
全失败时 compact 输出为 `(0, 0, D_full)`。时间统一为控制周期；短行末端保持，
导出 padding 的 dt 为 0。首样本严格保留完整实测关节状态，包括 passive mimic
约束残差，后续样本展开 mimic 仿射几何；主动或被动起点真不匹配仍拒绝。

此适配器处理调用者已有的 live-env 行，不创建副本、不重置场景、不管理
GenerationSession，也不伪造 articulation 的固定场景快照。原有
`AtomicTrajectoryGenerator` 仍负责固定场景、副本池、跨轮 compact 导出及有界会话；
其 PickUp-only 序列支持范围没有被 Slide 教程隐式扩展。

### 两个教程的技能序列与导出

- PickUp：不同 raw grasp 经技能自身的 roll/姿态规范化，生成 transit → approach →
  close/settle → lift；输出 `pickup.npz`、`pickup.json`。
- Slide：先生成并回放不同 grasp 的 pull。回放后逐环境读取**实际**把手位姿和机器人
  qpos，把该环境原选 grasp 的 handle-local 变换重定位，再规划 push；不重新选 winner，
  不广播 env 0，不假设把手已移动指令距离。pull 规划失败行不能进入 push；push 仍可独立
  过滤。分别输出 `slide_pull.npz/.json`、`slide_push.npz/.json`。

Slide 两个方向仍沿用显式配置的固定 `translation_distance`，并非按实测抽屉关节
行程逐环境闭环关到限位：实际 pull 不足时，等距离 push 目标可能超过闭合位置。
此教程不保证抽屉关节范围内的物理路径验收，也不宣称已经“关好抽屉”。

NPZ 包含 `qpos`（成功 batch）、`dt`、`valid_length`、`valid_mask`、`env_ids`、
`success_mask`、grasp 身份与位姿、候选列号。JSON 记录请求数/成功数、失败原因、
实际物理行及 Slide 的观测位姿/父 grasp 关系。

这里的成功表示规划通过。教程直接回放 qpos，没有使用 Atomic Runtime 的接触验证、
恢复或专家数据验收；不保证碰撞自由、实际抓取或滑动成功，报告显式标记
`physical_validation=False`。Slide 的后续输入使用实测几何，不将 pull 规划成功当作
物理效果证据。测试落在 `test_atomic_affordance.py`、`test_slide_candidates.py`、
`test_tutorial_affordance.py` 和 `test_affordance_tutorials.py`。

2026-09-10 教程实测（`embodichain2`，UR5，N=4，seed=13）：PickUp 输出
`(4,260,8)`，Slide pull 输出 `(4,140,8)`，均为四个不同 raw grasp，完成四环境
回放并正常退出。Slide 随后的 push 在完整轨迹关节限位检查中四行均被过滤，输出
`(0,0,8)`，原因 `trajectory:JOINT_LIMIT`，跳过 push 回放；没有用 hold 伪装成功。
这是生成、回放与失败处理的联调结果，不是完整 pull→push 物理成功证明。

相关 CPU 回归为 1334 passed、1 skipped、15 deselected、1 failed；唯一失败仍为
上述 Python 3.10 `__notes__` 兼容性测试。新增两项真实教程回归已通过
（2 passed），覆盖 N 覆盖原 `--num_envs`、真实候选/IK/回放、compact 输出和失败处理。
可串行运行：

```bash
pytest -q tests/sim/atomic_actions/test_affordance_tutorials.py \
  --run-gpu -m gpu -k test_real_affordance_tutorial
```

## 1. 目标与交付边界

将 graspkit 输出的多个 valid grasp 保留为不同逻辑候选，通过原子技能生成多条完整关节轨迹，返回 `positions: (B_out, N_max, D_full)`。每行对应一条候选技能序列，附带时间、有效长度、阶段、来源和验证结果。

区分三个数量：

- `S`：输入固定场景/初态数量。
- `E`：当前真实机器人实例数，即现有 AtomicActionEngine/MotionGenerator 的物理 batch。
- `C`：逻辑候选数量；`B_out <= C` 为最终保留的规划成功轨迹数。

例如 1 个固定场景、8 个 grasp、4 个同场景副本：`S=1, E=4, C=8`。先并行规划 4 条，再规划另外 4 条；全部通过时输出 `(8, N_max, D_full)`，其中只有 5 条通过时输出 `(5, N_max, D_full)`。若只有一个实例，现有路径只能分轮物化完整轨迹，虽然 grasp IK 已可在第二维批量求解。

首版交付包括多 grasp PickUp、兼容旧在线 winner 行为、真正的同场景副本并行、批量结果导出及接入既有生成会话。通用任意场景物理验收、所有原子技能一次性迁移、独立候选容量桶和 Atomic Runtime 全序列候选回放分别推进。

## 2. 当前代码中的实际限制

| 代码 | 当前行为 | 设计影响 |
|---|---|---|
| `toolkits/graspkit/pg_grasp/_antipodal_backend.py::get_valid_grasp_poses` | 返回 success、多个 pose、opening lengths、costs；已经过局部几何筛选/NMS/数量上限 | 保留私有 backend 定位；valid 不代表整臂 IK、整条路径或物理任务成功 |
| `toolkits/graspkit/pg_grasp/pose_generator.py::AntipodalGraspPoseGenerator.get_valid_grasp_poses` | 返回逐输入行的 ragged `list[(poses, costs)]`，丢弃 opening lengths | 从公共 facade 增加 richer 候选接口，保留旧接口 |
| `atomic_actions/primitives/pick_up.py::_select_feasible_grasp_variants` | 对 `(E,G,2)` 做 pre/grasp/lift/downstream IK 后，每 grasp 选一个 roll；只返回 pose/mask | 候选评估和选择分离，保留 variant 和关节解 |
| `PickUp._resolve_grasp_pose` | 按 cost 再选每环境一个 grasp | 旧入口保留 winner，新入口允许 all/top-K |
| `PickUp._get_full_pickup_trajectory` | 用 EEF 目标重新调用 MotionGenerator，未消费预筛选关节解 | 需要明确的已选分支物化路径 |
| `Robot.compute_batch_ik` | 已支持 `(E,K,4,4)`，当前 `return_all_solutions=False` | 复用候选维；多个 grasp 不等于每 grasp 的全部 IK 解 |
| `PlanningContext`、`TimedTrajectory`、engine context 校验 | env_ids 唯一，context batch 匹配真实 robot batch | 不重复 env_id，不修改 num_instances 来伪装候选数 |
| `GenerationRunner._validate_templates` | 一物理行一个模板，并要求 case/初态组合不同 | 增加候选来源入口和同 case 副本绑定 |
| `EnvRowMotionPlanner` | C 候选可按来源行分轮，但只支持 free/no-held 路径 | 可复用分轮思想；PickUp 不走其 free-motion 验收 |

现有 `AtomicActionEngine.compile()` 已能按真实 E 行顺序编译技能、传播 projected qpos/TaskState、维持 alive mask，并返回 `(E,N,D_full)`。应抽取复用其组合逻辑。

## 3. 分层接口与所有权

### 3.1 graspkit：`GraspCandidateBatch`

拟新增 `toolkits/graspkit/candidates.py`，由 graspkit 公共入口导出：

```python
@dataclass(frozen=True)
class GraspCandidateBatch:
    poses: torch.Tensor                  # (S, G_max, 4, 4)
    costs: torch.Tensor                  # (S, G_max)
    valid_mask: torch.Tensor             # (S, G_max), bool
    opening_widths: torch.Tensor | None  # (S, G_max), metres
    grasp_ids: tuple[tuple[str, ...], ...]
    frame: str                           # explicit reference-frame convention
```

接口设计要点：

1. `GraspPoseGenerator.get_grasp_candidates(...)` 提供基于旧 ragged 返回的默认适配，不新增 abstract method 使旧自定义 generator 无法实例化。Antipodal 实现 override，以保留 backend widths。
2. `get_valid_grasp_poses()` 和 `get_best_grasp_poses()` 的签名与返回结构保持兼容；新旧接口共享底层采样，避免每次调用重复抽样。
3. 工具层的 S 只是输入对象位姿行，不引入 SceneCase、仿真槽或 Gym。来源到 case 的映射归集成层。
4. 有效行必须 finite 且为合法 SE(3)；padding 使用合法有限 pose、`cost=inf`、`mask=False` 和不可选占位 ID。允许 G=0；不得将占位当成 grasp。
5. 原始 grasp pose 与传入 object pose 同参考系。进入 atomic 前记录/转换为 object-relative canonical grasp，并保留 frame 与标定身份。不要混用全局 world、local arena 和 solver root。
6. IDs 在候选集冻结时确定，不由当前数组下标、物理 slot 或 chunk 生成。记录物体局部 pose、width、采样参数与 seed；排序采用确定性的 cost/tie-break，截断记录原因。
7. 给 sampler 增加显式局部 generator，贯穿当前 `antipodal_sampler.py` 的随机方向/角度生成；几何缓存与随机样本缓存分开，后者包含 seed/采样配置。既有调用可保留原默认行为。
8. width 不直接作为 joint qpos。首版沿用已标定 OPEN/GRASP endpoint 命令；若要按候选 width 控制夹爪，必须提供 width→joint command 的 embodiment 映射，并重新检查 mimic 和碰撞几何。旧 generator 缺 width 时不启用该模式。

### 3.2 Atomic：候选评估和计划物化分离

拟在 `atomic_actions/candidates.py` 放小型公共候选协议/选择映射，具体技能数据结构放在对应 primitive 旁。候选是独立值对象，不持有 live robot、planner 或 ExecutionSession。

PickUp 首先抽取 `evaluate_grasp_candidates()`：

```text
raw grasp (S,G), geometry-only enumeration
 → source-to-slot binding at the current invocation
 → selected grasp candidates on real rows (E,K)
 → symmetric roll variants (E,K,V)
 → upright adjustment + grasp_frame_to_eef calibration
 → pre/grasp/lift/downstream batch IK from projected context
 → evaluated candidates + masks + per-stage qpos
```

`PickUpCandidate` 至少保存 grasp_id、variant_id、最终 object_to_eef、pre/grasp/lift EEF 与 qpos、可选 downstream qpos、cost、失败阶段、来源/标定/初态修订。单次扩展先固定每 grasp/variant 一条有效 IK 分支；多 seed/all-solutions 另加明确配置和构型去重。

几何 grasp 可以提前冻结，IK/路径证书则必须绑定当前 invocation 及其配置修订、起始 qpos、projected TaskState、物体位姿、solver root、控制周期、标定与碰撞场景修订。现有 `Robot.compute_batch_ik()` 仅接受真实 E 行；将 K×V 展平为其候选维，不直接把 S 或全部逻辑 C 当作环境维。后续技能在前缀编译完成后的逐分支 context 中评估；初态上算出的可达性只能作为预筛选，不能替代正确起点的规划验收。失效证书须重算，不静默沿用。

旧 `_resolve_grasp_pose()` 调用同一个 evaluator，再按原有 variant 旋转偏好和 grasp cost 选 winner。新候选入口在整个候选路径/技能序列完成前不进行单 winner reduction；仅做显式预算截断、去重和失败过滤。

拟增加 engine 的 `enumerate_candidates()` 和 `plan_candidate()` 服务入口。`plan_candidate()` 每次处理真实 E 行的一组已选候选，返回普通 `ActionPlan`，仍经框架的 request 解析、场景绑定、endpoint 授权及 `_validate_plan()`。具体技能以可选候选规划 hook 实现，不覆盖框架拥有的 `AtomicAction.plan()`，不由集成层直接调用 private planner。

序列编译可在 `compile()` 增加可选 `candidate_selections`，按 invocation ID 和 participant 标识选择；同时增加 `eligible_mask: Tensor | None = None`，默认全行参与，兼容原行为。当前 compile 的 alive 固定初始化为全 True，新路径改为从 eligible_mask 初始化。`enumerate_candidates()`/`plan_candidate()` 显式接收当前 active_mask=alive，逐技能沿用同一个 qpos/TaskState/alive 传播循环。

mask 必须贯穿候选评估和技能物化，不只是输出时过滤：空闲行不采样、不要求有效 selected candidate、不影响其他行的任务前置条件检查，并只生成安全 hold 占位；失败行不能被后续技能重新激活。结构、维度、场景绑定和 endpoint 授权校验仍须成立，不能借 inactive 行绕过契约。effect 只应用于 active 且规划成功行。

已有 `GraspGoal.grasp_xpos` 可用于初步接线，但含义是最终 EEF pose。它目前绕过 sampled-grasp 分支，且启用 rotate_upright 时仍会再次调姿。正式 selected-candidate 路径必须明确“规范化已完成”，保证 upright/calibration 只执行一次，并避免 AxisAlign 等技能重新选择对称解。

关节分支保持必须成为可验收契约：

- 使用 evaluator 保存的已解 qpos 作为阶段锚点，规划完验证实际关节锚点没有跳解。
- 自由 transit 可使用关节空间或碰撞规划；approach/lift 保留其 Cartesian 路径约束。
- 逐 Cartesian 样本 IK 使用同候选前一有效 seed，并做 FK/限位/连续性检查。
- 不能只对 pre/grasp/lift 的 qpos 做线性插值，就声称保持了直线接近和垂直抬升。
- 无法约束已选关节分支的后端标记能力缺失，或明确重新规划后生成新的分支身份、重验全路径；不得静默复用旧分支结论。

### 3.3 集成层：`AtomicTrajectoryGenerator`

在 `lab/trajectory_generation/integrations/atomic.py` 增加离线入口，复杂时将编译器放到相邻 `atomic_candidates.py`，旧 `export_pickup_templates()` 保留为兼容包装。

职责是生成/接收候选集、按能力分组、绑定真实环境行、调用 engine 编译、规范化时间/阶段、汇总结果。原子技能拥有目标和效果语义；Generator 不复制每个技能的轨迹算法。

`AtomicCandidateBranch` 保存：候选身份、canonical case/initial-state、按 invocation/participant 关联的 selected candidates、逐分支 projected state 与父分支关系。所有分支都从同一个已冻结初态启动；一个分支内部的后续技能使用该分支终态。下一轮 grasp 不从上轮 compiled.projected_context 起步。

结果拟为 `AtomicGenerationResult`：

```text
trajectories: CandidateTrajectoryBatch   # 规划成功的 B_out 行
planning_checks: per-row results        # IK/path/dynamics 等逐项状态
branch_metadata: per-row provenance     # grasp/variant/skill/parent
rejections: bounded lightweight audit   # 未通过或未支持的候选
summary: counts/status/stop_reason      # complete / partial / empty，非后端故障状态
```

positions、dt、valid_length、phases 和 identities 直接复用现有 `motion.expansion.contracts.CandidateTrajectoryBatch`。工具层 grasp mask、IK 成功、轨迹规划成功、路径验收和物理接受各有独立状态；`planning_checks` 未通过的结果不得算作可采集专家轨迹。若请求的必需规划能力 unavailable，明确返回失败/诊断。

输出协议：

- `positions: (B_out,N_max,D_full)`；完整机器人 joint order，主动夹爪和 mimic 几何均有明确语义。
- `dt: (B_out,N_max)`；有效首样本为 0，后续有效 arrival intervals 为正。
- `valid_length: (B_out,)` 和 `valid_mask: (B_out,N_max)`；末端重复 padding，padding dt 为 0。
- 跨轮不同长度只做 padding；不为凑同一 waypoint 数量擅自改变接触停留时间。需要控制周期规范化时保留阶段锚点/最短 hold，并重验动力学和碰撞。
- action 拼接处相同 qpos 的零时长重复锚点可合并并重映射边界；不连续则规划连接段或拒绝，不凭空增加/删除时间。
- 全失败返回 `(0,0,D_full)` 和完整计数/有界失败原因。

输出为普通 tensor 的调用方式是 `result.trajectories.positions`，同时提供 `valid_length/dt`。把结果送给执行器时必须消费这些字段。

### 3.4 grasp / IK 失败时的分阶段过滤

默认策略是逐候选拒绝、其余候选继续，最终只导出完整通过的轨迹。筛选单位为 `(case, grasp_id, variant_id, attempt_id)`；某个 variant 失败不淘汰同 grasp 的其他 variant，某一场景无可行候选也不连带淘汰其他场景。IK 返回失败表示本次有限搜索未找到解，不宣称目标数学上无解。

| 阶段 | 拒绝条件 / 原因码示例 | 过滤动作 |
|---|---|---|
| grasp 生成 | 正常返回零候选 / `NO_GRASP_CANDIDATES` | 记录 source/case 级结果，不制造 candidate ID 或轨迹；按预算换候选或结束该 case |
| grasp 输入与几何 | pose/cost 非有限、非 SE(3)、提供的 width 非法，或不满足夹爪/方向/局部碰撞约束 / `INVALID_GRASP`、`GRASP_CONSTRAINT_FAILED` | IK 前屏蔽；width 缺失只在启用 width 控制时拒绝 |
| 阶段 IK | pre-grasp、grasp、lift、后续目标或 Cartesian 中间样本搜索失败 / `IK_NOT_FOUND` | 记录 invocation、stage、可选 waypoint_index，终止当前分支 attempt |
| IK 解校验 | solver 声称成功但 qpos 非有限、越限，FK 位置/姿态超差或不满足构型连续性 / `IK_INVALID_RESULT` | 拒绝返回解；不得作为下一阶段 seed 或轨迹锚点 |
| 完整路径 / 后续技能 | 路径规划失败、碰撞、时间/动力学不合格、必需技能前置条件失败 / `PATH_NOT_FOUND`、`PATH_VALIDATION_FAILED`、`SKILL_PRECONDITION_FAILED` | 拒绝整条技能序列，不导出只有前半段成功的轨迹 |

这些过滤分别由 grasp facade、primitive evaluator 和集成层计划验证负责。当前 `_resolve_grasp_pose()` 已检查 pose/cost，`_compute_batch_candidate_ik()` 已用最后有效 seed 替换失败解；新路径复用这些保护，同时将 candidate mask 提前传入逐阶段评估，而不是只在最后选优时合并。

**阶段掩码与数值隔离。** 在固定物理 E 行内维护 `(E,K,V)` 的 alive；每一阶段只会减少有效候选：

```text
alive_0 = eligible & input_valid & geometry_valid
stage_ok = solver_success & finite_qpos & joint_limits_ok & fk_ok & continuity_ok
alive_next = alive_previous & stage_ok
next_seed = where(alive_next, checked_qpos, last_valid_seed)
```

finite 检查必须先于对该解的 FK/连续性计算；不合格数据先替换为有限安全占位，再计算并与原始 mask 合并，不能靠 `NaN * 0` 清除坏数据。空闲/失败行保留当前环境的安全 seed 与 FK pose，不能一律填零或单位矩阵。后端可用候选压紧时只评估 active 项；后端强制 dense batch 时允许计算安全占位，但对应逻辑阶段始终为 `NOT_ATTEMPTED`，占位解即使可达也不能重新激活候选。若没有 active 项，跳过该阶段后端调用。

每个 attempt 保留首个真实失败原因；未执行的后续阶段不能重复计为 IK 失败。任何必需中间 waypoint 失败都使整条分支失败，不能删掉该点再连接两侧轨迹。仅最终 alive 行进入 `CandidateTrajectoryBatch`，其 identities、phases、dt、lengths、来源及验证结果使用同一索引同步筛选。保留的轨迹按冻结的候选顺序稳定汇总，后续 top-K/多样性选择另行进行。

**补位与有界重试。** 先从剩余候选队列补位；新候选在下一轮从冻结初态重新开始，不能在已执行一半的技能序列中顶替失败行。默认 `ik_max_attempts_per_candidate=1`、`grasp_resample_rounds=0`，这里 candidate 指一个 grasp/variant 分支。可选 alternate-seed 重试只针对几何有效且正常返回 `IK_NOT_FOUND` 的候选；非法输入、损坏结果、后端故障不自动换 seed 重试。重试或重新采样均计入各自预算与总生成预算；seed 由逻辑身份和 attempt ordinal 决定，不由物理 slot 决定。重试是独立 attempt，保留原失败记录；重新采样产生新 grasp ID。首版每次重试从该 invocation 的同一合法 projected 起点重评整个阶段链，成功的其他分支不重算；任何改变关节分支的重试都重验相关阶段连接和完整路径，不能拼接不同重试中不连续的阶段解。不得通过放宽碰撞、FK 或关节限位阈值补足数量。

候选数、分支 attempt 数和实际阶段 IK 查询数分别计数；padding 计算不计为逻辑 attempt。预算耗尽时尚未尝试的分支标记 `NOT_ATTEMPTED`，不能计为 IK 失败；同一候选重试两次失败也不能算成两个不同候选被淘汰。

**输出与记账。** `summary` 给出请求上限、实际输出数、候选/IK attempt 数、各阶段首败计数及 `stop_reason`（目标数量达到、候选耗尽或预算耗尽）。例如 8 个候选中 2 个 grasp 无效、1 个 IK 失败，输出 B=5；不能复制成功行或用 hold 轨迹补成 B=8。全部被过滤时返回 `(0,0,D_full)`、空长度/时间/mask 和原因统计；有输出但不足目标为 partial，全空为 empty。`rejections` 保存有界的候选身份/attempt、invocation、stage、reason_code 与必要误差指标，不缓存整条失败轨迹；统计不随明细截断丢失。过滤前已由 GenerationSession 分配身份的 attempt 调用 `release(reason=...)` 终结，不进入 ready、rollout 或 confirmed coverage；只有已通过必需检查的结果才调用 `add_planned()`。

**无解与系统错误分开。** 以上过滤适用于后端按契约返回的候选级失败。缺失 solver、shape/device/场景绑定错误、共享初态/标定非法、后端抛异常、GPU 故障或任务必需能力缺失，默认显式失败并保留诊断，不能用宽泛异常捕获伪装成“所有 grasp 无解”。`success=False` 即使伴随 NaN qpos 仍归为 `IK_NOT_FOUND` 并隔离返回值；`success=True` 却给出非有限或其他不合格解才记录 `IK_INVALID_RESULT`。若后端契约无法保证其他行可靠，整次调用不得产生可接受结果。零候选、部分失败是正常结果，不应触发整批异常。

## 4. 两条 batch 路径

### 4.1 首版：真实环境行 + 同场景副本

保留 `PlanningContext.env_ids` 唯一和 engine batch=E 的约束。新增 `SceneReplicaPool`/`CandidateSlotAssignment`，放在 `lab/trajectory_generation/replicas.py`。

副本池由宿主配置或调用者显式提供真实实例；验证布局、几何、机器人/工具、完整初态、物理参数和控制周期等价。全局 arena offset 经 local arena/robot root 变换消除。同一 case 的副本共享 case ID 和 coverage，不能改名成不同 case 来绕过限制。

每轮从一个 case 的不同 grasp 分支取最多 E 条，绑定成 E 行目标和独立 task state，保持同一技能序列、binding、物体语义/实体身份、控制周期、机器人模型及可兼容碰撞 profile。仅共享不可变几何；可变障碍位姿和 attachment 逐行隔离。空闲行以 active mask 和安全锚点占位，effect 只更新 active 且成功行。

每个 assignment 保存 candidate_id、source case/row、目标 slot、host epoch 和坐标映射。逻辑候选的来源信息保持不变；在规划/执行边界生成目标槽兼容的副本。调整当前依赖 `source_row_indices == physical row` 的检查，通过显式 assignment 验证身份和场景，不能仅重写 row index。

轮内调用一次 E 行编译/规划服务，轮间汇总成功行。各候选组持有独立 projected task state；复用 planner 时串行绑定 world/model 状态。不用 Python 线程同时操作同一个 engine/planner。

这条路径给出首个真实并行验收：单 case × 4 实例 × 至少 8 个 grasp 候选，两轮批量生成；真实 rollout 继续使用 full_batch，整批恢复后才重用槽。

### 4.2 后续：独立候选容量桶

为了在 E=1 时也同时规划 C 条完整轨迹，需扩展 `motion` 的候选规划端口，显式输入 full start states、robot/solver root poses、source mapping、world index、已解关节锚点、phase/attachment profiles 和 valid mask。

后端能力至少分开声明 candidate-batch planning、solved-branch preservation、sampled path validation、per-candidate world、held-object geometry；不能用一个 supports_batch 替代。

cuRobo 适配需要修改其真实环境 batch 检查、root/world 获取和缓存键；缓存按机器人模型、锁定关节、世界/attachment profile、控制部分与桶容量区分。仅修改 `max_batch_size` 或 flatten tensor 不足以实现该能力。

可提供有限桶容量和显存/候选字节预算。分阶段调度的 primitive 可复用第一阶段提取的候选目标/关节锚点构建函数，经候选运动端口生成轨迹，再由 atomic 层组装候选结果。此路径不能将逻辑 C 行强塞进现有物理 E 行 `ActionPlan`；普通 ActionPlan 仅在选中候选绑定执行槽时物化。

首阶段没有该能力时使用明确的 env_rows/replica 路径，不宣称单实例已实现完整候选并行。规划 tensor batch 的支持也不自动带来 E=1 时多条轨迹的同时物理执行。

## 5. 各个 primitive 的接入范围

| 技能 | 候选如何进入 | 实施范围 |
|---|---|---|
| PickUp | 共享 grasp evaluator；候选含最终 object_to_eef 和阶段关节解 | 首版，显式保留多 grasp/roll |
| MoveEndEffector、MoveJoints | 每个分支的 pre-grasp/retreat/关节目标 | 首版复用普通逐行 goal |
| MoveHeldObject、Pour | 继承当前分支 HeldObjectState.object_to_eef | 同一分支内继续，不重复采 grasp |
| Place | PlaceGoal.xpos 是 EEF release pose；相同物体放置目标需逐分支转换 | 验证后续目标受抓法影响；AssembleGoal 已有持物变换路径 |
| AxisAlign | 已有显式 grasp_xpos，提取自己的轴约束、upright 和 symmetry 筛选 | 第二阶段，不能直接套 PickUp 变换顺序 |
| Slide | `SlideCandidateBatch` 和 selected `ik_interp`；每 raw grasp 一个候选 | 已接入 N 环境教程，保留 handle 局部平移轴和 Cartesian 滑动路径；不等同物理效果验收 |
| OpenDoor | 目前直接 get_best_grasp_poses；新增技能自己的 selected grasp 入口 | 后续工作，保留关节轴、handle、开度语义及接触路径约束 |
| CoordinatedPickment | 一候选为兼容的左右 grasp pair | 有界配对/筛选，联合 IK、自碰撞和持物约束 |
| CoordinatedPlacement | 继承 placing/support 资源各自的 held transforms | 联合规划两臂；不假定双臂共持同一物体或必须接在 CoordinatedPickment 后 |
| HandOver | pickup grasp、receiving grasp、交接方向形成候选 | 接收抓取绑定预测交接物体姿态；现实现包含 pickup/transfer/place，要求开始时两臂空，不能默认接在外部 PickUp 后 |
| Twist | TwistAffordance 中轴/接触受限的专用候选 | 共享批量输出和调度，使用专用几何候选 |
| Press、PushObject | 按压/起推接触点和工具姿态候选 | 共享框架，不将任意 antipodal grasp 解释成按压/推动目标 |

后续技能必须携带分支特定的持物关系。采用 `T_A_B` 表示将 B 坐标变换到 A：

```text
T_object_eef[c] = inverse(T_arena_object) @ T_arena_eef[c]
T_arena_eef_target[c] = T_arena_object_target @ T_object_eef[c]
```

相同物体目标位置，可能因不同 grasp 而需要不同 EEF 目标。`StateDelta.apply()` 只在该分支投影状态上按 mask 应用。当前 `PlanningContext.project()` 不更新 scene，后续碰撞规划还需显式维护该分支预测的 object pose/attached geometry；不能仅传播 TaskState 却继续检查物体初始障碍位姿。

每个 invocation 的规划模型还必须同步 projected full qpos 中的夹爪、mimic 和锁定关节几何。不能在 PickUp 闭爪后，仍从未执行的宿主初始 robot 状态读取张开的夹爪做后续碰撞检查。

## 6. 接入既有扩增/执行系统

给 `GenerationRunner` 增加候选 source 入口，例如 `run_source(cases, source, ...)`，保留现有 `run(cases, templates)` 作为 handwritten 模板适配。source 接收 Runner 拥有的 GenerationSession，先分配身份/预算再规划；不能另建 session 产生 Runner 不认识的候选。

独立离线调用 `AtomicTrajectoryGenerator` 可自持一个只做提议/规划记账的 session，不要求 sink、reset 或 physics；以规划输出条数和预算终止，不将其当作 committed 覆盖。

原子来源输出多条基础轨迹后，再对每条允许的 free 阶段做 residual/retime，谱系为：

```text
scene/initial state
 → grasp_id / variant_id / actual joint branch
 → compiled reference trajectory
 → residual or timing child
```

相同几何的 retime 子分支共享 geometry family；不同候选执行槽不改变 ID/随机流；实际近重复判断仍由 CoverageIndex 负责。按照完整技能序列可行性做最终 top-K/多样性选择，不能把局部低 grasp cost 当作完整序列最优。

Runner 启动以 source/planner/validator 的实际能力匹配，替代当前对 source kind 和 PickUp 具体类型的封闭组合。多候选对应多模板，不再要求一个物理行永远绑定一个 reference。原有预算、ready/pending 上限、失败回执和 confirmed 覆盖计数复用。

PickUp 首版继续使用同一个 `PickUpMotionValidator` 进行计划与实际接触检查；现有 validator 仅支持其 URDF/cuboid/CPU profile。任意 mesh 抓取、AxisAlign、OpenDoor 等有自己的验证 profile，未具备时可明确返回规划结果，不可直接入专家集。

当前 `export_pickup_templates()` 限定 MoveEndEffector→PickUp 且全批成功；新导出路径应按成功行转换、保留真实 segment 边界。旧函数保持兼容包装，不能用猜测 phase 或 all-success 要求阻断混合成功 batch。

当前 `initial_plan_provider` 只供应首个 invocation 的初始计划。完整 Atomic Runtime 回放多技能候选还需逐 invocation 的正式消费契约和效果/恢复验收，不通过它塞入整个长序列替代原有状态机。

## 7. 拟议调用示例

以下为接口草案，类型和方法需按本文实现后才可运行。`replica_pool` 已绑定经过验证的真实同场景实例，`candidate_inputs` 按 invocation 和 participant 关联，不把 grasp 应用到无关技能。

```python
grasp_batch = grasp_generator.get_grasp_candidates(
    mesh_vertices=vertices,
    mesh_triangles=triangles,
    obj_poses=object_poses,
    approach_direction=approach_direction,
    generator=local_generator,
)

source = AtomicTrajectoryGenerator(engine, replica_pool=replica_pool)
result = source.generate(
    invocations=(pick_up, move_held_object, place),
    context=initial_context,
    candidate_inputs={(pick_up.invocation_id, "primary"): grasp_batch},
    cfg=AtomicCandidateGenerationCfg(
        max_grasps_per_case=16,
        variants="feasible_rolls",
        max_output_trajectories=8,
        max_proposals=64,
        ik_max_attempts_per_candidate=1,
        grasp_resample_rounds=0,
        batch_mode="env_rows",
        pool_mode="grouped_replicas",
    ),
)

qpos = result.trajectories.positions       # (B_out, N_max, robot.dof)
dt = result.trajectories.dt
lengths = result.trajectories.valid_length
mask = result.trajectories.valid_mask
```

`place` 等依赖 grasp 的目标由注册的技能适配逻辑在分支上下文中绑定，不能用同一个固定 EEF tensor 覆盖全部 grasp。PickUp 自身已构建 pre-grasp→grasp→lift；如需额外 MoveEndEffector 前缀，先从几何候选绑定该前缀目标，编译后在其 projected context 中评估 PickUp，避免重复接近段或把旧起点的 IK 证书当成新计划。

新增配置使用 `@configclass`，分层解码并拒绝未知字段。Atomic 专用限制放在集成配置中；`motion.expansion.cfg` 仅扩展共享 pool/planning 能力。数量是有界上限，不保证有限候选能产生指定条数；预算耗尽应返回实际输出数量及原因。

## 8. 实施次序与验收

1. **保留 grasp 信息与 PickUp 分支。** 新增 GraspCandidateBatch、局部采样 RNG；抽取 evaluator；旧 winner 行为回归；证明 2 grasps × 2 feasible rolls 不被提前压缩，宽度/标定/seed 对齐。
2. **轨迹物化和 batch 输出。** Selected-candidate 路径消费已解锚点；保持 Cartesian 接近/抬升；一场景多候选分轮编译；混合成功与全失败输出，验证 dt、padding、阶段和 joint order。
3. **同场景副本真正并行。** 一个 case、4 个真实 slots、至少 8 个候选，两轮调用；相同 case 不伪造 ID；不同 arena offset 正确映射；逐行失败隔离；最后输出 `(B_out,N_max,D_full)`。这是首版“并行”退出条件。
4. **接入 Session/Runner。** 多 reference source、预算/去重和 full_batch rollout；重复恢复后初态一致；只有通过实际接触/任务验证的轨迹确认提交。另验未扩增的多 grasp 基础轨迹，避免只测试 residual 路径。
5. **后续技能。** 先 MoveHeldObject/Place，测试相同物体目标、不同 grasp 导出不同 EEF 目标；再 AxisAlign、Slide/OpenDoor，最后双臂 pair/HandOver。每个技能分别验收，不能因协议可接入就声明全部已实现。
6. **候选容量桶。** 在 E=1、C>1 条件下验证独立批量完整轨迹规划；按相同候选集对比分轮模式。记录 IK/完整规划时间、候选有效率、轨迹吞吐和显存峰值，实测后决定默认。

建议测试位置：`tests/toolkits/` 的 grasp facade 测试、`tests/sim/atomic_actions/test_actions.py` 和 engine 测试、`tests/lab/trajectory_generation/test_planning.py`、新增 `test_atomic_candidates.py`/`test_replicas.py`、`test_runner.py` 和真实 PickUp 示例测试。

必须覆盖：空 grasp、非有限 cost/非法 SE(3)、失败 seed 不污染其他分支、标定一次、grasp/variant/IK branch ID 对齐、关节锚点不跳解、整个技能序列才判成功、IK 证书绑定正确 invocation 起点、空闲槽不采样/应用 effect、跨轮不继承上轮 projected state、per-candidate 持物碰撞几何、不同长度/接触 hold、同 case 多副本共享覆盖、chunk/slot 改变不改变已冻结候选身份。

过滤专项验收：8→5 的混合结果与元数据对齐；单 variant 失败仍保留其兄弟分支；分别注入 pre/grasp/lift/后续 waypoint IK 失败；solver 返回 False+任意有限 qpos 或 True+NaN/越限/FK 超差；后续阶段不得重新激活失败行；全失败不调用后续 planner、输出合法空 batch；有限重试耗尽、补位初态和 Session release 正确；后端异常必须显式报错，不能计作正常零轨迹。

## 9. 主要修改文件

| 位置 | 拟议改动 |
|---|---|
| `toolkits/graspkit/candidates.py`（新增）、公共 `pose_generator.py`、`pg_grasp/pose_generator.py` | 候选契约和兼容 facade，保留 widths |
| `pg_grasp/antipodal_sampler.py`、`_antipodal_backend.py` | 显式局部 RNG、稳定候选来源、有限 cost；后端不依赖 atomic |
| `sim/atomic_actions/candidates.py`（新增）、`core.py`、`engine.py` | 可选技能候选协议、受保护的场景绑定入口和共享编译循环 |
| `sim/atomic_actions/primitives/pick_up.py` | evaluator、逐阶段 mask/IK 解过滤、旧 winner 包装、已选 grasp/IK 分支物化 |
| `lab/trajectory_generation/integrations/atomic.py` | AtomicTrajectoryGenerator、失败记录/有界补位、通用成功行导出、旧 export 兼容 |
| `lab/trajectory_generation/replicas.py`（新增）、`initial_state.py`、`integrations/sim.py` | 真实副本等价性、source→slot/epoch 映射和恢复 |
| `lab/trajectory_generation/runner.py`、`execution.py` | 多候选 source、按 assignment 执行、保留原模板入口 |
| `sim/motion/expansion/cfg.py` | 在对应实现落地后开放 grouped_replicas 等共享选项 |
| `sim/motion/motion_generator.py`、候选规划适配、`planners/curobo/curobo_planner.py` | 后续独立 candidate-batch 能力、显式 roots/worlds/锁定关节/attachment |
| 对应 primitive、tests、API 文档、agent_context | 按每一阶段实际暴露的能力同步 |

参考：[固定场景设计](fixed_scene_expert_trajectory_augmentation_design.md)、[实施计划](fixed_scene_expert_trajectory_augmentation_implementation_plan.md)。本文细化其中多 grasp、候选保留、同 case 副本和独立候选 batch 的工作；不改变计划成功与真实任务成功的区别。
