# 原子技能模块重构计划

> 状态：Draft
>
> 优先级：P0/P1
>
> 最后更新：2026-08-02

## 1. 背景

当前原子技能模块已经具备以下基础能力：

- 类型化的 Action Target；
- 基于 `WorldState` 的技能串联；
- 环境批量化的运动规划；
- 完整机器人 DoF 轨迹输出；
- TOPPRA、Neural Planner 和 cuRobo 等可插拔 planner；
- cuRobo 规划前的动态障碍物位姿更新。

这些能力可以继续复用，但当前模块本质上仍是一个**离线轨迹编译器**：

1. `AtomicAction.execute()` 实际执行的是规划，而不是物理执行；
2. `AtomicActionEngine.run()` 一次性规划并拼接完整轨迹；
3. 示例通过逐 waypoint 调用 `robot.set_qpos()` 开环回放；
4. `WorldState` 主要保存预测的 `last_qpos` 和持物关系，而不是实时观测；
5. 规划成功、轨迹执行成功和技能语义成功没有明确区分；
6. 抓取、释放等任务状态在规划完成时就被乐观提交；
7. 动态障碍物只能在新一次 `plan()` 开始前刷新，执行过程中没有轨迹失效检测与重规划；
8. 控制部件、夹爪开合位置和运动参数在多个 Action 配置中重复声明。

相关现有实现：

- [`AtomicActionEngine`](../embodichain/lab/sim/atomic_actions/engine.py)
- [`WorldState` 和 `ActionResult`](../embodichain/lab/sim/atomic_actions/core.py)
- [`TrajectoryBuilder`](../embodichain/lab/sim/atomic_actions/trajectory.py)
- [`MotionGenerator`](../embodichain/lab/sim/planners/motion_generator.py)
- [`PlanState` 和 `PlanResult`](../embodichain/lab/sim/planners/utils.py)
- [cuRobo planner](../embodichain/lab/sim/planners/curobo/curobo_planner.py)
- [原子技能教程回放逻辑](../scripts/tutorials/atomic_action/tutorial_utils.py)

## 2. 重构目标

本次重构优先解决现有架构问题，并建立动态场景和错误恢复能力。

### 2.1 P0 目标

- 统一批量环境下的成功状态和状态更新语义；
- 区分规划、执行和技能结果；
- 拆分目标、绑定、策略和运行时状态；
- 保留 planner 的时间、速度和加速度信息；
- 为复杂技能暴露明确的 phase 边界；
- 增加反馈式轨迹执行；
- 支持轨迹跟踪误差检测、安全保持和重新规划；
- 支持移动障碍物和移动目标触发的停止—重规划；
- 将抓取、释放等任务效果改成验证后提交；
- 保持现有静态接口和示例可兼容迁移。

### 2.2 P1 目标

- 支持不同环境独立进入执行、重规划、恢复和终止状态；
- 支持抓取失败、物体丢失和释放失败等技能级恢复；
- 减少现有 Action 配置中的控制部件和末端执行器参数重复；
- 完成所有现有原子技能向新接口迁移；
- 增加结构化执行事件、诊断信息和性能指标。

### 2.3 P2 目标

- 支持障碍物预测轨迹；
- 支持滚动时域规划；
- 支持 warm start 和异步预规划；
- 扩展通用工具能力接口；
- 支持动态增加、删除或改变障碍物几何形状。

## 3. 非目标

以下内容暂时降低优先级，不纳入 P0/P1 实施范围：

- 下半身运动技能；
- 浮动基座运动控制；
- 足底接触和步态规划；
- WBC、MPC 和质心动力学；
- 硬实时控制系统；
- 完整的实时工具热插拔。

新的状态和控制协议可以为 `root_pose`、contact、不同 command mode 等预留字段，但当前阶段不要求实现对应 planner 或 controller。

## 4. 设计原则

### 4.1 保留兼容路径

现有静态调用方式不立即删除：

```text
旧静态路径：
AtomicActionEngine.run()
        ↓
兼容调用 Engine.compile()
        ↓
输出完整位置轨迹
```

新的动态执行路径为：

```text
Engine.start()
        ↓
ExecutionSession
        ↓
反馈、监控、重规划、恢复
```

### 4.2 Action 描述意图，不负责运行时循环

Action 应负责：

- 定义技能目标；
- 生成 phase；
- 描述完成条件；
- 声明预期任务效果；
- 声明允许的恢复策略。

Action 不应负责：

- 推进仿真；
- 控制循环；
- 读取实时反馈；
- 在内部无限重试；
- 直接提交未经验证的任务状态。

### 4.3 不为动态场景继续堆叠 Action 参数

动态障碍物、跟踪阈值、重规划次数和 planner 选项属于场景、运动策略或恢复策略，不应复制到每个原子技能配置中。

### 4.4 第一版使用顺序 phase，不实现通用 DAG

现有复杂技能已经包含 approach、close、lift、release、retreat 等阶段。第一版只需要把这些边界显式化，并允许当前 phase 重规划或跳转到有限的恢复 phase。

## 5. 目标架构

```text
ActionInvocation
  ├── Goal
  ├── Binding
  ├── MotionPolicy
  └── RecoveryPolicy
          │
          ▼
     AtomicAction.plan()
          │
          ▼
       ActionPlan
  ├── PlannedPhase[]
  ├── Expected StateDelta
  └── PlannerDiagnostics
          │
          ▼
    ExecutionSession
  ├── TrajectoryExecutor
  ├── ExecutionMonitor
  ├── ReplanCoordinator
  └── EffectVerifier
          │
          ▼
    Robot / Simulation
          ▲
          │
RobotObservation + SceneSnapshot
```

## 6. 核心数据模型

### 6.1 `ActionGoal`

将 `ActionTarget` 的概念收敛为只描述目标的 `ActionGoal`：

```python
class ActionGoal:
    """描述技能要达到的目标，不携带硬件绑定和执行策略。"""
```

典型目标包括：

```python
PoseGoal(pose=...)
JointGoal(qpos=...)
EntityPoseGoal(entity_id="box", offset=...)
GraspGoal(entity_id="box", grasp_hint=...)
```

动态目标应保存实体引用和相对变换，而不是只保存已经解析的世界坐标 pose：

```python
EntityPoseGoal(
    entity_id="moving_box",
    frame="grasp_frame",
    offset=object_to_goal,
)
```

每次重规划时，通过最新 `SceneSnapshot` 重新解析目标。

迁移期间：

- 保留 `ActionTarget` 作为兼容别名；
- `targets.py` 继续导出旧名称；
- 删除没有数据、只承担标记作用的空 Target；
- 避免建立包含大量可选字段的单一 Goal 类型。

### 6.2 `ActionBinding`

`ActionBinding` 将技能语义角色绑定到机器人控制资源：

```python
ActionBinding(
    manipulators={"primary": "left_arm"},
    tools={"primary": "left_hand"},
)
```

双臂技能可以使用：

```python
ActionBinding(
    manipulators={
        "source": "left_arm",
        "destination": "right_arm",
    },
    tools={
        "source": "left_hand",
        "destination": "right_hand",
    },
)
```

以下字段应逐步从 ActionCfg 移入 Binding：

- `control_part`；
- `hand_control_part`；
- `left_arm_control_part`；
- `right_arm_control_part`；
- transfer/receive arm/hand control part。

### 6.3 `MotionPolicy`

`MotionPolicy` 统一管理运动生成策略：

```python
MotionPolicy(
    planner="curobo",
    velocity_limit=...,
    acceleration_limit=...,
    interpolation=...,
    collision_check=True,
)
```

适合放入 `MotionPolicy` 的内容包括：

- planner backend；
- planner options；
- 速度和加速度限制；
- 插值策略；
- planner attempts；
- 通用轨迹采样策略；
- 动态障碍物更新策略。

### 6.4 `RecoveryPolicy`

`RecoveryPolicy` 管理运行时阈值和恢复预算：

```python
RecoveryPolicy(
    max_replans=3,
    max_phase_retries=2,
    tracking_error_threshold=...,
    goal_translation_threshold=...,
    goal_rotation_threshold=...,
    phase_timeout=...,
)
```

所有重试都必须受到次数和总时间预算限制。

### 6.5 `EndEffectorProfile`

P0/P1 阶段先建立面向现有夹爪和手部关节的轻量 profile：

```python
EndEffectorProfile(
    control_resource="left_hand",
    open_command=...,
    close_command=...,
    hold_command=...,
)
```

以下参数应从 PickUp、Place、HandOver 等 ActionCfg 移入 profile：

- hand open/close qpos；
- transfer/receive hand open/close qpos；
- 默认开合插值策略；
- 末端执行器保持命令。

该 profile 后续可以扩展为灵巧手 synergy 或关节命令，但本轮不要求实现完整工具框架。

### 6.6 状态模型

将当前 `WorldState` 拆分为观测状态、任务状态和场景快照。

```python
@dataclass
class RobotObservation:
    timestamp: float
    qpos: torch.Tensor
    qvel: torch.Tensor
    qeffort: torch.Tensor | None
    root_pose: torch.Tensor | None
    root_twist: torch.Tensor | None
```

```python
@dataclass
class TaskState:
    held_objects: Mapping[str, HeldObjectState]
    coordinated_held_objects: Mapping[tuple[str, str], CoordinatedHeldObjectState]
```

```python
@dataclass
class SceneSnapshot:
    timestamp: float
    version: int
    entities: Mapping[str, EntityState]
```

```python
@dataclass
class PlanningContext:
    robot: RobotObservation
    task: TaskState
    scene: SceneSnapshot
    env_ids: torch.Tensor
```

迁移期间保留 `WorldState`，并提供到 `PlanningContext` 的兼容转换。

### 6.7 `StateDelta`

Action 不再直接返回已经提交的 `next_state`，而是声明预期变化：

```python
StateDelta(
    attach_objects=...,
    detach_objects=...,
)
```

状态提交流程为：

```text
规划 PickUp
    ↓
生成 expected attachment
    ↓
执行 close/lift
    ↓
验证抓取结果
    ↓
按成功 env mask 提交 attachment
```

如果执行或验证失败，`TaskState` 保持不变。

### 6.8 `TimedTrajectory`

使用带时间信息的轨迹替代单一位置 tensor：

```python
@dataclass
class TimedTrajectory:
    positions: torch.Tensor
    velocities: torch.Tensor | None
    accelerations: torch.Tensor | None
    dt: torch.Tensor
    duration: torch.Tensor
    env_ids: torch.Tensor
```

旧接口需要完整位置轨迹时，可使用 `timed_trajectory.positions` 投影回 `(B, N, DOF)`。

### 6.9 `ActionPlan` 和 phase

```python
@dataclass
class PhaseSpec:
    name: str
    goal: ActionGoal
    replannable: bool
    completion_condition: CompletionCondition
    recovery_policy: RecoveryPolicy
```

```python
@dataclass
class PlannedPhase:
    spec: PhaseSpec
    trajectory: TimedTrajectory
    planned_scene_version: int
    diagnostics: PlannerDiagnostics
```

```python
@dataclass
class ActionPlan:
    plan_success: torch.Tensor
    phases: tuple[PlannedPhase, ...]
    expected_effects: StateDelta
```

以 PickUp 为例：

| Phase | 是否允许运动重规划 | 完成条件 |
|---|---:|---|
| `approach` | 是 | 到达抓取位姿容差 |
| `close` | 否，允许 phase 重试 | 末端执行器达到关闭状态 |
| `lift` | 是 | 到达抬升目标 |
| `verify` | 否 | 确认物体仍被持有 |

## 7. 规划、执行和技能结果

必须拆分当前单一 `success` 的语义。

### 7.1 规划结果

```text
PLANNED
INFEASIBLE
PLANNER_ERROR
CANCELLED
```

### 7.2 执行状态

```text
CREATED
PLANNING
EXECUTING
REPLANNING
VERIFYING
RECOVERING
SUCCEEDED
FAILED
CANCELLED
```

### 7.3 技能结果

技能结果应包含：

- 是否完成语义目标；
- 最终失败分类；
- 执行过的 phase；
- 重规划和重试次数；
- 已提交的任务效果；
- planner 和执行诊断信息。

## 8. 分阶段实施计划

### 8.1 阶段 1：现有正确性与契约收敛

涉及文件：

- [`core.py`](../embodichain/lab/sim/atomic_actions/core.py)
- [`engine.py`](../embodichain/lab/sim/atomic_actions/engine.py)
- [`trajectory.py`](../embodichain/lab/sim/atomic_actions/trajectory.py)

任务：

1. 将 `ActionResult.success` 强制规范为 `(B,) bool tensor`；
2. 修复失败环境只冻结 `last_qpos`、却仍更新持物关系的问题；
3. 禁止对 Goal/Target tensor 做原地修改；
4. 统一所有 Action 对 planner cfg 和 plan options 的传递；
5. 明确实现或移除未使用的 `interpolation_type`；
6. 保留 planner diagnostics；
7. 为 `WorldState` 和 `ActionResult` 增加严格 shape 校验；
8. 增加每环境状态更新测试。

验收标准：

- 现有公开调用方式不变；
- 原有静态原子技能测试全部通过；
- 某环境失败后，该环境的 qpos 和任务状态均不发生变化；
- Engine 内部不再接受标量 `success`。

### 8.2 阶段 2：引入 V2 数据模型和兼容适配

建议新增：

```text
atomic_actions/
├── goals.py
├── bindings.py
├── state.py
├── effects.py
├── plans.py
└── adapters/
    └── legacy.py
```

任务：

1. 新增 `ActionGoal`，旧 `targets.py` 继续 re-export；
2. 新增 `ActionBinding`；
3. 新增 `RobotObservation`、`TaskState` 和 `SceneSnapshot`；
4. 新增 `StateDelta`；
5. 新增 `TimedTrajectory`；
6. 新增 `ActionPlan`、`PhaseSpec` 和 `PlannedPhase`；
7. 提供新旧类型之间的转换函数；
8. 首先迁移 MoveJoints 和 MoveEndEffector 验证接口。

验收标准：

- 两个简单 Action 能同时走 V2 和旧接口；
- 转换为旧 `ActionResult` 后位置轨迹保持一致；
- 新接口保留 `PlanResult` 的 dt、velocity 和 acceleration。

### 8.3 阶段 3：拆分规划与执行语义

新增规划接口：

```python
class AtomicAction:
    def plan(
        self,
        invocation: ActionInvocation,
        context: PlanningContext,
    ) -> ActionPlan:
        ...
```

Engine 增加：

```python
engine.compile(steps, context) -> CompiledTrajectory
engine.start(steps, context) -> ExecutionSession
```

兼容关系：

```text
AtomicAction.execute()    -> plan() + 旧 ActionResult 适配
AtomicActionEngine.run()  -> compile()
```

Action 迁移顺序：

1. MoveJoints；
2. MoveEndEffector；
3. MoveHeldObject；
4. Press；
5. PickUp；
6. Place；
7. HandOver；
8. Coordinated actions。

验收标准：

- 新代码不再使用“execute 表示规划”的语义；
- `execute()` 和 `run()` 只承担兼容职责；
- PickUp、Place 和 HandOver 返回明确的 phase。

### 8.4 阶段 4：实现静态闭环执行

建议新增：

```text
atomic_actions/runtime/
├── session.py
├── executor.py
├── monitor.py
├── events.py
└── status.py
```

`ExecutionSession` 使用 tick 风格接口，不在内部调用 `sleep`：

```python
event = session.tick(
    observation=robot_observation,
    scene=scene_snapshot,
)
```

P0 监控能力：

- qpos tracking error；
- qvel tracking error；
- phase timeout；
- trajectory completion；
- trajectory 时间有效性；
- 外部 cancel。

执行事件至少包括：

```text
PhaseStarted
WaypointDispatched
TrackingErrorDetected
TrajectoryInvalidated
ReplanStarted
ReplanSucceeded
EffectVerified
ActionSucceeded
ActionFailed
```

验收标准：

- 静态轨迹通过 ExecutionSession 执行时与旧 replay 结果一致；
- 注入 qpos 偏差时能够检测并进入安全 hold；
- cancel 后不再下发后续 waypoint；
- 每个失败都有结构化原因。

### 8.5 阶段 5：动态场景和停止—重规划

新增场景提供接口：

```python
class SceneProvider:
    def snapshot(self, env_ids: torch.Tensor) -> SceneSnapshot:
        ...
```

新增 planner 能力声明：

```python
PlannerCapabilities(
    dynamic_obstacle_pose_update=True,
    subset_env_planning=True,
    timed_trajectory=True,
    warm_start=False,
)
```

cuRobo 适配器负责将 `SceneSnapshot` 中声明为动态障碍物的实体转换为 `CuroboPlanOptions.dynamic_obstacle_poses`。Action 不应依赖 cuRobo 专属字段。

轨迹失效判断使用：

- 障碍物平移和旋转阈值；
- 动态目标平移和旋转阈值；
- 剩余轨迹碰撞复检；
- 轨迹有效期；
- 周期性重规划策略。

不能仅依赖 scene version，因为物理仿真中的场景状态可能每个 tick 都发生变化。

停止—重规划流程：

```text
检测轨迹失效
    ↓
下发安全 hold
    ↓
读取实际 qpos/qvel
    ↓
刷新 SceneSnapshot
    ↓
重新解析动态 Goal
    ↓
重规划当前 phase
    ↓
恢复执行
```

批量环境规划请求增加 `env_ids`：

```python
PlanningRequest(
    env_ids=invalid_env_ids,
    ...,
)
```

旧 planner 不支持 subset batch 时，通过 gather/pad/scatter 适配，不能要求所有环境因为一个环境失效而整体重启。

验收标准：

- 障碍物在规划后进入路径时触发 hold 和重规划；
- 移动目标超过阈值后重新解析目标并规划；
- 重规划起点来自实际观测，而不是预测 `last_qpos`；
- 重规划失败后产生明确结果，不发生无限循环。

### 8.6 阶段 6：错误恢复和任务状态事务

错误分类：

```text
PlanningFailed
TrajectoryInvalidated
TrackingError
GoalDrift
ExecutionTimeout
EffectVerificationFailed
UnexpectedContact
SafetyViolation
```

恢复动作：

```text
REPLAN_CURRENT_PHASE
RETRY_PHASE
TRANSITION_TO_RECOVERY_PHASE
HOLD
ABORT
```

默认规则：

- `TrajectoryInvalidated`：重规划当前 phase；
- `TrackingError`：hold 后从观测状态重规划；
- `PlanningFailed`：按预算重试或进入撤退 phase；
- `EffectVerificationFailed`：进入重新抓取或释放恢复 phase；
- `SafetyViolation`：直接终止，不自动重试。

首批技能级恢复：

1. PickUp 抓取验证失败：张开、撤退、重新接近；
2. MoveHeldObject 检测到物体丢失：hold 并终止当前技能；
3. Place 释放未完成：重试释放或安全撤退；
4. Press 未检测到预期接触：按策略重试或终止。

验收标准：

- 所有恢复都有次数和总时间预算；
- 抓取未验证时不提交 held state；
- 释放未验证时不删除 held state；
- SafetyViolation 不会被普通 retry 捕获；
- 最终结果同时包含规划、执行和技能三层状态。

### 8.7 阶段 7：参数治理和迁移收尾

参数归属：

| 当前参数 | 新归属 |
|---|---|
| `control_part` | `ActionBinding` |
| `hand_control_part` | `ActionBinding` |
| hand open/close qpos | `EndEffectorProfile` |
| velocity/acceleration | `MotionPolicy` |
| planner options | `MotionPolicy` |
| tracking/replan threshold | `RecoveryPolicy` |
| retry 次数和 timeout | `RecoveryPolicy` |
| object/pose/qpos | `ActionGoal` |
| 当前 qpos 和实体 pose | `PlanningContext` |
| phase waypoint 比例 | Action phase policy |

收尾任务：

- 删除只用于标记且没有数据的 Target；
- 合并语义重复的 Target；
- 将 HandOver 和 Coordinated actions 中重复的左右部件字段迁移到角色绑定；
- ActionCfg 只保留真正属于技能行为的参数；
- 增加配置 schema 校验和弃用提示；
- 完成所有教程和公共 API 文档迁移。

## 9. 动态执行状态机

每个环境独立维护执行状态：

```text
                 ┌──────────────┐
                 │  REPLANNING  │
                 └──────┬───────┘
                        │
                        ▼
PLANNING ──► EXECUTING ──► VERIFYING ──► SUCCEEDED
               │              │
               │              └────► RECOVERING
               │                         │
               └─────────────────────────┘

任意状态 ──► CANCELLED
任意不可恢复错误 ──► FAILED
```

当前 Engine 中单向递减的 `alive` mask 应替换为 per-env `ExecutionStatus`。为了保留批量规划效率，运行时可以将处于相同 phase 和相同操作的环境重新分桶后调用 planner。

## 10. 测试计划

### 10.1 单元测试

- success tensor shape；
- `StateDelta` 按 env mask 提交；
- `TimedTrajectory` 时间一致性；
- 动态 Goal 解析；
- phase 状态机；
- tracking error monitor；
- retry/replan budget；
- subset env gather/scatter；
- 旧接口兼容转换；
- Goal 和 frozen tensor 不被原地修改。

### 10.2 集成测试

至少覆盖：

1. 静态轨迹通过新 ExecutionSession 执行；
2. 障碍物在规划后进入路径；
3. 目标物体在 approach 阶段移动；
4. 执行中注入关节位置误差；
5. 一个环境需要重规划，其他环境继续执行；
6. PickUp 规划成功但抓取验证失败；
7. Place 释放验证失败；
8. SafetyViolation 直接终止且不重试；
9. 动态重规划超过预算后正确失败；
10. 旧静态示例通过兼容接口保持行为一致。

### 10.3 示例

建议新增：

```text
scripts/tutorials/atomic_action/
├── dynamic_obstacle_recovery.py
├── moving_target_replan.py
├── tracking_error_recovery.py
└── grasp_failure_recovery.py
```

## 11. 可观测性

每个 ExecutionSession 应记录：

- action 和 phase 名称；
- env id；
- planner backend；
- scene snapshot/version；
- 规划耗时；
- 重规划和重试次数；
- tracking error 峰值；
- trajectory invalidation 原因；
- recovery transition；
- 最终失败分类；
- `StateDelta` 是否提交。

建议将这些信息组织为 `ExecutionTrace`，用于调试、性能分析和回归测试。

## 12. PR 拆分

不建议通过单个大 PR 完成重构，推荐顺序如下：

1. **PR 1：现有正确性修复与契约统一**
2. **PR 2：V2 state/goal/plan/effect 数据模型**
3. **PR 3：MoveJoints、MoveEndEffector 迁移和兼容适配**
4. **PR 4：AtomicAction.plan 与 Engine.compile**
5. **PR 5：静态 ExecutionSession**
6. **PR 6：SceneProvider 和 cuRobo 动态障碍适配**
7. **PR 7：停止—重规划和轨迹错误恢复**
8. **PR 8：PickUp/Place 的任务状态事务和技能恢复**
9. **PR 9：其他复杂 Action 迁移与参数治理**
10. **PR 10：教程、文档、性能和恢复回归测试**

PR 1–7 构成动态运动生成 MVP；PR 8–10 完成现有技能体系的整体收敛。

## 13. MVP 完成定义

动态运动生成 MVP 完成时必须满足：

- 旧静态 API 仍可使用；
- 新 API 明确区分规划和执行；
- 轨迹包含时间信息；
- 能够检测 tracking error；
- 能够安全 hold 和 cancel；
- 能够在障碍物或目标移动后重规划当前 phase；
- 重规划从实际观测状态开始；
- 重规划次数和执行时间有界；
- 失败环境不会错误更新任务状态；
- 至少有一个多环境独立恢复测试；
- 至少有动态障碍、移动目标和跟踪误差三个可运行示例；
- 所有原有原子技能测试保持通过。

## 14. 主要风险与控制措施

| 风险 | 控制措施 |
|---|---|
| 重构范围过大导致现有技能长期不可用 | 保留 legacy adapter，按 Action 渐进迁移 |
| scene 每 tick 变化导致频繁重规划 | 使用位移阈值、碰撞复检和 cooldown，而不是只比较 version |
| 多环境恢复破坏批量效率 | 使用 `env_ids` 和同状态环境分桶 |
| planner 延迟期间继续执行不安全轨迹 | 先下发 hold，再异步或同步重规划 |
| 恢复形成无限循环 | 强制 retry、replan 和总时间预算 |
| 规划成功但任务状态错误 | 使用 `StateDelta` 和执行后验证 |
| Action 再次堆积运行时参数 | Binding、MotionPolicy 和 RecoveryPolicy 独立配置 |
| 过早设计通用工作流引擎 | P0/P1 仅实现顺序 phase 和有限恢复转移 |

## 15. 后续扩展边界

完成本计划后，下半身和 WBC 可以在以下扩展点接入，而不要求本轮实现：

- `RobotObservation.root_pose/root_twist`；
- `SceneSnapshot` 中的地形和 contact 信息；
- `ActionBinding` 中的新控制资源；
- `TimedTrajectory` 之外的 task-space reference；
- planner capability 中的 multi-frame、floating-base 和 contact；
- ExecutionSession 中的多 command mode ControllerRouter。

本轮不应因为这些未来需求扩大实施范围，但当前协议也不应继续假设所有输出永远都是固定基座机器人的位置轨迹。
