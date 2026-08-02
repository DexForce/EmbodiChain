# 原子技能模块重构计划

> 状态：已确认，PR1 实施中
>
> 实现分支：`refactor/atomic-action-pr1-contracts`
>
> 文档分支：`docs/atomic-action-refactor-plan`
>
> 最后更新：2026-08-02

## 1. 决策摘要

本轮重构采用一次性切换，不保留旧架构兼容层。原计划中的 PR1、PR2、PR3 合并为当前 PR1，并同时建立动态执行与错误恢复所需的最小运行时骨架。

核心决策如下：

1. 删除 `ActionTarget`、`WorldState`、`ActionResult`、`AtomicAction.execute()` 和 `AtomicActionEngine.run()`；
2. 使用 `ActionGoal`、`ActionBinding`、`MotionPolicy`、`RecoveryPolicy` 和 `PlanningContext` 分离不同职责；
3. `AtomicAction.plan()` 只做无副作用规划；
4. `AtomicActionEngine.compile()` 负责离线轨迹编译；
5. `AtomicActionEngine.start()` 创建 `ExecutionSession`，由 `tick()` 完成反馈式执行、失效检测和有限恢复；
6. 规划成功、轨迹执行成功和技能语义成功分别建模；
7. 抓取、释放等任务效果只在外部验证成功后提交；
8. 目标可以晚绑定到版本化场景实体，重规划时使用最新观测；
9. 下半身和 WBC 暂时降低优先级，但资源绑定、全机器人轨迹和机器人观测不再硬编码为单臂模型；
10. MLLM/Action Agent 不直接构造运行时对象，而是通过单独的语义协议和确定性编译层接入。详见[动作 Agent 对接设计](atomic_action_agent_integration.md)。

## 2. 背景和问题

旧模块已经支持批量环境、IK/运动规划器和多个复合操作技能，但其接口把规划、状态预测和执行语义混在一起：

- `execute()` 实际只生成轨迹，名称暗示它会驱动机器人；
- `run()` 一次性串联动作并返回静态轨迹，无法表达执行中的暂停、重规划或恢复；
- `WorldState` 同时承担观测状态和乐观预测状态，状态来源不清晰；
- 抓取和释放在规划成功后立即改变持物关系，即使真实执行尚未发生；
- `ActionTarget` 类型层级包含空标记类、硬件字段和运动参数，调用方参数量较大；
- `control_part`、planner 选项、采样数量和夹爪参数在多个 Action 配置中重复；
- 轨迹只消费位置采样，planner 的时间、速度和加速度信息容易丢失；
- 动态目标或障碍变化后没有正在执行轨迹的失效机制；
- 多环境中的局部失败容易被压缩成一个全局布尔值。

这些问题在静态示例中尚可绕过，但会直接阻碍移动目标、轨迹跟踪错误恢复、灵巧手接入以及后续 Action Agent 调度。

## 3. 目标与非目标

### 3.1 当前 P0

- 建立职责单一、类型明确且无旧兼容包袱的公共接口；
- 统一所有现有原子技能的调用和规划结果；
- 正确保留每个环境的规划成功状态；
- 保留轨迹时间、速度和加速度信息；
- 将任务状态更新改为声明式 `StateDelta`；
- 支持移动目标晚绑定、轨迹跟踪错误检测和停止—重规划；
- 为恢复设置次数、时间和误差边界，避免无限循环；
- 为 Action Agent 提供稳定技能标识、可枚举元数据和结构化运行事件。

### 3.2 后续 P1

- 将复合技能内部的大轨迹进一步拆成明确的顺序 phase；
- 接入真实控制循环的 hold、cancel、command acknowledgement 和安全检查；
- 建立 `SceneProvider`，让动态障碍物更新进入统一快照和轨迹失效流程；
- 增加抓取失败、物体丢失、释放失败等技能级恢复；
- 补充动态目标、动态障碍和跟踪误差的可运行示例；
- 增加 `ExecutionTrace`、延迟指标和恢复统计。

### 3.3 暂缓

- 下半身技能和步态规划；
- 浮动基座、接触规划、WBC、MPC 和质心动力学；
- 硬实时控制系统；
- 任意运行时工具热插拔；
- 由 MLLM 直接生成关节轨迹或任意 Python 参数。

接口为 `root_pose`、`root_twist`、全机器人 DoF、语义资源角色和不同执行后端预留扩展点，但当前 PR1 不实现对应控制器。

## 4. 总体架构

```text
语义请求（应用或 Action Agent）
             │
             ▼
      ActionInvocation
  ┌──────────┼───────────┐
  │          │           │
 Goal     Binding      Policies
             │
             ▼
      AtomicAction.plan()
             │
             ▼
         ActionPlan
  ┌──────────┼──────────────┐
  │          │              │
 Phases  TimedTrajectory  StateDelta
             │
       ┌─────┴─────┐
       ▼           ▼
 Engine.compile  Engine.start
 离线静态编译    ExecutionSession.tick
                   │
                   ▼
       command / event / verified state
```

`plan()`、`compile()` 和 `start()/tick()` 的语义必须保持严格区分：

| 接口 | 职责 | 是否推进仿真 | 是否提交真实任务状态 |
|---|---|---:|---:|
| `AtomicAction.plan()` | 将一次已绑定调用规划为 `ActionPlan` | 否 | 否 |
| `Engine.compile()` | 串联多个计划，生成离线完整轨迹和预测状态 | 否 | 否 |
| `Engine.start()` | 创建有状态执行会话 | 否 | 否 |
| `ExecutionSession.tick()` | 消费最新观测并产生下一控制命令 | 由调用方执行命令 | 仅验证后提交 |

## 5. 核心数据模型

### 5.1 `ActionGoal`

Goal 只描述技能要达到的目标。每个技能拥有自己的不可变 dataclass，例如：

- `EndEffectorPoseGoal`；
- `JointPositionGoal`；
- `NamedJointPositionGoal`；
- `GraspGoal`；
- `HeldObjectPoseGoal`；
- `PlaceGoal`；
- `AssembleGoal`；
- `PressGoal`；
- `CoordinatedPickGoal`；
- `CoordinatedPlacementGoal`。

不再保留无字段的 Target 标记类型，也不建立一个包含大量可选字段的万能 Goal。

动态 pose 使用实体引用：

```python
EndEffectorPoseGoal(
    xpos=SceneEntityPose(
        entity_id="moving_fixture",
        relative_pose=tool_offset,
        minimum_confidence=0.8,
    )
)
```

调用意图保持不变；每次 `plan()` 或重规划都从最新 `SceneSnapshot` 解析世界坐标。

### 5.2 `ActionBinding`

Binding 把技能语义角色映射到具体机器人资源：

```python
ActionBinding(
    manipulators={"primary": "left_arm"},
    end_effectors={"primary": "left_hand"},
)
```

双臂技能使用 `source`、`destination` 或其他稳定角色，而不是让上层直接依赖某个机器人命名：

```python
ActionBinding(
    manipulators={
        "source": "left_arm",
        "destination": "right_arm",
    },
    end_effectors={
        "source": "left_hand",
        "destination": "right_hand",
    },
)
```

当前 PR1 先让简单运动技能完全由 Binding 决定资源；部分复合技能仍需将 Binding 与实例配置进行一致性校验。后续应把夹爪/灵巧手姿态和复合控制组迁入机器人 capability/profile，最终消除复合技能配置中的硬件资源名称。

### 5.3 `MotionPolicy`

跨技能重复的运动参数统一放入 MotionPolicy：

- planner backend 约束；
- `motion_source`；
- interpolation；
- sample count；
- control period；
- velocity/acceleration limit；
- collision checking；
- 类型化 planner options。

技能配置只保留技能算法自身的参数，例如抓取接近距离、提起高度和放置 retreat 高度。

### 5.4 `RecoveryPolicy`

恢复策略独立于技能目标：

- 最大重规划次数；
- 最大 phase/技能重试次数；
- tracking error 阈值；
- 动态目标平移和旋转阈值；
- phase timeout。

策略必须有界。预算耗尽后产生结构化失败事件，并把决策权交还上层任务图或 Action Agent。

### 5.5 状态分层

```text
RobotObservation  实际测量的 qpos/qvel/根部状态
TaskState         已验证的持物等符号关系
SceneSnapshot     带 timestamp/version/confidence 的场景实体状态
PlanningContext   一次规划使用的完整不可变输入
```

离线 `compile()` 可以生成 `projected_context`，但该状态明确是预测值，不能覆盖执行会话中的已验证状态。

### 5.6 `ActionPlan` 和 `TimedTrajectory`

`ActionPlan` 包含：

- 稳定 `skill_id`；
- 每环境 `plan_success`；
- 一个或多个 `PlannedPhase`；
- 未提交的 `StateDelta`；
- planner diagnostics；
- 可选 invocation correlation id。

`TimedTrajectory` 始终是全机器人 joint space 表示，包含：

- positions；
- 可选 velocities；
- 可选 accelerations；
- 每个采样点的 `dt`；
- 每环境 duration；
- env ids。

全机器人轨迹并不表示所有关节都必须由一个 planner 控制。未参与技能的关节保持观测值；以后可以由资源仲裁层组合手臂、手、底盘或 WBC command。

### 5.7 `StateDelta`

规划只声明预期效果，例如：

```python
StateDelta(held_object_updates={"left_arm": held_object})
```

静态编译时可以把它应用到预测状态，以便规划后继动作。动态执行时必须等待感知、接触或任务验证器给出 `effect_success`，只对验证成功的环境行提交。

## 6. 动态场景与错误恢复

### 6.1 执行循环

`ExecutionSession` 不直接操作仿真器或设备。调用方循环执行：

```text
observe robot + scene
        │
        ▼
session.tick(context, effect_success?)
        │
        ├── JointCommand  → controller/simulation
        └── ExecutionEvent → trace/task graph/agent
```

这使同一状态机可以服务仿真和真实机器人，同时保持控制后端可替换。

### 6.2 轨迹失效来源

PR1 支持或预留以下失效来源：

| 来源 | 检测依据 | 默认处理 |
|---|---|---|
| tracking error | 实际 qpos 与上一 command 的最大偏差 | 当前 phase 重规划 |
| moving goal | 引用实体相对规划快照的位姿变化 | 从最新观测重规划 |
| timeout | 当前 phase 已执行时间 | 技能重试或失败 |
| planning failure | 每环境 `plan_success` | 有界重试 |
| effect failure | 外部验证结果 | 技能重试或失败 |
| moving obstacle | 场景/碰撞世界版本变化 | 后续 SceneProvider PR 接入 |

所有重规划必须以最新 `RobotObservation` 为起点，不能从旧轨迹上的预测 waypoint 继续。

### 6.3 恢复分层

恢复分为三层：

1. **运动层本地恢复**：hold、当前 phase 重规划、有限重试，不调用 MLLM；
2. **任务图已知恢复**：进入预先校验的恢复分支，例如重新观察、重新抓取或换候选；
3. **未知失败升级**：本地预算和已知分支都耗尽后，在安全边界向 Action Agent 请求重规划。

“预先规划恢复”指恢复结构、约束和监控条件预先确定，不等于提前冻结一条轨迹。实际轨迹仍在进入分支时结合最新场景生成。

### 6.4 批量环境语义

- 成功、失败、active、verified 和 recovery budget 均使用形状为 `(B,)` 的布尔或整数 tensor；
- 失败环境输出 hold command，不更新任务状态；
- 静态编译中的后续技能只对仍成功的环境行生效；
- 当前执行实现使用动作边界同步，以避免不同环境在同一 action sequence 中悄然错位；
- 真正的每环境异步 phase 调度可在后续 runtime PR 中增加。

## 7. 参数治理

参数按照“谁拥有语义”归属：

| 参数类别 | 归属 |
|---|---|
| 目标 pose、对象、关节目标 | Goal |
| 语义角色到机器人资源 | Binding |
| planner、采样、速度、碰撞选项 | MotionPolicy |
| 阈值、超时、重规划预算 | RecoveryPolicy |
| 手型、关节映射、控制模式 | Robot capability/profile |
| 接近距离、提起高度、技能 phase 行为 | ActionCfg |
| 实际机器人/场景/任务状态 | PlanningContext |

ActionCfg 不再接收一次调用才会变化的 pose、qpos、场景实体或运行时阈值。

对灵巧手的扩展不应向每个技能增加一组 `hand_open_qpos`/`hand_close_qpos`。后续 capability 应暴露语义命令，例如 `open`、`pregrasp`、`grasp`、`release`，由具体末端执行器适配为 joint command。

## 8. PR1 范围

当前 PR1 合并原计划的前三个基础 PR，并包含动态执行最小骨架：

- 新增 Goal、Binding、Policy、State、Effect、Invocation 和 Plan 模型；
- 删除旧 Target/WorldState/Result 以及 execute/run 接口；
- 引入稳定 `skill_id` 和 `SkillDescriptor`；
- 实现 `AtomicAction.plan()`、`Engine.compile()` 和 `Engine.start()`；
- 迁移所有现有原子技能、测试、教程和 benchmark；
- 修正 per-environment 成功传播和状态更新；
- 保留 planner timing 和 derivatives；
- 支持晚绑定移动目标；
- 实现 tracking error、目标变化、timeout、重规划预算和效果验证；
- 更新 API 文档、开发上下文和 `add-atomic-action` skill。

PR1 不承诺完成：

- 真实硬件 controller 接线；
- 动态障碍 `SceneProvider` 全链路；
- 所有复杂技能的细粒度 phase 拆分；
- 通用技能恢复图；
- Action Agent JSON adapter；
- 下半身或 WBC controller。

## 9. 后续实施顺序

### PR2：执行器与场景闭环

- 抽象 `CommandSink`/executor；
- 支持 command acknowledgement、cancel 和 safe hold；
- 引入 `SceneProvider` 和 collision-world revision；
- 动态障碍变化触发 trajectory invalidation；
- 增加移动目标、动态障碍和跟踪误差示例。

### PR3：复合技能 phase 与技能级恢复

- 将 PickUp、Place、Press、HandOver、双臂技能拆成明确 phase；
- 为 phase 定义 completion monitor；
- 增加 grasp lost、release failed、contact missing 等事件；
- 实现预定义 recovery transition。

### PR4：参数与 capability 收敛

- 把复合技能中的硬件资源名迁移到 Binding；
- 建立 gripper/dexterous-hand capability 和语义手型 profile；
- 让 descriptor 声明资源需求和可接受 goal schema；
- 完善机器人无关的 skill catalog。

### PR5：Action Agent adapter

- 定义 `SkillCallSpec`、`TaskGraphSpec` 和 `SkillResult`；
- 从 `SkillDescriptor` 生成 agent tool/schema；
- 实现 schema validator、entity grounder、capability binder 和 policy guard；
- 将 ExecutionEvent 映射为图执行和 MLLM 反馈；
- 建立 trace 和数据回流。

下半身和 WBC 在这些接口稳定后再单独规划，不阻塞当前主线。

## 10. 验收标准

### PR1

- 不再导出或引用旧架构类型和方法；
- 所有内置技能使用 `plan(invocation, context)`；
- 简单技能不在 ActionCfg 重复声明控制部件；
- `compile()` 不修改机器人和输入状态；
- 失败环境保持当前位置且不会错误获得任务效果；
- planner timing/velocity/acceleration 可进入 `TimedTrajectory`；
- 移动实体目标会基于新快照重规划；
- tracking error 和 timeout 的恢复次数有界；
- 非空 `StateDelta` 未验证时不能提交；
- 原子技能测试、受影响 planner 测试和格式检查通过；
- 教程、API 文档和开发 skill 与新接口一致。

### 动态执行完整 MVP

- controller 可以 hold/cancel 并确认 command；
- 动态障碍变化能够使当前轨迹失效；
- 至少有移动目标、动态障碍和 tracking error 三个可运行示例；
- 每个恢复路径都包含成功、预算耗尽和多环境局部失败测试；
- 结构化 trace 能还原每次规划、失效、重试、效果验证和状态提交。

## 11. 主要风险

| 风险 | 控制措施 |
|---|---|
| PR1 变更面大 | 无兼容层但保持单一原子提交；按模块测试并更新所有仓内调用方 |
| runtime 与控制频率耦合 | TimedTrajectory 保存时间，executor 明确定义 tick/command 时序 |
| 重规划造成抖动 | 位姿变化阈值、冷却策略和次数预算 |
| 场景快照与机器人观测不同步 | timestamp/version 校验，过期数据拒绝或降级 |
| 规划成功被误认为技能成功 | plan、execution、effect 三种结果分离 |
| MLLM 产生危险或无效参数 | 语义协议、白名单约束、确定性 grounding 和 policy guard |
| 灵巧手继续造成参数膨胀 | capability/profile 拥有硬件状态，技能只使用语义动作 |
| 为未来 WBC 过度设计 | 只保留资源和状态扩展点，不提前实现控制算法 |

## 12. 结论

这次重构的重点不是简单改名，而是把原子技能从“静态轨迹拼接函数”变成一个清晰的确定性运动后端：Goal 表达意图，Binding 选择机器人资源，Policy 描述可复用策略，PlanningContext 提供最新状态，ActionPlan 声明轨迹和未提交效果，ExecutionSession 负责有限闭环恢复。

这个边界既能支撑现有手臂技能，也能为动态场景、灵巧手、Action Agent 和更远期的全身控制提供稳定接口，同时避免现在为低优先级的下半身/WBC 引入过多实现复杂度。
