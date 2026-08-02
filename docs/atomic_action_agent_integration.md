# 原子技能与动作 Agent 对接设计

> 状态：Design proposal
>
> 依赖：原子技能重构 PR1
>
> 参考：[EmbodiChain PR #306](https://github.com/DexForce/EmbodiChain/pull/306)、[AgentChord paper](https://arxiv.org/html/2605.11951v1)
>
> 最后更新：2026-08-02

相关文档：[原子技能模块重构计划](atomic_action_refactoring_plan.md)。

## 1. 结论

重构后的原子技能架构对 MLLM/动作 Agent 的接入是有帮助的，但 `AtomicAction.plan()` 不应成为 MLLM 直接调用的接口。

推荐增加一个独立的 **Action Agent Adapter**：

```text
MLLM / Task Agent
      │  生成语义 JSON
      ▼
TaskGraphSpec / SkillCallSpec
      │  schema + graph + policy validation
      ▼
Grounder / Capability Binder
      │  确定性解析实体、资源和策略
      ▼
ActionInvocation
      │
      ▼
AtomicAction.plan() → ActionPlan
      │
      ▼
ExecutionSession.tick()
      │
      ├── ExecutionEvent → 图运行时 / Agent
      └── SkillResult    → 上层状态更新和下一步决策
```

这条边界带来四个直接收益：

1. MLLM 只处理稳定的语义对象，不需要知道 Python 类、tensor、planner 配置和机器人关节名；
2. 所有 agent 输出都经过确定性校验、grounding 和安全策略，再进入运动层；
3. 动态目标和错误恢复由执行运行时持续处理，避免每个控制 tick 都调用 MLLM；
4. 原子技能可以继续作为仿真、真实机器人、脚本和 agent 共用的确定性后端。

## 2. 模块边界

### 2.1 Action Agent 负责什么

- 根据用户目标和场景语义选择技能；
- 生成任务图或恢复分支；
- 使用逻辑实体引用，例如 `object:cup_1`、`location:tray_slot_2`；
- 选择允许的高层约束或 motion profile；
- 在未知失败或任务目标改变时重新规划任务图；
- 解释结构化技能结果，并决定继续、切换分支或停止。

### 2.2 原子技能层负责什么

- 校验具体 Goal 类型；
- 从最新 `PlanningContext` 生成可执行 `ActionPlan`；
- 保留每环境规划状态、轨迹时间和诊断信息；
- 声明尚未提交的 `StateDelta`；
- 在执行会话中检测跟踪误差、目标变化、超时和效果失败；
- 执行有界的本地 hold、重规划和重试；
- 输出结构化事件，不做开放式语言推理。

### 2.3 Adapter 负责什么

- 生成并发布技能目录；
- 校验 MLLM JSON；
- 把逻辑实体解析到版本化场景实体；
- 把语义角色绑定到机器人 capability；
- 将公开约束映射为受控的 `MotionPolicy`/`RecoveryPolicy`；
- 构造 Python 侧类型化 Goal 和 `ActionInvocation`；
- 把执行事件和结果序列化回 agent 协议；
- 拒绝越权资源、任意 planner options 和未知字段。

## 3. 为什么不能让 MLLM 直接调用 `AtomicAction.plan()`

`plan()` 是一个确定性的 Python 运行时契约：

```python
plan(
    invocation: ActionInvocation[GoalT],
    context: PlanningContext,
) -> ActionPlan
```

它要求调用方已经完成：

- 技能 ID 选择；
- Goal 类型构造；
- 场景实体解析；
- 机器人资源绑定；
- planner/recovery policy 选择；
- tensor 的 device、dtype、batch shape 处理；
- 安全和权限校验。

这些都不适合作为 MLLM 的自由输出。若直接暴露，prompt 会绑定内部类名和机器人实现，schema 会随着代码重构而频繁变化，也会允许模型绕过资源和策略约束。

因此需要三层表示：

| 层 | 表示 | 面向对象 | 特点 |
|---|---|---|---|
| Agent IR | `TaskGraphSpec` / `SkillCallSpec` | MLLM、服务接口 | 稀疏、JSON、语义化、稳定 |
| Grounded IR | `ActionInvocation` | 确定性 adapter、原子技能 | 类型化、已绑定、可审计 |
| Execution IR | `ActionPlan` / `ExecutionSession` | planner、controller、monitor | tensor、轨迹、时序、恢复状态 |

禁止把三层压成一个“万能 Action 参数对象”。

## 4. 与 PR #306 的关系

PR #306 正在探索 MLLM 动作生成和 JSON 校验，方向上可以复用以下思路：

- MLLM 输出结构化动作而不是自由文本；
- 动作进入执行前经过 schema/语义检查；
- 使用 task graph 表达多步任务；
- 将任务规划和底层执行分开；
- 保留可恢复、可审计的中间表示。

建议在其后续改造中收敛以下边界：

| Review 中可见的问题 | 建议调整 |
|---|---|
| JSON 暴露 Python Action 类或实现模块 | 只暴露稳定 `skill_id` |
| 上层指定 `left_arm`、具体 controller 等后端细节 | 使用 `primary/source/destination` 等语义角色，由 binder 解析 |
| target 中混合语义目标、raw config 和可空字段 | 每个技能生成独立、严格的 argument schema |
| 上层可以传任意 planner/config 字典 | 只允许命名 motion/recovery profile 或白名单约束 DSL |
| 图主要是线性动作列表 | 支持 monitor、条件边和预定义 recovery branch |
| 失败统一回到语言模型 | 先走运动层本地恢复和已知图分支，未知失败才升级 |
| 执行返回值信息不足 | 返回稳定 failure code、env mask、scene version、诊断和 trace id |

由于 PR #306 仍在 Review 和演进中，本设计不依赖它当前的具体 Python 类或 JSON 字段。推荐把它的 MLLM prompt/graph 能力放在 Agent IR 层，并使用本文 adapter 对接原子技能运行时。

## 5. 与 AgentChord 的关系

AgentChord 中动作 agent/任务图的核心启发适合放在原子技能之上，而不是塞进单个技能内部：

- 全局任务关系由图表达；
- 节点使用语义动作和对象引用；
- 执行前做结构与约束检查；
- 运行时按小段执行，并根据观测更新；
- 可预期失败使用预定义恢复分支；
- 只有未知失败才需要更高层重新推理。

映射到 EmbodiChain：

| AgentChord 概念 | EmbodiChain 建议实现 |
|---|---|
| Action/skill node | `SkillCallSpec`，编译为 `ActionInvocation` |
| Task graph | `TaskGraphSpec` + `TaskGraphRuntime` |
| Grounding | Entity resolver + capability binder |
| Action execution | `AtomicAction.plan()` + `ExecutionSession` |
| Monitor | `ExecutionEvent`、effect verifier、scene revision |
| Recovery branch | 图上的类型化 failure edge |
| Replan | 未知失败时由 MLLM 生成新图或图 patch |

原子技能内部仍采用顺序 phase，因为 approach、close、lift 等局部时序不需要通用 DAG。跨技能依赖、并行性和恢复分支则放在上层任务图。这避免同时维护两套重叠的图执行器。

## 6. Agent IR

### 6.1 `SkillCallSpec`

建议的 agent-facing 请求：

```json
{
  "call_id": "call-17",
  "skill_id": "pick_up",
  "arguments": {
    "object": {"entity_id": "cup_1"},
    "grasp_hint": "side"
  },
  "constraints": {
    "motion_profile": "safe",
    "avoid_regions": ["human_workspace"]
  }
}
```

设计规则：

- `skill_id` 是稳定公开 ID，不是类名；
- `arguments` schema 由具体技能拥有，禁止未知字段；
- 实体以稳定逻辑 ID 引用，不输出 world pose tensor；
- 不允许出现 robot joint name、device、Python callable 或任意 config；
- profile 是部署方注册的名字，不是展开后的 planner 参数；
- `call_id` 贯穿 invocation、plan、event 和 result。

对于移动目标：

```json
{
  "call_id": "call-18",
  "skill_id": "move_end_effector",
  "arguments": {
    "target": {
      "entity_id": "moving_fixture",
      "frame": "insertion_frame",
      "offset": {
        "translation": [0.0, 0.0, 0.05],
        "rotation_rpy": [0.0, 0.0, 0.0]
      }
    }
  },
  "constraints": {
    "motion_profile": "dynamic_target"
  }
}
```

Adapter 将其编译为 `SceneEntityPose`，而不是立刻固化成世界坐标。这样 ExecutionSession 重规划时仍能解析最新目标。

### 6.2 `TaskGraphSpec`

推荐使用显式节点和类型化边：

```json
{
  "graph_id": "serve-cup-42",
  "version": "1.0",
  "entry": "observe-cup",
  "nodes": [
    {
      "node_id": "observe-cup",
      "kind": "observe",
      "arguments": {"entity_id": "cup_1"},
      "on_success": "pick-cup",
      "on_failure": "abort"
    },
    {
      "node_id": "pick-cup",
      "kind": "skill",
      "call": {
        "call_id": "pick-1",
        "skill_id": "pick_up",
        "arguments": {"object": {"entity_id": "cup_1"}},
        "constraints": {"motion_profile": "safe"}
      },
      "on_success": "place-cup",
      "on_failure": {
        "grasp_not_verified": "reobserve-cup",
        "local_recovery_exhausted": "ask-agent"
      }
    },
    {
      "node_id": "reobserve-cup",
      "kind": "observe",
      "arguments": {"entity_id": "cup_1"},
      "on_success": "pick-cup",
      "on_failure": "abort"
    }
  ]
}
```

图验证至少检查：

- node ID 唯一；
- entry 和所有边目标存在；
- skill 在当前 catalog 中可见；
- arguments 符合技能 schema；
- 循环具有次数或时间上界；
- 并行节点的资源不冲突；
- failure code 合法且有默认处理；
- policy/capability 允许调用；
- 图不存在无法到达的危险动作。

### 6.3 约束 DSL

第一版只支持少量结构化约束：

- 命名 motion/recovery profile；
- 最大速度等级，例如 `slow/normal/fast`；
- 允许/禁止的语义区域；
- 是否要求 collision check；
- deadline 或最大技能时长；
- 目标置信度下限。

不要允许 MLLM 直接传 `PlanOptions`、碰撞缓存、采样核参数或任意 Python 表达式。高级参数仍由部署配置控制。

## 7. 技能目录

当前 `SkillDescriptor` 已提供最小基础：

- `skill_id`；
- `goal_type`；
- manipulator roles；
- end-effector roles；
- `agent_visible`。

Action Agent adapter 应从 registry 构建一个序列化 catalog。后续 descriptor 可扩展：

```python
@dataclass(frozen=True)
class AgentSkillDescriptor:
    skill_id: str
    description: str
    argument_schema: dict[str, object]
    required_capabilities: tuple[str, ...]
    resource_roles: tuple[str, ...]
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    failure_codes: tuple[str, ...]
    motion_profiles: tuple[str, ...]
```

这里的 `AgentSkillDescriptor` 是 adapter 导出的语义视图，不建议让原子技能核心依赖 JSON Schema 库。

Catalog 的用途：

1. 生成 MLLM tool definitions/prompt；
2. 在服务端校验 `SkillCallSpec`；
3. 根据机器人 capability 过滤不可用技能；
4. 为任务图做资源冲突和前置条件检查；
5. 固定训练、评测和 trace 使用的技能版本。

建议给 catalog 增加 `catalog_version` 和每技能 `schema_version`。技能 ID 稳定，schema 破坏性变化通过版本明确管理。

## 8. Grounding 和编译

### 8.1 编译流水线

```text
SkillCallSpec
  │
  ├─ 1. SchemaValidator
  ├─ 2. EntityResolver
  ├─ 3. CapabilityBinder
  ├─ 4. PolicyGuard
  ├─ 5. GoalFactory
  ▼
ActionInvocation
```

每一步都返回结构化错误，不使用模糊字符串作为控制流。

### 8.2 EntityResolver

输入逻辑实体引用，输出：

- 当前 scene 中的 stable entity ID；
- 类型、frame 和 affordance；
- confidence、timestamp 和 scene version；
- 是否保持 late-bound；
- 解析歧义及候选列表。

一次请求若有多个候选且无法确定，应返回 `ambiguous_entity`，由上层补充观察或重新选择；不得静默选择第一个对象。

### 8.3 CapabilityBinder

Binder 根据技能角色、机器人 capability 和资源占用情况生成 `ActionBinding`：

```text
required role: primary manipulator + grasp end effector
robot capability candidates:
  left_arm + left_gripper
  right_arm + right_dexterous_hand
             │
             ▼
policy/cost/resource arbitration
             │
             ▼
ActionBinding(...)
```

选择结果必须进入 trace。MLLM 可以表达偏好，例如“用空闲的手”，但最终绑定由确定性规则和资源仲裁决定。

### 8.4 GoalFactory

GoalFactory 是允许的 `skill_id → parser/factory` 映射。例如：

```python
def compile_move_eef(spec: SkillCallSpec, scene: SceneSnapshot) -> ActionInvocation:
    target = resolve_entity_reference(spec.arguments["target"], scene)
    goal = EndEffectorPoseGoal(
        xpos=SceneEntityPose(
            entity_id=target.entity_id,
            relative_pose=target.offset,
            minimum_confidence=target.minimum_confidence,
        )
    )
    return ActionInvocation(
        skill_id="move_end_effector",
        goal=goal,
        binding=bind_resources(spec),
        motion_policy=resolve_motion_profile(spec),
        recovery_policy=resolve_recovery_profile(spec),
        invocation_id=spec.call_id,
    )
```

上例是接口示意。生产实现应使用类型化 spec，而不是直接索引未经验证的 dict。

## 9. 执行、恢复和 Agent 重新规划

### 9.1 运行时事件

现有 ExecutionSession 已能输出如下类别的事件：

- action planned / replanned；
- tracking error；
- dynamic goal changed；
- phase timeout；
- effect verification required；
- action retry；
- recovery exhausted；
- action/session completed。

后续需要增加稳定 failure code，而不是让 Agent 解析 message：

```json
{
  "event_id": "evt-991",
  "trace_id": "trace-42",
  "call_id": "pick-1",
  "skill_id": "pick_up",
  "kind": "recovery_exhausted",
  "failure_code": "grasp_not_verified",
  "env_ids": [3, 7],
  "timestamp": 12.84,
  "scene_version": 105,
  "recoverable_locally": false,
  "diagnostics": {
    "replan_count": 3,
    "retry_count": 2
  }
}
```

### 9.2 三层恢复策略

#### 层 1：运动层本地恢复

适用：tracking error、移动目标、小范围碰撞世界变化、planner transient failure。

处理：hold → 更新快照 → 从实际状态重规划 → 继续。全过程不调用 MLLM。

#### 层 2：任务图已知恢复

适用：抓取未验证、目标暂时丢失、候选 grasp 失败、放置区域被占用。

处理：进入图中已校验的分支，例如 reobserve、select next grasp、clear region、retry skill。

#### 层 3：Action Agent 重新规划

适用：本地预算耗尽、已知分支不可用、任务条件发生非局部变化或用户目标改变。

升级必须发生在安全边界：机器人 hold/cancel 完成，执行上下文已快照化，未验证的 StateDelta 未提交。

提供给 MLLM 的上下文应该是压缩后的结构化事实：

- 当前任务目标和已完成图节点；
- verified TaskState；
- 可见实体、置信度和关键关系；
- 失败 code、次数和最近诊断；
- 当前可用技能/capability；
- 仍然有效的安全约束。

不要把完整关节轨迹、raw tensor 日志或无界历史全部放入 prompt。

### 9.3 “预编译恢复”的含义

恢复图预先确定的是：

- 可以走哪些分支；
- 触发分支的 failure/monitor 条件；
- 循环预算；
- 资源和安全约束；
- 每个分支可调用的技能。

它不预先冻结动态场景中的具体轨迹。进入恢复节点后，仍通过最新 `PlanningContext` 调用 `AtomicAction.plan()`。

## 10. `SkillResult`

每个技能终止时向图运行时返回统一结果：

```json
{
  "call_id": "pick-1",
  "skill_id": "pick_up",
  "status": "succeeded",
  "succeeded_env_ids": [0, 1, 2],
  "failed_env_ids": [],
  "effects": [
    {"predicate": "holding", "resource_role": "primary", "entity_id": "cup_1"}
  ],
  "scene_version": 108,
  "trace_id": "trace-42",
  "metrics": {
    "planning_attempts": 2,
    "replans": 1,
    "duration_s": 4.31
  }
}
```

状态建议分为：

- `succeeded`：效果已验证；
- `failed`：不可恢复或预算耗尽；
- `cancelled`：上层取消且已安全停止；
- `precondition_failed`：未开始执行；
- `partially_succeeded`：批量环境部分成功；
- `needs_replan`：安全停止后请求上层重新规划。

`planned` 不是技能成功状态。

## 11. 安全和确定性边界

在 Agent 请求进入 motion layer 前必须满足：

- JSON schema 严格校验并拒绝未知字段；
- 技能存在、版本兼容且 `agent_visible=True`；
- 所有实体引用存在或有明确的重新观察策略；
- capability 和资源绑定合法；
- 参数落在部署策略允许范围内；
- task graph 循环、并行和恢复都有界；
- 规划器及 controller 的碰撞和速度策略不可被 MLLM 绕过；
- 高风险技能可以要求外部 approval；
- 所有编译决策、schema 版本和绑定写入 trace。

MLLM 的输出永远是“候选程序”，不是直接控制命令。

## 12. 可观测性和数据闭环

建议 trace 至少记录：

- prompt/request ID、graph ID、catalog/schema version；
- 原始 SkillCallSpec 和校验后的 canonical form；
- entity grounding 结果及置信度；
- capability binding 及候选淘汰原因；
- ActionInvocation correlation ID；
- planner backend、scene version、规划耗时和诊断；
- command/observation 时间戳；
- tracking error 峰值；
- 每次 invalidation、replan、retry 和 recovery transition；
- effect verification 结果和 StateDelta commit；
- 最终 SkillResult。

这些数据可用于：

- 调试 agent 与 motion layer 的责任边界；
- 构造任务图生成和失败恢复训练数据；
- 离线回放 agent 决策；
- 比较不同 MLLM 的 schema 合法率和任务成功率；
- 发现频繁升级到 MLLM 的失败类别，并将其下沉为确定性恢复策略。

隐私和数据量控制应在 trace 层完成；默认不把原始图像和全频控制日志回传给 MLLM。

## 13. 推荐代码结构

```text
embodichain/agents/action/
├── schemas.py          # SkillCallSpec / TaskGraphSpec / SkillResult
├── catalog.py          # descriptor → agent catalog/schema
├── validation.py       # JSON、graph、constraint validation
├── grounding.py        # entity reference → semantic target
├── binding.py          # capability/resource arbitration
├── policies.py         # named profile registry and guard
├── compiler.py         # SkillCallSpec → ActionInvocation
├── graph_runtime.py    # graph state and typed transitions
├── events.py           # ExecutionEvent → graph/agent event
└── tracing.py          # correlation and replay records
```

核心 `embodichain.lab.sim.atomic_actions` 不依赖 agents 包。依赖方向只能是：

```text
agents.action → lab.sim.atomic_actions
```

这样原子技能仍可在不安装或不启用 MLLM 的环境中独立使用。

## 14. 实施计划

### 阶段 A：协议和目录

- 冻结稳定 skill IDs；
- 定义 `SkillCallSpec`、`TaskGraphSpec`、`SkillResult`；
- 从 `SkillDescriptor` 生成只读 catalog；
- 为 3 个代表性技能提供 schema：MoveEndEffector、PickUp、Place；
- 增加 strict validation 和版本测试。

### 阶段 B：Grounding 和 Binding

- 接入 SceneSnapshot/entity registry；
- 实现 entity ambiguity 和 stale scene 错误；
- 建立 robot capability inventory；
- 编译为 `ActionInvocation`；
- 增加 round-trip/negative tests。

### 阶段 C：图运行和恢复

- 执行线性图与条件 failure edge；
- 将 ExecutionEvent 映射为 graph transition；
- 实现本地恢复、已知图恢复、MLLM 升级三层边界；
- 支持安全 cancel/hold 后 graph patch。

### 阶段 D：PR #306 对接

- 让现有 MLLM prompt 输出新的 Agent IR；
- 移除 Python 类、robot resource 和 raw config 暴露；
- 使用 catalog 动态生成 tool schema；
- 将原 PR 的 executor 接口替换为 graph runtime + compiler；
- 保留其有效的 prompt、task decomposition 和验证逻辑。

### 阶段 E：评测和数据闭环

- schema validity；
- grounding accuracy；
- task graph validation rate；
- skill plan/execution/effect success 分项指标；
- local recovery rate；
- known-branch recovery rate；
- MLLM escalation rate 和二次成功率；
- 安全策略拒绝准确率。

## 15. MVP 验收标准

- MLLM 输出中不包含 Python 类名、tensor、joint name 或 raw planner config；
- 每个公开技能有稳定 ID、严格 arguments schema 和 capability 需求；
- `SkillCallSpec` 能确定性编译为 `ActionInvocation`；
- 相同 canonical spec、catalog、scene version 和 robot profile 可复现相同 grounding；
- 移动实体使用 late-bound goal，重规划时不会继续使用旧 world pose；
- planner、execution 和 effect failure 可以区分；
- tracking error 不会立即触发 MLLM；
- 已知抓取失败能进入图中恢复分支；
- 本地与图恢复耗尽后能安全停止并生成 `needs_replan`；
- 非法技能、未知字段、越权资源和危险策略被拒绝；
- invocation、plan、events、result 能通过 trace ID 串联；
- 仿真脚本和非 Agent 调用不依赖 agents 包。

## 16. 当前缺口

原子技能重构 PR1 已经或正在提供：

- 稳定 `skill_id` 和基础 `SkillDescriptor`；
- `ActionInvocation`；
- Goal/Binding/Policy 分层；
- versioned `SceneSnapshot` 和 late-bound `SceneEntityPose`；
- `ActionPlan`、`StateDelta` 和有界 `ExecutionSession`；
- 结构化 ExecutionEvent 和 correlation ID。

仍需在后续 PR 完成：

- Agent IR 的正式 dataclass/JSON schema；
- 自动 catalog/schema 生成；
- entity resolver 和 capability inventory；
- policy guard；
- task graph validator/runtime；
- 稳定 failure code 和 SkillResult；
- PR #306 的 adapter 迁移；
- controller/SceneProvider 的真实闭环接线。

## 17. 最终建议

将动作 Agent 看作“受约束任务程序生成器”，将原子技能模块看作“确定性、可观测、可恢复的运动执行后端”。两者通过稳定的 Agent IR 和编译器连接，而不是互相泄漏内部对象。

这一结构既保留 PR #306 的 MLLM 任务生成价值，也吸收 AgentChord 中任务图、监控和恢复分层的思想，并能利用本次原子技能重构已有的 Goal/Binding/Policy/Plan/ExecutionSession 边界。它同时为灵巧手和未来全身 capability 留出扩展空间，但不会把当前 PR1 扩张为 WBC 项目。
