# Failure-Monitor-Recovery 框架设计

本文面向 EmbodiChain 当前代码结构，系统说明项目中的一套运行时故障注入、故障监测与恢复执行框架。它不是一个抽象的“异常处理”概念，而是一条从任务规划、失败预判、代码生成，到执行时监测与恢复的完整链路。

需要特别强调的是，这套框架的运行时能力并不依赖 agent 才能使用。`TaskAgent`、`FailureAnticipationAgent`、`CodeAgent` 提供的是一条“自动生成 failure-monitor-recovery 代码”的上层工作流；但底层执行接口本身是开放的。即使完全不用 agent，开发者也可以手写 `drive(...)`、`error_functions=[...]`、`monitor_sequences=[...]`、`recovery_sequences=[...]`，直接构造同样的故障注入、监测和恢复逻辑。

这套框架的核心目标有三个：

1. 在正常任务步骤之上，显式注入“可恢复、可观测、可用于收集数据”的失败案例。
2. 把失败处理从“重新整体规划”降级为“在当前 step 内局部恢复”，降低代价并提高可控性。
3. 让 failure、monitor、recovery 三者都可以由 LLM 生成，但仍然被运行时语义和原子动作系统严格约束。

---

## 1. 框架总览

从代码结构上看，这套机制分成两个层面：

- 生成层：决定“哪些 step 值得注入失败、如何监测、如何恢复”
- 运行层：真正执行 `drive(...)`，注入错误，检测 monitor，触发 recovery

对应的主要代码入口如下：

- `embodichain/lab/gym/envs/tasks/tableware/base_agent_env.py`
- `embodichain/agents/hierarchy/failure_anticipation_agent.py`
- `embodichain/agents/mllm/prompt/failure_anticipation_prompt.py`
- `embodichain/agents/hierarchy/code_agent.py`
- `embodichain/agents/mllm/prompt/code_prompt.py`
- `embodichain/lab/sim/agent/atom_actions.py`
- `embodichain/lab/sim/agent/monitor_functions.py`
- `embodichain/agents/hierarchy/error_agent.py`

如果从使用方式上划分，这套框架其实支持两种工作模式：

- Agent 驱动模式：由 `TaskAgent + FailureAnticipationAgent + CodeAgent` 自动生成带 recovery 的执行代码
- 手写脚本模式：开发者直接手写 `drive(...)` 调用，把 monitor 和 recovery 显式写进脚本

后者同样是完整、一等公民的使用方式，而不是仅供调试的临时旁路。

可以把整条链路概括为：

```text
任务目标
  -> TaskAgent 生成 task plan
  -> FailureAnticipationAgent 为每个 step 设计潜在失败与恢复策略
  -> CodeAgent 把 plan + anticipated failures 编译成 drive(...) 代码
  -> drive(...) 执行动作
       -> 注入 error
       -> 每一步运行 monitor
       -> monitor 触发后执行对应 recovery sequence
```

---

## 2. 生成层：如何把“失败恢复”编译进动作代码

### 2.1 调用入口

在 `BaseAgentEnv.generate_code_for_actions()` 中，系统会按顺序执行：

1. `TaskAgent` 生成任务步骤
2. 如果 `recovery=True`，则调用 `FailureAnticipationAgent`
3. `CodeAgent` 根据 task plan 和 anticipated failures 生成最终 Python 代码

简化后的流程如下：

```python
task_plan = self.task_agent.generate(...)

anticipated_failures = ""
if recovery:
    anticipated_failures = self.failure_anticipation_agent.generate(...)

code_file_path, kwargs, code = self.code_agent.generate(
    task_plan=task_plan,
    anticipated_failures=anticipated_failures,
    ...
)
```

这个设计很关键。它说明：

- recovery 不是执行时“临时瞎猜”的
- recovery 也不是 validation 失败后才统一重做的
- 它是在代码生成之前，就被明确写进每个 step 的执行策略里

### 2.2 FailureAnticipationAgent 的职责

`FailureAnticipationAgent` 的作用不是穷举所有理论失败，而是筛选“高价值、可恢复、可监测”的失败场景，并输出结构化文本。

对应实现位于：

- `embodichain/agents/hierarchy/failure_anticipation_agent.py`
- `embodichain/agents/mllm/prompt/failure_anticipation_prompt.py`

它的输出格式是严格 step 对齐的：

```text
[ANTICIPATED_FAILURES]:
Step 1:
Errors:
E1.1 [error_type=misplaced_object] ...
Policies:
M1.1: monitor_object_moved(...)
R1.1: grasp(...) -> move_...

Step 2:
None
```

这意味着 failure anticipation 的产物本质上不是“自由文本解释”，而是一种半结构化中间表示：

- `E<i>.<j>`：该 step 允许注入什么错误
- `M<i>.<j>`：用什么 monitor 判断该失败是否发生
- `R<i>.<j>`：触发后该执行哪一串恢复动作

### 2.3 Prompt 如何约束 failure anticipation

`FailureAnticipationPrompt` 明确要求模型只输出：

- 物理上合理的失败
- 当前系统已有 `error_functions` 支持的失败
- 当前系统已有 `monitor_functions` 可检测的失败
- 能用已有 `atom_actions` 在短序列中恢复的失败

也就是说，这个 Agent 不是自由创作，而是被如下三类资源约束：

1. `failure_recovery/error_functions.txt`
2. `failure_recovery/monitor_functions.txt`
3. `atom_actions.txt`

这让 failure anticipation 具备了“生成式设计 + 运行时可落地”的双重属性。

---

## 3. CodeAgent：如何把 failure spec 编译成 `drive(...)`

### 3.1 生成 monitor 代码的 prompt

当 agent 配置使用：

```json
"FailureAnticipationAgent": {
    "prompt_name": "anticipate_potential_failure"
},
"CodeAgent": {
    "prompt_name": "generate_monitor_code"
}
```

说明代码生成不再只是普通动作代码，而是会把 failure-monitor-recovery 机制一起编译进去。

这个配置可以在下面文件中看到：

- `configs/gym/agent/pour_water_agent/agent_config_dual.json`

### 3.2 `generate_monitor_code` 的关键约束

`embodichain/agents/mllm/prompt/code_prompt.py` 中的 `generate_monitor_code()` 给了非常明确的结构约束：

- 必须先保留原始 task plan 的 step 顺序
- 只能对同一个 step 添加 `error_functions`、`monitor_sequences`、`recovery_sequences`
- `monitor_sequences` 与 `recovery_sequences` 外层索引一一对应
- recovery 必须写成 `partial(drive, ...)`
- recovery 内部的原子动作必须延迟求值，例如 `partial(grasp, ...)`

这实际上把 LLM 的自由度压缩到了“如何做 step-level augmentation”，而不是让它重新设计运行时引擎。

### 3.3 CodeAgent 的执行方式

`embodichain/agents/hierarchy/code_agent.py` 中，`CodeAgent.act()` 支持两种生成代码形态：

1. 生成 `create_agent_action_list()` 函数，然后调用它
2. 直接生成模块级别的 `drive(...)` 调用，并直接执行

为了让 recovery 可延迟执行，`CodeAgent` 还做了两件重要的事：

- 通过 AST 把 `partial(monitor_fn(...))` 归一化为 `partial(monitor_fn, ...)`
- 给模块级别的函数调用自动注入 `**kwargs`

因此，生成代码看起来像普通 Python，但在执行语义上已经适配了这套框架。

这里也要再次强调：`CodeAgent` 的职责是“把这套机制自动编译出来”，而不是“提供唯一入口”。从框架边界上说，agent 只是 code author，`drive(...)` 才是真正的 runtime。只要手写代码遵循相同的数据结构和调用约定，完全可以脱离 agent 单独使用这套 failure-monitor-recovery 机制。

---

## 4. 运行层：`drive(...)` 是整个框架的执行内核

核心运行时入口位于：

- `embodichain/lab/sim/agent/atom_actions.py`

其中 `drive(...)` 是整个框架最重要的执行器。它同时负责：

- 解析左右臂动作
- 对齐双臂轨迹长度
- 注入 action-level / object-level error
- 按 step 执行仿真
- 调用 monitor
- 触发 recovery

从设计上看，`drive(...)` 相当于一个小型的 step runtime。

### 4.1 `drive(...)` 的输入不是只有动作

`drive(...)` 的典型调用形式如下：

```python
drive(
    left_arm_action=...,
    right_arm_action=...,
    error_functions=[...],
    monitor_sequences=[...],
    recovery_sequences=[...],
    env=env,
)
```

这里每个字段的语义不同：

- `left_arm_action/right_arm_action`：当前 step 的执行轨迹
- `error_functions`：该 step 可能被注入的干扰
- `monitor_sequences`：该 step 的失败检测器
- `recovery_sequences`：该 step 的恢复策略

这四者共同定义了一个“可恢复 step”，而不是一个单纯的轨迹执行命令。

因此，哪怕没有任何 agent 参与，开发者也可以直接写出如下风格的脚本：

```python
from functools import partial
from embodichain.lab.sim.agent.atom_actions import *

drive(
    left_arm_action=grasp(
        robot_name="left_arm",
        obj_name="cup",
        pre_grasp_dis=0.10,
    ),
    right_arm_action=None,
    error_functions=[
        partial(
            inject_object_error,
            error_type="misplaced_object",
            error_obj="cup",
        ),
    ],
    monitor_sequences=[
        [
            partial(
                monitor_object_moved,
                obj_name="cup",
                threshold=0.02,
            )
        ],
    ],
    recovery_sequences=[
        [
            partial(
                drive,
                left_arm_action=partial(
                    grasp,
                    robot_name="left_arm",
                    obj_name="cup",
                    pre_grasp_dis=0.10,
                ),
                right_arm_action=None,
            ),
        ],
    ],
    env=env,
)
```

也就是说，agent 负责的是“帮你生成这段代码”，而不是“替代这段代码对应的底层能力”。

### 4.2 动作可以是立即值，也可以是延迟调用

在 `drive(...)` 开头，系统会调用：

```python
left_arm_action = resolve_action(left_arm_action, env, kwargs)
right_arm_action = resolve_action(right_arm_action, env, kwargs)
```

其意义是：

- 如果传进来的是 ndarray，就直接执行
- 如果传进来的是 `partial(grasp, ...)` 这类可调用对象，就在运行时求值

这正是 recovery 能被定义成“延迟执行动作”的基础。

---

## 5. Error 注入语义

error 的实现位于：

- `embodichain/agents/hierarchy/error_agent.py`

当前支持的错误类型分为两类：

### 5.1 object-level error

例如：

- `misplaced_object`
- `fallen_object`

它们属于 `object_error_types`，在 `drive(...)` 的每个执行 step 内检查是否触发。

这类错误的特点是：

- 改变场景中的物体状态
- 更接近外界扰动或目标物位置漂移
- 适合配合 `monitor_object_moved(...)`

### 5.2 action-level error

例如：

- `wrong_affordance`

它属于 `action_error_types`，在整段轨迹执行前注入一次。

`wrong_affordance` 的实现方式不是直接改场景，而是重新生成一条“被扰动后的动作轨迹”。简化逻辑如下：

```python
left_arm_action, right_arm_action = error_function(
    left_arm_action=left_arm_action,
    right_arm_action=right_arm_action,
    env=env,
    **kwargs,
)
```

这说明 action-level error 更像“执行意图偏了”，而 object-level error 更像“环境状态变了”。

---

## 6. Monitor 语义：统一使用 `True == failure occurred`

monitor 定义位于：

- `embodichain/lab/sim/agent/monitor_functions.py`

当前最常用的两个 monitor 是：

### 6.1 `monitor_object_moved`

适用于：

- 抓取前物体被碰歪
- 静止目标发生位移

典型代码：

```python
partial(
    monitor_object_moved,
    obj_name="cup",
    threshold=0.02,
)
```

它比较的是：

- 当前物体 pose
- 上一帧或缓存中的 pose

只要位移超过阈值，就返回 `True`。

### 6.2 `monitor_object_held`

适用于：

- 已抓住物体后，在移动、旋转、放置途中发生脱手

典型代码：

```python
partial(
    monitor_object_held,
    robot_name="right_arm",
    obj_name="bottle",
    threshold=0.05,
)
```

虽然名字叫 `held`，但它的统一语义依然是：

- `True`：失败发生
- `False`：未检测到失败

对这个函数而言，`True` 的实际含义是“物体已经不再被稳定持有”。

### 6.3 为什么统一语义很重要

这套框架强依赖“所有 monitor 都能被统一调度”。因此它不能接受某个 monitor 用 `True` 表示成功、另一个 monitor 用 `True` 表示失败。

统一语义后，`drive(...)` 可以写成非常简单的逻辑：

```python
for function in monitor_sequence:
    result = function()
    if result == True:
        # 触发对应 recovery
```

这也是你这套框架很干净的一点：运行时不关心 monitor 的内部实现，只关心它是否触发。

---

## 7. `monitor_sequences` / `recovery_sequences` 的配对机制

这是整套框架最关键的运行时约定之一。

外层列表按索引一一对应：

```python
monitor_sequences = [
    [monitor_a],
    [monitor_b1, monitor_b2],
]

recovery_sequences = [
    [recovery_for_a_step1, recovery_for_a_step2],
    [recovery_for_b],
]
```

含义是：

- `monitor_sequences[0]` 触发时，执行 `recovery_sequences[0]`
- `monitor_sequences[1]` 触发时，执行 `recovery_sequences[1]`

而每个内层 `monitor_sequence` 可以包含多个 monitor callable。它们应表示：

- 同一种失败的多个替代检测器
- 共享同一条恢复链

当前 `drive(...)` 的运行逻辑是：

1. 依次遍历外层 monitor sequence
2. 只要某个 sequence 中任一 monitor 返回 `True`
3. 就触发同索引下的 recovery sequence
4. 然后 `drive(...)` 直接返回，不再继续当前原始轨迹

因此这是一个“单次触发、单条恢复链接管当前 step”的模型，而不是多故障并行处理器。

---

## 8. Recovery 为什么写成 `partial(drive, ...)`

你这套设计里最漂亮的一点，是 recovery 不是一个单独 API，而是把恢复动作重新表达成“新的 step 序列”。

例如：

```python
recovery_sequences=[
    [
        partial(
            drive,
            left_arm_action=partial(
                grasp,
                robot_name="left_arm",
                obj_name="cup",
                pre_grasp_dis=0.10,
            ),
            right_arm_action=None,
        ),
        partial(
            drive,
            left_arm_action=partial(
                move_by_relative_offset,
                robot_name="left_arm",
                dx=0.0,
                dy=0.0,
                dz=0.10,
                mode="extrinsic",
            ),
            right_arm_action=None,
        ),
    ]
]
```

这样设计的优点非常明显：

1. recovery 不需要新的执行器，直接复用已有 `drive(...)`
2. recovery 本身仍然是由原子动作构成，语义统一
3. 单臂恢复和双臂恢复可以复用同一套接口
4. recovery step 内部也能继续使用 `drive(...)` 的同步与动作求值机制

换句话说，你的 recovery 不是“逃逸分支”，而是“同构的二级执行图”。

---

## 9. Recovery 的几个关键实现细节

### 9.1 lazy materialization

recovery 里的动作不是预先执行，而是只有在 monitor 真触发后才求值。

例如：

```python
partial(
    drive,
    right_arm_action=partial(grasp, robot_name="right_arm", obj_name="bottle"),
)
```

这里 `grasp(...)` 不会在生成代码时立即跑掉，而是 recovery 被触发时才执行。

这带来两个好处：

- recovery 使用的是“触发当下”的最新环境状态
- 不会为了一个根本没触发的失败去提前生成无用轨迹

### 9.2 进入 recovery 前会先刷新环境中的当前状态

在 `atom_actions.py` 中，进入每个 recovery step 前会读取机器人当前 `qpos`，并刷新：

- `env.left_arm_current_qpos`
- `env.right_arm_current_qpos`
- `env.left_arm_current_xpos`
- `env.right_arm_current_xpos`
- `env.left_arm_current_gripper_state`
- `env.right_arm_current_gripper_state`

这保证 recovery 的起点来自真实执行后的机器人状态，而不是原始计划中的理想状态。

### 9.3 recovery 可以接管另一只手剩余轨迹

这是运行时层面非常实用的一点。

如果一个双臂 step 里只有左手失败，右手原本剩余轨迹可能仍然应该继续。`drive(...)` 会在 monitor 触发后截出未执行完的剩余动作：

```python
remaining_left_arm_action = left_arm_action[i + 1 :]
remaining_right_arm_action = right_arm_action[i + 1 :]
```

然后在 recovery step 的 `left_arm_action=None` 或 `right_arm_action=None` 情况下，把剩余轨迹注入给对应手臂。

这个机制让你的 recovery 不是“两个手都全部重来”，而是支持局部接管。

---

## 10. 一个完整实例：DualPourWater

最好的示例在：

- `embodichain/database/agent_generated_content/DualPourWater_example/agent_anticipated_failures.txt`
- `embodichain/database/agent_generated_content/DualPourWater_example/agent_generated_code.py`

### 10.1 failure anticipation 文本

以 Step 3 为例：

```text
Step 3:
Errors:
E3.1 [error_type=wrong_affordance] left_arm transport toward [0.55, 0.05] is perturbed and the cup is lost during motion
Policies:
M3.1: monitor_object_held(robot_name="left_arm", obj_name="cup", threshold=0.05)
R3.1: grasp(robot_name='left_arm', obj_name='cup', pre_grasp_dis=0.10) -> move_by_relative_offset(...) -> move_to_absolute_position(...)
```

这段文字已经清楚表达了一个完整策略：

- 错误源：运输轨迹偏移
- 检测器：杯子是否仍被左手持有
- 恢复策略：重新抓取，再抬起，再回到目标位置

### 10.2 编译后的运行时代码

对应生成代码如下：

```python
drive(
    left_arm_action=move_to_absolute_position(
        robot_name="left_arm",
        x=0.55,
        y=0.05,
        z=None,
    ),
    right_arm_action=None,
    error_functions=[
        partial(
            inject_object_error,
            error_type="misplaced_object",
            error_obj="cup",
        ),
    ],
    monitor_sequences=[
        [
            partial(
                monitor_object_held,
                robot_name="left_arm",
                obj_name="cup",
                threshold=0.05,
            )
        ],
    ],
    recovery_sequences=[
        [
            partial(
                drive,
                left_arm_action=partial(
                    grasp,
                    robot_name="left_arm",
                    obj_name="cup",
                    pre_grasp_dis=0.10,
                ),
                right_arm_action=None,
            ),
            partial(
                drive,
                left_arm_action=partial(
                    move_by_relative_offset,
                    robot_name="left_arm",
                    dx=0.0,
                    dy=0.0,
                    dz=0.10,
                    mode="extrinsic",
                ),
                right_arm_action=None,
            ),
            partial(
                drive,
                left_arm_action=partial(
                    move_to_absolute_position,
                    robot_name="left_arm",
                    x=0.55,
                    y=0.05,
                    z=None,
                ),
                right_arm_action=None,
            ),
        ],
    ],
)
```

这里可以看到一个非常清晰的编译映射：

- `E3.1` 被翻译成 `error_functions`
- `M3.1` 被翻译成 `monitor_sequences[0]`
- `R3.1` 被翻译成 `recovery_sequences[0]`

也就是说，failure anticipation 的文本结构和 runtime code 的数据结构是同构的。

---

## 11. 为什么这套设计适合数据收集

从“收集 error-recovery demonstration”这个目标来看，这套框架有几个明显优势。

### 11.1 故障是显式设计出来的

失败不是随机噪声，而是：

- 由 error function 明确注入
- 由 monitor 明确检测
- 由 recovery 明确修复

因此每条 recovery 数据都带有清晰标签：

- 失败类型是什么
- 何时被检测到
- 如何恢复

### 11.2 同一个任务 step 可以生成多种失败样本

一个 step 可以绑定多个：

- `error_functions`
- `monitor_sequences`
- `recovery_sequences`

这使得同一个 task plan 可以扩展出多样的 recovery demonstration，而不需要重新设计整条任务。

### 11.3 recovery 保持原子动作空间一致

恢复不是另写一套“专家脚本”，而是复用现有 `grasp`、`move_to_absolute_position`、`place_on_table` 等原子动作。

这样带来的好处是：

- 数据空间一致
- 模型更容易学习“正常动作”和“恢复动作”之间的关系
- 未来可以把 recovery 也看成一种普通技能序列

---

## 12. 已有测试验证了哪些运行时语义

`tests/sim/agent/test_atom_actions.py` 已经覆盖了这套框架里几个很关键的性质：

- monitor 触发后，原始失败轨迹会停止，切入 recovery
- recovery 是 lazy materialization 的，不会提前求值
- `monitor_sequences` 和 `recovery_sequences` 通过外层索引对齐
- 单臂 recovery 时，另一只手可以继承原始剩余轨迹
- 剩余轨迹只会被消费一次，不会重复注入
- error function 的触发概率支持默认值与局部覆盖

这很重要，因为它说明你的框架不是只有 prompt 约定，而是有明确的 runtime contract 和单测支撑。

---

## 13. 这套框架最核心的设计思想

如果要用一句话总结，这套 failure-monitor-recovery 框架的核心思想是：

> 把“故障恢复”从一个模糊的后处理问题，重构为 task-step 对齐、可编译、可执行、可监测、可学习的数据生成机制。

它有几个非常突出的特点：

- step 对齐：每个 failure 都对应到具体 task step
- 语义分层：error 是扰动源，monitor 是判定器，recovery 是动作策略
- 同构执行：recovery 继续使用 `drive(...)`
- 双臂兼容：单臂和双臂都能纳入同一机制
- 可测试：关键行为已经能被单元测试验证

---

## 14. 当前实现的边界与后续可演进方向

虽然这套框架已经非常完整，但从当前代码看，还存在几个明确边界。

### 14.1 当前是“单次触发、单条恢复链接管”

一旦某个 monitor sequence 触发，当前 `drive(...)` 会执行对应 recovery，然后返回，不会继续在同一层 `drive(...)` 中处理后续 monitor。

这意味着它更像：

- 一个 step 内的单分支恢复器

而不是：

- 一个支持多重故障并发调度的行为树

### 14.2 monitor 仍然是 step-level，而不是 phase-level

当前 monitor 检查发生在轨迹逐步执行中，但 recovery 的组织粒度仍然是 step。它很适合“抓取失败”“运输脱手”“目标物位移”这类明确事件，但对更细粒度的时序控制还有提升空间。

### 14.3 failure anticipation 依赖 prompt discipline

虽然运行时结构是刚性的，但 anticipated failures 仍然来自 LLM 输出。因此你通过：

- prompt 规则
- `error_functions.txt`
- `monitor_functions.txt`
- `recovery_code_prompt.txt`

来约束它，这是目前很合理的做法。未来如果要进一步提高稳定性，可以考虑把 anticipated failures 变成更强结构化的 DSL 或 JSON schema。

---

## 15. 小结

EmbodiChain 当前的 failure-monitor-recovery 框架并不是简单地给 `drive(...)` 多加了两个参数，而是建立了一条完整的“失败数据生成与执行闭环”：

1. 先由任务规划定义正常步骤
2. 再由 failure anticipation 为 step 设计高价值失败
3. 再由代码生成把失败策略编译成可执行 runtime code
4. 最后由 `drive(...)` 在执行中完成注入、监测与恢复

从架构角度看，这套系统最大的价值在于：它把 recoverable failure 从“偶发异常”提升成了“可编排的训练信号”。

如果后续要继续扩展，我认为最值得演进的方向有三个：

1. 把 anticipated failures 从半结构化文本升级成更强的结构化表示
2. 让 recovery 支持更复杂的多分支或层级式策略
3. 把 monitor 从 step-level 进一步细化到更稳定的 phase-level / condition-level

但就当前代码而言，这套框架已经具备了非常清晰的设计边界、很强的可解释性，以及不错的工程闭环能力。
