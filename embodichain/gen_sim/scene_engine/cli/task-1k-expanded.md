# Task 1K 扩增方案（Scene 服务 Action 版）

> 版本：v0.4  
> 日期：2026-08-04  
> 依据：1k+ tasks.md、task-1k-revision-recommendations.md、taskspec-semantic-protocol.md

## 0. 核心目标

### 0.1 先认识几个词

| 词 | 含义 | 例子 |
|---|---|---|
| Scene | 机器人要操作的具体世界 | 一个关闭的抽屉、一个可抓取物体和一个把手 |
| Scene 模式 | 一类具有相同操作条件的 Scene | “关闭抽屉”；“打开抽屉且有可放置区域” |
| Action | 一个动作或一条动作序列 | E9 打开抽屉；E9 → E1 → E10 |
| Action 编排 | 按前置条件和动作结果连接 Easy Action | E9 打开后，E1 才能放入，最后 E10 关闭 |
| Scene Agent | 生成、恢复或编辑 Scene | 从抽屉图片恢复抽屉、把手和内部空间 |
| Action Agent | 在 Scene 中执行和编排 Action | 在打开的抽屉中执行 E1 放入物体 |
| TaskSpec | 规定 Scene 初始条件、goal 和允许 Action 的任务说明 | goal 是 contain(drawer, object) 且 drawer closed |
| Episode | 一次具体执行或扰动后的执行记录 | 同一任务中注入一次抓取滑移后的执行 |
| 任务扩增 | TaskSpec 的目标、条件或因果结构改变，形成有意义的新任务 | “放到桌面”改为“放入抽屉并关闭” |
| 数据扩大 | TaskSpec 不变，只增加 Scene、轨迹或 Episode | 同一任务换桌面、机器人或成功轨迹 |

### 0.2 哪些算任务扩增

| 变化 | 分类 | 例子 |
|---|---|---|
| 改变最终 goal | 任务扩增 | on(object, table) → contain(drawer, object) |
| 增加数量、姿态、顺序或恢复要求 | 任务扩增 | 一个物体 → all；place → open → place → close |
| Scene 出现新状态，但 TaskSpec 和 goal 不变 | 数据扩大 | 同一个放置任务使用打开的不同抽屉场景 |
| 只更换物体、桌面、机器人或动作轨迹 | 数据扩大 | 杯子 A → 杯子 B；UR5 → Franka；轨迹 A → B |

判断标准：

```text
TaskSpec 的目标、必要条件或因果结构改变 → 任务扩增
TaskSpec 不变，只增加 Scene、Action 轨迹或 Episode → 数据扩大
```

Task 1K 的重点不是制造很多不同的物体或桌面，而是：

```
Scene Agent 提供能执行原子动作的场景条件
        ↓
Action Agent 使用 Easy 原子动作
        ↓
组合成更多有意义的 TaskSpec
```

Scene 是 Action 的基础；Action 是改变 Scene 的方式；TaskSpec 负责把二者组合成一个任务。

### 0.3 方案先看

Task 1K 按以下顺序扩增：

```text
P0  固定 E1–E12，以及每个 Easy 需要的 Scene 条件
P1  为每个 Easy 设计多种可执行 Scene 模式
P2  用一个 Easy 生成最简单的 TaskSpec
P3  用前一个 Action 的结果连接下一个 Easy，形成短动作链
P4  改变 goal、数量、姿态、顺序或铰接状态，形成有意义的新任务
P5  检查 Scene、Action 和 goal，统计任务和数据数量
```

核心不是把同一个任务换成很多物体，而是：

```text
一个 Easy
→ 多种可执行 Scene
→ 多种 goal
→ 合法的 Easy Action 组合
→ 更多语义不同的 TaskSpec
```

## 1. 最小流程

Task 生成有两条入口，最后都通过 TaskSpec 连接 Scene 和 Action。

### 1.1 TaskSpec 驱动

适合从 Easy 动作组合出新的任务：

```
组合 Easy Action 和目标
    ↓
写出 TaskSpec：目标、动作和必要条件
    ↓
Scene Agent 根据 TaskSpec 生成或恢复 Scene
    ↓
Action Agent 执行动作链
    ↓
检查最终 goal
```

例如：

```
目标：把物体放入抽屉并关闭抽屉
动作：E9 → E1 → E10
TaskSpec：要求有关闭的抽屉、可抓取物体，最终 contain 且 closed
Scene Agent：生成满足这些条件的 Scene
Action Agent：执行 E9 → E1 → E10
```

### 1.2 Scene 驱动

适合从图像、纯文字或编辑后的 Scene 中发现可以生成的任务：

```
图像 / 纯文字 / Scene 编辑输入
    ↓
Scene Agent 生成或恢复 Scene
    ↓
TaskSpec 根据 Scene 中可用的物体、关系和状态，选择 goal 和 Easy Action
    ↓
Action Agent 执行 TaskSpec 规定的动作
    ↓
检查最终 goal
```

例如，Scene Agent 从图像中发现一个关闭的抽屉、一个可抓取物体和一个把手，TaskSpec 可以选择：

```
goal：打开抽屉                 → E9
goal：打开并放入物体           → E9 → E1
goal：打开、放入并关闭          → E9 → E1 → E10
```

两条入口的区别是：

```
TaskSpec 驱动：先决定要做什么，再生成 Scene
Scene 驱动：先知道有什么，再决定能做什么
```

## 2. 三个核心部分

### 2.1 Scene Agent

Scene Agent 负责根据 Action 的需要生成、恢复或编辑 Scene。

它需要提供：

- 需要被操作的物体；
- 物体的可抓取、可放置或可交互状态；
- 物体之间的关系；
- 桌面和容器的位置；
- 铰接式物体的开关状态；
- 下一步 Action 所需的目标区域或对象。

Scene Agent 当前支持的主要关系：

| 关系 | 含义 |
|---|---|
| left、right、center | 物体相对于桌面的区域 |
| left_front、left_back | 桌面的左前、左后区域 |
| right_front、right_back | 桌面的右前、右后区域 |
| left_center、right_center | 桌面的左中、右中区域 |
| on | 物体在桌面或支撑面上 |
| stack_on | 一个刚体叠放在另一个刚体上 |
| contain | 刚体或铰接式容器包含刚体物体 |

Scene Agent 可以生成以下场景：

```
关闭的抽屉
打开的抽屉
打开的抽屉 + 可放入的刚体物体
打开的烤箱托盘 + 可放入的物体
多个可堆叠物体
有指定左右、前后或桌面区域关系的物体
```

### 2.2 Action Agent

Action Agent 只使用 1k+ tasks.md 中的 Easy 原子动作：

| ID | Easy Action | 作用 |
|---|---|---|
| E1 | PickUp + 旧 Place | 抓取、移动到目标位姿并释放 |
| E2 | MoveHeldObject—upright | 将倒下物体扶正 |
| E3 | MoveHeldObject—horizontal | 将物体水平摆正 |
| E4 | MoveHeldObject—pour | 将容器移动到目标上方并倾倒 |
| E5 | Handover—vertical | 以竖直姿态交接物体 |
| E6 | Handover—horizontal | 以水平姿态交接物体 |
| E7 | CoordinatedPickUp | 双臂共同抓取盘子或托盘 |
| E8 | AssemblePlace | 将手机摆在手机支架上 |
| E9 | PullArticulatedPart | 沿关节轴拉开抽屉或烤箱托盘 |
| E10 | PushArticulatedPart | 沿关节轴推闭抽屉或烤箱托盘 |
| E11 | TurnKnob | 将旋钮转到目标角度或档位 |
| E12 | PressButton | 沿按钮法向按压并触发 |

Action Agent 有两层能力：

```text
原子能力：E1–E12
编排能力：选择、排序、连接和执行多个 Easy Action
```

编排能力不是新的原子技能。它只负责判断：前一个 Action 的结果，是否满足下一个 Action 的前置条件。

### 2.3 TaskSpec：连接 Scene 和 Action

TaskSpec 是二者之间的共同任务说明，不是第三个独立执行 Agent。

TaskSpec 至少说明：

```yaml
TaskSpec:
  scene:        # Scene 的初始条件
  goal:         # 最终要达到的条件
  constraints:  # 执行限制和不变量
  actions:      # 需要或允许的 Easy Action
```

它的连接关系是：

```text
TaskSpec.scene 和 TaskSpec.constraints
→ Scene Agent 生成满足条件的 Scene

TaskSpec.goal 和 TaskSpec.actions
→ Action Agent 选择并编排 Easy Action

Scene + Action 执行结果
→ 检查 TaskSpec.goal 是否成立
```

因此，TaskSpec 是任务扩增的主要位置：改变 TaskSpec 的目标、必要条件或动作因果关系，才可能产生有意义的新任务。

## 3. Scene 如何服务原子 Action

每个 Easy Action 都对应一类 Scene 条件。

| Easy Action | Scene 需要提供的条件 | 动作结果 |
|---|---|---|
| E1 | 可抓取物体、目标位置、支撑面或容器 | 物体被放到目标位置 |
| E2 | 倒下且可抓取的物体、稳定的 upright 姿态 | 物体被扶正 |
| E3 | 可抓取的长物体、合法 horizontal 姿态 | 物体被水平摆正 |
| E4 | 可倾倒容器、接收目标、对齐空间 | 物料被倒入目标 |
| E5 | 适合竖直交接的物体和两个可达区域 | 竖直交接完成 |
| E6 | 适合水平交接的物体和两个可达区域 | 水平交接完成 |
| E7 | 可由双臂共同抓取的盘子或托盘 | 双臂抓取完成 |
| E8 | 物体和匹配的支架或 socket | 物体完成装配放置 |
| E9 | 关闭的抽屉、柜门或烤箱托盘，以及可操作部件 | 铰接部件打开 |
| E10 | 已打开的铰接部件 | 铰接部件关闭 |
| E11 | 可旋转旋钮和目标档位 | 旋钮达到目标档位 |
| E12 | 可按压按钮和触发位置 | 按钮被触发 |

扩增的基本问题是：

```
为了让某个 Easy Action 成立，Scene 还可以提供哪些不同的状态、关系和目标？
```

## 4. 按规则进行底向上扩增

### 4.1 基本组合规则

TaskSpec 先规定 goal 和允许的 Action；Scene Agent 再提供满足这些 Action 前置条件的 Scene。若 Scene 已由图像或文字输入得到，TaskSpec 则从已有 Scene 中选择可执行的 goal 和 Action。

无论从哪条入口开始，每一步都必须满足：

```
pre(Action) ⊆ 当前 Scene 状态
```

执行后更新 Scene：

```
下一个 Scene 状态
= 当前 Scene 状态 + Action 的结果
```

然后继续选择下一个可以执行的 Easy Action。

```
Scene_0
→ Easy Action_1
→ Scene_1
→ Easy Action_2
→ Scene_2
→ ...
→ goal
```

第一阶段限制动作链长度：

```
长度 1：一个 Easy Action
长度 2：两个有明确前后关系的 Easy Action
长度 3：三个以内的线性组合
```

### 4.2 实际扩增步骤

TaskSpec 驱动时：

```
Step 1  组合 Easy Action，确定 goal
Step 2  写出 TaskSpec 的 scene、goal 和 constraints
Step 3  Scene Agent 生成满足 TaskSpec 的 Scene
Step 4  Action Agent 执行动作链
Step 5  检查最终 goal
```

Scene 驱动时：

```
Step 1  Scene Agent 从图像、文字或编辑指令得到 Scene
Step 2  根据 Scene 中的物体、关系和状态列出可用 Easy Action
Step 3  TaskSpec 选择 goal 和 Action 组合
Step 4  Action Agent 执行动作链
Step 5  检查最终 goal
```

### 4.3 组合例子

**例 1：扶正并放置**

```
Scene_0：倒下的罐头 + 桌面目标区域
E2：罐头 → upright
E1：罐头 → 目标区域
goal：upright(can) ∧ on(can, target_region)
```

动作链是 E2 → E1。只更换罐头、桌面或轨迹，不增加新的任务。

**例 2：打开抽屉、放入物体、关闭抽屉**

```
Scene_0：关闭的抽屉 + 抽屉把手 + 可抓取物体
E9：打开抽屉
E1：把物体放入抽屉
E10：关闭抽屉
goal：contain(drawer, object) ∧ closed(drawer)
```

Scene Agent 提供“铰接式容器可以放置物体”的场景能力；Action Agent 使用 E9、E1、E10 完成任务。

**例 3：姿态和区域同时变化**

```
Scene_0：倒下的红色罐头 + 杯子 + 桌面左右区域
E2：红色罐头 → upright
E1：红色罐头 → 杯子左侧
goal：upright(red_can) ∧ left(red_can, cup)
```

把 left 改成 right 会改变目标关系，可以形成新的任务；只改变罐头模型或桌面材质，不形成新任务。

## 5. 哪些组合可以形成新任务

任务扩增必须同时满足三件事：

```text
1. TaskSpec 的 goal、必要条件或因果结构发生了变化
2. Scene 能提供满足新要求的条件或状态
3. Action 原子能力或动作编排能够完成新要求
```

Action 可以保持同一个原子动作；只要 TaskSpec 的目标或必要条件改变，并且 Scene 和 Action 能够支持它，就可以形成新任务。只有 Scene 变化或只有 Action 轨迹变化，不足以称为任务扩增。

只保留以下规则：

```
Scene 提供不同的 Action 前置条件
→ 可以生成不同的 Action 组合

Action 结果支持新的后续 Action
→ 可以增加动作链长度和因果关系

最终 goal、物体数量、必要顺序、姿态要求或必要能力改变
→ 可以形成新的 TaskSpec
```

以下变化不作为新任务来源：

```
只更换物体模型
只更换桌面外观
只更换机器人
只更换动作轨迹
只改写语言
```

这些变化只能称为数据量扩大：

```text
Scene 数据更多，但 TaskSpec 不变
Action 轨迹更多，但 TaskSpec 不变
Episode 更多，但 TaskSpec 不变
```

Scene 的变化只有在改变任务所需的状态、关系或目标，并被 TaskSpec 和 Action 编排使用时，才会参与新的任务组合。

## 6. 底向上扩增公式

设：

```
E                  = 12 个 Easy 原子 Action
S_e                = 能支持原子 Action e 的 Scene 模式数
G(e,s)             = Scene 模式 s 下可以定义的语义 goal 数
L(e,s,g)           = 能完成 goal 的合法 Action 链数量
Y                  = Scene、Action 和 goal 检查通过率
```

候选任务数估算为：

```
C_task_bottomup
  ≈ Σe∈E Σs∈S_e G(e,s) × L(e,s,g)
```

通过检查的任务数估算为：

```
N_task_bottomup
  ≈ C_task_bottomup × Y
```

这里的 L 只统计由 E1–E12 组成的合法动作链。不同轨迹不增加任务数量。

### 6.1 简单估算例子

如果平均每个 Easy Action 有：

```
S = 4 个 Scene 模式
G = 3 个语义 goal
L = 2 条合法 Action 链
Y = 0.8 的通过率
```

则：

```
C_task_bottomup
  ≈ 12 × 4 × 3 × 2
  = 288

N_task_bottomup
  ≈ 288 × 0.8
  ≈ 230
```

要达到 1,000 个任务，需要增加不同的 Scene 状态、语义 goal 和合法 Action 组合，而不是只增加物体或轨迹。

### 6.2 铰接式容器估算例子

假设 Scene Agent 可以提供 4 种铰接式 Scene 模式：

```
1. 抽屉关闭
2. 抽屉打开
3. 抽屉打开且有可放置区域
4. 抽屉打开且已有物体
```

每种模式平均支持 3 个不同 goal、2 条合法 Action 链：

```
C_hinged
  ≈ 4 × 3 × 2
  = 24 个候选任务

N_hinged
  ≈ 24 × Y
```

可形成的任务包括：

```
打开抽屉
打开并放入物体
打开、放入并关闭
打开、取出并恢复关闭
```

差异来自 goal 和必要动作顺序，而不是抽屉材质或物体模型。

## 7. 执行顺序

```
P0  固定 E1–E12 和 Scene 条件表
P1  每个 Easy 先设计 3–4 种 Scene 模式
P2  每个 Scene 模式先生成单动作任务
P3  用 Action 结果连接长度为 2 和 3 的动作链
P4  改变 goal、数量、顺序、姿态和铰接状态
P5  检查 Scene、Action 和 goal
P6  统计任务、Scene、Action 和 Episode
```

第一轮可以用下面的预算估算：

```text
12 个 Easy
× 每个 4 个 Scene 模式
× 每个模式 3 个语义 goal
× 平均 2 条合法 Action 链
× 0.8 通过率
≈ 230 个候选任务
```

之后继续增加 Scene 模式、goal 变化和合法 Action 组合，直到达到 Task 1K 的目标。

核心原则：

```
Scene Agent 负责提供 Action 能力成立所需的世界；
Action Agent 负责在这个世界中组合 Easy 原子动作；
TaskSpec 记录组合后的目标和必要条件；
只有目标或因果结构发生变化，才算新的任务。
```
