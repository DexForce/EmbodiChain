# WholeBodyIKSolver 全身IK求解器

`WholeBodyIKSolver` 是一个基于 [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) +
[CasADi/IPOPT](https://web.casadi.org/) 的全身逆运动学求解器，专为具有**多个末端执行器**
（双臂 + 头部 + 腰腿）的人形机器人设计。

与其他单链求解器不同，它允许在初始化时配置任意数量的末端帧，每次调用时只需指定
**一个目标末端**，同时通过腿部稳定性代价约束整体姿态。

---

## 核心概念

### 三个层次

```
整体机器人 URDF
    │
    ├─ 关节锁定 → Reduced Model（只保留活动关节）
    │
    ├─ 末端帧注册（每个 EE 挂载到对应父关节）
    │    ├── "left"  → 挂在 LEFT_J7 上，带偏移变换
    │    ├── "right" → 挂在 RIGHT_J7 上
    │    └── "head"  → 挂在 NECK2 上
    │
    └─ CasADi 优化问题（每个 EE 独立一套，预构建缓存）
         ├── 末端跟踪代价（位置 + 姿态）
         ├── 关节正则化代价（防止极端关节角）
         ├── 平滑代价（抑制帧间跳变）
         └── 腿部稳定性代价（维持自然站姿）
```

### 优化问题形式

每次调用 `get_ik()` 时，求解以下带约束的非线性最小二乘问题：

```
min  w_pos  * ‖p(q) - p*‖²        ← 末端位置误差
   + w_rot  * ‖log(R(q) Rᵀ*)‖²    ← 末端姿态误差（SO(3) 对数映射）
   + w_reg  * ‖W·q‖²               ← 关节正则化
   + w_smo  * ‖q - q_last‖²        ← 平滑（靠近上一帧解）
   + Σ leg_costs                   ← 腿部/躯干稳定性

s.t.  q_lower ≤ q ≤ q_upper        ← 关节限位约束
```

其中 `p*`、`R*` 来自输入的目标位姿矩阵，`q_last` 为上一帧关节角。

---

## 配置说明

### EndEffectorCfg — 末端帧配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `frame_name` | `str` | 帧名称（在 Pinocchio 模型中注册，调用时也用此名索引） |
| `parent_joint` | `str` | 父关节名，必须存在于 reduced model 中 |
| `rotation` | `List[List[float]]` | 父关节系到末端帧的旋转矩阵（3×3，行主序嵌套列表） |
| `translation` | `List[float]` | 父关节系到末端帧的平移 `[x, y, z]`（米） |
| `w_pos` | `float` | 位置代价权重，默认 `50.0` |
| `w_rot` | `float` | 姿态代价权重，默认 `1.0` |

### LegCostCfg — 腿部稳定性代价

定义一条形如 `weight * (c₀q[j₀] + c₁q[j₁] + ...)²` 的代价项。

| 参数 | 类型 | 说明 |
|------|------|------|
| `joint_names` | `List[str]` | 参与的关节名列表 |
| `coefficients` | `List[float]` | 各关节对应的系数 |
| `weight` | `float` | 代价权重，默认 `10.0` |

**示例**（令腰部与膝部关节保持等幅反向，维持重心）：

```python
LegCostCfg(joint_names=["WAIST", "KNEE"], coefficients=[1.0, -1.0], weight=10.0)
```

### WholeBodyIKSolverCfg — 求解器总配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `urdf_path` | `str` | 机器人 URDF 文件路径（继承自 `SolverCfg`） |
| `urdf_dir` | `str` | 网格文件目录，`None` 时自动取 `urdf_path` 所在目录 |
| `joint_names` | `List[str]` | **保留为活动关节**的关节名列表（其余关节被锁定为 0） |
| `end_effectors` | `Dict[str, EndEffectorCfg]` | 末端执行器字典，键为自定义名称 |
| `active_ee` | `str` | 默认激活的末端名称 |
| `leg_mode` | `int` | 腿部模式：`2`=标准站立，`3`=行走（切换稳定性代价集合） |
| `max_iter` | `int` | IPOPT 最大迭代次数，默认 `10` |
| `tol` | `float` | IPOPT 收敛容忍度，默认 `1e-4` |
| `w_regularization` | `float` | 正则化代价系数，默认 `0.02` |
| `w_smooth` | `float` | 平滑代价系数，默认 `0.1` |
| `joint_reg_extra` | `Dict[str, float]` | 对特定关节追加正则化权重，如 `{"WAIST": 9.0}` 表示该关节总权重为 `1 + 9 = 10` |
| `leg_costs_mode2` | `List[LegCostCfg]` | `leg_mode=2` 时的稳定性代价列表 |
| `leg_costs_mode3` | `List[LegCostCfg]` | `leg_mode=3` 时的稳定性代价列表 |

---

## 完整配置示例

以下示例对应 `dex_w1` 机器人（正交型腕关节，20 个活动关节）：

```python
from embodichain.lab.sim.solvers import (
    WholeBodyIKSolverCfg,
    EndEffectorCfg,
    LegCostCfg,
)
import numpy as np

cfg = WholeBodyIKSolverCfg(
    urdf_path="/path/to/dex_w1.urdf",
    urdf_dir="/path/to/meshes",

    # 保留活动关节（其余全部锁定为 0）
    joint_names=[
        "LEFT_J1",  "LEFT_J2",  "LEFT_J3",  "LEFT_J4",
        "LEFT_J5",  "LEFT_J6",  "LEFT_J7",
        "RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4",
        "RIGHT_J5", "RIGHT_J6", "RIGHT_J7",
        "WAIST", "BUTTOCK", "KNEE", "ANKLE",
        "NECK1", "NECK2",
    ],

    # 末端执行器定义
    end_effectors={
        "left": EndEffectorCfg(
            parent_joint="LEFT_J7",
            rotation=[[0, 1, 0], [0.707, 0, 0.707], [0.707, 0, -0.707]],
            translation=[-0.15, 0, 0],
            w_pos=50.0,
            w_rot=1.0,
        ),
        "right": EndEffectorCfg(
            parent_joint="RIGHT_J7",
            rotation=[[0, 1, 0], [0.707, 0, 0.707], [0.707, 0, -0.707]],
            translation=[-0.15, 0, 0],
        ),
        "head": EndEffectorCfg(
            parent_joint="NECK2",
            rotation=[[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
            translation=[0, -0.1, 0],
            w_pos=50.0,
            w_rot=1.0,
        ),
    },

    active_ee="left",   # 默认求解左手
    leg_mode=2,

    # 对腰部等关节加大正则化（保持直立姿态）
    joint_reg_extra={"WAIST": 9.0, "BUTTOCK": 9.0, "ANKLE": 9.0},

    # 腿部稳定性代价（mode 2）
    leg_costs_mode2=[
        # 约束 WAIST 与 KNEE 等幅反向
        LegCostCfg(
            joint_names=["WAIST", "KNEE"],
            coefficients=[1.0, -1.0],
            weight=10.0,
        ),
        # 约束三关节线性组合为 0（保持重心竖直）
        LegCostCfg(
            joint_names=["WAIST", "BUTTOCK", "KNEE"],
            coefficients=[1.0, 1.0, 1.0],
            weight=10.0,
        ),
    ],
)

solver = cfg.init_solver()
```

---

## 主要方法

### `get_ik()` — 逆运动学求解

```python
success, q = solver.get_ik(
    target_xpos,          # np.ndarray 或 torch.Tensor，形状 (4, 4)
    qpos_seed=current_q,  # 当前关节角，形状 (nq,)，可选
    active_ee="left",     # 本次求解哪个末端，可选（默认用 cfg.active_ee）
)
```

**返回值：**

- `success`：`torch.BoolTensor`，形状 `(1,)`，`True` 表示 IPOPT 收敛
- `q`：`torch.FloatTensor`，形状 `(nq,)`，求解到的关节角

> 即使 `success=False`（未收敛），`q` 仍返回 IPOPT 当前最优点，可作为下一帧初始值继续迭代。

**切换末端示例：**

```python
# 同一个 solver，不同末端，无需重新初始化
success_l, q = solver.get_ik(left_target,  qpos_seed=q, active_ee="left")
success_r, q = solver.get_ik(right_target, qpos_seed=q, active_ee="right")
success_h, q = solver.get_ik(head_target,  qpos_seed=q, active_ee="head")
```

### `set_active_ee()` — 全局切换默认末端

```python
solver.set_active_ee("right")
# 之后调用 get_ik() 不传 active_ee，默认求解右手
```

### `get_fk()` — 正运动学

```python
pose_4x4 = solver.get_fk(q, ee_name="left")   # torch.Tensor (4, 4)
```

### `get_all_ee_names()` — 查询已配置末端列表

```python
print(solver.get_all_ee_names())  # ['left', 'right', 'head']
```

---

## 与现有框架的关系

```
BaseSolver (抽象基类)
    ├── PinocchioSolver    — 单链，Jacobian 迭代 IK
    ├── PinkSolver         — 单链，多任务优化 IK
    ├── DifferentialSolver — 单链，微分 IK 控制器
    └── WholeBodyIKSolver  — 多末端，全身 IPOPT 优化 IK  ← 本文档
```

`WholeBodyIKSolver` 继承 `BaseSolver`，`get_ik()` 返回值格式
`(torch.BoolTensor, torch.FloatTensor)` 与其他求解器完全一致，
可直接替换使用，无需修改上层调用代码。

---

## 性能与调参建议

| 目标 | 建议 |
|------|------|
| 提升跟踪精度 | 增大 `w_pos`（位置权重），如从 `50` 调到 `100` |
| 减少跳变 | 增大 `w_smooth`，如 `0.1 → 0.5` |
| 保持站立姿态 | 增大 `joint_reg_extra` 中腰腿关节的权重 |
| 加快求解速度 | 减小 `max_iter`（默认 `10` 已较激进，一般不超过 `20`） |
| 调试姿态漂移 | 检查 `leg_costs_mode2/3` 的系数，确保约束符合运动学意图 |

---

## 参考资料

- [Pinocchio 文档](https://stack-of-tasks.github.io/pinocchio/)
- [CasADi 文档](https://web.casadi.org/)
- [IPOPT 参数参考](https://coin-or.github.io/Ipopt/OPTIONS.html)
