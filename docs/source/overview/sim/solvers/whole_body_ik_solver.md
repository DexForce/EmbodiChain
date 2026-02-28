# WholeBodyIKSolver

`WholeBodyIKSolver` is a whole‑body inverse kinematics (IK) solver built on
[Pinocchio](https://stack-of-tasks.github.io/pinocchio/). It is designed for
redundant humanoid robots where a **single task‑space end‑effector** (e.g. left
hand) is controlled by a **larger set of joints** (torso + arm).

Unlike single‑chain solvers that assume a nearly square Jacobian, this solver:

- builds a *reduced* Pinocchio model that keeps torso + one arm joints active,
- uses **Damped Least Squares (DLS)** in task space, and
- adds **null‑space secondary objectives** (regularization, smoothness, leg
  stability) to keep the whole‑body motion natural and stable.

It inherits from `PinocchioSolver` at the API level (configuration keys,
`get_ik` / `get_fk` signatures) but completely re‑implements the internals to
handle redundancy.

---

## Key Ideas

### Reduced whole‑body model

At initialization we:

- load the full Pinocchio model from URDF,
- **lock all joints except those in `joint_names`**, and
- obtain a reduced model with `dof = len(active_joints)`:

```188:203:embodichain/lab/sim/solvers/whole_body_ik_solver.py
full_robot = pin.RobotWrapper.BuildFromURDF(cfg.urdf_path, urdf_dir)

active_set = set(cfg.joint_names or [])
joints_to_lock = [
    name
    for name in full_robot.model.names
    if name not in active_set and name != "universe"
]
self._reduced = full_robot.buildReducedRobot(
    list_of_joints_to_lock=joints_to_lock,
    reference_configuration=np.zeros(full_robot.model.nq),
)

self.dof = self._reduced.model.nq
```

For the W1 robot, a typical `joint_names` for left whole‑body IK is:

```text
ANKLE, KNEE, BUTTOCK, WAIST, LEFT_J1..LEFT_J7   → 4 + 7 = 11 DOF
```

controlling a 6D end‑effector pose. This makes the whole‑body IK **redundant**
\(11 > 6\), which is the main reason we need a dedicated solver.

### Null‑space secondary objectives

Because the Jacobian \(J \in \mathbb{R}^{6 \times \mathrm{dof}}\) is
rectangular, there is a non‑trivial null space. We exploit it to encode
secondary objectives:

- joint regularization (keep joints near zero),
- smoothness (stay close to the seed configuration),
- leg / torso stability (custom linear combinations of torso joints).

The configuration class exposes these weights:

```134:144:embodichain/lab/sim/solvers/whole_body_ik_solver.py
w_pos: float = 1.0
w_rot: float = 0.1

w_regularization: float = 0.02
w_smooth: float = 0.1

joint_reg_extra: Optional[Dict[str, float]] = None

leg_costs_mode2: Optional[List[LegCostCfg]] = None
leg_costs_mode3: Optional[List[LegCostCfg]] = None
leg_mode: int = 2
```

`LegCostCfg` encodes simple quadratic terms of linear joint combinations:

```69:80:embodichain/lab/sim/solvers/whole_body_ik_solver.py
class LegCostCfg:
    """A leg / torso stability cost of the form:

       weight * (sum_i coefficients[i] * q[joint_names[i]])^2
    """
```

The three parts are merged into a single secondary gradient:

```324:333:embodichain/lab/sim/solvers/whole_body_ik_solver.py
def _secondary_gradient(self, q: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    g = np.zeros(self.dof)
    g += 2.0 * cfg.w_regularization * self._reg_weights * q
    g += 2.0 * cfg.w_smooth * (q - q_seed)
    for indices, coeffs, weight in self._leg_cost_terms:
        val = float(np.dot(coeffs, q[indices]))
        g[indices] += 2.0 * weight * val * coeffs
    return g
```

This gradient is then projected into the null space of the Jacobian so that it
**does not change the end‑effector pose**.

---

## Configuration (`WholeBodyIKSolverCfg`)

`WholeBodyIKSolverCfg` inherits from `PinocchioSolverCfg` and keeps the same
interface for:

- `urdf_path` – robot URDF,
- `joint_names` – active joints (others are locked),
- `end_link_name` – end‑effector link already present in the URDF,
- `root_link_name` – root link for the kinematic chain,
- `tcp` – tool center point as a \(4 \times 4\) matrix.

Additional fields specific to whole‑body IK:

| Field               | Type                        | Description                                              |
|---------------------|-----------------------------|----------------------------------------------------------|
| `urdf_dir`          | `str \| None`              | Optional mesh directory; defaults to `mesh_path`/URDF dir |
| `w_pos`             | `float`                     | Position error weight in task‑space norm                 |
| `w_rot`             | `float`                     | Orientation error weight                                 |
| `max_iterations`    | `int`                       | Max DLS iterations per IK call (default 200)             |
| `dt`                | `float`                     | Integration step in configuration space (default 0.5)    |
| `damp`              | `float`                     | DLS damping factor                                       |
| `pos_eps`           | `float`                     | Convergence threshold on position error (meters)         |
| `rot_eps`           | `float`                     | Convergence threshold on orientation error (radians)     |
| `w_regularization`  | `float`                     | Weight for joint regularization                          |
| `w_smooth`          | `float`                     | Weight for staying close to seed configuration           |
| `joint_reg_extra`   | `Dict[str, float] \| None` | Per‑joint extra regularization weight                    |
| `leg_costs_mode2`   | `List[LegCostCfg] \| None` | Leg / torso costs for mode 2 (e.g., standing)           |
| `leg_costs_mode3`   | `List[LegCostCfg] \| None` | Leg / torso costs for mode 3 (e.g., walking)            |
| `leg_mode`          | `int`                       | Select which leg cost set to use (2 or 3)                |

> **Note on `joint_names` semantics**  
> For `WholeBodyIKSolver`, `joint_names` means “**joints to keep unlocked**” in
> the reduced model, *not* “the serial chain” as in some other solvers. The
> recommended way is to leave it as `None` in config and let `Robot.init_solver`
> fill it from the corresponding `control_parts` entry.

---

## Usage Examples

### Direct solver usage

Minimal example for left arm + torso on W1:

```python
import numpy as np
import torch
from embodichain.lab.sim.solvers import WholeBodyIKSolverCfg

URDF = "/path/to/DexforceW1_v02_1.urdf"
LEFT_TCP = np.array(
    [[-1.0, 0.0, 0.0, 0.012],
     [ 0.0, 0.0, 1.0, 0.0675],
     [ 0.0, 1.0, 0.0, 0.127],
     [ 0.0, 0.0, 0.0, 1.0]]
)

cfg = WholeBodyIKSolverCfg(
    urdf_path=URDF,
    joint_names=["ANKLE", "KNEE", "BUTTOCK", "WAIST"] +
                [f"LEFT_J{i+1}" for i in range(7)],
    end_link_name="left_ee",
    root_link_name="base_link",
    tcp=LEFT_TCP,
    max_iterations=200,
    dt=0.3,
    pos_eps=3e-3,
    rot_eps=0.05,
)

solver = cfg.init_solver()

# Forward kinematics
q = torch.zeros(solver.dof)
pose = solver.get_fk(q)      # (4, 4)

# Inverse kinematics (round‑trip)
success, q_ik = solver.get_ik(pose, qpos_seed=q)
```

### Integration with `Robot` and control parts

For the W1 robot we follow the framework’s pattern:

- `DexforceW1Cfg._build_default_solver_cfg` defines four solvers:
  - `"left_arm"`, `"right_arm"` → `SRSSolver` (single‑arm analytic IK),
  - `"left_arm_body"`, `"right_arm_body"` → `WholeBodyIKSolver`
    (torso + arm whole‑body IK).
- `control_parts["left_arm_body"]` / `["right_arm_body"]` provide the joint
  lists; `Robot.init_solver` fills `joint_names` automatically.

Usage:

```python
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

sim = SimulationManager(SimulationManagerCfg(headless=False, sim_device="cpu"))
robot_cfg = DexforceW1Cfg.from_dict({"version": "v021", "arm_kind": "anthropomorphic"})
robot = sim.add_robot(cfg=robot_cfg)

# Compute a FK target (world frame)
qpos_fk = torch.zeros(1, 11)  # torso(0:4) + left_arm(4:11)
target = robot.compute_fk(qpos_fk, name="left_arm_body", to_matrix=True)

# Solve IK via high‑level API
q_seed = robot.get_qpos(name="left_arm_body")
success, q_ik = robot.compute_ik(pose=target, name="left_arm_body", joint_seed=q_seed)
```

The `Robot` class handles:

- transforming world‑frame poses into `root_link_name` frame, and
- mapping `q_ik` back into the appropriate joint subset.

---

## Algorithm Details

At each IK call, we perform the following steps
(\(q \in \mathbb{R}^{\mathrm{dof}}\), typically 11):

1. **Forward kinematics** on the reduced model:

   ```python
   pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q)
   oMf = self._reduced.data.oMf[self._ee_frame_id]
   ```

2. **Transform target pose** into the end‑link frame, taking into account:

   - robot root vs Pinocchio universe (`_root_base_xpos`),
   - tool center point (`tcp_xpos`):

   ```402:407:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   compute_xpos = self._root_base_xpos @ target_xpos @ np.linalg.inv(self.tcp_xpos)
   target_R = compute_xpos[:3, :3]
   target_t = compute_xpos[:3, 3]
   target_se3 = pin.SE3(target_R.copy(), target_t.copy())
   ```

3. **6D pose error** in LOCAL frame using the SE(3) logarithm:

   ```432:433:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   err6 = pin.log6(oMf.actInv(target_se3)).vector
   ```

4. **Weighted DLS primary task**:

   ```444:453:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   J_full = pin.computeFrameJacobian(..., pin.LOCAL)
   J = J_full * w_diag[:, None]  # position/orientation weights

   JJt = J @ J.T
   JJt[np.diag_indices_from(JJt)] += cfg.damp
   J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))

   dq_primary = J_pinv @ err
   ```

5. **Null‑space secondary task** (the main difference from `PinocchioSolver`):

   ```455:457:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   N = np.eye(self.dof) - J_pinv @ J
   g_sec = self._secondary_gradient(q, q_seed)
   dq_secondary = -N @ g_sec
   ```

   Here \(N = I - J^\dagger J\) is the projector onto the null space of \(J\).
   It guarantees that the secondary motion **does not change the end‑effector
   pose**, and only affects redundant DOFs (e.g. torso).

6. **Integrate and clip joint limits**:

   ```459:463:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   q = pin.integrate(self._reduced.model, q, (dq_primary + dq_secondary) * cfg.dt)
   q = np.clip(
       q,
       self._reduced.model.lowerPositionLimit,
       self._reduced.model.upperPositionLimit,
   )
   ```

This loop repeats until both position and orientation errors fall below
`pos_eps` / `rot_eps`, or `max_iterations` is reached.

---

## Public Methods

### `get_ik(target_xpos, qpos_seed=None, qvel_seed=None, return_all_solutions=False)`

```python
success, q = solver.get_ik(
    target_xpos,          # np.ndarray or torch.Tensor, shape (4, 4) or (1, 4, 4)
    qpos_seed=current_q,  # seed configuration in cfg joint order, optional
)
```

**Returns**

- `success`: `torch.BoolTensor`, shape `(1,)`, `True` if convergence criteria met
- `q`: `torch.FloatTensor`, shape `(nq,)`, IK solution in cfg joint order

> Even if `success == False`, `q` is the last iterate and can be used as a
> reasonable seed for the next step in an iterative controller.

When called through `robot.compute_ik`, the batch dimension is handled by the
`Robot` class and the return value is normalized to `(n_envs, dof)`.

### `get_fk(qpos)`

```python
pose = solver.get_fk(q)         # (4, 4) if q is (nq,)
poses = solver.get_fk(q_batch)  # (B, 4, 4) if q_batch is (B, nq)
```

The method computes FK on the reduced model and applies `tcp_xpos` to return
the TCP pose.

### `describe_chain()`

Returns a textual description of the active kinematic chain from
`root_link_name` to `end_link_name`, marking each joint as `[active]` or
`[locked]`. This is useful for debugging which joints are controlled by the
solver.

---

## Design Notes (Short Design Document)

### Why a separate whole‑body solver?

`PinocchioSolver` is designed for **single chains** where the number of joints
is close to the 6D task space (non‑redundant or lightly redundant). Its DLS
update:

\[
  \Delta q = J^\dagger e
\]

provides a unique solution and works well for arms.

For whole‑body IK on W1:

- torso + one arm → 11 DOF,
- end‑effector pose → 6D,

so the Jacobian is \(6 \times 11\) and the problem is **redundant**. Without
secondary objectives, the extra 5 DOFs can drift arbitrarily, leading to
unnatural torso motion and unstable poses.

This solver adds:

- a reduced whole‑body model (torso + arm),
- configurable secondary costs (regularization, smoothness, leg stability), and
- null‑space projection to keep these costs from affecting the end‑effector.

### Differences from `PinocchioSolver`

- **Same external API**:
  - uses `end_link_name`, `root_link_name`, `tcp`, `joint_names` like other
    solvers,
  - `get_ik` / `get_fk` signatures and return types are compatible.
- **Different internals**:
  - builds a reduced model with multiple branches (torso + arm) instead of a
    single serial chain,
  - uses null‑space control `(I - J†J)` with `_secondary_gradient`,
  - supports leg/torso stability via `LegCostCfg`.

### When to use `WholeBodyIKSolver`

Use this solver when:

- you want **whole‑body cooperation** (torso + arm) to reach hand targets,
- you need to keep **standing / walking posture** reasonable while solving IK,
- you want an API consistent with existing solvers (`Robot.compute_ik`,
  `solver_cfg`, `control_parts`), but with redundancy resolution built‑in.

For non‑redundant arms or when an analytic solution exists, `SRSSolver` or
`PinocchioSolver` remain more efficient choices.

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
