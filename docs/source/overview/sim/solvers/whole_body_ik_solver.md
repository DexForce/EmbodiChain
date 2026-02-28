# WholeBodyIKSolver

`WholeBodyIKSolver` is a whole‑body inverse kinematics (IK) solver built on [Pinocchio](https://stack-of-tasks.github.io/pinocchio/). It is designed for redundant humanoid robots where a single task‑space end‑effector (e.g. a hand) is controlled by a larger set of joints (torso + arm).

Unlike single‑chain solvers that assume a nearly square Jacobian, this solver:

- builds a *reduced* Pinocchio model that keeps torso + one arm joints active,
- uses **Damped Least Squares (DLS)** in task space, and
- adds **null‑space secondary objectives** (regularization, smoothness, leg / torso stability) to keep the whole‑body motion natural and stable.

It inherits from `PinocchioSolver` at the API level (configuration keys, `get_ik` / `get_fk` signatures) but completely re‑implements the internals to handle redundancy.

---

## Key Ideas

### Reduced whole‑body model

At initialization we load the full Pinocchio model from URDF, lock all joints except those in `joint_names`, and obtain a reduced model whose DOF equals the number of active joints:

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

controlling a 6D end‑effector pose. This makes the whole‑body IK **redundant** (11 > 6), which is the main reason we need a dedicated solver.

### Null‑space secondary objectives

Because the Jacobian \(J \in \mathbb{R}^{6 \times \mathrm{dof}}\) is rectangular, there is a non‑trivial null space. We exploit it to encode secondary objectives:

- joint regularization (keep joints near zero),
- smoothness (stay close to the seed configuration),
- leg / torso stability (custom linear combinations of torso joints).

The configuration class exposes the relevant weights:

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
    """Configuration of a single leg / torso stability secondary cost.

       The cost has the form weight * (sum_i coefficients[i] * q[joint_names[i]])^2.
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

This gradient is then projected into the null space of the Jacobian so that it does **not change the end‑effector pose**.

---

## Configuration (`WholeBodyIKSolverCfg`)

`WholeBodyIKSolverCfg` inherits from `PinocchioSolverCfg` and keeps the same interface for:

- `urdf_path` – robot URDF,
- `joint_names` – active joints (others are locked),
- `end_link_name` – end‑effector link already present in the URDF,
- `root_link_name` – root link for the kinematic chain,
- `tcp` – tool center point as a 4x4 matrix.

Additional fields specific to whole‑body IK:

| Field              | Type                        | Description                                                                 |
|--------------------|-----------------------------|-----------------------------------------------------------------------------|
| `urdf_dir`         | `str \| None`              | Optional mesh directory; defaults to `mesh_path` or the URDF directory     |
| `w_pos`            | `float`                     | Position error weight in task‑space norm                                    |
| `w_rot`            | `float`                     | Orientation error weight                                                    |
| `max_iterations`   | `int`                       | Max DLS iterations per IK call (default 200)                                |
| `dt`               | `float`                     | Integration step in configuration space (default 0.5)                       |
| `damp`             | `float`                     | DLS damping factor                                                          |
| `pos_eps`          | `float`                     | Convergence threshold on position error (meters)                            |
| `rot_eps`          | `float`                     | Convergence threshold on orientation error (radians)                        |
| `w_regularization` | `float`                     | Weight for joint regularization                                             |
| `w_smooth`         | `float`                     | Weight for staying close to seed configuration                              |
| `joint_reg_extra`  | `Dict[str, float] \| None` | Per‑joint extra regularization weight                                       |
| `leg_costs_mode2`  | `List[LegCostCfg] \| None` | Leg / torso costs for mode 2 (e.g., standing)                               |
| `leg_costs_mode3`  | `List[LegCostCfg] \| None` | Leg / torso costs for mode 3 (e.g., walking)                                |
| `leg_mode`         | `int`                       | Select which leg cost set to use (`2` or `3`)                               |

> **Note on `joint_names` semantics**  
> For `WholeBodyIKSolver`, `joint_names` means “joints to keep unlocked” in the reduced model, not “the serial chain” as in some other solvers. The recommended way is to leave it as `None` in config and let `Robot.init_solver` populate it from the corresponding `control_parts` entry.

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
    joint_names=["ANKLE", "KNEE", "BUTTOCK", "WAIST"] + [f"LEFT_J{i+1}" for i in range(7)],
    end_link_name="left_ee",
    root_link_name="base_link",
    tcp=LEFT_TCP,
    max_iterations=200,
    dt=0.3,
    pos_eps=3e-3,
    rot_eps=0.05,
)

solver = cfg.init_solver()

q = torch.zeros(solver.dof)
pose = solver.get_fk(q)      # (4, 4)

success, q_ik = solver.get_ik(pose, qpos_seed=q)
```

### Integration with `Robot` and control parts

For the W1 robot we follow the framework’s pattern:

- `DexforceW1Cfg._build_default_solver_cfg` defines four solvers:
  - `"left_arm"`, `"right_arm"` → `SRSSolver` (single‑arm analytic IK),
  - `"left_arm_body"`, `"right_arm_body"` → `WholeBodyIKSolver` (torso + arm whole‑body IK).
- `control_parts["left_arm_body"]` / `["right_arm_body"]` provide the joint lists; `Robot.init_solver` fills `joint_names` automatically.

Usage:

```python
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

sim = SimulationManager(SimulationManagerCfg(headless=False, sim_device="cpu"))
robot_cfg = DexforceW1Cfg.from_dict({"version": "v021", "arm_kind": "anthropomorphic"})
robot = sim.add_robot(cfg=robot_cfg)

qpos_fk = torch.zeros(1, 11)  # torso(0:4) + left_arm(4:11)
target = robot.compute_fk(qpos_fk, name="left_arm_body", to_matrix=True)

q_seed = robot.get_qpos(name="left_arm_body")
success, q_ik = robot.compute_ik(pose=target, name="left_arm_body", joint_seed=q_seed)
```

The `Robot` class handles transforming world‑frame poses into the `root_link_name` frame and mapping `q_ik` back into the appropriate joint subset.

---

## Algorithm Details

At each IK call, the solver performs the following steps (\(q \in \mathbb{R}^{\mathrm{dof}}\), typically 11 for W1 torso + arm):

1. Run forward kinematics on the reduced model to obtain the current end‑effector pose:

   ```python
   pin.framesForwardKinematics(self._reduced.model, self._reduced.data, q)
   oMf = self._reduced.data.oMf[self._ee_frame_id]
   ```

2. Transform the target pose into the end‑link frame, taking into account:
   - robot root vs Pinocchio universe (`_root_base_xpos`),
   - tool center point (`tcp_xpos`):

   ```402:407:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   compute_xpos = self._root_base_xpos @ target_xpos @ np.linalg.inv(self.tcp_xpos)
   target_R = compute_xpos[:3, :3]
   target_t = compute_xpos[:3, 3]
   target_se3 = pin.SE3(target_R.copy(), target_t.copy())
   ```

3. Compute 6D pose error in LOCAL frame using the SE(3) logarithm:

   ```432:433:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   err6 = pin.log6(oMf.actInv(target_se3)).vector
   ```

4. Build the weighted DLS primary task:

   ```444:453:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   J_full = pin.computeFrameJacobian(..., pin.LOCAL)
   J = J_full * w_diag[:, None]  # position/orientation weights

   JJt = J @ J.T
   JJt[np.diag_indices_from(JJt)] += cfg.damp
   J_pinv = J.T @ np.linalg.solve(JJt, np.eye(6))

   dq_primary = J_pinv @ err
   ```

5. Add the null‑space secondary task (the main difference from `PinocchioSolver`):

   ```455:457:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   N = np.eye(self.dof) - J_pinv @ J
   g_sec = self._secondary_gradient(q, q_seed)
   dq_secondary = -N @ g_sec
   ```

   Here \(N = I - J^\dagger J\) is the projector onto the null space of \(J\). It guarantees that the secondary motion does not change the end‑effector pose and only affects redundant DOFs (e.g. torso).

6. Integrate and clip joint limits:

   ```459:463:embodichain/lab/sim/solvers/whole_body_ik_solver.py
   q = pin.integrate(self._reduced.model, q, (dq_primary + dq_secondary) * cfg.dt)
   q = np.clip(
       q,
       self._reduced.model.lowerPositionLimit,
       self._reduced.model.upperPositionLimit,
   )
   ```

This loop repeats until both position and orientation errors fall below `pos_eps` / `rot_eps`, or `max_iterations` is reached.

---

## Public Methods

### `get_ik(target_xpos, qpos_seed=None, qvel_seed=None, return_all_solutions=False)`

```python
success, q = solver.get_ik(
    target_xpos,          # np.ndarray or torch.Tensor, shape (4, 4) or (1, 4, 4)
    qpos_seed=current_q,  # seed configuration in cfg joint order, optional
)
``+

**Returns**

- `success`: `torch.BoolTensor`, shape `(1,)`, `True` if convergence criteria are met.
- `q`: `torch.FloatTensor`, shape `(nq,)`, IK solution in cfg joint order.

Even if `success == False`, `q` is the last iterate and can be used as a reasonable seed for the next step in an iterative controller.

When called through `robot.compute_ik`, the batch dimension is handled by the `Robot` class and the return value is normalized to `(n_envs, dof)`.

### `get_fk(qpos)`

```python
pose = solver.get_fk(q)         # (4, 4) if q is (nq,)
poses = solver.get_fk(q_batch)  # (B, 4, 4) if q_batch is (B, nq)
```

The method computes FK on the reduced model and applies `tcp_xpos` to return the TCP pose.

### `describe_chain()`

Returns a human‑readable description of the kinematic chain from `root_link_name` to `end_link_name`, marking each joint as `[active]` or `[locked]`. This is useful for understanding which joints are controlled by the solver.

---

## Design Notes (Short Design Document)

### Why a separate whole‑body solver?

`PinocchioSolver` is designed for single chains where the number of joints is close to the 6‑DOF task space (non‑redundant or lightly redundant). Its DLS update \(\Delta q = J^\dagger e\) provides a unique solution and works well for arms. For whole‑body IK on W1, however, a torso + one arm uses 11 DOF to control a 6‑D end‑effector pose; the Jacobian is \(6 \times 11\) and the problem is redundant. Without secondary objectives, the extra 5 DOFs can drift arbitrarily, leading to unnatural torso motion and unstable poses.

`WholeBodyIKSolver` addresses this by building a reduced whole‑body model (torso + arm), adding configurable secondary costs (regularization, smoothness, leg stability), and projecting them into the null space of the Jacobian so that they do not disturb the primary end‑effector task.

### Differences from `PinocchioSolver`

- Same external API:
  - uses `end_link_name`, `root_link_name`, `tcp`, `joint_names` like other solvers;
  - `get_ik` / `get_fk` signatures and return types are compatible.
- Different internals:
  - builds a reduced model with multiple branches (torso + arm) instead of a single serial chain;
  - uses null‑space control `(I - J†J)` with `_secondary_gradient`;
  - supports leg/torso stability via `LegCostCfg`.

### When to use `WholeBodyIKSolver`

Use this solver when you want whole‑body cooperation (torso + arm) to reach hand targets, need to keep standing / walking posture reasonable while solving IK, and prefer an API consistent with existing solvers (`Robot.compute_ik`, `solver_cfg`, `control_parts`) but with redundancy resolution built‑in. For non‑redundant arms or when an analytic solution exists, `SRSSolver` or `PinocchioSolver` remain more efficient choices.