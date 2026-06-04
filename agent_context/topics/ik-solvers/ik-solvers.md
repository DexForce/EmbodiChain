# ik-solvers

> Topic: Inverse-kinematics solver subsystem — solver hierarchy,
> available algorithms, configuration, seed sampling, and failure modes.

---

## Entry Points

| File | Role |
|---|---|
| `embodichain/lab/sim/solvers/__init__.py` | Public re-exports for all solver classes and configs |
| `embodichain/lab/sim/solvers/base_solver.py` | `BaseSolver` ABC + `SolverCfg` base config |
| `embodichain/lab/sim/cfg.py` | `RobotCfg.solver_cfg` — where solver config is wired into a robot |
| `embodichain/lab/sim/solvers/qpos_seed_sampler.py` | `QposSeedSampler` — random joint-seed generation |
| `embodichain/lab/sim/solvers/null_space_posture_task.py` | `NullSpacePostureTask` — Pink null-space posture objective |
| `embodichain/lab/sim/utility/solver_utils.py` | Helpers: `create_pk_serial_chain`, `build_reduced_pinocchio_robot`, `validate_iteration_params`, `compute_pinocchio_fk` |

---

## Overview

Each robot can have one or more IK solvers (one per control part).
Solvers share a common `BaseSolver` interface for FK, IK, Jacobian, TCP,
and joint-limit management.  A `SolverCfg` subclass is instantiated
inside `RobotCfg` and its `init_solver()` factory method produces the
concrete `BaseSolver` instance at runtime.

All solvers use a `pytorch_kinematics` serial chain (`pk_serial_chain`)
for FK and Jacobian computation. `torch.compile` is applied to the FK
path for performance.

---

## Solver Hierarchy

```
SolverCfg  (@configclass, abstract)
  ├── SRSSolverCfg
  ├── OPWSolverCfg
  ├── PytorchSolverCfg
  ├── PinocchioSolverCfg
  ├── PinkSolverCfg
  └── DifferentialSolverCfg

BaseSolver  (ABCMeta)
  ├── SRSSolver
  ├── OPWSolver
  ├── PytorchSolver
  ├── PinocchioSolver
  ├── PinkSolver
  └── DifferentialSolver
```

`SolverCfg.init_solver()` is the abstract factory; each subclass
overrides it to construct the matching `BaseSolver` subclass.

---

## Available Solvers

| Solver | Algorithm | When to use | Key dependencies |
|---|---|---|---|
| **SRSSolver** | SRS analytical IK (7-DOF, elbow-sampling) | DexForce W1 arms; fast GPU-batched analytical solve via Warp kernels | `warp` |
| **OPWSolver** | OPW analytical IK (6-DOF) | 6-DOF industrial arms with OPW kinematic structure | `warp`, `polars` |
| **PytorchSolver** | Iterative damped-least-squares via `pytorch_kinematics` | General-purpose GPU solver; good default for arbitrary URDFs | `pytorch_kinematics` |
| **PinocchioSolver** | Iterative IK via Pinocchio + optional CasADi | High-accuracy IK with full rigid-body dynamics model | `pinocchio`, `casadi` (optional) |
| **PinkSolver** | Task-based IK via Pink (QP optimisation) | Multi-task IK (e.g., dual-arm, posture + EE control, null-space tasks) | `pinocchio`, `pink` |
| **DifferentialSolver** | Differential IK (Jacobian pseudo-inverse / SVD / DLS) | Real-time velocity-level IK; supports relative-mode commands | (none beyond core) |

---

## Solver Interface

### `SolverCfg` (base config)

Key fields shared by all configs:

| Field | Type | Purpose |
|---|---|---|
| `class_type` | `str` | Solver class name (used by `from_dict` factory) |
| `urdf_path` | `str \| None` | Path to robot URDF |
| `joint_names` | `list[str] \| None` | Joints to include; `None` = all |
| `end_link_name` | `str` | End-effector link name |
| `root_link_name` | `str` | Base link name |
| `tcp` | `np.ndarray` | 4×4 tool-center-point transform |
| `ik_nearest_weight` | `list[float] \| None` | Per-joint weight for nearest-solution selection |
| `user_qpos_limits` | `list[float] \| None` | Optional custom joint limits `[2, DOF]` or `[DOF, 2]` |

Factory: `cfg.init_solver(device=..., **kwargs) → BaseSolver`

Dict factory: `SolverCfg.from_dict(dict) → SolverCfg` resolves `class_type` dynamically.

### `BaseSolver` (abstract base)

| Method | Signature | Notes |
|---|---|---|
| `get_ik` | `(target_pose: Tensor[4,4], joint_seed, num_samples) → (success: Tensor, joints: Tensor)` | Abstract. Returns `(num_envs,)` bool + `(num_envs, dof)` joint positions |
| `get_fk` | `(qpos: Tensor) → Tensor[batch,4,4]` | Concrete. Uses compiled `pk_serial_chain` FK + TCP |
| `get_jacobian` | `(qpos, locations, jac_type) → Tensor` | Concrete. Returns `(batch, 6, dof)` full / `(batch, 3, dof)` trans/rot |
| `set_tcp` / `get_tcp` | `(np.ndarray[4,4])` | Set/get tool-center-point |
| `set_qpos_limits` / `get_qpos_limits` | limits as list/Tensor | Set/get per-joint limits |
| `update_with_robot_limit` | `(robot_qpos_limits: Tensor[DOF,2])` | Clamp solver limits to robot limits |
| `set_ik_nearest_weight` / `get_ik_nearest_weight` | per-joint weights | Controls nearest-solution ranking |

---

## Configuration

### In `RobotCfg` (`embodichain/lab/sim/cfg.py`)

```python
@configclass
class RobotCfg(ArticulationCfg):
    solver_cfg: Union[SolverCfg, Dict[str, SolverCfg], None] = None
```

- **Single-part robot**: `solver_cfg = PytorchSolverCfg(...)`.
- **Multi-part robot** (e.g., dual-arm): `solver_cfg = {"right_arm": SRSSolverCfg(...), "left_arm": SRSSolverCfg(...)}`.
  Keys must match `control_parts` names in `RobotCfg`.

### Iterative solver common params

`PytorchSolverCfg`, `PinocchioSolverCfg`, `PinkSolverCfg`, and
`DifferentialSolverCfg` share these fields:

| Field | Default | Purpose |
|---|---|---|
| `pos_eps` | `5e-4` | Position convergence tolerance |
| `rot_eps` | `5e-4` | Rotation convergence tolerance |
| `max_iterations` | 500–1000 | Iteration cap |
| `dt` | `0.1` | Numerical integration step |
| `damp` | `1e-6` | Damping for numerical stability |
| `is_only_position_constraint` | `False` | Ignore orientation in IK |
| `num_samples` | 5–30 | Random seeds per solve |

### DifferentialSolver-specific

- `ik_method`: `"pinv"`, `"svd"`, `"trans"`, `"dls"` — Jacobian inversion strategy.
- `ik_params`: auto-populated defaults per method (e.g., `k_val`, `lambda_val`).
- `command_type`: `"position"` or `"pose"`.
- `use_relative_mode`: delta commands relative to current pose.

### PinkSolver-specific

- `variable_input_tasks` / `fixed_input_tasks`: lists of `pink.tasks.FrameTask` for QP optimisation.
- `mesh_path`: path for Pinocchio URDF mesh loading.
- `show_ik_warnings` / `fail_on_joint_limit_violation`: error-handling behaviour.

### SRSSolver-specific

- `dh_params`, `link_lengths`, `rotation_directions`, `T_b_ob`, `T_e_oe`: kinematic model params.
- `sort_ik`: whether to rank solutions by distance to seed.
- Requires `num_envs` in `init_solver()`.

### OPWSolver-specific

- `a1, a2, b, c1–c4, offsets, flip_axes, has_parallelogram`: OPW kinematic parameters.
- `safe_margin`: joint-limit safety margin in radians.

---

## Seed Sampling and Null-Space Tasks

### `QposSeedSampler` (`qpos_seed_sampler.py`)

Used by iterative solvers (e.g., `PytorchSolver`) to generate joint-seed
batches for IK multi-start:

- `__init__(num_samples, dof, device)`
- `sample(qpos_seed, lower_limits, upper_limits, batch_size) → Tensor[batch*num_samples, dof]`
  - First sample = provided seed; remaining are uniform-random within limits.
- `repeat_target_xpos(target_xpos, num_samples)` — repeats target poses to match expanded seed batch.

### `NullSpacePostureTask` (`null_space_posture_task.py`)

A `pink.tasks.Task` subclass for posture control in the null space of
higher-priority tasks.

- Error: `e(q) = M · (q* − q)` where `M` is a joint-selection mask.
- Jacobian: null-space projector `N(q) = I − J_primary⁺ · J_primary`.
- Used exclusively with `PinkSolver` for multi-task QP IK (e.g., controlling
  posture while satisfying end-effector constraints).

---

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `get_ik` returns `success=False` for all envs | Target pose unreachable or too few samples | Increase `num_samples`; verify target is within workspace |
| Joint limits mismatch warnings at init | `user_qpos_limits` shape is wrong | Must be `[2, DOF]` or `[DOF, 2]` |
| `"Kinematic chain is not initialized"` in FK | `pk_serial_chain` creation failed | Check `urdf_path`, `end_link_name`, `root_link_name` are valid |
| Solver picks distant IK solution | `ik_nearest_weight` not tuned | Set per-joint weights to prefer important joints |
| `ImportError` for pinocchio / pink / warp | Optional dependency missing | Install: `pip install pin==2.7.0`, `pip install pin-pink==3.4.0`, or `pip install warp-lang` |
| `solver_cfg` keys don't match `control_parts` | Multi-part robot misconfiguration | Dict keys in `solver_cfg` must exactly match `control_parts` names |
| DifferentialSolver oscillates near target | `damp` too low or `dt` too large | Increase `damp` or decrease `dt` |
| Pink solver ignores orientation | `is_only_position_constraint = True` | Set to `False` for full-pose IK |
