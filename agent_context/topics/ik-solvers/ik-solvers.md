# ik-solvers

> Topic: Inverse-kinematics solver subsystem — solver hierarchy,
> available algorithms, configuration, seed sampling, and failure modes.

---

## Entry Points

| File | Role |
|---|---|
| `embodichain/lab/sim/motion/solvers/__init__.py` | Public re-exports for all solver classes and configs |
| `embodichain/lab/sim/motion/solvers/base_solver.py` | `BaseSolver` ABC + `SolverCfg` base config |
| `embodichain/lab/sim/cfg.py` | `RobotCfg.solver_cfg` — where solver config is wired into a robot |
| `embodichain/lab/sim/motion/solvers/qpos_seed_sampler.py` | `QposSeedSampler` — random joint-seed generation |
| `embodichain/lab/sim/motion/solvers/null_space_posture_task.py` | `NullSpacePostureTask` — Pink null-space posture objective |
| `embodichain/lab/sim/utility/solver_utils.py` | Helpers: `create_pk_serial_chain`, `build_reduced_pinocchio_robot`, `validate_iteration_params`, `compute_pinocchio_fk` |

---

## Overview

Each robot can have one or more IK solvers (one per control part).
Solvers share a common `BaseSolver` interface for FK, IK, Jacobian, TCP,
and joint-limit management.  A `SolverCfg` subclass is instantiated
inside `RobotCfg` and its `init_solver()` factory method produces the
concrete `BaseSolver` instance at runtime.

Import solver classes and configs from `embodichain.lab.sim.motion.solvers`.
`RobotCfg.from_dict()` resolves configured `class_type` names against
that public module. The `motion` parent loads subpackages lazily; adding solver
exports must not eagerly import planners or workspace analyzers into the Robot
initialization path. Warp kernels live under `embodichain/compute/kinematics/_warp/`.
Focused solver tests live under `tests/sim/motion/solvers/`.

All solvers use a `pytorch_kinematics` serial chain (`pk_serial_chain`)
for FK and Jacobian computation. `torch.compile` is applied to the FK
path for performance.

---
## Available Solvers

| Solver | Algorithm | When to use | Key dependencies |
|---|---|---|---|
| **SRSSolver** | SRS analytical IK (7-DOF, elbow-sampling) | DexForce W1 arms; fast GPU-batched analytical solve via Warp kernels | `warp` |
| **OPWSolver** | OPW analytical IK (6-DOF) | 6-DOF industrial arms with OPW kinematic structure | `warp`, `polars` |
| **PytorchSolver** | Iterative damped-least-squares via `pytorch_kinematics` | General-purpose GPU solver; good default for arbitrary URDFs | `pytorch_kinematics` |
| **PinocchioSolver** | Iterative IK via Pinocchio + optional CasADi | High-accuracy IK with full rigid-body dynamics model | `pinocchio`, `casadi` (optional) |
| **PinkSolver** | Task-based IK via Pink (QP optimisation) | Multi-task IK (e.g., dual-arm, posture + EE control, null-space tasks) | `pinocchio`, `pink` |
| **URSolver** | UR analytical IK | UR kinematic families | `warp` |
| **NeuralIKSolver** | Learned IK | Model-dependent inference; inspect its config/checkpoint contract | `torch` |
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
| `get_ik` | `(target_pose: Tensor[4,4], joint_seed, num_samples) → (success: Tensor, joints: Tensor)` | Concrete solvers determine candidate axes; see the shape contract below |
| `get_fk` | `(qpos: Tensor) → Tensor[batch,4,4]` | Concrete. Uses compiled `pk_serial_chain` FK + TCP |
| `get_jacobian` | `(qpos, locations, jac_type) → Tensor` | Concrete. Returns `(batch, 6, dof)` full / `(batch, 3, dof)` trans/rot |
| `set_tcp` / `get_tcp` | `(np.ndarray[4,4])` | Set/get tool-center-point |
| `set_qpos_limits` / `get_qpos_limits` | limits as list/Tensor | Set/get per-joint limits |
| `update_with_robot_limit` | `(robot_qpos_limits: Tensor[DOF,2])` | Clamp solver limits to robot limits |
| `set_ik_nearest_weight` / `get_ik_nearest_weight` | per-joint weights | Controls nearest-solution ranking |

---

## Shape and coordinate contract

Do not infer a universal `(N, dof)` result from the abstract interface. Pink
returns joint candidates as `(N, 1, dof)`. On Pytorch's normal result path,
nearest mode retains that singleton solution axis and `return_all_solutions=True`
returns `(N, num_samples, dof)`. Its early all-targets-failed path currently
returns `(N, dof)` even when all solutions were requested. Check the concrete
solver, failure branch and success-mask contract
before selecting/squeezing a candidate. Preserve batch size one.

The solver consumes poses relative to its configured chain root and applies its
TCP transform; integrations own world/root conversion and robot/solver joint
ordering. Validate frames, radians, device and joint limits at that boundary.

## Wiring and detailed tuning

`SolverCfg.from_dict()` resolves `class_type`; `init_solver()` creates the concrete solver.
Wire configs through `RobotCfg.solver_cfg`; multi-part keys follow `control_parts`.
Read [solver details](solver-details.md) for iterative parameters, Pink/SRS/OPW behavior,
`QposSeedSampler` and null-space posture tasks. Gizmo adaptation belongs to
`objects/gizmo.py`; see [native gizmos](../sim-visualization/native-gizmos.md).

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `get_ik` returns `success=False` for all envs | Target pose unreachable or too few samples | Increase `num_samples`; verify target is within workspace |
| Joint limits mismatch warnings at init | `user_qpos_limits` shape is wrong | Must be `[2, DOF]` or `[DOF, 2]` |
| `"Kinematic chain is not initialized"` in FK | `pk_serial_chain` creation failed | Check `urdf_path`, `end_link_name`, `root_link_name` are valid |
| Solver picks distant IK solution | `ik_nearest_weight` not tuned | Set per-joint weights to prefer important joints |
| `ImportError` for pinocchio / pink / warp | Optional dependency missing | Use the dependency declarations in `pyproject.toml` / `setup.py` for this checkout |
| `solver_cfg` keys don't match `control_parts` | Multi-part robot misconfiguration | Dict keys in `solver_cfg` must exactly match `control_parts` names |
| DifferentialSolver oscillates near target | `damp` too low or `dt` too large | Increase `damp` or decrease `dt` |
| Pink solver ignores orientation | `is_only_position_constraint = True` | Set to `False` for full-pose IK |

## Computation ownership

OPW, SRS, and UR Warp implementations live in
`embodichain/compute/kinematics/_warp/{opw,srs,ur}.py`. The existing
`lab.sim.motion.solvers` classes own configuration, state, device buffers, and
solver interfaces. The compute kernels do not import simulation modules.
`utils/warp/kinematics/*_solver.py` are compatibility aliases.
Validate kernel import/compilation with `tests/compute/test_imports.py` and
solver behavior with the corresponding `tests/sim/motion/solvers/` tests.
