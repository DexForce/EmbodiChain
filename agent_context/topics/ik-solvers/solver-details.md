# Solver tuning, seeds and null-space tasks

Read this when the request needs these details. [Topic overview](ik-solvers.md).

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

- `variable_input_tasks` contains exactly one `pink.tasks.FrameTask` targeted
  by the single-pose IK API and may include other task types. Additional fixed
  frame constraints belong in `fixed_input_tasks`.
- `mesh_path`: path for Pinocchio URDF mesh loading.
- `show_ik_warnings` / `fail_on_joint_limit_violation`: error-handling behaviour.
- Supports single-pose and sequential batch IK while preserving one seed per
  target and returning a consistent `(N,)`, `(N, 1, dof)` result contract.
- End-effector targets are interpreted as TCP poses; the configured TCP is
  removed before setting the controlled Pink frame target. Targets remain
  relative to `root_link_name` even when that link is offset from the URDF root.
- Convergence is checked against the single targeted variable FrameTask. Maximum-iteration,
  stagnation, and solver-exception exits report failure and preserve the
  corresponding input seed.
- Adaptive controls (`stagnation_tolerance`, `stagnation_iterations`,
  `max_backtracks`, `damping_growth`, `damping_decay`, and `max_damping`)
  use a lexicographic merit that prioritizes FrameTask progress and considers
  only controllable projected null-space posture error as a secondary term.
  They increase regularization and terminate stalled solves early.
- Effective limits intersect URDF, user-configured, and runtime robot limits,
  then synchronize the result into the reduced Pinocchio model in Pink order.

### SRSSolver-specific

- `dh_params`, `link_lengths`, `rotation_directions`, `T_b_ob`, `T_e_oe`: kinematic model params.
- `sort_ik`: whether to rank solutions by distance to seed.
- `search_mode`: `"seeded"` computes the seed's geometric shoulder-elbow-wrist
  arm angle and searches redundancy angles radially around it; if the configured
  radial step cannot produce `num_samples` distinct angles in one revolution,
  the incomplete radial prefix is replaced by a complete seed-centered uniform
  full-circle grid. `"full"` samples the complete `[-pi, pi)` interval directly.
- `redundancy_step`: angular increment used by seed-centered search.
- Requesting all solutions always uses full-space redundancy sampling.
- CPU and CUDA derive the reference plane in the base frame and use the same
  signed arm-angle, shoulder-azimuth degeneracy rule, and periodic
  nearest-solution formulas. Shoulder azimuth is set to zero only when the
  shoulder-to-wrist projection onto the XY plane is near zero.
- At shoulder or wrist Euler singularities, both analytical backends preserve
  the seed's free coupled joint and solve the remaining coupled angle, avoiding
  arbitrary equivalent-angle jumps near singular configurations.
- Candidate revolute angles are shifted by integer multiples of `2*pi` into
  the configured joint limits, choosing the representation nearest the seed.
- Runtime `set_tcp()` and `set_ik_nearest_weight()` calls synchronize the CPU
  and Warp analytical-backend caches immediately.
- CPU target/reference-plane geometry is precomputed per target and elbow
  branch. CUDA derives target/config/angle indices directly from the Warp
  thread id, and all-solution sorting uses device-side tensor sorting rather
  than a serial quadratic Warp sort.
- Warp arm-angle and IK scratch arrays are reused by shape within a solver
  instance to avoid repeated device allocations during steady-state calls.
- Periodic-equivalent all-solutions candidates are greedily deduplicated
  against retained representatives on CPU before indexing the original device
  tensor, preserving order without allocating quadratic GPU scratch space.
- Requires `num_envs` in `init_solver()`.

Focused performance and accuracy validation is available at
`scripts/benchmark/robotics/kinematic_solver/srs_solver.py`; it compares CPU
and available CUDA backends in seeded and full redundancy-search modes.

### OPWSolver-specific

- `a1, a2, b, c1–c4, offsets, flip_axes, has_parallelogram`: OPW kinematic parameters.
- `safe_margin`: joint-limit safety margin in radians.

---

## Seed Sampling and Null-Space Tasks

### Default seed (`BaseSolver.get_default_qpos_seed`)

When `get_ik` receives no seed, every solver falls back to the joint-range
midpoint from `BaseSolver.get_default_qpos_seed()` — never a zero
configuration, which violates the limits of some robots (Franka FR3 joints 4
and 6) and biases nearest-solution selection toward the bounds. Pinocchio
resets its internal `init_qpos` to this default on seedless calls instead of
reusing the previous call's seed; UR accepts a missing seed instead of
crashing. Pink keeps its own limit-projected neutral configuration.

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

- Error: a Pinocchio manifold difference masked in tangent space (`nv`), with
  an empty joint selection meaning all actuated joints while floating-base
  coordinates are always excluded.
- Jacobian: null-space projector `N(q) = I − J_primary⁺ · J_primary`.
- Add it to either `variable_input_tasks` or `fixed_input_tasks`; PinkSolver
  initializes it, includes it in QP solving, and uses its controllable projected
  error only as a secondary backtracking merit behind FrameTask progress. It
  updates its simulator-ordered target through
  `update_null_space_joint_targets()`.

---
