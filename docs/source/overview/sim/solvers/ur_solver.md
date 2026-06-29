# URSolver

`URSolver` is a specialized analytical inverse kinematics (IK) solver for the
Universal Robots family of 6-DOF manipulators (UR3, UR3e, UR5, UR5e, UR10,
UR10e). Unlike numerical solvers, it derives closed-form joint solutions from
the robot's Denavit–Hartenberg (DH) parameters and evaluates them in batch on
the GPU through a [NVIDIA Warp](https://github.com/NVIDIA/warp) kernel. This
makes it fast, deterministic, and well suited to large-scale batch IK tasks and
real-time control where exact solutions are preferred over iterative refinement.

The analytic formulation follows the
[ur-analytic-ik](https://github.com/Victorlouisdg/ur-analytic-ik) repository,
and the DH parameters for each robot variant are taken from
[`dh_parameters.hh`](https://github.com/Victorlouisdg/ur-analytic-ik/blob/main/src/ur_analytic_ik/dh_parameters.hh).

## Key Features

* Analytical, closed-form IK for the full UR series (UR3/UR3e/UR5/UR5e/UR10/UR10e)
* DH parameters auto-populated from the selected `ur_type`, sourced from `ur-analytic-ik`
* Batched GPU computation via a Warp kernel for many target poses at once
* Up to 8 base analytical solutions, expanded to 512 FK-equivalent candidates per pose through per-joint `±2π` shifts
* Strict enforcement of joint limits; invalid candidates are flagged and filtered out
* Forward-kinematics (FK) error check that validates each candidate against the target pose
* Tool Center Point (TCP) support via `set_tcp`
* Flexible configuration via `URSolverCfg`

## DH Parameters

The UR kinematic chain is described with the standard DH convention:

| Joint | theta | d   | a   | alpha   |
|-------|-------|-----|-----|---------|
| 1     | q1    | d1  | 0   | +π/2    |
| 2     | q2    | 0   | a2  | 0       |
| 3     | q3    | 0   | a3  | 0       |
| 4     | q4    | d4  | 0   | +π/2    |
| 5     | q5    | d5  | 0   | -π/2    |
| 6     | q6    | d6  | 0   | 0       |

Setting `ur_type` in `URSolverCfg` selects the matching `(d1, a2, a3, d4, d5, d6)`
values and the corresponding URDF asset. The values are reproduced from
[`dh_parameters.hh`](https://github.com/Victorlouisdg/ur-analytic-ik/blob/main/src/ur_analytic_ik/dh_parameters.hh).

## Configuration

The solver is configured using the `URSolverCfg` class. Selecting `ur_type`
automatically fills in the DH parameters and the URDF path; the remaining fields
default to sensible values and rarely need to be overridden.

```python
import torch
from embodichain.lab.sim.solvers.ur_solver import URSolver, URSolverCfg

cfg = URSolverCfg(
    ur_type="ur10",       # one of: ur3, ur3e, ur5, ur5e, ur10, ur10e
    end_link_name="ee_link",
    root_link_name="base_link",
    # DH parameters are set automatically from ur_type; override only if needed:
    # d1=0.1273, a2=-0.612, a3=-0.5723, d4=0.163941, d5=0.1157, d6=0.0922,
    tcp=[
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.12],
        [0.0, 0.0, 0.0, 1.0],
    ],
)

solver = cfg.init_solver(device="cuda")
```

.. tip::
    ``ur_type`` is the only parameter that usually needs to be set. The
    ``__post_init__`` hook raises ``ValueError`` for any unrecognized value, so
    an invalid robot variant fails fast at configuration time.

## Main Methods

* `get_fk(self, qpos: torch.Tensor) -> torch.Tensor`
  Computes the end-effector pose (homogeneous transformation matrix) for the
  given joint positions. Inherited from `BaseSolver` and computed from the URDF
  kinematic chain, with the TCP applied.

  **Parameters:**
  + `qpos` (`torch.Tensor` or `list[float]`): Joint positions, shape `(num_envs, num_joints)` or `(num_joints,)`.

  **Returns:**
  + `torch.Tensor`: End-effector pose(s), shape `(num_envs, 4, 4)`.

  **Example:**

```python
  fk = solver.get_fk(qpos=[0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0])
  print(fk)
  # Output:
  # tensor([[[ -1.0000,  -0.0000,   0.0000,   0.0000],
  #          [ -0.0000,  -0.0000,  -1.0000,  -0.2561],
  #          [  0.0000,  -1.0000,   0.0000,   1.4273],
  #          [  0.0000,   0.0000,   0.0000,   1.0000]]], device='cuda:0')
```

* `get_ik(self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor, return_all_solutions: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`
  Computes joint positions (inverse kinematics) for the given target
  end-effector pose(s) using the closed-form UR solution evaluated in a Warp
  kernel.

  For each target pose the kernel first derives the 8 base analytical solutions
  (combinations of the two `theta1` branches, the two `theta5` sign branches,
  and the elbow-up/down `theta3` branch). Because UR joints are `2π`-periodic,
  each base solution is then expanded into `2**6 = 64` per-joint `±2π` shift
  variants that are FK-equivalent, yielding `8 * 64 = 512` candidate solutions.
  A candidate is flagged valid only if its forward kinematics matches the target
  pose *and* every joint lies within the configured joint limits.

  When `return_all_solutions=False`, the valid candidate closest to `qpos_seed`
  (by Euclidean distance in joint space) is returned for each target.

  **Parameters:**
  + `target_xpos` (`torch.Tensor`): Target end-effector pose(s), shape `(4, 4)` or `(num_envs, 4, 4)`.
  + `qpos_seed` (`torch.Tensor`): Reference joint positions used to pick the closest solution, shape `(num_envs, num_joints)` or `(num_joints,)`.
  + `return_all_solutions` (`bool`, optional): If `True`, returns all 512 candidates and their validity flags instead of the single closest solution. Default is `False`.
  + `**kwargs`: Additional arguments for future extensions.

  **Returns:**
  + `Tuple[torch.Tensor, torch.Tensor]`:
    - If `return_all_solutions=False`:
      - First element: validity flag per environment, shape `(num_envs,)`.
      - Second element: closest joint positions, shape `(num_envs, num_joints)`.
    - If `return_all_solutions=True`:
      - First element: validity flag per candidate, shape `(num_envs, 512)`.
      - Second element: all candidate joint positions, shape `(num_envs, 512, num_joints)`.

  **Example:**

```python
  import torch
  # Reuse the FK pose above as the IK target.
  xpos = torch.tensor([[[ -1.0000,  -0.0000,   0.0000,   0.0000],
                        [ -0.0000,  -0.0000,  -1.0000,  -0.2561],
                        [  0.0000,  -1.0000,   0.0000,   1.4273],
                        [  0.0000,   0.0000,   0.0000,   1.0000]]],
                      device=solver.device)
  qpos_seed = torch.zeros((1, 6), device=solver.device)
  valid, ik_qpos = solver.get_ik(target_xpos=xpos, qpos_seed=qpos_seed)
  print("IK valid:", valid)
  print("IK solution:", ik_qpos)
  # IK valid: tensor([True], device='cuda:0')
  # IK solution: tensor([[ 0.0004, -1.5704, -0.0007, -1.5705, -0.0004, -0.0000]],
  #                     device='cuda:0')

  # Return every candidate instead of just the closest one:
  valid_all, all_solutions = solver.get_ik(
      target_xpos=xpos, qpos_seed=qpos_seed, return_all_solutions=True
  )
  print(all_solutions.shape)   # torch.Size([1, 512, 6])
```

* `set_tcp(self, tcp: np.ndarray)`
  Sets the Tool Center Point transform and precomputes its inverse so that IK
  targets are mapped into the flange frame before solving.

* `dh_matrix(theta_i, d_i, a_i, alpha_i) -> torch.Tensor` *(staticmethod)*
  Computes a single 4×4 Denavit–Hartenberg transformation matrix from the
  standard DH parameters `(theta, d, a, alpha)`. Useful for building custom FK
  chains or debugging the kinematic model.

## How It Works

1. **DH model**: The robot is parameterized by `(d1, a2, a3, d4, d5, d6)` and
   the fixed twists `alpha ∈ {+π/2, 0, -π/2}` (see the table above).
2. **Analytic solve**: For a target pose, `theta1` has two branches; for each,
   `theta5` (sign) and `theta6` are recovered, then `theta2/3/4` are solved in
   closed form with an elbow-up/down split — yielding 8 base solutions.
3. **Shift expansion**: Each base solution is expanded into 64 per-joint `±2π`
   shift variants that preserve the end-effector pose, producing 512 candidates.
4. **Validation**: A candidate is valid only if its forward kinematics matches
   the target (translation ≤ 1e-2, rotation ≤ 1e-1) *and* all joints lie within
   the joint limits. The Warp kernel runs one thread per target pose, so the
   whole batch is evaluated in parallel on the GPU.

## References

* [ur-analytic-ik (GitHub)](https://github.com/Victorlouisdg/ur-analytic-ik) — reference implementation of the analytic UR IK
* [DH parameters source (`dh_parameters.hh`)](https://github.com/Victorlouisdg/ur-analytic-ik/blob/main/src/ur_analytic_ik/dh_parameters.hh)
* [NVIDIA Warp Documentation](https://github.com/NVIDIA/warp)
