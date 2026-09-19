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
from embodichain.lab.sim.motion.solvers.ur_solver import URSolver, URSolverCfg

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

`ur_type` selects the robot variant; an unrecognized value raises `ValueError`
at configuration time.

## Solution selection

See {ref}`shared FK/IK conventions <motion-solver-conventions>` for the
common call pattern. UR IK produces eight analytical branches and expands
joint-period shifts into 512 candidates per pose. Each candidate must satisfy
joint limits and an FK check against the target.

| Mode | Validity shape | Joint solution shape |
|---|---|---|
| `return_all_solutions=False` | `(B,)` | `(B, 6)`; valid candidate closest to `qpos_seed`. |
| `return_all_solutions=True` | `(B, 512)` | `(B, 512, 6)`; inspect validity before selecting. |

```python
qpos_seed = torch.tensor(
    [[0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]], device=solver.device
)
target_pose = solver.get_fk(qpos_seed)
valid, ik_qpos = solver.get_ik(target_pose, qpos_seed=qpos_seed)
valid_all, candidates = solver.get_ik(
    target_pose, qpos_seed=qpos_seed, return_all_solutions=True
)
```

`set_tcp()` updates the tool transform used by FK and IK. Use `dh_matrix()`
when inspecting the model's individual DH transforms.

## References

* [ur-analytic-ik (GitHub)](https://github.com/Victorlouisdg/ur-analytic-ik) — reference implementation of the analytic UR IK
* [DH parameters source (`dh_parameters.hh`)](https://github.com/Victorlouisdg/ur-analytic-ik/blob/main/src/ur_analytic_ik/dh_parameters.hh)
* [NVIDIA Warp Documentation](https://github.com/NVIDIA/warp)
