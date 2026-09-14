# FEPSolver

`FEPSolver` implements numerical FEP configuration search for seven-revolute-joint
serial chains, including fixed geometric offsets. `backend="auto"` uses Torch
on CPU and a fused Warp kernel on CUDA. FK, TCP and joint limits use the shared
`BaseSolver` interface.

The search policy follows the local HolisticMotion reference
`src/kinematics/fep/FEPKinematics.cpp` and
`include/holistic_motion/kinematics/fep/FEPKinematics.h` at commit
`ef044f52e77bfcea044a25df4f703382e3f5b1c3`. Its current FEP implementation uses
numerical correction, with no validated analytical backend. This port adds no
HolisticMotion runtime dependency.

## Configuration and use

```python
import torch
from embodichain.lab.sim.motion.solvers import FEPSolverCfg

cfg = FEPSolverCfg(
    urdf_path="/path/to/robot.urdf",
    root_link_name="base_link",
    end_link_name="tool_link",
    solve_method="all_configurations",
)
solver = cfg.init_solver(device="cpu")
seed = torch.zeros(1, 7)
target = solver.get_fk(seed)  # TCP transform relative to chain root
success, qpos = solver.get_ik(target, seed)
validity, candidates = solver.get_ik(target, seed, return_all_solutions=True)
configuration = solver.get_configuration(seed)
success, qpos = solver.get_ik(
    target, seed, solve_method="configuration", configuration=configuration
)
```

Robot configurations select `class_type: FEPSolver` under `solver_cfg`.
If `joint_names` is supplied it must match the seven chain joints in order.
Models with prismatic joints or another active DOF count are rejected.

## Numerical backends

Select `backend="torch"` or `backend="warp"` at construction to force a backend.
The Warp backend runs FK, Jacobian, damped least squares and backtracking in
one kernel per numerical batch. Double precision protects the small normal
equations; returned positions remain float32. The first use compiles the kernel.

Native candidates are verified with the shared FK and the public position/angle
tolerances. A candidate satisfying these checks is accepted even if it missed
the kernel's stricter internal stopping threshold. Failed candidates retry the
Torch correction from the original seed. This verification does not relax the
public accuracy contract.

Native geometry is packed at solver construction, including fixed links and
oblique axes. Construct a new solver after modifying the chain. TCP and joint
limit updates remain live. Launches use the current Torch CUDA stream and
returned tensors remain valid across subsequent calls.

## Search methods

| `solve_method` | Behavior | Candidate count |
| --- | --- | --- |
| `seeded_numerical` | Correct the input seed with bounded damped least squares (default) | 1 |
| `compatibility` | Same numerical correction policy | 1 |
| `configuration` | Initialize and verify signs of joints 2, 4 and 6 | 1 |
| `all_configurations` | Try the eight sign combinations, deduplicate and rank valid solutions | 8 |
| `nearest_redundancy` | Scan joint-7 seeds outwards, stopping at the first successful radius | 1 |

Configuration descriptors contain shoulder/elbow/wrist signs (-1 or +1) and
the seventh-joint seed angle in radians. Zero belongs to the positive branch.
The redundancy angle initializes the solve; it is not held fixed. Branch
enumeration is local numerical search and can miss reachable solutions.
Redundancy scanning defaults to 5-degree increments through 180 degrees in
each direction. Both candidates at the first successful radius are ranked.

## Shapes and tolerances

`get_ik` accepts `(N, 4, 4)` or one `(4, 4)` transform and `(N, 7)` or one
broadcast `(7,)` seed. A missing seed defaults to zero. Nearest mode returns
`(N,)` validity and `(N, 7)` positions. All mode returns `(N, K)` validity
and `(N, K, 7)` positions. Read validity before using candidates: invalid
slots contain the input seed clamped to current limits. Ranking uses weighted
wrapped joint distance, with duplicate solutions masked out.

Correction defaults to 200 iterations, damping 0.01, and a maximum per-joint
update of 0.35 rad. Backtracking evaluates four scales together and retains
the lowest-error improving update. Converged rows remain unchanged; rows with
no improving update stop as failures. Acceptance
requires FK errors at most 1e-5 m and 1e-5 rad, finite joints and current joint
limits. `batch_size=None` automatically uses 256 targets on CPU or a budget of
16,384 candidate seeds on CUDA (2,048 targets for eight-configuration search).
An explicit positive `batch_size` overrides the number of targets per chunk.
The Torch numerical loop reuses accepted FK residuals and compacts active rows once
per iteration to reduce redundant FK calls and CUDA synchronization.
Changing TCP with `set_tcp` or limits with `set_qpos_limits` affects later
calls. Inherited `get_fk_batch` / `get_ik_batch` preserve leading batch axes.
Continuous batch path selection is unsupported.

## Validation and benchmark

Run `pytest tests/sim/motion/solvers/test_fep_solver.py` for CPU regression
coverage; add `--run-gpu -m gpu` for the CUDA smoke test.
Run `python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s fep`
for latency, memory and accuracy on a local synthetic offset 7R chain.
The benchmark uses nearby perturbed seeds and measures local correction.
Add `--fep-backends torch warp` to compare both backends on identical fixtures,
including the optional native CPU path. Timings exclude warm-up/compilation
and include public FK verification and any numerical fallback.

## API reference

See the [FEP API reference](../../../../api_reference/fep_solver.rst).
