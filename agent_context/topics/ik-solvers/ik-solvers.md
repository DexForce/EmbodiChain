# IK Solvers

## Find the owner

| Change or question | Start here |
|---|---|
| Public solver/config names | `embodichain/lab/sim/motion/solvers/__init__.py` |
| FK/IK adapters, TCP, limits and capabilities | `embodichain/lab/sim/motion/solvers/base_solver.py` → `BaseSolver`, `SolverCfg` |
| Robot config decoding and solver binding | `embodichain/lab/sim/cfg/robot.py`, `embodichain/lab/sim/objects/robot.py` |
| Chain/reduced-model construction | `embodichain/lab/sim/utility/solver_utils.py` |
| Seed generation and retrieval | `embodichain/lab/sim/motion/solvers/qpos_seed_sampler.py`, `qpos_seed_sel_sampler.py` in that directory |
| Pink posture integration | `embodichain/lab/sim/motion/solvers/null_space_posture_task.py` |
| Analytical kernels | `embodichain/compute/kinematics/_warp/` |

`SolverCfg.from_dict()` resolves a configured type; `init_solver()` creates its
stateful instance. Robot binds solvers to control parts and synchronizes robot
limits. Read [robot configuration](../robot-system/robot-system.md) for part
resolution and asset/chain agreement, rather than duplicating that assembly
inside a solver.

Import public APIs through `embodichain.lab.sim.motion.solvers`. The motion
parent must remain lazy: solver exports must not import planners, the motion
generator or offline workspace analysis during Robot initialization.

## Choose the implementation

The registry is the inventory; inspect the matching module and config for
current options and dependencies. The main change sites are:

| Need | Solver module under `motion/solvers/` |
|---|---|
| Analytical structure for SRS, OPW or UR arms | `srs_solver.py`, `opw_solver.py`, `ur_solver.py` |
| General iterative PK solve and multi-start seeds | `pytorch_solver.py` |
| Reduced Pinocchio model or task-based QP | `pinocchio_solver.py`, `pink_solver.py` |
| Jacobian differential commands | `differential_solver.py` |
| Checkpoint-dependent learned inference | `neural_ik_solver.py` |

Use [add-solver](../../../.agents/skills/add-solver/SKILL.md) for a new solver.
Read [solver details](solver-details.md) for cross-call state, analytical buffer
ownership, Pink frame/limit adaptation or seed-cache invalidation.

## Frame and result contracts

Solvers consume poses relative to the configured chain root and apply the TCP
transform. Robot/integrations own arena/root conversion and robot/solver joint
ordering. Check those frames, device, radians and effective limits before
changing numerical convergence settings. Shared quaternion conventions belong
to [simulation](../simulation-system/simulation-system.md); external adapters
own any conversion.

`BaseSolver.get_fk_batch()` and `get_ik_batch()` flatten and restore arbitrary
leading batch axes. FK returns `(..., 4, 4)`; nearest IK returns boolean success
`(...)` and qpos `(..., dof)` without a candidate axis. Robot broadcasts root
transforms over those axes without materializing per-target copies.

Concrete `get_ik()` APIs retain their own candidate and failure shapes. Pink
returns `(N, 1, dof)`. Pytorch's normal nearest path retains a singleton
candidate axis and its all-solutions path returns `(N, samples, dof)`, but its
early all-targets-failed path returns `(N, dof)`. Do not squeeze blindly or
assume all-solutions masks share one universal shape. Preserve batch size one;
prefer the batch adapter when only the nearest solution is needed.

## Continuous batch IK

`Robot.compute_batch_ik(..., continuous=True)` first checks the typed
`supports_continuous_batch_ik` capability, before frame conversion or candidate
generation. Unsupported solvers raise `ValueError`; independent per-target IK
remains available without that capability.

A supporting solver supplies all-candidate IK and `_select_continuous_ik_path`.
The hook consumes candidates `(B, N, K, dof)`, validity `(B, N, K)` and initial
seeds `(B, dof)`, returning validity `(B, N)` and a path `(B, N, dof)`.
Candidate count `K` belongs to the solver; OPW implements this capability.

## Computation and state ownership

Stateful configuration, device buffers, seed policy, TCP and limit updates stay
in `lab/sim/motion/solvers`. Pure OPW/SRS/UR kernels belong to
`compute/kinematics/_warp/` and cannot import simulation modules.
`utils/warp/kinematics/*_solver.py` retains compatibility aliases only.

Analytical scratch reuse must not let later calls mutate earlier public results.
See [buffer lifecycle](solver-details.md#analytical-buffers-and-selection) before
changing allocation, stream behavior or nearest/all-solutions dispatch.

## Focused validation

| Changed boundary | Existing coverage |
|---|---|
| Typed capability and solver config decoding | `tests/sim/motion/solvers/test_base_solver.py` |
| Concrete algorithm and failure behavior | Matching `test_*_solver.py` under `tests/sim/motion/solvers/` |
| Batch/frame restoration, scratch reuse, CUDA streams | `tests/sim/motion/solvers/test_analytic_batching.py` |
| Seedless calls and seed database invalidation | `tests/sim/motion/solvers/test_default_qpos_seed.py`, `test_qpos_seed_sel_sampler.py` in that directory |
| Compute import boundary | `tests/compute/test_imports.py` |

For viewer-only IK adaptation, follow
[native gizmos](../sim-visualization/native-gizmos.md) instead of coupling solver
code to rendering. Algorithm parameters and benchmark setup remain in source
and `scripts/benchmark/robotics/kinematic_solver/`.
