# Solver state and adaptation contracts

Read when changing solver state across calls, frame adaptation or candidate
selection. Entry points and validation are in the [IK overview](ik-solvers.md).
Parameter defaults and numerical algorithms remain in concrete solver modules
and their tests, not in this page.

## Analytical buffers and selection

UR all-solutions and OPW use `solvers/_buffers.py`. `prepare_buffers(max_batch)`
reserves high-water capacity; allocation is lazy and grows for larger requests.
Public outputs remain independent of later calls, including all-solutions
outputs. A lock and CUDA events serialize reuse, Warp runs on the current Torch
stream, and storage dies with the solver. OPW packs current joint limits on each
call, so limit updates must not require buffer invalidation.

UR nearest-solution dispatch uses `ur_ik_nearest_kernel` without a full-batch
candidate tensor. Numerically ambiguous distances fall back to bounded chunks
using legacy Torch ranking, preserving candidate order and rounding at branch
bisectors. The ambiguity threshold selects the fallback; it must not decide a
tie. The all-solutions path preserves the full ordered periodic candidates,
including repeated representatives when no shifted value fits the limits.
Kernel formulas/counts and fallback chunk limits are source-owned in
`ur_solver.py` and `compute/kinematics/_warp/ur.py`.

SRS CPU and Warp paths must agree on reference frames, periodic representatives
and seed-dependent singularity handling. Runtime TCP and nearest-weight updates
synchronize both caches. Shape-reused scratch and all-solution deduplication
must preserve ordering without introducing quadratic GPU storage. Inspect
`srs_solver.py` and `compute/kinematics/_warp/srs.py` together for changes; use
`tests/sim/motion/solvers/test_srs_solver.py` for parity and singularity cases.
OPW continuous-path selection is the typed capability described in the overview,
not an untyped special case in Robot.

## Pink frame and limit adaptation

`pink_solver.py` accepts exactly one targeted variable `FrameTask` for its
single-pose IK API; additional fixed frame constraints belong in
`fixed_input_tasks`. Targets are TCP poses relative to `root_link_name`, even
when that root is offset from the URDF root. Remove TCP before setting the Pink
frame target and preserve the root transform in reduced-model adaptation.

Convergence is measured against that targeted frame. Stagnation, iteration-limit
and solver-exception exits report failure while preserving the input seed for
the corresponding target. Multi-target calls preserve per-target seeds and the
`(N,)`, `(N, 1, dof)` result contract.

Effective limits intersect URDF, user and runtime robot limits before mapping
into reduced Pinocchio joint order. Backtracking prioritizes FrameTask progress;
only controllable projected posture error is a secondary merit. Keep this
ordering when changing convergence or regularization controls.

`NullSpacePostureTask` computes manifold error in tangent space (`nv`), excludes
floating-base coordinates, and treats an empty joint selection as all actuated
joints. Pink initializes the task from either task list and updates its target
through `update_null_space_joint_targets()` in simulator joint order. Do not
substitute configuration-space subtraction or let posture improvement override
failure to progress on the controlled frame.

## Seeds and cached retrieval

`BaseSolver.get_default_qpos_seed()` returns the joint-range midpoint. Seedless
Pinocchio calls reset to that default instead of retaining the preceding call's
seed; Pink uses its own limit-projected neutral configuration. Inspect the
concrete seedless path rather than assuming zero is legal for every robot.

`QposSeedSampler` preserves the caller seed as the first multi-start sample;
remaining slots sample within limits. Target repetition must use the same
expanded batch ordering.

`QposSeedSelSampler` is an opt-in Pytorch multi-start extension. Its lazy Sobol FK
database stores flange poses, so retrieval receives the TCP-stripped target and
`set_tcp()` does not invalidate the database. Joint-limit changes do trigger a
rebuild. Without a target it falls back to ordinary random sampling. Optional
Jacobian ranking consumes the same pose/joint ordering. Analytical solvers do
not use this database; they still use seeds for their own branch selection.

Validate retrieval with `test_qpos_seed_sel_sampler.py`, seedless behavior with
`test_default_qpos_seed.py`, and Pink adaptation with `test_pink_solver.py`, all
under `tests/sim/motion/solvers/`. Keep measured performance figures in benchmark
results rather than treating one machine's timing as a solver contract.
