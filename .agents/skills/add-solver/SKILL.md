---
name: add-solver
description: Use when adding a new kinematic (IK/FK) solver to EmbodiChain — implements the solver module, its Sphinx docs page, the unit test, and the benchmark entry together
---

# Add Solver

Scaffold a new kinematic solver — and its three required companion artifacts
(docs, unit test, benchmark) — following EmbodiChain's `SolverCfg` / `BaseSolver`
pattern. The reference implementation is the UR analytic solver; use it as the
gold standard for structure and style.

## When to Use

- User asks to "add a solver", "add an IK solver", "add a kinematic solver"
- A new robot family needs a closed-form / numerical / Warp-kernel IK backend
- The request names a solver that does not yet exist under
  `embodichain/lab/sim/solvers/`

## The Four Artifacts

A new solver is **not complete** until all four exist and pass. Each must follow
the conventions below.

| # | Artifact | Path |
|---|----------|------|
| 1 | Solver module | `embodichain/lab/sim/solvers/<name>_solver.py` |
| 2 | (GPU/Warp only) Warp kernel | `embodichain/utils/warp/kinematics/<name>_solver.py` |
| 3 | Sphinx docs page | `docs/source/overview/sim/solvers/<name>_solver.md` |
| 4 | Unit test | `tests/sim/solvers/test_<name>_solver.py` |
| 5 | Benchmark entry | extend `scripts/benchmark/robotics/kinematic_solver/run_benchmark.py` |

Plus two registration edits:

- Export the new `Cfg` + `Solver` classes from
  `embodichain/lab/sim/solvers/__init__.py`.
- Add the docs page to the toctree in
  `docs/source/overview/sim/solvers/index.rst`.

## Steps

### 1. Gather Solver Requirements

Ask the user (only what is not already stated):

1. **Solver name** (`<name>_solver`) and a one-line description.
2. **Robot family / kinematic type** — which robot(s) it targets, DOF.
3. **Approach** — analytical closed-form (UR, OPW, SRS), numerical/Jacobian
   (Pytorch, Differential, Pink, Pinocchio), or neural (NeuralIK).
4. **Backend** — pure PyTorch, NVIDIA Warp GPU kernel, or both.
5. **URDF / DH / OPW parameter source** — link or file the parameters come
   from (cite it in the docs, as the UR solver cites `ur-analytic-ik`).

### 2. Write the Solver Module

File: `embodichain/lab/sim/solvers/<name>_solver.py`

Required pieces (mirror `ur_solver.py`):

- Apache 2.0 copyright header (the 15-line block).
- `from __future__ import annotations` after the header.
- A `@configclass` config class `<Name>SolverCfg(SolverCfg)`:
  - Fields for any robot-specific parameters (DH params, OPW params, etc.).
  - A `__post_init__` that populates derived fields (e.g. per-variant DH
    parameters) and **raises `ValueError` for unknown variants** — fail fast.
  - An `init_solver(self, device=..., **kwargs) -> "<Name>Solver"` that
    constructs the solver and calls `solver.set_tcp(self._get_tcp_as_numpy())`.
- A `<Name>Solver(BaseSolver)` class:
  - `__init__(self, cfg, device, **kwargs)` calls `super().__init__(...)`,
    sets `self.dof`, and initializes solver-specific state.
  - Implements `get_ik(self, target_xpos, qpos_seed, return_all_solutions=False, **kwargs)`
    returning `(success, ik_qpos)` (or `(validity, all_solutions)` when
    `return_all_solutions=True`). Shapes follow the `BaseSolver.get_ik`
    contract.
  - Reuses `get_fk`, `set_tcp`, `get_qpos_limits`, etc. from `BaseSolver` —
    do **not** reimplement FK unless the solver has a custom chain.
  - Add static helpers (e.g. `dh_matrix`) only when genuinely needed.
- `__all__ = ["<Name>SolverCfg", "<Name>Solver"]`.

Template:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# ... (full Apache 2.0 header) ...
# ----------------------------------------------------------------------------

from __future__ import annotations

import torch
import numpy as np
from embodichain.utils import configclass
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver
from embodichain.data import get_data_path


@configclass
class FooSolverCfg(SolverCfg):
    # robot-specific parameters, with sensible defaults
    robot_type: str = "foo"
    urdf_path: str = get_data_path("Foo/foo.urdf")

    def __post_init__(self):
        super().__post_init__()
        if self.robot_type == "foo":
            ...
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    def init_solver(self, device: torch.device = torch.device("cpu"), **kwargs) -> "FooSolver":
        """Initialize the solver with the configuration.

        Args:
            device: The device to use for the solver. Defaults to CPU.
            **kwargs: Additional keyword arguments for solver initialization.

        Returns:
            FooSolver: An initialized solver instance.
        """
        solver = FooSolver(cfg=self, device=device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class FooSolver(BaseSolver):
    def __init__(self, cfg: FooSolverCfg, device: str, **kwargs):
        super().__init__(cfg, device, **kwargs)
        self.dof = 6
        # init solver-specific state / Warp params here

    def get_ik(self, target_xpos, qpos_seed, return_all_solutions: bool = False, **kwargs):
        """Compute target joint positions.

        Args:
            target_xpos (torch.Tensor): Target end-effector pose, shape (n_sample, 4, 4).
            qpos_seed (torch.Tensor): Reference joint positions, shape (n_sample, num_joints).
            return_all_solutions (bool): Return all candidates instead of the closest. Defaults to False.
            **kwargs: Additional arguments for future extensions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (success, target_joints).
        """
        ...
        return ik_validity, ik_qpos


__all__ = ["FooSolverCfg", "FooSolver"]
```

### 3. (GPU/Warp only) Write the Warp Kernel

For analytic solvers evaluated in batch on the GPU (UR, OPW, SRS), put the
`@wp.kernel` / `@wp.func` / `@wp.struct` definitions in
`embodichain/utils/warp/kinematics/<name>_solver.py` — **not** in the solver
module. The solver module imports the kernel and any param struct from there
and launches it with `wp.launch`.

Conventions (see `ur_solver.py` under `utils/warp/kinematics/`):

- Apache header + `from __future__ import annotations`.
- Define a `@wp.struct` for solver parameters (e.g. `URParam`) and pass it
  to the kernel as an input.
- One kernel thread per target pose (`dim=(n_sample,)`).
- Write candidate solutions and validity flags into preallocated Warp arrays,
  then convert back to torch with `wp.to_torch(...)`.
- Use `standardize_device_string(self.device)` from
  `embodichain.utils.device_utils` to get the Warp device string.

For pure-PyTorch / numerical solvers, skip this step entirely and implement
`get_ik` directly with torch ops.

### 4. Register in `__init__.py`

Add the import + keep `__all__` (if present) consistent in
`embodichain/lab/sim/solvers/__init__.py`:

```python
from .foo_solver import FooSolverCfg, FooSolver
```

### 5. Write the Docs Page

File: `docs/source/overview/sim/solvers/<name>_solver.md` — mirror the structure
of `ur_solver.md`:

1. `# <Name>Solver` — one-paragraph intro (what it solves, why it's fast / what
   approach it uses, the GPU/numerical backend).
2. Cite the reference implementation / parameter source with a link.
3. **Key Features** — bullet list.
4. **Kinematic model / parameters** (DH table, OPW params, etc.) as needed.
5. **Configuration** — a `python` code block constructing the `Cfg` and calling
   `cfg.init_solver(device=...)`. Use a `.. tip::` Sphinx directive for the
   one parameter that usually matters.
6. **Main Methods** — document `get_fk` (inherited), `get_ik` (with full
   signature, parameters, returns, and a runnable **Example** code block showing
   both `return_all_solutions=False` and `True`), `set_tcp`, and any static
   helpers. Use Google-style param lists and `+` bullets as in `ur_solver.md`.
7. **How It Works** — numbered explanation of the solve pipeline.
8. **References** — markdown links.

Then add the page to the toctree in `docs/source/overview/sim/solvers/index.rst`:

```rst
.. toctree::
    :maxdepth: 1

    pytorch_solver.md
    ...
    foo_solver.md
```

### 6. Write the Unit Test

File: `tests/sim/solvers/test_<name>_solver.py` — follow `test_ur_solver.py`
exactly:

- Apache header + `from __future__ import annotations` (after header).
- A `grid_sample_qpos_from_limits(...)` helper (reuse the one from
  `test_ur_solver.py`) to sample joint configs within limits with a safety
  margin.
- A `BaseSolverTest` class with:
  - `setup_simulation(self, device)` — builds a `SimulationManagerCfg`,
    a `RobotCfg` whose `solver_cfg={"arm": <Name>SolverCfg(...)}` uses the new
    solver, and adds the robot via `self.sim.add_robot(cfg=cfg)`.
  - `test_ik(self)` — the round-trip contract:
    1. Sample qpos from the robot's joint limits.
    2. `compute_batch_fk` → `fk_xpos` (both matrix and xyzquat forms).
    3. `compute_batch_ik` on both pose forms; assert the two IK results match.
    4. Re-run FK on the IK output; assert `sample_qpos ≈ ik_qpos` and
       `fk_xpos ≈ ik_xpos` with `atol=5e-3, rtol=5e-3`.
    5. Feed an unreachable pose; assert `res[0] == False` and the output shape.
  - `teardown_method` calling `self.sim.destroy()`.
- Two concrete subclasses driving `setup_method`:
  - `class TestFooSolverCUDA(BaseSolverTest): setup_method → "cuda"`
  - `class TestFooSolver(BaseSolverTest): setup_method → "cpu"`
- `if __name__ == "__main__":` block running `pytest.main(["-v", "-s", __file__])`.

### 7. Add the Benchmark Entry

Extend `scripts/benchmark/robotics/kinematic_solver/run_benchmark.py` (do **not**
create a separate benchmark file — the kinematic-solver benchmark is unified):

1. Add module-level constants for the new solver's joint limits, joint names,
   TCP, etc. (mirror `UR_LOWER_LIMITS` / `UR_UPPER_LIMITS` / `UR_TCP`).
2. If the solver is not yet in `SUPPORTED_SOLVERS`, add its short name there
   and update `_normalize_selected_solvers` / the `--solvers` argparse choices.
3. Write `_init_<name>_solver(device) -> <Name>Solver` and
   `_timed_<name>_ik_call(solver, fk_xpos, qpos_seed)` helpers, mirroring
   `_init_ur_solver` / `_timed_ur_ik_call` (3-iteration timing skipping the
   first run, `_sync_cuda()`, `_reset_peak_gpu_memory()`,
   `_memory_snapshot()`).
4. Write `benchmark_<name>_solver() -> (perf_rows, metric_rows)` mirroring
   `benchmark_ur_solver`: iterate `SAMPLE_SIZES`, run CPU (+ optional CUDA),
   verify accuracy via `get_pose_err`, and append rows for both the
   `Time & Memory` and `Success & Other Metrics` tables.
5. Wire it into `run_all_benchmarks()` behind an `if "<name>" in solvers_to_run:`
   guard, extending `perf_rows` / `metric_rows`. The leaderboard
   (`_build_leaderboard_rows`) and the markdown report
   (`_write_markdown_report`) are shared — the report must contain exactly the
   three tables (`Time & Memory`, `Success & Other Metrics`, `Leaderboard`).

For the full benchmark conventions (timing, memory, three-table report), defer
to the `benchmark` skill (`.agents/skills/benchmark/SKILL.md`).

### 8. Format and Verify

```bash
conda activate embodichain
black embodichain/lab/sim/solvers/<name>_solver.py
black embodichain/utils/warp/kinematics/<name>_solver.py   # if added
black tests/sim/solvers/test_<name>_solver.py
black scripts/benchmark/robotics/kinematic_solver/run_benchmark.py
```

Run the unit test (CPU class is enough for a quick check):

```bash
pytest tests/sim/solvers/test_<name>_solver.py::TestFooSolver -v
```

Smoke-run the benchmark for the new solver only:

```bash
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s <name>
```

Finally, run the `/pre-commit-check` skill to catch all CI violations locally.

## Code Style Checklist

- [ ] Apache 2.0 header on every new file (solver, warp kernel, test)
- [ ] `from __future__ import annotations` after the header
- [ ] `@configclass` on the Cfg, inheriting `SolverCfg`
- [ ] `__post_init__` raises `ValueError` on unknown variants
- [ ] `init_solver` constructs the solver and calls `set_tcp(...)`
- [ ] Solver inherits `BaseSolver`, sets `self.dof`, implements `get_ik`
- [ ] `__all__` declared in the solver module
- [ ] Google-style docstrings with Sphinx directives (`.. tip::`, `.. attention::`)
- [ ] Cfg + Solver exported from `embodichain/lab/sim/solvers/__init__.py`
- [ ] Docs page added to `index.rst` toctree
- [ ] Test covers FK↔IK round-trip + unreachable-pose failure, CPU & CUDA classes
- [ ] Benchmark entry added with CPU (+CUDA) timing, accuracy, and the 3-table report
- [ ] `black` run on all changed files

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Reimplementing FK / TCP / joint limits in the solver | Reuse `BaseSolver.get_fk`, `set_tcp`, `set_qpos_limits` |
| Putting the Warp kernel inside the solver module | Put `@wp.kernel`/`@wp.struct` in `embodichain/utils/warp/kinematics/<name>_solver.py` and import it |
| Not exporting the Cfg/Solver from `__init__.py` | Add the import line so `from embodichain.lab.sim.solvers import FooSolverCfg` works |
| Forgetting the docs toctree entry | Add the `.md` to `docs/source/overview/sim/solvers/index.rst` |
| Test only checks happy-path IK | Must verify FK↔IK round-trip equality AND an unreachable-pose returns `False` |
| Creating a separate benchmark file | Extend the unified `run_benchmark.py` instead |
| Skipping `black` / pre-commit | CI checks every file including tests and benchmarks |
| Missing `__post_init__` validation | Unknown robot variants must `raise ValueError` at config time |

## Quick Reference

| Action | Command / Path |
|--------|----------------|
| Reference solver | `embodichain/lab/sim/solvers/ur_solver.py` |
| Reference Warp kernel | `embodichain/utils/warp/kinematics/ur_solver.py` |
| Reference docs | `docs/source/overview/sim/solvers/ur_solver.md` |
| Reference test | `tests/sim/solvers/test_ur_solver.py` |
| Benchmark file | `scripts/benchmark/robotics/kinematic_solver/run_benchmark.py` |
| Python env | `conda activate embodichain` |
| Run test (CPU) | `pytest tests/sim/solvers/test_<name>_solver.py::TestFooSolver -v` |
| Run benchmark | `python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark -s <name>` |
| Format | `black <changed files>` |
| Pre-commit | `/pre-commit-check` |
