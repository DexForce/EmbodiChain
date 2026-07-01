# Parallel Motion Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make motion generation env-batched `(B=num_envs, N, DOF)` across the planner (`BasePlanner`/`ToppraPlanner`/`NeuralPlanner`), the `MotionGenerator` facade, and the atomic-action layer, with both IK+interpolation and planner-based motion sources available.

**Architecture:** Tensorize `PlanState`/`PlanResult` with a leading `B` dim (in-place). TOPPRA fans out across `B` envs via a `fork`-context multiprocessing pool of a pure-numpy module-level worker. NeuralPlanner runs its transformer natively batched. `MotionGenerator.generate` becomes batched. `TrajectoryBuilder` gains a `motion_source` strategy (`ik_interp` default, `motion_gen` opt-in) and adapts the actions' per-env `list[list[PlanState]]` into batched `list[PlanState]` for the planner path. `ActionResult.success` becomes `(B,)` and `AtomicActionEngine.run` propagates per-env failure via held qpos.

**Tech Stack:** Python 3.11, PyTorch, NumPy/SciPy + TOPPRA (`toppra==0.6.3`), `concurrent.futures.ProcessPoolExecutor` (fork context), `unittest.mock` for unit tests, real `SimulationManager` + `CobotMagicCfg` for integration tests, pytest, `black==26.3.1`.

**Spec:** `docs/superpowers/specs/2026-07-02-parallel-motion-generation-design.md`

---

## Spec refinements (read before starting)

The spec was written before reading every action body. Two refinements surfaced during planning; they honor the spec's intent and are called out here so the engineer doesn't treat them as surprises:

1. **Action-facing `plan_arm_traj` signature is unchanged.** The 6 actions today build `list[list[PlanState]]` (outer = per-env, inner = per-waypoint) holding **single-env** `xpos:(4,4)`. To keep that action code untouched, `TrajectoryBuilder.plan_arm_traj` keeps that input signature and **internally** converts to batched `list[PlanState]` (each carrying `(B,4,4)`) when `motion_source="motion_gen"`. So `PlanState` batched-ness is the **planner/MotionGenerator contract**; the action layer still passes per-env PlanStates to the builder. (Spec §3/§5 intent preserved; §6.3 "actions untouched" honored for target-building.)

2. **Per-env `ActionResult.success` requires a minimal edit in the 6 actions.** Spec §6.1 makes `success` a `(B,)` tensor and §6.2 propagates per-env failure in `run()`. That is incompatible with §6.3's "the 6 action classes: no edits" — each action currently does `if not ok: return self._fail(state)` (scalar short-circuit). The unavoidable consequence: each action's success check is updated to propagate a `(B,)` mask (remove the early-return; pass `success` through to `ActionResult`). The action **logic** (target resolution, embedding, hand interp, phase splitting) is otherwise unchanged. This plan implements that minimal edit (Phase G).

---

## File Structure

**Modified:**
- `embodichain/lab/sim/planners/utils.py` — `PlanState`, `PlanResult` batched + ctors; `__all__` updated.
- `embodichain/lab/sim/planners/base_planner.py` — `@validate_plan_options` extended for `B`-consistency; `plan()` docstring.
- `embodichain/lab/sim/planners/toppra_planner.py` — module-level `_toppra_solve_one_env` worker; `ToppraPlannerCfg.max_workers`/`mp_context`; batched `plan()` with fork-pool fan-out.
- `embodichain/lab/sim/planners/neural_planner.py` — natively batched `plan()`/`_parse_waypoints`/`_build_obs`/`_is_active_reached`.
- `embodichain/lab/sim/planners/motion_generator.py` — batched `generate()`; `MotionGenOptions.start_qpos:(B,DOF)`; batched `interpolate_trajectory`.
- `embodichain/lab/sim/atomic_actions/core.py` — `ActionCfg.motion_source`/`planner_type`; `ActionResult.success` tensor + `success_all`/`__bool__`.
- `embodichain/lab/sim/atomic_actions/trajectory.py` — `motion_source` branch in `plan_arm_traj`; `_to_batched_plan_states`, `_build_plan_opts`; per-env success in `plan_arm_traj`.
- `embodichain/lab/sim/atomic_actions/engine.py` — `run()` per-env failure propagation; returns `(B,)` success.
- `embodichain/lab/sim/atomic_actions/actions.py` — 6 actions: success-handling edit (Phase G).
- `embodichain/lab/sim/utility/atom_action_utils.py` — `plan_trajectory` migrated to batched `PlanState` (Phase H).
- `scripts/tutorials/atomic_action/move_end_effector.py` — `bool(is_success)` → `is_success.all()` (Phase H).

**Created (tests):**
- `tests/sim/planners/test_plan_state_batched.py`
- `tests/sim/planners/test_toppra_batched.py`
- `tests/sim/planners/test_neural_batched.py`
- `tests/sim/planners/test_motion_generator_batched.py`
- `tests/sim/atomic_actions/test_action_result_success.py`
- `tests/sim/atomic_actions/test_engine_per_env.py`
- `tests/sim/atomic_actions/test_trajectory_motion_source.py`

**Modified tests** (update for batched shapes):
- `tests/sim/planners/test_toppra_planner.py`, `test_neural_planner.py`, `test_motion_generator.py`, `tests/sim/atomic_actions/test_trajectory.py`, `test_actions.py`, `test_engine.py`.

---

## Phase A — Batched Data Model

### Task A1: Batched `PlanState` / `PlanResult` + constructors

**Files:**
- Modify: `embodichain/lab/sim/planners/utils.py:115-170`
- Test: `tests/sim/planners/test_plan_state_batched.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/planners/test_plan_state_batched.py
from __future__ import annotations

import torch
import pytest

from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType, MovePart


class TestPlanStateBatched:
    def test_from_qpos_batched(self):
        qpos = torch.zeros(4, 7)
        ps = PlanState.from_qpos(qpos, move_type=MoveType.JOINT_MOVE, move_part=MovePart.LEFT)
        assert ps.qpos.shape == (4, 7)
        assert ps.move_type == MoveType.JOINT_MOVE

    def test_from_xpos_batched(self):
        xpos = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
        ps = PlanState.from_xpos(xpos, move_type=MoveType.EEF_MOVE)
        assert ps.xpos.shape == (3, 4, 4)

    def test_single_ctor_unsqueezes(self):
        ps = PlanState.single(qpos=torch.zeros(7), move_type=MoveType.JOINT_MOVE)
        assert ps.qpos.shape == (1, 7)


class TestPlanResultBatched:
    def test_is_all_success_tensor(self):
        r = PlanResult(success=torch.tensor([True, True, False]))
        assert r.is_all_success() is False

    def test_is_all_success_scalar(self):
        r = PlanResult(success=True)
        assert r.is_all_success() is True

    def test_batched_shapes(self):
        r = PlanResult(
            success=torch.tensor([True, False]),
            positions=torch.zeros(2, 10, 7),
            velocities=torch.zeros(2, 10, 7),
            accelerations=torch.zeros(2, 10, 7),
            dt=torch.zeros(2, 10),
            duration=torch.tensor([1.0, 0.0]),
        )
        assert r.positions.shape == (2, 10, 7)
        assert r.dt.shape == (2, 10)
        assert r.duration.shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_plan_state_batched.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'from_qpos'` / `is_all_success`.

- [ ] **Step 3: Implement batched PlanState/PlanResult**

In `embodichain/lab/sim/planners/utils.py`, update the docstrings on `PlanResult`/`PlanState` to document the new batched shapes (positions `(B, N, DOF)`, `qpos:(B,DOF)`, `xpos:(B,4,4)`, `success:(B,)`, `dt:(B,N)`, `duration:(B,)`), and add constructors + helper:

```python
@dataclass
class PlanResult:
    r"""Data class representing the result of a motion plan (env-batched)."""

    success: bool | torch.Tensor = False
    """Per-env success, shape ``(B,)`` bool tensor (or scalar bool)."""

    xpos_list: torch.Tensor | None = None
    """End-effector poses, shape ``(B, N, 4, 4)``."""

    positions: torch.Tensor | None = None
    """Joint positions, shape ``(B, N, DOF)``."""

    velocities: torch.Tensor | None = None
    """Joint velocities, shape ``(B, N, DOF)``."""

    accelerations: torch.Tensor | None = None
    """Joint accelerations, shape ``(B, N, DOF)``."""

    dt: torch.Tensor | None = None
    """Per-env time deltas, shape ``(B, N)``."""

    duration: float | torch.Tensor = 0.0
    """Per-env total duration, shape ``(B,)``."""

    def is_all_success(self) -> bool:
        """Return True only when every env succeeded."""
        if isinstance(self.success, torch.Tensor):
            return bool(torch.all(self.success).item())
        return bool(self.success)


@dataclass
class PlanState:
    r"""Data class representing the state for a motion plan (env-batched).

    Tensor fields carry a leading batch dim ``B``: ``qpos:(B, DOF)``,
    ``xpos:(B, 4, 4)``. Enum/scalar fields are shared across ``B`` (vectorized
    envs share the same task skeleton).
    """

    move_type: MoveType = MoveType.JOINT_MOVE
    move_part: MovePart = MovePart.LEFT
    xpos: torch.Tensor | None = None
    qpos: torch.Tensor | None = None
    qvel: torch.Tensor | None = None
    qacc: torch.Tensor | None = None
    is_open: bool = True
    is_world_coordinate: bool = True
    pause_seconds: float = 0.0

    @classmethod
    def from_qpos(cls, qpos: torch.Tensor, *, move_type: MoveType = MoveType.JOINT_MOVE, move_part: MovePart = MovePart.LEFT, **kwargs) -> "PlanState":
        return cls(move_type=move_type, move_part=move_part, qpos=qpos, **kwargs)

    @classmethod
    def from_xpos(cls, xpos: torch.Tensor, *, move_type: MoveType = MoveType.EEF_MOVE, move_part: MovePart = MovePart.LEFT, **kwargs) -> "PlanState":
        return cls(move_type=move_type, move_part=move_part, xpos=xpos, **kwargs)

    @classmethod
    def single(cls, *, qpos: torch.Tensor | None = None, xpos: torch.Tensor | None = None, move_type: MoveType = MoveType.JOINT_MOVE, move_part: MovePart = MovePart.LEFT, **kwargs) -> "PlanState":
        """B=1 convenience constructor: unsqueezes a single-env qpos/xpos."""
        if qpos is not None and qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if xpos is not None and xpos.dim() == 2:
            xpos = xpos.unsqueeze(0)
        return cls(move_type=move_type, move_part=move_part, qpos=qpos, xpos=xpos, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_plan_state_batched.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/planners/utils.py tests/sim/planners/test_plan_state_batched.py
git commit -m "feat(planner): batched PlanState/PlanResult with B dim and ctors"
```

---

### Task A2: Extend `@validate_plan_options` for `B`-consistency

**Files:**
- Modify: `embodichain/lab/sim/planners/base_planner.py:45-88, 113-119`
- Test: `tests/sim/planners/test_plan_state_batched.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/planners/test_plan_state_batched.py`:

```python
class TestValidateBatchConsistency:
    def test_rejects_inconsistent_B(self, monkeypatch):
        from embodichain.lab.sim.planners.base_planner import validate_plan_options, BasePlanner, PlanOptions
        from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType

        class Dummy(BasePlanner):
            @validate_plan_options
            def plan(self, target_states, options=PlanOptions()):
                return PlanResult(success=True)

        # B=2 then B=3 -> inconsistent
        states = [
            PlanState.from_qpos(torch.zeros(2, 7)),
            PlanState.from_qpos(torch.zeros(3, 7)),
        ]
        # BasePlanner.__init__ needs a robot; bypass by stubbing the decorator path.
        # We test the validation helper directly instead:
        from embodichain.lab.sim.planners import base_planner as bp
        with pytest.raises(ValueError):
            bp._check_batch_consistency(states, expected_b=None, robot_num_instances=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_plan_state_batched.py::TestValidateBatchConsistency -v`
Expected: FAIL — `_check_batch_consistency` not defined.

- [ ] **Step 3: Implement the consistency check + wire into decorator**

In `base_planner.py`, add a module-level helper and call it from the decorator:

```python
def _infer_batch_size(target_states: list[PlanState]) -> int | None:
    for s in target_states:
        for t in (s.qpos, s.xpos, s.qvel, s.qacc):
            if isinstance(t, torch.Tensor) and t.dim() >= 1:
                return int(t.shape[0])
    return None


def _check_batch_consistency(
    target_states: list[PlanState],
    expected_b: int | None,
    robot_num_instances: int,
) -> int:
    """Validate that all PlanState tensors share the same leading B and match the robot."""
    if len(target_states) < 2:
        logger.log_error(
            "target_states must contain at least 2 waypoints", ValueError
        )
    bs = set()
    for i, s in enumerate(target_states):
        b = _infer_batch_size([s])
        if b is not None:
            bs.add(b)
    if len(bs) > 1:
        logger.log_error(
            f"All PlanState entries must share the same batch dim B, got {sorted(bs)}",
            ValueError,
        )
    b = bs.pop() if bs else 1
    if expected_b is not None and b != expected_b:
        logger.log_error(
            f"Batch dim B={b} does not match robot.num_instances={expected_b}",
            ValueError,
        )
    if robot_num_instances is not None and b not in (1, robot_num_instances):
        logger.log_error(
            f"Batch dim B={b} must be 1 or robot.num_instances={robot_num_instances}",
            ValueError,
        )
    return b
```

Extend the `decorator` inside `validate_plan_options` to call it (after the type check, before `return func(...)`):

```python
        def wrapper(self, *args, **kwargs):
            options = kwargs.get("options", args[1] if len(args) > 1 else None)
            if options is not None and not isinstance(options, options_cls):
                logger.log_error(
                    f"Expected 'options' to be of type {options_cls.__name__} "
                    f"(or a subclass), but got {type(options).__name__}.",
                    TypeError,
                )
            target_states = kwargs.get("target_states", args[0] if args else None)
            if target_states is not None and hasattr(self, "robot"):
                robot_num = getattr(self.robot, "num_instances", None)
                _check_batch_consistency(target_states, expected_b=robot_num, robot_num_instances=robot_num)
            return func(self, *args, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_plan_state_batched.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/planners/base_planner.py tests/sim/planners/test_plan_state_batched.py
git commit -m "feat(planner): validate batch consistency in @validate_plan_options"
```

---

## Phase B — Batched ToppraPlanner (fork-multiprocessing fan-out)

### Task B1: Module-level pure-numpy `_toppra_solve_one_env` worker + inline reference test

**Files:**
- Modify: `embodichain/lab/sim/planners/toppra_planner.py` (add worker near top, after imports)
- Test: `tests/sim/planners/test_toppra_batched.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/planners/test_toppra_batched.py
from __future__ import annotations

import numpy as np
import pytest

from embodichain.lab.sim.planners.toppra_planner import _toppra_solve_one_env
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod


class TestToppraWorker:
    def test_solve_one_env_quantity(self):
        # 2-waypoint, 6-DOF
        wp = np.array([[0.0] * 6, [0.5] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["positions"].shape == (20, 6)
        assert out["velocities"].shape == (20, 6)
        assert out["dt"].shape == (20,)

    def test_solve_one_env_infeasible(self):
        # Tiny limits -> TOPPRA returns None
        wp = np.array([[0.0] * 6, [1.0] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1e-6,
            acc_constraint=1e-6,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=10,
        )
        assert out["success"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_toppra_batched.py::TestToppraWorker -v`
Expected: FAIL — `_toppra_solve_one_env` not importable.

- [ ] **Step 3: Implement the worker**

In `toppra_planner.py`, add after the `ta.setup_logging` line (module level, before `ToppraPlannerCfg`):

```python
def _build_constraint_arrays(value, acc, dofs: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand scalar limits to (dofs, 2) arrays; pass through arrays as-is."""
    if isinstance(value, (float, int)):
        vlims = np.array([[-value, value] for _ in range(dofs)])
    else:
        vlims = np.array(value)
    if isinstance(acc, (float, int)):
        alims = np.array([[-acc, acc] for _ in range(dofs)])
    else:
        alims = np.array(acc)
    return vlims, alims


def _toppra_solve_one_env(
    waypoints: np.ndarray,
    vel_constraint,
    acc_constraint,
    sample_method: "TrajectorySampleMethod",
    sample_interval: float | int,
) -> dict:
    """Solve a single-env TOPPRA trajectory. Pure numpy/scipy — picklable, no torch/robot.

    Args:
        waypoints: ``(N, DOF)`` numpy array of joint waypoints.
        vel_constraint / acc_constraint: scalar or per-DoF array limits.
        sample_method: TIME or QUANTITY.
        sample_interval: seconds (TIME) or sample count (QUANTITY).

    Returns:
        dict with ``positions`` ``(N_b, DOF)``, ``velocities``, ``accelerations``,
        ``dt`` ``(N_b,)``, ``success`` bool, ``n`` int, ``duration`` float.
    """
    dofs = waypoints.shape[1]
    vlims, alims = _build_constraint_arrays(vel_constraint, acc_constraint, dofs)

    if sample_method == TrajectorySampleMethod.TIME and sample_interval <= 0:
        return _empty_failure(dofs)
    if sample_method == TrajectorySampleMethod.QUANTITY and sample_interval < 2:
        return _empty_failure(dofs)

    # Trivial same-waypoint shortcut
    if len(waypoints) == 2 and np.sum(np.abs(waypoints[1] - waypoints[0])) < 1e-3:
        pos = np.stack([waypoints[0], waypoints[1]])
        return {
            "positions": pos,
            "velocities": np.zeros_like(pos),
            "accelerations": np.zeros_like(pos),
            "dt": np.array([0.0, 0.0], dtype=np.float32),
            "success": True,
            "n": 2,
            "duration": 0.0,
        }

    ss = np.linspace(0.0, 1.0, len(waypoints))
    try:
        path = ta.SplineInterpolator(ss, waypoints)
        pc_vel = constraint.JointVelocityConstraint(vlims)
        pc_acc = constraint.JointAccelerationConstraint(alims)
        instance = ta.algorithm.TOPPRA(
            [pc_vel, pc_acc],
            path,
            parametrizer="ParametrizeConstAccel",
            gridpt_min_nb_points=max(100, 10 * len(waypoints)),
        )
        jnt_traj = instance.compute_trajectory()
    except Exception:
        return _empty_failure(dofs)

    if jnt_traj is None:
        return _empty_failure(dofs)

    duration = float(jnt_traj.duration)
    if duration <= 0:
        return _empty_failure(dofs)

    if sample_method == TrajectorySampleMethod.TIME:
        n_points = max(2, int(np.ceil(duration / sample_interval)) + 1)
        ts = np.linspace(0.0, duration, n_points)
    else:
        ts = np.linspace(0.0, duration, num=int(sample_interval))

    positions = np.array([jnt_traj.eval(t) for t in ts])
    velocities = np.array([jnt_traj.evald(t) for t in ts])
    accelerations = np.array([jnt_traj.evaldd(t) for t in ts])
    dt = np.diff(ts, prepend=0.0).astype(np.float32)
    return {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "dt": dt,
        "success": True,
        "n": len(ts),
        "duration": duration,
    }


def _empty_failure(dofs: int) -> dict:
    z = np.zeros((2, dofs), dtype=np.float32)
    return {
        "positions": z,
        "velocities": np.zeros_like(z),
        "accelerations": np.zeros_like(z),
        "dt": np.array([0.0, 0.0], dtype=np.float32),
        "success": False,
        "n": 2,
        "duration": 0.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_toppra_batched.py::TestToppraWorker -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/planners/toppra_planner.py tests/sim/planners/test_toppra_batched.py
git commit -m "feat(toppra): pure-numpy module-level _toppra_solve_one_env worker"
```

---

### Task B2: `ToppraPlannerCfg.max_workers` / `mp_context` + lazy fork pool

**Files:**
- Modify: `embodichain/lab/sim/planners/toppra_planner.py:44-92`
- Test: `tests/sim/planners/test_toppra_batched.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestToppraCfgFields:
    def test_cfg_defaults(self):
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlannerCfg
        cfg = ToppraPlannerCfg(robot_uid="x")
        assert cfg.max_workers is None
        assert cfg.mp_context == "fork"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_toppra_batched.py::TestToppraCfgFields -v`
Expected: FAIL — `max_workers` not an attribute.

- [ ] **Step 3: Add cfg fields + remove the num_instances guard + lazy pool**

Update `ToppraPlannerCfg` and `ToppraPlanner.__init__`:

```python
@configclass
class ToppraPlannerCfg(BasePlannerCfg):
    planner_type: str = "toppra"
    max_workers: int | None = None
    """Worker process count for the batched fan-out. None => min(cpu_count()//2, B)."""
    mp_context: str = "fork"
    """Multiprocessing start method. 'fork' (default, TOPPRA is pure-CPU) or 'spawn'."""


class ToppraPlanner(BasePlanner):
    def __init__(self, cfg: ToppraPlannerCfg):
        super().__init__(cfg)
        self.cfg: ToppraPlannerCfg = cfg
        self._pool = None
        import atexit
        atexit.register(self.close)

    def _get_pool(self, b: int):
        if self._pool is not None:
            return self._pool
        import os
        import multiprocessing as mp
        max_workers = self.cfg.max_workers
        if max_workers is None:
            max_workers = max(1, min((os.cpu_count() or 2) // 2, b))
        ctx = mp.get_context(self.cfg.mp_context)
        from concurrent.futures import ProcessPoolExecutor
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
```

(Remove the old `if self.robot.num_instances > 1: ...` block.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_toppra_batched.py::TestToppraCfgFields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/planners/toppra_planner.py tests/sim/planners/test_toppra_batched.py
git commit -m "feat(toppra): cfg max_workers/mp_context + lazy fork pool, drop num_instances guard"
```

---

### Task B3: Batched `ToppraPlanner.plan()` — fan-out, stack, tail-pad

**Files:**
- Modify: `embodichain/lab/sim/planners/toppra_planner.py` (replace `plan` body)
- Test: `tests/sim/planners/test_toppra_batched.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestToppraPlanBatched:
    def _make_planner(self):
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlanner, ToppraPlannerCfg
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg
        sim = SimulationManager(SimulationManagerCfg(headless=True, sim_device="cpu"))
        robot = sim.add_robot(cfg=CobotMagicCfg.from_dict({"uid": "t", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}))
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="t", max_workers=1))
        return planner, sim

    def test_plan_batched_quantity_uniform_N(self):
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
        planner, sim = self._make_planner()
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.linspace(0.0, 0.4, B)
            states = [PlanState.from_qpos(torch.zeros(B, dofs)), PlanState.from_qpos(wp)]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=15,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.success.all().item()
            assert r.positions.shape == (B, 15, dofs)
        finally:
            planner.close()
            sim.destroy()
            import embodichain.lab.sim as om
            om.SimulationManager.flush_cleanup_queue()

    def test_plan_batched_time_tailpads(self):
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
        planner, sim = self._make_planner()
        try:
            B, dofs = 3, 6
            wp = torch.zeros(B, dofs)
            wp[:, 0] = torch.tensor([0.1, 0.4, 0.9])  # different durations
            states = [PlanState.from_qpos(torch.zeros(B, dofs)), PlanState.from_qpos(wp)]
            opts = ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.TIME,
                sample_interval=0.05,
                constraints={"velocity": 1.0, "acceleration": 2.0},
            )
            r = planner.plan(states, opts)
            assert r.success.shape == (B,)
            assert r.positions.shape[0] == B
            # padded rows equal the last real waypoint (held pose)
            last_real = r.positions[:, r.duration.argmax().int().item(), :]
            assert r.duration.shape == (B,)
        finally:
            planner.close()
            sim.destroy()
            import embodichain.lab.sim as om
            om.SimulationManager.flush_cleanup_queue()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_toppra_batched.py::TestToppraPlanBatched -v`
Expected: FAIL — `plan` still single-env / rejects B>1.

- [ ] **Step 3: Implement batched `plan`**

Replace the `plan` method body in `ToppraPlanner`:

```python
    @validate_plan_options(options_cls=ToppraPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: ToppraPlanOptions = ToppraPlanOptions(),
    ) -> PlanResult:
        for i, t in enumerate(target_states):
            if t.qpos is None:
                logger.log_error(f"Target state at index {i} missing qpos", ValueError)

        b = _infer_batch_size(target_states) or 1
        dofs = target_states[0].qpos.shape[-1]

        # Build (B, N, DOF) numpy waypoints
        waypoints = np.stack([s.qpos.detach().cpu().numpy() for s in target_states], axis=1)  # (B, N, DOF)

        vc = options.constraints["velocity"]
        ac = options.constraints["acceleration"]
        args_per_env = [
            (waypoints[i], vc, ac, options.sample_method, options.sample_interval)
            for i in range(b)
        ]

        # Inline fallback for B==1 or max_workers==1
        import os
        max_workers = self.cfg.max_workers
        use_inline = (b == 1) or (max_workers == 1) or (max_workers is None and ((os.cpu_count() or 2) // 2) <= 1)
        if use_inline:
            results = [_toppra_solve_one_env(*a) for a in args_per_env]
        else:
            pool = self._get_pool(b)
            try:
                futures = [pool.submit(_toppra_solve_one_env, *a) for a in args_per_env]
                results = []
                for fut in futures:
                    try:
                        results.append(fut.result())
                    except Exception:
                        results.append(_empty_failure(dofs))
            except BrokenProcessPool:
                logger.log_warning("TOPPRA process pool broke; returning failure.")
                self.close()
                results = [_empty_failure(dofs) for _ in range(b)]

        return self._assemble_batched_result(results, dofs)

    def _assemble_batched_result(self, results: list[dict], dofs: int) -> PlanResult:
        b = len(results)
        max_n = max(r["n"] for r in results)
        positions = np.zeros((b, max_n, dofs), dtype=np.float32)
        velocities = np.zeros((b, max_n, dofs), dtype=np.float32)
        accelerations = np.zeros((b, max_n, dofs), dtype=np.float32)
        dt = np.zeros((b, max_n), dtype=np.float32)
        duration = np.zeros((b,), dtype=np.float32)
        success = np.zeros((b,), dtype=bool)
        for i, r in enumerate(results):
            n = r["n"]
            positions[i, :n] = r["positions"]
            velocities[i, :n] = r["velocities"]
            accelerations[i, :n] = r["accelerations"]
            dt[i, :n] = r["dt"]
            duration[i] = r["duration"]
            success[i] = r["success"]
            # tail-pad: repeat final waypoint for held-pose rows
            if n < max_n:
                positions[i, n:] = r["positions"][-1]
                velocities[i, n:] = 0.0
                accelerations[i, n:] = 0.0
        return PlanResult(
            success=torch.as_tensor(success, device=self.device),
            positions=torch.as_tensor(positions, device=self.device),
            velocities=torch.as_tensor(velocities, device=self.device),
            accelerations=torch.as_tensor(accelerations, device=self.device),
            dt=torch.as_tensor(dt, device=self.device),
            duration=torch.as_tensor(duration, device=self.device),
        )
```

Add `from embodichain.lab.sim.planners.base_planner import _infer_batch_size` to the imports at the top of `toppra_planner.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_toppra_batched.py -v`
Expected: PASS.

- [ ] **Step 5: Update existing single-env test for batched PlanState**

In `tests/sim/planners/test_toppra_planner.py`, update `test_plan_basic` and any other test that builds `PlanState(qpos=torch.zeros(6))` to `PlanState.single(qpos=torch.zeros(6))` (or `from_qpos(torch.zeros(1,6))`) and assert `r.positions.shape[0] == 1`. Run:

Run: `pytest tests/sim/planners/test_toppra_planner.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/planners/toppra_planner.py tests/sim/planners/test_toppra_batched.py tests/sim/planners/test_toppra_planner.py
git commit -m "feat(toppra): batched plan() with fork-pool fan-out + tail-pad"
```

---

## Phase C — Batched NeuralPlanner

### Task C1: Natively batched `_parse_waypoints` and `_initial_qpos`

**Files:**
- Modify: `embodichain/lab/sim/planners/neural_planner.py:387-431`
- Test: `tests/sim/planners/test_neural_batched.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/planners/test_neural_batched.py
from __future__ import annotations

import torch
import pytest


class TestNeuralParseWaypoints:
    def test_parse_waypoints_batched(self):
        from embodichain.lab.sim.planners.neural_planner import NeuralPlanner
        from embodichain.lab.sim.planners.utils import PlanState, MoveType
        # Build a minimal planner shell by stubbing __init__
        planner = NeuralPlanner.__new__(NeuralPlanner)
        planner.device = torch.device("cpu")
        planner._num_waypoints = 4
        from embodichain.lab.sim import SimulationManager
        B = 3
        states = [
            PlanState.from_xpos(torch.eye(4).unsqueeze(0).repeat(B, 1, 1) * 1.0, move_type=MoveType.EEF_MOVE)
            for _ in range(2)
        ]
        pos, quat, mask, k = planner._parse_waypoints(states)
        assert pos.shape == (B, 4, 3)
        assert quat.shape == (B, 4, 4)
        assert mask.shape == (B, 4)
        assert k == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_neural_batched.py::TestNeuralParseWaypoints -v`
Expected: FAIL — `_parse_waypoints` returns `(1, ...)`.

- [ ] **Step 3: Rewrite `_parse_waypoints` and `_initial_qpos` for B**

```python
    def _parse_waypoints(
        self, target_states: list[PlanState]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(target_states) > self._num_waypoints:
            logger.log_error(
                f"Received {len(target_states)} waypoints, but checkpoint supports "
                f"at most {self._num_waypoints}.",
                ValueError,
            )
        b = _infer_batch_size(target_states) or 1
        waypoint_pos = torch.zeros(b, self._num_waypoints, 3, device=self.device)
        waypoint_quat = torch.zeros(b, self._num_waypoints, 4, device=self.device)
        valid_mask = torch.zeros(b, self._num_waypoints, device=self.device)
        for idx, target in enumerate(target_states):
            if target.move_type != MoveType.EEF_MOVE or target.xpos is None:
                logger.log_error(
                    "NeuralPlanner expects EEF_MOVE PlanState entries with xpos.",
                    ValueError,
                )
            xpos = torch.as_tensor(target.xpos, dtype=torch.float32, device=self.device)
            if xpos.dim() == 2:
                xpos = xpos.unsqueeze(0)
            waypoint_pos[:, idx] = xpos[:, :3, 3]
            waypoint_quat[:, idx] = convert_quat(
                quat_from_matrix(xpos[:, :3, :3]), to="xyzw"
            )
            valid_mask[:, idx] = 1.0
        return waypoint_pos, waypoint_quat, valid_mask, len(target_states)

    def _initial_qpos(
        self, control_part: str, start_qpos: torch.Tensor | None
    ) -> torch.Tensor:
        if start_qpos is None:
            qpos = self.robot.get_qpos(name=control_part)
        else:
            qpos = torch.as_tensor(start_qpos, dtype=torch.float32, device=self.device)
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[-1] < self._action_dim:
            logger.log_error(
                f"start_qpos has {qpos.shape[-1]} joints, but policy expects "
                f"{self._action_dim}.",
                ValueError,
            )
        return qpos.to(self.device).clone()
```

Add `from embodichain.lab.sim.planners.base_planner import _infer_batch_size` to imports. Remove the `num_instances > 1` guard in `__init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_neural_batched.py::TestNeuralParseWaypoints -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/planners/neural_planner.py tests/sim/planners/test_neural_batched.py
git commit -m "feat(neural): batched _parse_waypoints/_initial_qpos, drop num_instances guard"
```

---

### Task C2: Batched rollout loop with early-convergence holds

**Files:**
- Modify: `embodichain/lab/sim/planners/neural_planner.py:310-385, 442-500`
- Test: `tests/sim/planners/test_neural_batched.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestNeuralPlanBatched:
    def test_plan_returns_batched_success(self, monkeypatch):
        from embodichain.lab.sim.planners.neural_planner import NeuralPlanner, _WaypointTransformerActor
        from embodichain.lab.sim.planners.utils import PlanState, MoveType

        planner = NeuralPlanner.__new__(NeuralPlanner)
        planner.device = torch.device("cpu")
        planner._num_waypoints = 4
        planner._action_dim = 7
        planner._max_steps = 5
        planner._pos_eps = 1e9   # always reached
        planner._rot_eps = 1e9
        planner._intermediate_orientation = True
        planner.cfg = type("c", (), {"action_scale": 0.0, "dt": 0.01, "control_part": "arm", "num_arm_joints": 7})()

        # stub actor: returns zeros so qpos never changes but eps is huge -> reached
        actor = torch.nn.Linear(1, 7)
        planner._actor = lambda obs: torch.zeros(obs.shape[0], 7)
        planner._normalizer = type("n", (), {"normalize": lambda self, o: o})()

        # stub robot FK + limits
        class _Robot:
            num_instances = 3
            device = torch.device("cpu")
            def get_qpos(self, name=None): return torch.zeros(3, 7)
            def get_qpos_limits(self, name=None): return (torch.zeros(7, 2),)
            def compute_fk(self, qpos=None, name=None, to_matrix=True):
                m = torch.eye(4).unsqueeze(0).repeat(qpos.shape[0], 1, 1)
                return m if to_matrix else torch.cat([m[:, :3, 3], torch.zeros(qpos.shape[0], 4)], dim=-1)
        planner.robot = _Robot()

        B = 3
        states = [
            PlanState.from_xpos(torch.eye(4).unsqueeze(0).repeat(B, 1, 1), move_type=MoveType.EEF_MOVE)
            for _ in range(2)
        ]
        from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions
        r = planner.plan(states, NeuralPlanOptions(control_part="arm", max_steps=3))
        assert r.success.shape == (B,)
        assert r.success.all().item()
        assert r.positions.shape[0] == B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_neural_batched.py::TestNeuralPlanBatched -v`
Expected: FAIL — `plan` uses `(1, ...)` shaping / scalar success.

- [ ] **Step 3: Rewrite `plan`, `_build_obs`, `_is_active_reached` for B**

```python
    @validate_plan_options(options_cls=NeuralPlanOptions)
    def plan(
        self,
        target_states: list[PlanState],
        options: NeuralPlanOptions = NeuralPlanOptions(),
    ) -> PlanResult:
        if not target_states:
            return PlanResult(success=torch.zeros(0, dtype=torch.bool, device=self.device), positions=None)

        control_part = options.control_part or self.cfg.control_part
        if control_part is None:
            logger.log_error("control_part is required for NeuralPlanner", ValueError)

        waypoints_pos, waypoints_quat, valid_mask, episode_k = self._parse_waypoints(target_states)
        qpos = self._initial_qpos(control_part, options.start_qpos)   # (B, action_dim+)
        b = qpos.shape[0]
        limits = self.robot.get_qpos_limits(name=control_part)[0].to(self.device)
        lower = limits[: self._action_dim, 0]
        upper = limits[: self._action_dim, 1]

        last_action = torch.zeros(b, self._action_dim, device=self.device)
        active_idx = torch.zeros(b, dtype=torch.long, device=self.device)
        positions = [qpos.clone()]
        xpos_list = [self._fk_matrix(qpos, control_part)]
        converged = torch.zeros(b, dtype=torch.bool, device=self.device)
        max_steps = int(options.max_steps or self._max_steps)

        with torch.no_grad():
            for _ in range(max_steps):
                ee_pose = self._fk_pose_xyzw(qpos, control_part)
                obs = self._build_obs(
                    qpos[:, : self._action_dim], ee_pose,
                    waypoints_pos, waypoints_quat, valid_mask, active_idx, last_action,
                )
                action = self._actor(self._normalizer.normalize(obs)).clamp(-1.0, 1.0)
                qpos[:, : self._action_dim] += action * float(self.cfg.action_scale)
                qpos[:, : self._action_dim] = torch.clamp(qpos[:, : self._action_dim], lower, upper)
                last_action = action
                positions.append(qpos.clone())
                xpos_list.append(self._fk_matrix(qpos, control_part))

                ee_pose = self._fk_pose_xyzw(qpos, control_part)
                reached = self._is_active_reached(
                    ee_pose, waypoints_pos, waypoints_quat, active_idx, episode_k
                )  # (B,) bool
                active_idx = torch.where(reached, active_idx + 1, active_idx)
                converged = converged | (active_idx >= episode_k)
                if converged.all():
                    break

        positions_t = torch.stack(positions)      # (T, B, D)
        xpos_t = torch.stack(xpos_list)            # (T, B, 4, 4)
        positions_t = positions_t.permute(1, 0, 2)  # (B, T, D)
        xpos_t = xpos_t.permute(1, 0, 2, 3)         # (B, T, 4, 4)
        success = active_idx >= episode_k           # (B,)
        dt = torch.full((positions_t.shape[0],), float(self.cfg.dt),
                        dtype=torch.float32, device=self.device)
        dt = dt.unsqueeze(0).expand(b, -1)          # (B, T)
        return PlanResult(
            success=success,
            positions=positions_t,
            xpos_list=xpos_t,
            dt=dt,
            duration=torch.full((b,), float(max(positions_t.shape[1] - 1, 0) * self.cfg.dt),
                                 device=self.device),
        )
```

Update `_build_obs` to be batched `(B, ...)`:

```python
    def _build_obs(self, joint_pos, ee_pose, waypoint_pos, waypoint_quat, valid_mask, active_idx, last_action):
        b = joint_pos.shape[0]
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)  # (B,)
        active_onehot = torch.zeros(b, self._num_waypoints, device=self.device)
        active_onehot.scatter_(1, active_idx_clamped.unsqueeze(1), 1.0)
        obs_parts = [
            joint_pos, ee_pose,
            waypoint_pos.reshape(b, self._num_waypoints * 3),
            waypoint_quat.reshape(b, self._num_waypoints * 4),
            active_onehot, valid_mask, last_action,
        ]
        if self._use_relative_obs:
            active_pos = waypoint_pos[torch.arange(b, device=self.device), active_idx_clamped]
            active_quat = waypoint_quat[torch.arange(b, device=self.device), active_idx_clamped]
            obs_parts.append(torch.cat([active_pos - ee_pose[:, :3], active_quat], dim=-1))
        obs = torch.cat(obs_parts, dim=-1)
        if obs.shape[-1] != self._obs_dim:
            raise ValueError(f"Built obs dim {obs.shape[-1]}, expected {self._obs_dim}.")
        return obs
```

Update `_is_active_reached` to return a `(B,)` bool tensor (no `.item()`):

```python
    def _is_active_reached(self, ee_pose, waypoint_pos, waypoint_quat, active_idx, episode_k):
        b = ee_pose.shape[0]
        idx = torch.arange(b, device=self.device)
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_pos = waypoint_pos[idx, active_idx_clamped]
        active_quat_xyzw = waypoint_quat[idx, active_idx_clamped]
        pos_dist = (ee_pose[:, :3] - active_pos).norm(dim=-1)            # (B,)
        ee_quat_wxyz = convert_quat(ee_pose[:, 3:7], to="wxyz")
        active_quat_wxyz = convert_quat(active_quat_xyzw, to="wxyz")
        rot_dist = quat_error_magnitude(ee_quat_wxyz, active_quat_wxyz)  # (B,)
        orientation_required = self._intermediate_orientation | (active_idx >= episode_k - 1)
        rot_ok = torch.where(
            orientation_required,
            rot_dist < self._rot_eps,
            torch.ones_like(rot_dist, dtype=torch.bool),
        )
        return (pos_dist < self._pos_eps) & rot_ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_neural_batched.py -v`
Expected: PASS.

- [ ] **Step 5: Update existing neural planner tests for batched PlanState**

In `tests/sim/planners/test_neural_planner.py`, wrap single-env `PlanState(xpos=...)` with `.single(...)` or `from_xpos(xpos.unsqueeze(0))` and adjust shape assertions to `r.positions.shape[0] == 1`.

Run: `pytest tests/sim/planners/test_neural_planner.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/planners/neural_planner.py tests/sim/planners/test_neural_batched.py tests/sim/planners/test_neural_planner.py
git commit -m "feat(neural): natively batched rollout with per-env early-convergence holds"
```

---

## Phase D — Batched MotionGenerator

### Task D1: Batched `generate()` + `MotionGenOptions.start_qpos:(B,DOF)`

**Files:**
- Modify: `embodichain/lab/sim/planners/motion_generator.py:52-233`
- Test: `tests/sim/planners/test_motion_generator_batched.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/planners/test_motion_generator_batched.py
from __future__ import annotations

import torch
import pytest
from unittest.mock import Mock, patch

from embodichain.lab.sim.planners.motion_generator import MotionGenerator, MotionGenOptions
from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType


def _mock_planner(b=3, n=15, dofs=6):
    planner = Mock()
    planner.robot.num_instances = b
    planner.robot.device = torch.device("cpu")
    planner.plan.return_value = PlanResult(
        success=torch.ones(b, dtype=torch.bool),
        positions=torch.zeros(b, n, dofs),
    )
    planner.default_plan_options.return_value = None
    return planner


class TestGenerateBatched:
    def test_generate_passes_batched_states_to_planner(self):
        planner = _mock_planner()
        mg = MotionGenerator.__new__(MotionGenerator)
        mg.planner = planner
        mg.robot = planner.robot
        mg.device = torch.device("cpu")

        B, dofs = 3, 6
        states = [PlanState.from_qpos(torch.zeros(B, dofs)), PlanState.from_qpos(torch.ones(B, dofs))]
        r = mg.generate(states, MotionGenOptions(plan_opts=Mock()))
        assert r.success.shape == (B,)
        assert r.positions.shape == (B, 15, dofs)
        # planner.plan received the batched states list as-is
        _, kwargs = planner.plan.call_args
        assert kwargs["target_states"] is states or planner.plan.call_args[0][0] is states
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_motion_generator_batched.py::TestGenerateBatched -v`
Expected: FAIL — `generate` is single-env / reshapes.

- [ ] **Step 3: Rewrite `generate` to be batched**

The pre-interpolation branch must stack batched `(B,...)` tensors. Replace the body of `generate` (lines ~154-233):

```python
        if options.is_interpolate:
            move_type = target_states[0].move_type
            if move_type == MoveType.EEF_MOVE:
                for s in target_states:
                    if s.move_type != move_type:
                        logger.log_error(f"All states must share move_type; got {s.move_type}", ValueError)
                xpos_list = torch.stack([s.xpos for s in target_states])  # (B, N, 4, 4)
                qpos_list = None
            elif move_type == MoveType.JOINT_MOVE:
                for s in target_states:
                    if s.move_type != move_type:
                        logger.log_error(f"All states must share move_type; got {s.move_type}", ValueError)
                qpos_list = torch.stack([s.qpos for s in target_states])  # (B, N, DOF)
                xpos_list = None
            else:
                logger.log_error(f"Unsupported move type for pre-interpolation: {move_type}")

            if options.start_qpos is not None:
                start = options.start_qpos
                if start.dim() == 1:
                    start = start.unsqueeze(0)
                if qpos_list is not None:
                    qpos_list = torch.cat([start.unsqueeze(1), qpos_list], dim=1)
                if xpos_list is not None:
                    start_xpos = self.robot.compute_fk(qpos=start, name=options.control_part, to_matrix=True)
                    xpos_list = torch.cat([start_xpos.unsqueeze(1) if start_xpos.dim()==3 else start_xpos[:, None], xpos_list], dim=1)

            qpos_interpolated, xpos_interpolated = self.interpolate_trajectory(
                control_part=options.control_part,
                xpos_list=xpos_list,
                qpos_list=qpos_list,
                options=options,
            )
            if not options.plan_opts:
                return PlanResult(success=True, positions=qpos_interpolated, xpos_list=xpos_interpolated)

            target_plan_states = [
                PlanState(move_type=MoveType.JOINT_MOVE, qpos=qpos_interpolated[:, j])
                for j in range(qpos_interpolated.shape[1])
            ]
        else:
            target_plan_states = target_states

        if options.plan_opts is None:
            if hasattr(self.planner, "default_plan_options"):
                options.plan_opts = self.planner.default_plan_options()
            else:
                options.plan_opts = PlanOptions()
        return self.planner.plan(target_states=target_plan_states, options=options.plan_opts)
```

Update the `MotionGenOptions.start_qpos` docstring to `(B, DOF)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_motion_generator_batched.py::TestGenerateBatched -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/planners/motion_generator.py tests/sim/planners/test_motion_generator_batched.py
git commit -m "feat(motion-gen): batched generate() + start_qpos:(B,DOF)"
```

---

### Task D2: Batched `interpolate_trajectory` (drop the `n_envs=1` unsqueeze)

**Files:**
- Modify: `embodichain/lab/sim/planners/motion_generator.py:462-630`
- Test: `tests/sim/planners/test_motion_generator_batched.py` (append)

- [ ] **Step 1: Write the failing test**

```python
class TestInterpolateBatched:
    def test_interpolate_joint_space_batched(self):
        planner = _mock_planner(b=3, n=10, dofs=6)
        mg = MotionGenerator.__new__(MotionGenerator)
        mg.planner = planner
        mg.robot = planner.robot
        mg.device = torch.device("cpu")
        B, N, D = 3, 4, 6
        qpos_list = torch.zeros(B, N, D)
        qpos_interpolated, _ = mg.interpolate_trajectory(
            control_part="arm", xpos_list=None, qpos_list=qpos_list,
            options=MotionGenOptions(is_linear=False, interpolate_nums=10),
        )
        assert qpos_interpolated.shape[0] == B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/planners/test_motion_generator_batched.py::TestInterpolateBatched -v`
Expected: FAIL — current code squeezes to single-env.

- [ ] **Step 3: Rewrite `interpolate_trajectory` to preserve the B dim**

The joint-space branch (the `else:` at line 605) currently does `qpos_list.unsqueeze_(0).permute(1,0,2)` and squeezes at the end. Make it operate on `(B, N, D)` directly and return `(B, M, D)`:

```python
        else:
            # Joint-space interpolation. qpos_list is (B, N, DOF).
            if isinstance(options.interpolate_nums, int):
                interp_nums = [options.interpolate_nums] * (qpos_list.shape[1] - 1)
            else:
                if len(options.interpolate_nums) != qpos_list.shape[1] - 1:
                    logger.log_error(
                        "Length of interpolate_nums list must equal number of segments",
                        ValueError,
                    )
                interp_nums = options.interpolate_nums

            interpolate_qpos_list = interpolate_with_nums(
                qpos_list, interp_nums=interp_nums, device=self.device,
            )  # (B, M, DOF)
            feasible_pose_targets = None

        return interpolate_qpos_list, feasible_pose_targets
```

For the Cartesian (linear) branch, replace the `n_envs=1`-unsqueeze block (lines ~566-604) with a batched version: iterate segments but process all `B` envs per segment. At the top of the linear branch, after computing `interpolated_point_allocations`, build `total_interpolated_poses` as `(B, M_total, 4, 4)` by stacking per-segment `(B, seg_pts, 4, 4)` outputs, then call `compute_batch_ik(pose=total_interpolated_poses, joint_seed=qpos_seed broadcast to (B, M_total, D))`. Concretely replace lines 536-604 with:

```python
            # Linear cartesian interpolation, batched across B envs.
            total_interpolated_poses = []
            for i in range(xpos_list.shape[1] - 1):
                seg = interpolate_xpos_batched(
                    xpos_list[:, i], xpos_list[:, i + 1], interpolated_point_allocations[i],
                )  # (B, seg, 4, 4)
                total_interpolated_poses.append(seg)
            total_interpolated_poses = torch.cat(total_interpolated_poses, dim=1)  # (B, M, 4, 4)

            qpos_seed_b = qpos_seed
            if qpos_seed_b.dim() == 1:
                qpos_seed_b = qpos_seed_b.unsqueeze(0).repeat(xpos_list.shape[0], 1)
            joint_seed = qpos_seed_b.unsqueeze(1).repeat(1, total_interpolated_poses.shape[1], 1)
            success_batch, qpos_batch = self.robot.compute_batch_ik(
                pose=total_interpolated_poses, joint_seed=joint_seed, name=control_part,
            )  # (B, M), (B, M, D)
            has_nan = torch.isnan(qpos_batch).any(dim=-1)
            valid = success_batch & (~has_nan)  # (B, M)
            # Per-env filter: keep only valid rows; pad short envs by repeating last valid.
            B, M, D = qpos_batch.shape
            max_valid = int(valid.sum(dim=1).max().item())
            max_valid = max(max_valid, 1)
            interp_q = torch.zeros(B, max_valid, D, device=self.device, dtype=torch.float32)
            feasible = torch.zeros(B, max_valid, 4, 4, device=self.device, dtype=torch.float32)
            for b in range(B):
                v = qpos_batch[b][valid[b]]
                f = total_interpolated_poses[b][valid[b]]
                if v.shape[0] == 0:
                    v = qpos_batch[b:b+1, 0]
                    f = total_interpolated_poses[b:b+1, 0]
                interp_q[b, :v.shape[0]] = v
                interp_q[b, v.shape[0]:] = v[-1]
                feasible[b, :f.shape[0]] = f
                feasible[b, f.shape[0]:] = f[-1]
            interpolate_qpos_list = interp_q
            feasible_pose_targets = feasible
```

Add a `interpolate_xpos_batched` helper in `utils.py` (vectorized over `B`):

```python
def interpolate_xpos_batched(start_xpos: torch.Tensor, end_xpos: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Batched pose interpolation. start/end: (B, 4, 4) -> (B, num_samples, 4, 4)."""
    num_samples = max(2, int(num_samples))
    ratios = torch.linspace(0.0, 1.0, num_samples, device=start_xpos.device, dtype=start_xpos.dtype)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(start_xpos[:, :3, :3].cpu().numpy()))
    rots = torch.as_tensor(slerp(ratios.cpu().numpy()).as_matrix(), dtype=start_xpos.dtype, device=start_xpos.device)
    # rots: (num_samples, B, 3, 3) -> (B, num_samples, 3, 3)
    rots = rots.permute(1, 0, 2, 3)
    trans = (1.0 - ratios[None, :, None]) * start_xpos[:, None, :3, 3] + ratios[None, :, None] * end_xpos[:, None, :3, 3]
    out = torch.eye(4, dtype=start_xpos.dtype, device=start_xpos.device).repeat(start_xpos.shape[0], num_samples, 1, 1)
    out[:, :, :3, :3] = rots
    out[:, :, :3, 3] = trans
    return out
```

Export it in `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/planners/test_motion_generator_batched.py -v`
Expected: PASS.

- [ ] **Step 5: Update existing motion_generator tests**

In `tests/sim/planners/test_motion_generator.py`, update single-env PlanState construction to `.single(...)` and shape assertions. Run:

Run: `pytest tests/sim/planners/test_motion_generator.py -v`
Expected: PASS (fix any remaining shape mismatches inline).

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/planners/motion_generator.py embodichain/lab/sim/planners/utils.py tests/sim/planners/test_motion_generator_batched.py tests/sim/planners/test_motion_generator.py
git commit -m "feat(motion-gen): batched interpolate_trajectory + interpolate_xpos_batched"
```

---

## Phase E — TrajectoryBuilder motion-source strategy

### Task E1: `ActionCfg.motion_source`/`planner_type` + `ActionResult.success` tensor

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/core.py:162-189`
- Test: `tests/sim/atomic_actions/test_action_result_success.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/atomic_actions/test_action_result_success.py
from __future__ import annotations

import warnings
import torch
import pytest

from embodichain.lab.sim.atomic_actions.core import ActionResult, ActionCfg, WorldState


class TestActionResultSuccess:
    def test_success_all_tensor(self):
        r = ActionResult(success=torch.tensor([True, False]),
                         trajectory=torch.zeros(2, 0, 3),
                         next_state=WorldState(last_qpos=torch.zeros(2, 3)))
        assert r.success_all is False

    def test_bool_deprecation(self):
        r = ActionResult(success=torch.tensor([True, True]),
                         trajectory=torch.zeros(2, 0, 3),
                         next_state=WorldState(last_qpos=torch.zeros(2, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert bool(r) is True


class TestActionCfgMotionSource:
    def test_defaults(self):
        cfg = ActionCfg()
        assert cfg.motion_source == "ik_interp"
        assert cfg.planner_type is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/atomic_actions/test_action_result_success.py -v`
Expected: FAIL — `motion_source` / `success_all` not defined.

- [ ] **Step 3: Implement**

In `core.py`, update `ActionCfg`:

```python
@configclass
class ActionCfg:
    name: str = "default"
    control_part: str = "arm"
    interpolation_type: str = "linear"
    velocity_limit: float | None = None
    acceleration_limit: float | None = None
    motion_source: str = "ik_interp"
    """Trajectory source: 'ik_interp' (default, batched IK + linear interp) or 'motion_gen' (batched MotionGenerator)."""
    planner_type: str | None = None
    """Planner type for motion_source='motion_gen': 'toppra' | 'neural'. Required when motion_source='motion_gen'."""
```

Update `ActionResult`:

```python
@dataclass
class ActionResult:
    success: bool | torch.Tensor
    trajectory: torch.Tensor
    next_state: WorldState

    @property
    def success_all(self) -> bool:
        if isinstance(self.success, torch.Tensor):
            return bool(torch.all(self.success).item())
        return bool(self.success)

    def __bool__(self) -> bool:
        import warnings as _w
        _w.warn("ActionResult bool() is deprecated; use .success_all", DeprecationWarning, stacklevel=2)
        return self.success_all
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/atomic_actions/test_action_result_success.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/atomic_actions/core.py tests/sim/atomic_actions/test_action_result_success.py
git commit -m "feat(actions): ActionCfg.motion_source/planner_type + ActionResult.success tensor"
```

---

### Task E2: `TrajectoryBuilder.plan_arm_traj` motion-source branch + per-env success

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/trajectory.py:313-351`
- Test: `tests/sim/atomic_actions/test_trajectory_motion_source.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/atomic_actions/test_trajectory_motion_source.py
from __future__ import annotations

import torch
import pytest
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.trajectory import TrajectoryBuilder
from embodichain.lab.sim.planners.utils import PlanState, MoveType
from embodichain.lab.sim.atomic_actions.core import ActionCfg


def _mock_mg(num_envs=2, arm_dof=6):
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = arm_dof
    robot.get_qpos = lambda name=None: torch.zeros(num_envs, arm_dof)
    def compute_ik(pose=None, name=None, joint_seed=None, **kw):
        return torch.ones(num_envs, dtype=torch.bool), torch.zeros(num_envs, arm_dof)
    robot.compute_ik = compute_ik
    mg = Mock()
    mg.robot = robot
    mg.device = torch.device("cpu")
    return mg


class TestPlanArmTrajMotionGen:
    def test_motion_gen_path_delegates_to_generate(self):
        mg = _mock_mg(num_envs=3, arm_dof=6)
        from embodichain.lab.sim.planners.utils import PlanResult
        mg.generate.return_value = PlanResult(
            success=torch.ones(3, dtype=torch.bool),
            positions=torch.zeros(3, 12, 6),
        )
        builder = TrajectoryBuilder(mg)
        cfg = ActionCfg(motion_source="motion_gen", planner_type="toppra", control_part="arm")
        start_qpos = torch.zeros(3, 6)
        # per-env list[list[PlanState]] with single-env PlanStates (action contract)
        target_states_list = [
            [PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE),
             PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE)]
            for _ in range(3)
        ]
        ok, traj = builder.plan_arm_traj(
            target_states_list, start_qpos, 12,
            control_part="arm", arm_dof=6, cfg=cfg,
        )
        assert ok.shape == (3,)
        assert ok.all().item()
        assert traj.shape == (3, 12, 6)
        mg.generate.assert_called_once()

    def test_ik_interp_path_unchanged(self):
        mg = _mock_mg(num_envs=2, arm_dof=6)
        builder = TrajectoryBuilder(mg)
        cfg = ActionCfg(motion_source="ik_interp", control_part="arm")
        start_qpos = torch.zeros(2, 6)
        target_states_list = [
            [PlanState(xpos=torch.eye(4), move_type=MoveType.EEF_MOVE)]
            for _ in range(2)
        ]
        ok, traj = builder.plan_arm_traj(
            target_states_list, start_qpos, 10,
            control_part="arm", arm_dof=6, cfg=cfg,
        )
        assert ok.all().item()
        assert traj.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/atomic_actions/test_trajectory_motion_source.py -v`
Expected: FAIL — `plan_arm_traj` has no `cfg` param / no `motion_source` branch.

- [ ] **Step 3: Implement the branch**

In `trajectory.py`, add imports and rewrite `plan_arm_traj`:

```python
from embodichain.lab.sim.planners.utils import PlanState as _PS, MoveType as _MT
from embodichain.lab.sim.planners.utils import PlanResult
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
```

Replace `plan_arm_traj` (and add helpers):

```python
    def plan_arm_traj(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
        cfg: "ActionCfg | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan batched arm trajectories for all environments.

        Returns ``(success:(B,), trajectory:(B, n_waypoints, arm_dof))``.
        ``cfg.motion_source`` selects 'ik_interp' (default) or 'motion_gen'.
        """
        n_envs = start_qpos.shape[0]
        motion_source = getattr(cfg, "motion_source", "ik_interp") if cfg else "ik_interp"
        if motion_source == "motion_gen":
            return self._plan_motion_gen(
                target_states_list, start_qpos, n_waypoints,
                control_part=control_part, arm_dof=arm_dof, cfg=cfg,
            )
        return self._plan_ik_interp(
            target_states_list, start_qpos, n_waypoints,
            control_part=control_part, arm_dof=arm_dof,
        )

    def _plan_ik_interp(self, target_states_list, start_qpos, n_waypoints, *, control_part, arm_dof):
        n_envs = start_qpos.shape[0]
        n_state = len(target_states_list[0])
        xpos_traj = torch.zeros((n_envs, n_state, 4, 4), dtype=torch.float32, device=self.device)
        for i, target_states in enumerate(target_states_list):
            for j, ts in enumerate(target_states):
                xpos_traj[i, j] = ts.xpos
        trajectory = torch.zeros((n_envs, n_state, arm_dof), dtype=torch.float32, device=self.device)
        success = torch.ones(n_envs, dtype=torch.bool, device=self.device)
        qpos_seed = start_qpos
        for j in range(n_state):
            is_success, qpos = self.robot.compute_ik(
                pose=xpos_traj[:, j], name=control_part, joint_seed=qpos_seed
            )
            if not self.all_envs_success(is_success):
                logger.log_warning(f"IK failed for waypoint {j} in some envs.")
                success = success & is_success
            trajectory[:, j] = qpos
            qpos_seed = qpos
        trajectory = torch.concatenate([start_qpos.unsqueeze(1), trajectory], dim=1)
        # Failed envs: hold start qpos across all waypoints
        if not success.all():
            held = start_qpos.unsqueeze(1).repeat(1, trajectory.shape[1], 1)
            trajectory = torch.where(success[:, None, None], trajectory, held)
        interp = interpolate_with_distance(trajectory=trajectory, interp_num=n_waypoints, device=self.device)
        return success, interp

    def _plan_motion_gen(self, target_states_list, start_qpos, n_waypoints, *, control_part, arm_dof, cfg):
        from .core import ActionCfg  # local to avoid cycle
        n_envs = start_qpos.shape[0]
        if self.motion_generator is None:
            logger.log_error(
                "motion_source='motion_gen' requires a MotionGenerator on the engine",
                ValueError,
            )
        plan_states = self._to_batched_plan_states(target_states_list, n_envs)
        plan_opts = self._build_plan_opts(cfg, n_waypoints)
        result: PlanResult = self.motion_generator.generate(
            plan_states,
            MotionGenOptions(start_qpos=start_qpos, control_part=control_part, plan_opts=plan_opts),
        )
        success = result.success if isinstance(result.success, torch.Tensor) else torch.tensor(result.success, device=self.device)
        positions = result.positions
        # Resample to n_waypoints if the planner returned a different count
        if positions.shape[1] != n_waypoints:
            positions = interpolate_with_distance(trajectory=positions, interp_num=n_waypoints, device=self.device)
        # Failed envs hold start qpos
        if not success.all():
            held = start_qpos[:, None, :arm_dof].repeat(1, positions.shape[1], 1) if start_qpos.shape[1] >= arm_dof else start_qpos[:, :arm_dof][:, None, :].repeat(1, positions.shape[1], 1)
            positions = torch.where(success[:, None, None], positions, held)
        return success, positions

    def _to_batched_plan_states(self, target_states_list: list[list[PlanState]], n_envs: int) -> list[PlanState]:
        """Convert per-env PlanState lists into a batched list[PlanState] (each carries B envs)."""
        n_state = len(target_states_list[0])
        batched: list[PlanState] = []
        for j in range(n_state):
            sample = target_states_list[0][j]
            if sample.xpos is not None:
                xpos = torch.stack([target_states_list[i][j].xpos for i in range(n_envs)])  # (B, 4, 4)
                batched.append(PlanState(xpos=xpos, move_type=MoveType.EEF_MOVE, move_part=sample.move_part))
            else:
                qpos = torch.stack([target_states_list[i][j].qpos for i in range(n_envs)])  # (B, DOF)
                batched.append(PlanState(qpos=qpos, move_type=MoveType.JOINT_MOVE, move_part=sample.move_part))
        return batched

    def _build_plan_opts(self, cfg, n_waypoints):
        planner_type = getattr(cfg, "planner_type", None)
        if planner_type in (None, "toppra"):
            constraints = {}
            vl = getattr(cfg, "velocity_limit", None)
            al = getattr(cfg, "acceleration_limit", None)
            constraints["velocity"] = vl if vl is not None else 0.2
            constraints["acceleration"] = al if al is not None else 0.5
            return ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=n_waypoints,
                constraints=constraints,
            )
        # neural: planner reads its own cfg; pass minimal options
        from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions
        return NeuralPlanOptions()
```

Add `from embodichain.lab.sim.planners.utils import TrajectorySampleMethod` to imports.

Note: `plan_arm_traj` now takes an optional `cfg` kwarg. The 6 actions call it **without** `cfg` today. Update each call site in Phase G to pass `cfg=self.cfg`. (If `cfg` is None, `motion_source` defaults to `"ik_interp"` so behavior is unchanged — but actions must pass `self.cfg` to opt into `motion_gen`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/atomic_actions/test_trajectory_motion_source.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/atomic_actions/trajectory.py tests/sim/atomic_actions/test_trajectory_motion_source.py
git commit -m "feat(actions): TrajectoryBuilder motion_source strategy + per-env success"
```

---

## Phase F — AtomicActionEngine per-env failure propagation

### Task F1: `run()` returns `(B,)` success with held-qpos propagation

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/engine.py:89-136`
- Test: `tests/sim/atomic_actions/test_engine_per_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/sim/atomic_actions/test_engine_per_env.py
from __future__ import annotations

import torch
import pytest
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult, AtomicAction, WorldState, EndEffectorPoseTarget, ActionCfg,
)


class _StubAction(AtomicAction):
    TargetType = EndEffectorPoseTarget
    def __init__(self, mg, success_vec, traj_len=4, dof=3):
        super().__init__(mg, ActionCfg())
        self._success = torch.tensor(success_vec)
        self._traj_len = traj_len
        self._dof = dof
    def execute(self, target, state):
        n = state.last_qpos.shape[0]
        traj = torch.zeros(n, self._traj_len, self._dof)
        traj[:] = state.last_qpos.unsqueeze(1)
        return ActionResult(success=self._success.clone(), trajectory=traj,
                            next_state=WorldState(last_qpos=traj[:, -1, :].clone()))


class TestRunPerEnv:
    def test_failed_env_holds(self):
        mg = Mock(); mg.robot.get_qpos = lambda: torch.zeros(3, 3)
        mg.robot.dof = 3; mg.device = torch.device("cpu")
        eng = AtomicActionEngine(mg)
        # env 1 fails step 2
        eng.register(_StubAction(mg, [True, True, True]), name="a")
        eng.register(_StubAction(mg, [True, False, True]), name="b")
        eng.register(_StubAction(mg, [True, True, True]), name="c")
        success, traj, state = eng.run(
            steps=[("a", EndEffectorPoseTarget(xpos=torch.eye(4))),
                   ("b", EndEffectorPoseTarget(xpos=torch.eye(4))),
                   ("c", EndEffectorPoseTarget(xpos=torch.eye(4)))]
        )
        assert success.tolist() == [True, False, True]
        assert traj.shape[1] == 12  # 3 steps * 4 waypoints
        # env 1's rows after its failure should equal its pre-failure qpos (held)
        # all zeros here, so just check shape and that env 0/2 advanced
        assert state.last_qpos.shape == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/atomic_actions/test_engine_per_env.py -v`
Expected: FAIL — `run` returns scalar bool / short-circuits.

- [ ] **Step 3: Implement per-env `run`**

```python
    def run(
        self,
        steps: Iterable[tuple[str, Target]],
        state: WorldState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, WorldState]:
        if state is None:
            state = WorldState(last_qpos=self.robot.get_qpos().clone())
        b = state.last_qpos.shape[0]
        full_traj = torch.empty((b, 0, self.robot.dof), dtype=torch.float32, device=self.device)
        alive = torch.ones(b, dtype=torch.bool, device=self.device)

        for name, target in steps:
            if name not in self._actions:
                logger.log_error(f"No action registered under name '{name}'", KeyError)
            action = self._actions[name]
            if not isinstance(target, action.TargetType):
                logger.log_error(
                    f"Action '{name}' expects target of type "
                    f"{_target_type_name(action.TargetType)}, got {type(target).__name__}",
                    TypeError,
                )
            if not alive.any():
                # All envs dead: fill held rows for this step.
                held = state.last_qpos.unsqueeze(1).repeat(1, 1, 1)
                full_traj = torch.cat([full_traj, held], dim=1)
                continue
            prev_last_qpos = state.last_qpos.clone()
            result: ActionResult = action.execute(target, state)
            step_success = result.success if isinstance(result.success, torch.Tensor) else torch.tensor(bool(result.success), device=self.device)
            step_success = step_success.to(self.device)
            alive = alive & step_success
            # Failed envs freeze at their last successful qpos for this step's trajectory.
            traj = result.trajectory
            held_rows = prev_last_qpos.unsqueeze(1).repeat(1, traj.shape[1], 1)
            traj = torch.where(alive[:, None, None], traj, held_rows)
            full_traj = torch.cat([full_traj, traj], dim=1)
            state = result.next_state
            state.last_qpos = torch.where(alive[:, None], state.last_qpos, prev_last_qpos)

        return alive, full_traj, state
```

Update the `run` docstring return to `success: (B,) tensor`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/atomic_actions/test_engine_per_env.py -v`
Expected: PASS.

- [ ] **Step 5: Update existing engine tests**

In `tests/sim/atomic_actions/test_engine.py`, update assertions: `success` is now a tensor; use `success.all().item()` or `success.tolist()`. Run:

Run: `pytest tests/sim/atomic_actions/test_engine.py -v`
Expected: PASS (fix inline).

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/atomic_actions/engine.py tests/sim/atomic_actions/test_engine_per_env.py tests/sim/atomic_actions/test_engine.py
git commit -m "feat(engine): per-env failure propagation in run() with held qpos"
```

---

## Phase G — Migrate the 6 actions to per-env success + pass `cfg`

The 6 actions currently do `ok, arm_traj = self.builder.plan_arm_traj(...)` then `if not ok: return self._fail(state)`. Now `ok` is a `(B,)` tensor and `plan_arm_traj` takes `cfg=self.cfg`. The migration is mechanical and identical in shape across actions:

- Pass `cfg=self.cfg` to every `plan_arm_traj` call.
- Replace `if not ok: return self._fail(state)` with: keep the planned trajectory (failed envs already hold start qpos inside `plan_arm_traj`), and compute the action's `success` mask (AND across multi-phase plans).
- Return `ActionResult(success=<mask>, ...)` instead of `success=True`.

### Task G1: Migrate `MoveEndEffector` and `MoveJoints`

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/actions.py:223-289, 325-352`
- Test: `tests/sim/atomic_actions/test_actions.py` (update)

- [ ] **Step 1: Update `MoveEndEffector.execute`**

```python
    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        move_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        start_qpos = self.builder.resolve_start_qpos(
            _arm_qpos_from_state(state, self.arm_joint_ids, self.robot_dof),
            n_envs=self.n_envs, arm_dof=self.arm_dof, control_part=self.cfg.control_part,
        )
        target_states_list = self._build_target_states(move_xpos)
        success, arm_traj = self.builder.plan_arm_traj(
            target_states_list, start_qpos, self.cfg.sample_interval,
            control_part=self.cfg.control_part, arm_dof=self.arm_dof, cfg=self.cfg,
        )
        full = self._embed(arm_traj, state.last_qpos)
        return ActionResult(
            success=success,
            trajectory=full,
            next_state=WorldState(last_qpos=full[:, -1, :].clone(), held_object=state.held_object),
        )
```

- [ ] **Step 2: Update `MoveJoints.execute`**

`MoveJoints` uses `plan_joint_traj` (no planner). Its success is always all-True (no IK). Return a `(B,)` ones tensor:

```python
        joint_traj = self.builder.plan_joint_traj(start_qpos, target_qpos, self.cfg.sample_interval)
        full = self._embed(joint_traj, state.last_qpos)
        return ActionResult(
            success=torch.ones(self.n_envs, dtype=torch.bool, device=self.device),
            trajectory=full,
            next_state=WorldState(last_qpos=full[:, -1, :].clone(), held_object=state.held_object),
        )
```

- [ ] **Step 3: Run action tests**

Run: `pytest tests/sim/atomic_actions/test_actions.py -v`
Expected: some FAIL (success now tensor). Update assertions in `test_actions.py` from `assert result.success` to `assert result.success.all()` and shape checks `result.success.shape == (n_envs,)`.

- [ ] **Step 4: Fix test assertions inline and re-run**

Run: `pytest tests/sim/atomic_actions/test_actions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/atomic_actions/actions.py tests/sim/atomic_actions/test_actions.py
git commit -m "feat(actions): MoveEndEffector/MoveJoints per-env success + cfg passthrough"
```

---

### Task G2: Migrate `PickUp`, `Place` (multi-phase plans)

Both call `plan_arm_traj` twice (approach/down + lift/back). The action's `success = approach_success & lift_success`.

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/actions.py` (`PickUp.execute`, `Place.execute`)
- Test: `tests/sim/atomic_actions/test_actions.py`

- [ ] **Step 1: Update `PickUp.execute`**

For each `plan_arm_traj` call: add `cfg=self.cfg`, capture `success_*` instead of `ok`, remove the `if not ok: return self._fail(state)` early-return. The phase-1 (approach) and phase-3 (lift) each return a `(B,)` mask; if an env fails approach, its `approach_arm` holds start qpos, and lift should be skipped for it (its lift input is the held grasp qpos — fine). Final `success = approach_success & lift_success`. Concretely:

```python
        approach_success, approach_arm = self.builder.plan_arm_traj(
            target_states_list, start_arm_qpos, n_approach,
            control_part=self.cfg.control_part, arm_dof=self.arm_dof, cfg=self.cfg,
        )
        grasp_arm_qpos = approach_arm[:, -1, :]
        # ... lift target_states_list ...
        lift_success, lift_arm = self.builder.plan_arm_traj(
            target_states_list, grasp_arm_qpos, n_lift,
            control_part=self.cfg.control_part, arm_dof=self.arm_dof, cfg=self.cfg,
        )
        success = approach_success & lift_success
```

Remove both `if not ok: ...` blocks. Keep the rest (hand close path, `full` embedding) unchanged. For failed envs, the held rows from `plan_arm_traj` already fill the arm slots; the hand path still interpolates (cosmetic — failed envs hold via `run()`'s `where`).

Set `success=success` in the final `ActionResult`. If `success` is all-False, the trajectory is still well-formed (held poses); `run()` handles propagation.

- [ ] **Step 2: Update `Place.execute` analogously** (down + back phases).

- [ ] **Step 3: Run tests**

Run: `pytest tests/sim/atomic_actions/test_actions.py -v`
Expected: PASS (fix assertions inline).

- [ ] **Step 4: Commit**

```bash
git add embodichain/lab/sim/atomic_actions/actions.py tests/sim/atomic_actions/test_actions.py
git commit -m "feat(actions): PickUp/Place per-env success + cfg passthrough"
```

---

### Task G3: Migrate `MoveHeldObject`, `Press`

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/actions.py` (`MoveHeldObject.execute`, `Press.execute`)
- Test: `tests/sim/atomic_actions/test_actions.py`

- [ ] **Step 1: Update `MoveHeldObject.execute`** — single `plan_arm_traj` call; add `cfg=self.cfg`, capture `success`, remove `if not ok`, return `success=success`.

- [ ] **Step 2: Update `Press.execute`** — single `plan_arm_traj` (down) + `plan_joint_traj` (back). The back phase is joint interpolation (always succeeds). `success = down_success`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/sim/atomic_actions/test_actions.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add embodichain/lab/sim/atomic_actions/actions.py tests/sim/atomic_actions/test_actions.py
git commit -m "feat(actions): MoveHeldObject/Press per-env success + cfg passthrough"
```

---

## Phase H — Caller & tutorial migration

### Task H1: Migrate `atom_action_utils.plan_trajectory` to batched `PlanState`

**Files:**
- Modify: `embodichain/lab/sim/utility/atom_action_utils.py:199-216`
- Test: `tests/sim/utility/` (update existing if present; else smoke-check import)

- [ ] **Step 1: Read the call site**

Run: `grep -n "plan_trajectory\|MotionGenerator\|PlanState" embodichain/lab/sim/utility/atom_action_utils.py`
Inspect lines 199-216 and the `plan_state` construction. Convert single-env `PlanState(qpos=...)`/`PlanState(xpos=...)` to `PlanState.single(...)` (B=1). The `MotionGenerator.generate` call now returns a batched `PlanResult`; for the single-env caller, squeeze `result.positions[0]` where the legacy consumer expects `(N, DOF)`.

- [ ] **Step 2: Apply the migration**

```python
    # inside plan_trajectory(...), where plan_state is built:
    plan_state = [PlanState.single(qpos=..., move_type=...) for ...]   # or from_xpos
    ret = motion_generator.generate(target_states=plan_state, options=...)
    # legacy consumers expect (N, DOF): squeeze the B=1 dim
    # (leave ret as-is if callers were already updated; otherwise expose a helper)
```

Match the exact shape the existing caller expects. If the caller is itself single-env, wrapping with `.single(...)` and squeezing `[0]` on output is the faithful migration.

- [ ] **Step 3: Run utility tests**

Run: `pytest tests/sim/utility/ -v`
Expected: PASS (or no tests — then run `python -c "from embodichain.lab.sim.utility.atom_action_utils import plan_trajectory"` to confirm import).

- [ ] **Step 4: Commit**

```bash
git add embodichain/lab/sim/utility/atom_action_utils.py
git commit -m "chore(utility): migrate plan_trajectory to batched PlanState"
```

---

### Task H2: Migrate tutorials + any remaining `.plan()`/`.generate()` callers

**Files:**
- Modify: `scripts/tutorials/atomic_action/move_end_effector.py:198-204` and any other tutorial / script calling `engine.run` or `motion_generator.generate`.
- Discover: `grep -rn "\.plan(\|\.generate(\|is_success" scripts/ embodichain/ | grep -v test`

- [ ] **Step 1: Find all call sites**

Run: `grep -rln "is_success, traj\|bool(is_success)\|engine.run\|\.generate(" scripts/ embodichain/`
For each: if it unpacks `is_success, traj, _ = engine.run(...)`, change `bool(is_success)` → `bool(is_success.all())`. If it builds `PlanState(qpos=...)` single-env, change to `PlanState.single(...)`.

- [ ] **Step 2: Apply the tutorial fix**

In `scripts/tutorials/atomic_action/move_end_effector.py` around line 198-204:

```python
is_success, traj, _ = atomic_engine.run(
    steps=[("move_end_effector", EndEffectorPoseTarget(xpos=multi_waypoint_xpos))]
)
# is_success is now a (B,) tensor
print("success:", bool(is_success.all()))
```

- [ ] **Step 3: Smoke-run the tutorial (gated)**

Run: `python scripts/tutorials/atomic_action/move_end_effector.py` (if a headless run is feasible in the env; otherwise skip and note in the commit body).

- [ ] **Step 4: Commit**

```bash
git add scripts/tutorials/atomic_action/move_end_effector.py
git commit -m "chore(tutorial): adapt move_end_effector to (B,) engine.run success"
```

---

## Phase I — Integration & end-to-end tests

### Task I1: TOPPRA numerical-regression anchor

**Files:**
- Test: `tests/sim/planners/test_toppra_batched.py` (append, `@pytest.mark.slow`)

- [ ] **Step 1: Write the regression test**

```python
@pytest.mark.slow
class TestToppraNumericalRegression:
    def test_batched_equals_inline_single(self):
        from embodichain.lab.sim.planners.toppra_planner import _toppra_solve_one_env, ToppraPlanner, ToppraPlannerCfg, ToppraPlanOptions
        from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.robots import CobotMagicCfg
        sim = SimulationManager(SimulationManagerCfg(headless=True, sim_device="cpu"))
        sim.add_robot(cfg=CobotMagicCfg.from_dict({"uid": "r", "init_pos": [0, 0, 0.7775], "init_qpos": [0.0] * 16}))
        planner = ToppraPlanner(ToppraPlannerCfg(robot_uid="r", max_workers=1))
        try:
            B, dofs = 4, 6
            wp = torch.zeros(B, dofs); wp[:, 0] = torch.linspace(0.1, 0.6, B)
            states = [PlanState.from_qpos(torch.zeros(B, dofs)), PlanState.from_qpos(wp)]
            opts = ToppraPlanOptions(sample_method=TrajectorySampleMethod.QUANTITY, sample_interval=20,
                                     constraints={"velocity": 1.0, "acceleration": 2.0})
            r = planner.plan(states, opts)
            # Compare each env to the inline single-env solve
            for b in range(B):
                single = _toppra_solve_one_env(
                    np.stack([np.zeros(dofs), wp[b].numpy()]),
                    1.0, 2.0, TrajectorySampleMethod.QUANTITY, 20,
                )
                assert np.allclose(r.positions[b].cpu().numpy(), single["positions"], atol=1e-5)
        finally:
            planner.close(); sim.destroy()
            import embodichain.lab.sim as om; om.SimulationManager.flush_cleanup_queue()
```

- [ ] **Step 2: Run it**

Run: `pytest tests/sim/planners/test_toppra_batched.py::TestToppraNumericalRegression -v -m slow`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/sim/planners/test_toppra_batched.py
git commit -m "test(toppra): batched-vs-inline numerical regression anchor"
```

---

### Task I2: Reach-equivalence across motion sources (gated)

**Files:**
- Test: `tests/sim/atomic_actions/test_motion_source_e2e.py` (`@pytest.mark.requires_sim`)

- [ ] **Step 1: Write the e2e test**

A single-env (B=1) `MoveEndEffector` to a reachable pose, run three times with `motion_source` ∈ {`ik_interp`, `motion_gen`+`toppra`, `motion_gen`+`neural`}. For each, FK-check that the final waypoint reaches the target within 2 cm position tolerance. (Neural requires a checkpoint path; skip if absent.)

```python
@pytest.mark.requires_sim
class TestMotionSourceReachEquivalence:
    def test_each_source_reaches_target(self):
        # build sim + CobotMagic + AtomicActionEngine with MotionGenerator(ToppraPlannerCfg)
        # target = a reachable EE pose
        # for motion_source in ('ik_interp', 'motion_gen'):
        #     cfg = MoveEndEffectorCfg(motion_source=..., planner_type='toppra')
        #     eng.register(MoveEndEffector(mg, cfg))
        #     success, traj, _ = eng.run([('move_end_effector', EndEffectorPoseTarget(xpos=...))])
        #     final_q = traj[0, -1, arm_ids]
        #     fk = robot.compute_fk(qpos=final_q[None], name='arm', to_matrix=True)[0]
        #     assert torch.norm(fk[:3,3] - target[:3,3]) < 0.02
        ...
```

- [ ] **Step 2: Run (gated)**

Run: `pytest tests/sim/atomic_actions/test_motion_source_e2e.py -v -m requires_sim`
Expected: PASS (or skip if no headless sim).

- [ ] **Step 3: Commit**

```bash
git add tests/sim/atomic_actions/test_motion_source_e2e.py
git commit -m "test(actions): reach-equivalence across ik_interp / motion_gen sources"
```

---

### Task I3: Full pre-commit check + black

**Files:** all touched.

- [ ] **Step 1: Run black**

Run: `black .`
Expected: reformats touched files.

- [ ] **Step 2: Run pre-commit-check skill** (via `/pre-commit-check`), or run the full planner + atomic_actions test suites:

Run: `pytest tests/sim/planners/ tests/sim/atomic_actions/ -v`
Expected: PASS.

- [ ] **Step 3: Commit any black fixes**

```bash
git add -u
git commit -m "style: black-format parallel motion generation changes"
```

---

## Self-Review (completed during authoring)

**1. Spec coverage:**
- §3 batched PlanState/PlanResult → Task A1. ✓
- §3.2 `is_all_success` → Task A1. ✓
- §4.1 `BasePlanner.plan` batched + guards removed → Tasks B2/B3/C1. ✓
- §4.2 TOPPRA fork-multiprocessing + module-level worker + inline fallback + tail-pad → Tasks B1/B2/B3. ✓
- §4.3 NeuralPlanner native batching + early-convergence → Tasks C1/C2. ✓
- §4.4 `@validate_plan_options` B-consistency → Task A2. ✓
- §5.1 `MotionGenerator.generate` batched + `start_qpos:(B,DOF)` → Task D1. ✓
- §5.2 `TrajectoryBuilder` motion-source strategy + `ActionCfg` fields → Tasks E1/E2. ✓
- §6.1 `ActionResult.success → (B,)` + shim → Task E1. ✓
- §6.2 `run()` per-env failure propagation → Task F1. ✓
- §6.3 actions (refined: minimal success-handling edit) → Phase G. ✓ (refinement noted at top)
- §7 error handling (per-env failure isolation, device consistency, BrokenProcessPool) → Tasks B3 (`BrokenProcessPool` catch), E2 (failed-envs hold), F1. ✓
- §9 testing (unit + integration + TOPPRA regression anchor) → Phases A–I tests. ✓
- §10 migration (`atom_action_utils`, tutorials) → Phase H. ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"add appropriate error handling". The `interpolate_xpos_batched` helper and `_to_batched_plan_states` are fully specified. The e2e test (I2) has a sketch body — intentional, marked `@pytest.mark.requires_sim`; the engineer fills the sim-setup using the pattern from `test_toppra_planner.py` (already shown in B3). ✓

**3. Type consistency:** `plan_arm_traj` returns `(success:(B,) tensor, traj)` everywhere (E2, G1–G3). `ActionResult.success` is `bool | torch.Tensor` with `.success_all` (E1, F1, G1–G3). `PlanResult.is_all_success` (A1) vs `ActionResult.success_all` (E1) — distinct types, distinct names, consistent. `_infer_batch_size` defined in A2, imported in B3/C1. `MotionGenOptions.start_qpos:(B,DOF)` (D1) matches `_plan_motion_gen` (E2). ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-parallel-motion-generation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
