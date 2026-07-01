# Parallel Motion Generation for EmbodiChain — Design Spec

**Date:** 2026-07-02
**Status:** Approved (brainstorming complete)
**Scope:** Env-batched (`B = num_envs`) parallel motion generation across the planner (`BasePlanner`), the motion-generation facade (`MotionGenerator`), and the atomic-action layer (`atomic_actions/`).

---

## 1. Context & Motivation

EmbodiChain's simulation module (`embodichain/lab/sim/`) already parallelizes IK across vectorized environments — the Warp-GPU solvers (SRS, OPW, UR) and torch-batched solvers (Pytorch, Neural, Differential) all accept a leading `B = num_envs` batch dimension and solve all envs in one call.

The motion-generation stack sitting *on top of* IK is **not** parallel:

- `BasePlanner.plan(target_states: list[PlanState]) -> PlanResult` operates on a **single trajectory**. `PlanState` holds `qpos:(DOF,)` / `xpos:(4,4)` — no batch dim. `PlanResult.positions:(N,DOF)` — no batch dim.
- `ToppraPlanner` (numpy/scipy, CPU-bound) and `NeuralPlanner` (torch `nn.Module`) **both explicitly reject** `robot.num_instances > 1` (`toppra_planner.py:88`, `neural_planner.py:209`).
- `MotionGenerator.generate()` is a single-trajectory facade. Its `interpolate_trajectory` even forces an artificial `n_envs=1` unsqueeze when calling batch-IK (`motion_generator.py:571`). (One outlier, `estimate_trajectory_sample_count`, already accepts `[B,N,4,4]`.)
- The **new** typed atomic-action layer (`atomic_actions/`) is env-batched via `TrajectoryBuilder` (IK + linear interpolation) and returns `(n_envs, n_waypoints, robot.dof)` trajectories — but it **never calls the planner**. Steps in `AtomicActionEngine.run()` are iterated **sequentially**. The legacy `atom_actions.py` / `action_bank` path *does* call `MotionGenerator.generate()` but is single-env.

**Goal:** make motion generation and atomic actions env-batched — `(B, N, DOF)` — with both motion sources (planner-based `MotionGenerator` *and* IK+interpolation) available to the atomic-action layer.

---

## 2. Design Decisions (settled in brainstorming)

| # | Decision | Choice |
|---|---|---|
| D1 | Primary batch axis | **Across sim envs** (`B = num_envs`), tensor shape `(B, N, DOF)`. |
| D2 | TOPPRA strategy (CPU-bound, serial per traj) | **Multiprocessing fan-out** — `B` independent single-env TOPPRA solves across a process pool. |
| D3 | TOPPRA multiprocessing context | **`fork`** — TOPPRA compute is pure CPU/numpy; workers never touch CUDA/torch/sim, so the fork-after-CUDA hazard does not apply to the worker execution path. `spawn` retained as an overrideable escape hatch. |
| D4 | MotionGenerator ↔ atomic-action relationship | **Both motion sources available.** `TrajectoryBuilder` becomes strategy-based: per-action `motion_source` cfg selects `ik_interp` (default, current path) or `motion_gen` (delegates to batched `MotionGenerator`). Both return the same `(B, N, DOF)` trajectory. |
| D5 | Batched API vs. existing single-env API | **In-place.** `PlanState`/`PlanResult`/`plan()`/`generate()` gain a leading `B` dim natively; `num_instances>1` guards removed; existing callers migrated. |
| D6 | Batch representation in planner interface | **Tensorize `PlanState`/`PlanResult` with leading `B`** (Approach A). The `list[PlanState]` indexes waypoints; each `PlanState` carries all `B` envs. Preserves per-waypoint semantics (MoveType, gripper, move_part); GPU-friendly. |

---

## 3. Batched Data Model (`PlanState` / `PlanResult`)

Location: `embodichain/lab/sim/planners/utils.py`.

### 3.1 `PlanState` — fields gain a batch dim

| Field | Current | New |
|---|---|---|
| `qpos` | `(DOF,)` or `None` | `(B, DOF)` or `None` |
| `xpos` | `(4, 4)` or `None` | `(B, 4, 4)` or `None` |
| `qvel` / `qacc` | `(DOF,)` | `(B, DOF)` |
| `move_type: MoveType` | scalar enum | scalar enum (**shared across `B`**) |
| `move_part: MovePart` | scalar | scalar (shared) |
| `is_open`, `is_world_coordinate`, `pause_seconds` | scalar | scalar (shared) |

**Shared scalars across `B`:** vectorized envs run the same task skeleton (same move types, same gripper open/close sequence, same `move_part`); only numeric targets differ. If a future use case needs per-env move-type divergence, the enums become `(B,)` tensors — explicitly deferred (see §8).

**New convenience constructors:**
- `PlanState.from_qpos(qpos:(B,DOF), move_type, move_part, ...)` and `PlanState.from_xpos(xpos:(B,4,4), ...)`.
- `PlanState.single(...)` — B=1 ctor for the common single-env case (replaces ad-hoc `.unsqueeze(0)` at call sites).

### 3.2 `PlanResult` — batched outputs

| Field | Current | New |
|---|---|---|
| `success` | `bool` | `(B,)` bool tensor — per-env success mask |
| `positions` | `(N, DOF)` | `(B, N, DOF)` |
| `velocities` / `accelerations` | `(N, DOF)` | `(B, N, DOF)` |
| `xpos_list` | `(N, 4, 4)` | `(B, N, 4, 4)` |
| `dt` | `(N,)` | `(B, N)` — TOPPRA produces per-env optimal timing |
| `duration` | `float` | `(B,)` |

**Helper:** `PlanResult.is_all_success() -> bool` preserves old boolean-returning ergonomics.

**Unchanged:** `is_satisfied_constraint` (`base_planner.py:140`) already accepts `(B,N,DOF)`.

---

## 4. Batched Planner Interface

### 4.1 `BasePlanner.plan()`

Location: `embodichain/lab/sim/planners/base_planner.py:115`.

```python
@validate_plan_options
@abstractmethod
def plan(
    self,
    target_states: list[PlanState],   # len = num_waypoints; each PlanState carries B envs
    options: PlanOptions = PlanOptions(),
) -> PlanResult:                       # batched: positions (B, N, DOF), success (B,)
```

The `num_instances > 1` guards in `ToppraPlanner` (line 88) and `NeuralPlanner` (line 209) are **removed**. `B` is inferred from `target_states[0].qpos.shape[0]` (or `.xpos`), validated consistent across the list and against `robot.num_instances`.

### 4.2 ToppraPlanner — fork-multiprocessing fan-out

TOPPRA stays scipy/numpy per-env; we fan out across `B`.

**Module-level pure-numpy worker** (toppra_planner.py, top-level so it is picklable for both fork and spawn):

```python
def _toppra_solve_one_env(
    waypoints: np.ndarray,          # (N, DOF)
    vel_constraint,
    acc_constraint,
    sample_method: TrajectorySampleMethod,
    sample_interval: float | int,
) -> dict:
    # Builds SplineInterpolator + TOPPRA, solves, samples. Returns
    # {"positions": (N_b, DOF), "velocities": ..., "accelerations": ...,
    #  "dt": (N_b,), "success": bool, "n": N_b}
```

TOPPRA needs no robot/FK — only waypoints + per-DoF limits — so this separation is clean. `self`/`robot`/sim objects are **never** sent to workers.

**Pool management:**
- `mp.get_context("fork")` (D3). Workers inherit already-imported `toppra`/`scipy`/`numpy` — cold-start drops to ~ms.
- Pool created **lazily** on first `plan()`, **long-lived**, reused across calls.
- `ToppraPlannerCfg.max_workers: int | None = None` — default `min(os.cpu_count() // 2, B)` at first call (leaves cores for physics/CPU-sim to avoid oversubscription); user-tunable.
- `ToppraPlannerCfg.mp_context: str = "fork"` — overrideable to `"spawn"` as the escape hatch if a future TOPPRA backend touches CUDA or a third-party lib raises fork-related driver warnings.
- `close()` method + `atexit` registration + `__del__` guard for teardown.

**Why fork is safe here:** workers only run `_toppra_solve_one_env` — pure numpy/scipy, no torch, no CUDA, no sim. The classic fork-after-CUDA hazard is "fork a process *and then call CUDA in the child*"; we don't. The lazy pool is created after sim init, so children inherit warmed sim state in COW memory — fine for read-only TOPPRA compute; workers must not mutate shared state (they return fresh numpy arrays).

**Inline fallback:** `B == 1` or `max_workers == 1` → run `_toppra_solve_one_env` in-process (no IPC). Same function, so trivially testable as the numerical reference.

**Fan-out + collect:**
1. Unstack waypoint list into `B` per-env arrays: `waypoints_per_env[b] = stack([s.qpos[b] for s in target_states])` → `(N, DOF)` numpy.
2. Submit `B` tasks to the pool.
3. Each returns its own `N_b`. `QUANTITY` sampling → all `N_b` equal → stack to `(B, N, DOF)`. `TIME` sampling → `N_b` varies → **pad shorter by repeating the final waypoint** to `max_N`, record real `duration:(B,)` so callers know the per-env endpoint.
4. Per-env exceptions → `success[b] = False`, `positions[b]` filled with `start_qpos` repeated, `duration[b] = 0`; other envs proceed. Re-raised only if **all** envs fail (likely config error).
5. `BrokenProcessPool` → all `success = False`, pool torn down and rebuilt on next call, error logged; sim does not crash.
6. Main process converts numpy → `torch` on `self.device`, assembles batched `PlanResult`.

### 4.3 NeuralPlanner — native batching

The actor is already a torch `nn.Module` under `torch.no_grad()`. Changes:
- Drop the `(1, ...)` shaping (`neural_planner.py:334`, `:451`) — use `(B, ...)` throughout.
- Waypoint tokens: `(B, num_waypoints, 3)` positions / `(B, num_waypoints, 4)` quats (currently built at `:397-413` for B=1).
- Autoregressive rollout loop (`:341`) iterates `max_steps`; each step's transformer forward and per-env reach checks (`pos_eps`/`rot_eps`) operate on `(B, ...)`. Loop continues until **all** envs satisfy reach **or** `max_steps`; envs that converge early **hold their last qpos** (no further updates). `success:(B,)` marks converged envs.
- `xpos_list` filled via the batched `robot.get_fk` (already batched, `base_solver.py:433`).

### 4.4 `@validate_plan_options` (extended)

Asserts: all `PlanState` share the same `B`; `B == robot.num_instances` (or `1`); `len(target_states) >= 2`; `move_type` consistent across the list where the subclass requires (NeuralPlanner: all `EEF_MOVE`). Mismatch → `ValueError` naming the offending field and shapes.

### 4.5 Unchanged

`BasePlannerCfg` (`robot_uid`, `planner_type`), `PlanOptions` base, the `register_planner_type` registry, `is_satisfied_constraint`, and the two `PlanOptions` subclasses. Only `ToppraPlannerCfg` gains `max_workers` and `mp_context`.

---

## 5. Batched `MotionGenerator` + Dual Motion-Source `TrajectoryBuilder`

### 5.1 `MotionGenerator.generate()` — batched in-place

Location: `embodichain/lab/sim/planners/motion_generator.py:137`.

```python
def generate(
    self,
    target_states: list[PlanState],          # list over waypoints; each carries B envs
    options: MotionGenOptions = MotionGenOptions(),
) -> PlanResult:                             # batched: positions (B, N, DOF), success (B,)
```

`MotionGenOptions` (`:53`) gains a batch dim on `start_qpos: (B, DOF)` (was `(DOF,)`); all other fields stay scalar/shared (`control_part`, `plan_opts`, `is_interpolate`, `interpolate_nums`, `is_linear`, `interpolate_position_step`, `interpolate_angle_step`).

**Internal flow** (rework of `:154-233`):
1. **Pre-interpolation branch** (`is_interpolate=True`): operate on `(B, num_seg, 4, 4)` / `(B, num_seg, DOF)` directly. The per-segment Python loop stays (segment count small, GPU-launch overhead dominates) but each iteration processes all `B` envs. `robot.compute_batch_ik` is already `(n_envs, ...)`-native — drop the `n_envs=1` unsqueeze (`:571`). Output: densified `(B, N_interp, DOF)` `PlanState`s. If no `plan_opts`, return directly as a batched `PlanResult` (preserves the `:204-210` short-circuit, now batched).
2. **Planner branch**: `self.planner.plan(target_states, options.plan_opts)` — now batched. `start_qpos:(B,DOF)` passed through (NeuralPlanner consumes it as rollout seed).
3. `estimate_trajectory_sample_count` (`:235`) already accepts `[B, N, 4, 4]` — used as-is.

`_support_planner_dict` and `register_planner_type` unchanged.

### 5.2 `TrajectoryBuilder` — motion-source strategy

Location: `embodichain/lab/sim/atomic_actions/trajectory.py`.

**New `ActionCfg` fields** (`atomic_actions/core.py:180`):

```python
@configclass
class ActionCfg:
    name: str = "default"
    control_part: str = "arm"
    interpolation_type: str = "linear"
    velocity_limit: float | None = None
    acceleration_limit: float | None = None
    motion_source: str = "ik_interp"   # "ik_interp" (default) | "motion_gen"
    planner_type: str | None = None    # "toppra" | "neural"; required when motion_source="motion_gen"
```

- `motion_source="ik_interp"` (default): current path — `plan_arm_traj` calls `robot.compute_ik` (already `(B, ...)`) + `interpolate_with_distance`. **Zero behavior change.**
- `motion_source="motion_gen"`: build a batched `list[PlanState]` from the action's waypoint targets and call `self.motion_generator.generate(...)`. Returned `(B, N, DOF)` `PlanResult.positions` is the trajectory segment — same shape the builder already produces, so downstream full-DoF embedding / hand-qpos interpolation / `WorldState` threading are **unchanged and mode-agnostic**.

**`TrajectoryBuilder.plan_arm_traj`** (`trajectory.py:313`) gains the branch:

```python
def plan_arm_traj(self, target_states, state, cfg):
    if cfg.motion_source == "ik_interp":
        return self._plan_ik_interp(target_states, state, cfg)   # current body, extracted
    elif cfg.motion_source == "motion_gen":
        plan_states = self._to_plan_states(target_states, state, cfg)   # list[PlanState], batched
        plan_opts = self._build_plan_opts(cfg)
        result = self.motion_generator.generate(
            plan_states,
            MotionGenOptions(start_qpos=state.last_qpos,
                             control_part=cfg.control_part,
                             plan_opts=plan_opts),
        )
        return result.positions, result.success          # (B, N, DOF), (B,)
    else:
        raise ValueError(f"unknown motion_source {cfg.motion_source}")
```

- `_to_plan_states` translates typed targets: `EndEffectorPoseTarget` → `EEF_MOVE` `PlanState`s with `xpos:(B,4,4)`; `JointPositionTarget` → `JOINT_MOVE`.
- `_build_plan_opts` maps `ActionCfg.velocity_limit`/`acceleration_limit`/`sample_interval` → `ToppraPlanOptions`/`NeuralPlanOptions`.
- The builder already holds a `motion_generator` reference (passed from `AtomicActionEngine`, `engine.py:73`) — it is now *used* on the `motion_gen` path. If `motion_gen` is selected and no `MotionGenerator` is wired → `ValueError` at `execute` time.

---

## 6. Atomic Action Layer Changes

### 6.1 `ActionResult.success` → per-env

`ActionResult.success` becomes `torch.Tensor` shape `(B,)` bool.

**Backward-compat shim:**
- Property `success_all: bool` = `bool(success.all())`.
- `__bool__` delegates to `success_all` (with `DeprecationWarning`) so `if result:` keeps working during migration.

### 6.2 `AtomicActionEngine.run()` — per-env failure propagation

Location: `embodichain/lab/sim/atomic_actions/engine.py:89`.

Revised loop:

```python
alive = torch.ones(B, dtype=bool, device=device)   # envs still executing
for name, target in steps:
    if not alive.any():
        break
    result = action.execute(target, state)
    prev_last_qpos = state.last_qpos.clone()
    alive = alive & result.success                   # (B,)
    concat_traj = cat(concat_traj, result.trajectory, dim=1)
    state = result.next_state
    # failed envs freeze at their last successful qpos
    state.last_qpos = where(alive[:, None], state.last_qpos, prev_last_qpos)
return alive, concat_traj, state                     # success is (B,)
```

**Holding behavior for failed envs:** a failed env at step *i* does not execute step *i+1*'s motion. Failed envs' trajectory rows for subsequent steps are filled by **repeating their last successful qpos** (held pose); `state.last_qpos` frozen via `where`. Keeps the `(B, N_total, DOF)` tensor rectangular (no raggedness) — physically sensible (failed env holds still while successful envs finish).

**Return contract:** `run()` returns `success: torch.Tensor (B,)` instead of `bool`. Callers do `bool(success.all())` or use a new `engine.run(...).all_success` property. Tutorial migration (`scripts/tutorials/atomic_action/move_end_effector.py:198`) is a one-liner.

### 6.3 What does NOT change

- The 6 action classes (`MoveEndEffector`, `MoveJoints`, `PickUp`, `MoveHeldObject`, `Place`, `Press`) — no edits. The builder's branch is invisible to them.
- Typed targets (`core.py:69-128`) — already `(B, ...)`.
- `WorldState` — already `(B, ...)`.
- `register_action` / per-engine registry.
- The legacy `atom_actions.py` layer — untouched (deprecated).
- `action_bank/` (gym task scheduler) — untouched in v1 (inherits batched API for free once it passes batched `PlanState`s; its own migration is a separate caller-update task).
- IK solvers — none needed; already batched.

---

## 7. Error Handling & Edge Cases

**Planner-level:**
- `B` consistency: `@validate_plan_options` asserts all `PlanState` share the same leading `B`, and `B == robot.num_instances` (or `1`). Mismatch → `ValueError` naming the offending field and shapes.
- Empty / single-waypoint `target_states`: `len < 2` → `ValueError` (guard in `BasePlanner.plan` before dispatch).
- TOPPRA worker exception: caught per-env; `success[b]=False`, `positions[b]` = `start_qpos` repeated, `duration[b]=0`; other envs proceed. Re-raised only if all envs fail.
- TOPPRA infeasible trajectory (limits too tight): TOPPRA returns failure → same per-env handling.
- NeuralPlanner non-convergence: `success[b]=False`, return last predicted qpos (best-effort).
- Pool failure (`BrokenProcessPool`): all `success=False`, pool torn down and rebuilt on next call, error logged; sim does not crash.

**TrajectoryBuilder / engine level:**
- `motion_source="motion_gen"` with no `MotionGenerator` wired → `ValueError` at `execute` time.
- `planner_type` missing/unknown when `motion_source="motion_gen"` → `ValueError` listing registered planner types.
- Mixed `motion_source` across steps in one `run()`: explicitly supported; no special handling.
- `start_qpos` shape mismatch (`(DOF,)` vs `(B,DOF)`): `generate` unsqueezes `(DOF,)`→`(1,DOF)` and broadcasts, with a warning if `B>1` (likely caller bug).

**Waypoint-count divergence (TIME sampling):** pad shorter per-env trajectories by repeating the final waypoint to `max_N`; record `duration:(B,)`. Document that trailing padded rows are **held poses**, not continued motion.

**Device consistency:** all `PlanResult` tensors on `self.device`. TOPPRA workers return numpy; main process converts. NeuralPlanner stays on device throughout. No silent cross-device copies.

---

## 8. Scope Guardrails (explicit out-of-scope for v1)

- `MoveJoints` with `motion_source="motion_gen"` (TOPPRA on joint-space moves) — deferred.
- Per-env divergent `move_type` (currently shared scalars across `B`) — deferred.
- `action_bank/` task-graph integration with the new batched atomic actions — deferred.
- Any change to the IK solvers — none needed.

---

## 9. Testing Strategy

Layered from cheap/unit → expensive/integration. Follows project conventions (`/add-test` skill, `tests/`, pytest).

### 9.1 Unit (no sim, no GPU) — `tests/lab/sim/planners/`

1. **`PlanState`/`PlanResult` batched shapes** — construct with `(B,DOF)`/`(B,4,4)`, assert output shapes `(B,N,DOF)`; `from_qpos`/`from_xpos`/`single` ctors.
2. **`@validate_plan_options`** — rejects mismatched `B`, mismatched `robot.num_instances`, `len<2`, mixed `move_type` for NeuralPlanner.
3. **`_toppra_solve_one_env`** (module-level worker, inline) — pure-numpy: correct keys/shapes; matches a reference scipy TOPPRA solve on a fixed seed (regression baseline saved as fixture).
4. **`ToppraPlanner` fan-out** — monkeypatch `_toppra_solve_one_env` to a deterministic stub; assert `B` calls made, results stacked to `(B,N,DOF)`, `QUANTITY`→uniform N, `TIME`→tail-pad to `max_N` with correct `duration`. Inline fallback (B=1) asserted separately.
5. **`NeuralPlanner` batching** — tiny random `nn.Module` actor fixture; assert `(B,...)` shapes, early-convergence envs hold while others continue, `success:(B,)` correct.
6. **`TrajectoryBuilder._to_plan_states` / `_build_plan_opts`** — typed-target → `PlanState` translation per target type; cfg → `PlanOptions` mapping.
7. **`ActionResult`** — `success:(B,)`, `success_all`, `__bool__` deprecation path.

### 9.2 Integration (mocked robot, no GPU) — `tests/lab/sim/atomic_actions/`

8. **`MoveEndEffector` both motion sources** — mocked `robot.compute_ik` and mocked `MotionGenerator.generate`; assert both paths return `(B,N,DOF)` trajectory and identical `WorldState` threading.
9. **`AtomicActionEngine.run` per-env failure** — 3-step plan, inject failure on env 2 at step 2; assert env 2's step-3 trajectory rows = held last qpos, envs 0/1 complete, returned `success == [True, False, True]`.
10. **Mixed motion-source steps** — `[(move_ee, ik_interp), (move_ee, motion_gen)]` in one `run()`; assert concat along dim=1, no shape errors.

### 9.3 End-to-end (real sim, gated) — `tests/lab/sim/integration/`

11. **TOPPRA vs NeuralPlanner vs ik_interp reach-equivalence** — on a 2-waypoint EE move, all three produce trajectories that reach the target within tolerance (FK-checked). Not bit-exact; reach-equivalent. `slow` / `requires-sim` markers; skipped in CI fast path.
12. **Throughput sanity** — `B=64`, assert `generate` completes in <T seconds (loose bound; guards against accidental serialization regressions). Not a hard perf SLA.
13. **Fork-safety smoke** — run TOPPRA batched alongside an initialized sim context; assert no crash/hang.

**TOPPRA numerical regression anchor:** the saved fixture (#3) is the key correctness anchor — multiprocessing fan-out must produce results identical (within fp tol) to the inline single-env solve, since both call the same `_toppra_solve_one_env`.

---

## 10. Migration Notes

- Callers of `plan()`/`generate()` passing single-env `PlanState`s: add `.unsqueeze(0)` (B=1) or use `PlanState.single(...)`.
  - `grep -rn "\.plan(" embodichain/` ; `grep -rn "\.generate(" embodichain/`
- `embodichain/lab/sim/utility/atom_action_utils.py:199` (`plan_trajectory`) — migrate to batched `PlanState`.
- `action_bank` callers — deferred (out of scope), but note they'll need migration when they adopt batched envs.
- Tutorials (`scripts/tutorials/atomic_action/`) — `bool(is_success)` → `is_success.all()` or `result.success_all`.

---

## 11. Summary

- **§3**: batched `PlanState`/`PlanResult` with leading `B` dim, shared scalar enums.
- **§4**: `BasePlanner.plan` batched; TOPPRA via fork-multiprocessing fan-out of a pure-numpy module-level worker (lazy long-lived pool, inline B=1 fallback); NeuralPlanner natively batched with early-convergence holds.
- **§5**: `MotionGenerator.generate` batched in-place; `TrajectoryBuilder` gains `motion_source` strategy (`ik_interp` default, `motion_gen` opt-in) reading from `ActionCfg`.
- **§6**: `ActionResult.success → (B,)`; `AtomicActionEngine.run` per-env failure propagation via held qpos; the 6 actions untouched.
- **§7/§9**: per-env error isolation, device consistency, layered tests with a TOPPRA numerical-regression anchor.
