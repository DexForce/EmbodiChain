# Newton Physics Backend PR — Design

Date: 2026-06-21
Branch: `feature/newton-physics-backend`
Companion: `design/newton-backend-design.md` (current-state record)

## 1. PR Targets and Scope

The PR has five targets:

1. Integrate the Newton physics backend on top of `dexsim` (`/root/sources/dexsim`).
2. Implement `RigidObject`, `Articulation`, and `Robot` on Newton.
3. Refactor `embodichain/lab/sim/cfg.py` to support both backends, including
   Newton solver configuration.
4. Support multiple-env parallel simulation on Newton.
5. Support a differentiable env for analytic policy gradient (APG), in the
   style of `dexsim/python/dexsim/engine/newton_physics/differentiable_stepper.py`.

### Status going into this design

Targets 1, 2, 3 are **already complete on the branch** (see
`design/newton-backend-design.md`):

- `PhysicsBackend` ABC + registry; `DefaultPhysicsBackend` / `NewtonPhysicsBackend`.
- `DefaultPhysicsCfg` / `NewtonPhysicsCfg` with full solver dispatch
  (`mujoco_warp` / `xpbd` / `semi_implicit` / `featherstone` / `vbd`),
  `requires_grad`, `broad_phase`, `visualizer_enabled`,
  `NewtonCollisionAttributesCfg`.
- Newton `RigidObject`, `Articulation`, `Robot` with batch views, runtime
  attribute mutation, per-link mass live push.
- Capability matrix pinned by `tests/sim/test_backend_parity.py`.
- Newton finalize/invalidate lifecycle owned by `NewtonPhysicsBackend`.

Targets 4 and 5 are **outstanding** and are the focus of this design.
`cfg.py` is otherwise left alone — Phase 3b legacy-`PhysicalAttr` removal is
deferred.

## 2. Target 4 — Multi-Env Parallel Simulation on Newton

### Mechanism

`dexsim` already exposes the primitive we need:
`arena_src.clone_arena_to(arena_i)`. The pattern (see
`/root/sources/dexsim/examples/python/physics/basic/hello_newton.py`) is:

1. Build a source arena and populate it with rigid bodies / articulations.
2. Add `num_envs - 1` additional empty arenas.
3. Call `arena_src.clone_arena_to(arena_i)` for each.
4. Newton finalize then sees `num_envs` parallel bodies and builds a single
   batched model.

EmbodiChain already builds `num_envs` arenas in
`SimulationManager._build_multiple_arenas` but does not clone — the existing
default-backend pattern is to call `add_*` once per `arena_index`. We add the
clone path on Newton only.

### User-facing API

No new public API. The flow is:

```python
sim_cfg = SimulationManagerCfg(
    physics_cfg=NewtonPhysicsCfg(...),
    num_envs=4,
)
sim = SimulationManager(sim_cfg)
sim.add_rigid_object(cube_cfg)       # spawns into arena_0 (source)
sim.add_robot(robot_cfg)             # spawns into arena_0 (source)
sim.finalize_newton_physics()        # clones arena_0 -> 1..3, then finalizes
```

Spawning with `cfg.arena_index > 0` on Newton raises with the message
"Newton spawn must target the source arena (arena_index in {-1, 0}); per-env
clones are produced at finalize." `arena_index == -1` (global) and
`arena_index == 0` both route to arena_0 on Newton.

### Implementation

`NewtonPhysicsBackend` gains an `_arenas_cloned: bool` flag (init `False`).
`prepare()` is extended:

```python
def prepare(self) -> None:
    if self._is_finalized and self._lifecycle_state() == "READY":
        return
    self._clone_source_arena_if_needed()
    # ... existing ensure_simulation_prepared_lazy + rebuild_newton_from_scene ...

def _clone_source_arena_if_needed(self) -> None:
    arenas = self._manager._arenas
    if len(arenas) <= 1 or self._arenas_cloned:
        return
    source = arenas[0]
    for arena in arenas[1:]:
        if self._arena_is_empty(arena):
            source.clone_arena_to(arena)
    self._arenas_cloned = True
```

`invalidate()` resets `_arenas_cloned` **only when scene topology changes**.
The two cases:

- Topology change (`add_*` / `remove_*`): the corresponding `SimulationManager`
  paths already call `self.physics.invalidate()`; we extend that to also
  clear `_arenas_cloned` so the next `prepare()` re-clones into the (possibly
  new) arenas. Attribute writes (`set_mass`, pose setters) keep
  `_arenas_cloned = True`.

Spawn guards live in `embodichain/lab/sim/utility/sim_utils.py`. Each
`add_rigid_object` / `add_articulation` / `add_robot` Newton path adds a
single guard:

```python
if _is_newton_backend_active() and cfg.arena_index > 0:
    logger.log_error(
        "Newton spawn must target the source arena "
        "(arena_index in {-1, 0}); per-env clones are produced at finalize."
    )
```

### Object Backend Views — Multi-Env Body-ID Resolution

`NewtonRigidBodyView` and `NewtonArticulationView` (`embodichain/lab/sim/
objects/backends/newton.py`) currently lazy-resolve a single body ID per
entity. We extend the resolver to return a `[num_envs]` index tensor after
finalize:

- Each `RigidObject` / `Articulation` records its `entity_name` from arena_0.
- After clone, dexsim produces parallel entities in arenas 1..N-1 with
  predictable per-arena names (the `clone_arena_to` namespacing scheme).
- The Newton view queries the finalized model's body/articulation registry
  for every name variant and assembles the `[num_envs]` tensor on the
  configured device.
- Pre-finalize fallback returns the scalar arena_0 ID, matching today's
  BUILDER-state behavior. The scalar path is kept for code that runs before
  finalize.

All batched accessors (`get_body_state`, `set_local_pose`,
`apply_force_torque`, ...) already accept `env_ids` and operate on
`[num_envs, ...]` tensors; the only changes are in the view's
`_resolve_body_ids` and a `_num_envs` field plumbed in from the manager.

### Default Backend

Default backend behavior is **unchanged**. The existing pattern (one `add_*`
per `arena_index`) continues to work. Source-arena cloning on the default
backend is deferred.

### Tests

`tests/sim/test_newton_multi_env.py` (new):

- Spawn a dynamic cube and a Franka URDF into arena_0 with `num_envs=4`.
- Finalize; verify rigid-object and articulation `get_body_state` return
  shape `[4, ...]` with positions offset by the arena grid spacing.
- Step 10 substeps; verify per-env states diverge under per-env force
  application.
- Spawn with `arena_index=1` raises.
- Mutating an attribute (`set_mass`) does not trigger re-clone (assert
  `_arenas_cloned` stays `True`); adding a new asset does (assert it
  becomes `False`).

## 3. Target 5 — DifferentiableEmbodiedEnv (APG)

### Reference Pattern

`/root/sources/analytic_policy_gradients/envs/franka_reach_env.py` shows
the bridge pattern: a `torch.autograd.Function` (`_NewtonStepFunc`) opens a
`wp.Tape()`, launches Warp kernels in the forward, saves the tape, and runs
`tape.backward()` in the backward to extract `action.grad`. The franka
example bypasses dynamics and takes the gradient through FK only (because
the Featherstone solver does not propagate gradients through control).

EmbodiChain will take the dynamics-grad path using
`dexsim.engine.newton_physics.DifferentiableStepper` with the
`semi_implicit` solver — this is the configuration `requires_grad=True`
already requires (see `NewtonPhysicsCfg.to_dexsim_cfg`). The FK-only path
is deferred as a future `grad_mode="kinematic"` option.

### Module Layout

- `embodichain/lab/sim/diff/__init__.py` — public surface.
- `embodichain/lab/sim/diff/bridge.py` — `_NewtonStepFunc(torch.autograd.Function)`,
  `differentiable_step(manager, action, substeps)` helper, `tape_context(manager)`
  context manager.
- `embodichain/lab/gym/envs/differentiable_env.py` — `DifferentiableEmbodiedEnv`
  subclass.
- `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` — example task.

`SimulationManager` gains two thin delegators (default backend raises):

```python
def create_differentiable_stepper(self):
    return self.physics.newton_manager.create_differentiable_stepper()

def create_gradient_rollout(self, *args, **kwargs):
    return self.physics.newton_manager.create_gradient_rollout(*args, **kwargs)
```

### DifferentiableEmbodiedEnv Contract

Construction validates the Newton requires-grad config:

```python
if not isinstance(cfg.sim_cfg.physics_cfg, NewtonPhysicsCfg):
    log_error("DifferentiableEmbodiedEnv requires Newton backend.")
if not cfg.sim_cfg.physics_cfg.requires_grad:
    log_error("DifferentiableEmbodiedEnv requires requires_grad=True.")
# solver_type=='semi_implicit' is already enforced by NewtonPhysicsCfg.to_dexsim_cfg.
```

### Step Pipeline

`step(action)` is overridden:

```python
def step(self, action):
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
    obs, reward, terminated, truncated, info = _NewtonStepFunc.apply(
        action, self._sim_state_dict()
    )
    # auto-reset done envs with torch.where, preserving gradient on live envs
    ...
    return obs, reward, terminated, truncated, info
```

Inside `_NewtonStepFunc.forward`:

1. Open `wp.Tape()`.
2. Apply the action to drive targets via a Warp kernel (replaces direct
   `set_drive_target` calls in `_step_action` where those calls are not
   tape-recorded — to be confirmed during implementation; if dexsim's
   drive setter already records into the tape, we skip the replacement).
3. Run `DifferentiableStepper.step(state_in, state_out, control, contacts,
   dt)` for `sim_steps_per_control` substeps, swapping `state_in`/`state_out`.
4. Evaluate the observation and reward managers, reading `joint_q` /
   `body_q` via `wp.to_torch` (zero-copy, autograd-aware).
5. Save `tape`, grad-tracked Warp arrays, and metadata in `ctx`.

`_NewtonStepFunc.backward`:

1. Copy upstream `grad_reward` / `grad_obs` into Warp tensors' `.grad`.
2. `ctx.tape.backward()`.
3. Return `wp.to_torch(action_wp.grad)` reshaped to the action shape.
4. `ctx.tape.zero()`.

### Reward and Observation Functors

Existing functors that read tensors via `wp.to_torch` or torch operations on
manager-provided state are autograd-compatible by construction (Warp tape
sees the kernel launches; torch ops just compose). Functors that detour
through CPU / NumPy break the graph and will be flagged. The audit is
scoped to the example task's needs; a full functor audit is out of scope.

The constraint is documented in `agent_context/` (new topic
`differentiable-env`) so future functor authors know the rule.

### Reset Path

`reset()` is non-differentiable. Wrap in `torch.no_grad()`, detach any
tensors written into Warp state. Auto-reset on `done` follows the franka
example: compute `obs_after_step`, then where `done_mask`:
`obs = torch.where(done_mask.unsqueeze(-1), fresh_obs.detach(), obs)`.
Live envs keep their gradient connection to the upstream action.

### Memory and Truncation

Each tape records all substeps in a single env step. For long
`sim_steps_per_control` or large `num_envs`, GPU memory can grow quickly.
`DifferentiableEmbodiedEnv` accepts an optional `truncate_backward_at`
argument (default `None` = full env step). When set, the tape is split
into chunks of N substeps; chunk boundaries are detached. This is a knob,
not a default behavior change.

### Example Task

`embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` mirrors the
APG reference env but is built on EmbodiChain primitives:

- Franka FR3 URDF spawned via `add_robot` into arena_0.
- `num_envs = 4`, `NewtonPhysicsCfg(requires_grad=True, solver_cfg={"solver_type":"semi_implicit"})`.
- Observation: joint positions + EE pose + target pose + last action.
- Reward: position+orientation tracking matching the reference env.
- `DifferentiableEmbodiedEnv` subclass overriding only task-specific
  reward/obs construction.

### Tests

`tests/gym/envs/test_differentiable_env.py`:

1. Constructing `DifferentiableEmbodiedEnv` with `requires_grad=False`
   raises.
2. `obs`/`reward` returned from `step(action)` have `requires_grad=True`
   and a non-None `grad_fn`.
3. `loss = reward.sum(); loss.backward()` produces `action.grad` of
   shape `[num_envs, action_dim]` with finite, non-zero values.
4. Finite-difference parity: per-env autograd gradient matches a
   two-sided finite-difference estimate within tolerance on a 2-step
   rollout (loose tolerance — `rtol=1e-1, atol=1e-2` — since the
   semi-implicit solver is not a smooth function of action).
5. One APG iteration reduces the smoke loss.

`tests/sim/test_differentiable_stepper.py`:

1. `manager.create_differentiable_stepper()` raises on default backend.
2. On Newton with `requires_grad=False`, raises with a clear message.
3. On Newton with `requires_grad=True`, one `step()` produces tape-recorded
   buffers and `tape.backward()` is callable.

## 4. SimulationManager and Backend Changes Summary

```
embodichain/lab/sim/physics/newton.py
    NewtonPhysicsBackend
        + _arenas_cloned: bool
        + _clone_source_arena_if_needed()
        + _arena_is_empty(arena)
        ~ prepare()              (call clone helper before rebuild)
        ~ invalidate()           (clear _arenas_cloned only on topology change)

embodichain/lab/sim/sim_manager.py
        + create_differentiable_stepper()   (delegates to NewtonManager)
        + create_gradient_rollout(*a, **kw) (delegates to NewtonManager)
        ~ add_rigid_object / add_articulation / add_robot
            also invalidate -> reset _arenas_cloned (already invalidates;
            additional flag-reset wired through invalidate())

embodichain/lab/sim/objects/backends/newton.py
    NewtonRigidBodyView, NewtonArticulationView
        + _num_envs
        ~ _resolve_body_ids -> returns [num_envs] tensor after finalize
        (no signature changes on public methods)

embodichain/lab/sim/utility/sim_utils.py
        + arena_index>0 guard on Newton spawn paths

embodichain/lab/sim/diff/        (new package)
    bridge.py
        _NewtonStepFunc(torch.autograd.Function)
        differentiable_step(manager, action, substeps)
        tape_context(manager)
    __init__.py
        re-exports

embodichain/lab/gym/envs/differentiable_env.py    (new)
    DifferentiableEmbodiedEnv(EmbodiedEnv)

embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py    (new)
    FrankaReachApgTask
```

`cfg.py` is **unchanged**.

## 5. Risks

1. **`clone_arena_to` semantics under post-finalize mutation.** Cloning runs
   at finalize. Attribute writes don't trigger re-clone; topology changes
   do (via `invalidate()` clearing `_arenas_cloned`). If a user mutates
   the source arena's *children list* without going through `add_*` /
   `remove_*` (raw dexsim calls), the clone state goes stale. We document
   the contract; we do not attempt to detect raw mutations.
2. **Drive-target write inside Tape.** `_step_action` currently writes
   joint drive targets via dexsim setters. Verify during implementation
   whether those writes are Warp-tape-recorded. If not, the differentiable
   path replaces them with a Warp kernel that writes into
   `control.joint_target` directly. Decision point at implementation; no
   user-facing impact either way.
3. **Tape memory.** Long rollouts × large `num_envs` × full backward can
   exhaust GPU memory. `truncate_backward_at` mitigates; we document
   recommended values for the example task.
4. **Functor autograd compatibility is opt-in per functor.** No mass
   refactor — only the functors needed by the example task are audited.
   The contract is documented in `agent_context/` so future authors know
   when to use torch ops vs. NumPy detours.
5. **Upstream dexsim.** Continues to depend on `yueci/adapt-embodichain`
   for active-joint indexing (existing risk; not introduced here). No new
   upstream dependencies — `clone_arena_to`, `DifferentiableStepper`,
   and `GradientRollout` are all on dexsim main paths.
6. **Body-ID resolution after clone.** The view extension assumes a
   predictable per-arena naming scheme from `clone_arena_to`. We verify
   the actual naming pattern during implementation and adjust the
   resolver accordingly; if `clone_arena_to` does not namespace bodies
   per-arena in a way EmbodiChain can rebuild, fall back to maintaining
   parallel entity lists per-`RigidObject` / `Articulation`.

## 6. Out of Scope (Deferred)

These targets are intentionally not in this PR:

- Default-backend cloning via `clone_arena_to`.
- Soft / cloth objects on Newton.
- `RigidObjectGroup` on Newton.
- FK-only differentiable mode (`grad_mode="kinematic"`).
- Per-link Newton-native contact params on articulations (waiting on
  dexsim per-link shape-material setter).
- Phase 3b legacy-`PhysicalAttr` removal.
- Functor-wide autograd audit.

## 7. PR Shape

Single feature branch `feature/newton-physics-backend`. Commit plan
(after squashing the existing `wip` commits):

1. `feat(sim/newton): clone source arena at finalize for multi-env`
   - `NewtonPhysicsBackend._clone_source_arena_if_needed`
   - View multi-env body-id resolution
   - Spawn guards for `arena_index>0` on Newton
   - `tests/sim/test_newton_multi_env.py`
2. `feat(sim/diff): NewtonStepFunc bridge for Warp tape -> torch autograd`
   - `embodichain/lab/sim/diff/` package
   - `SimulationManager.create_differentiable_stepper` /
     `create_gradient_rollout` delegators
   - `tests/sim/test_differentiable_stepper.py`
3. `feat(gym): DifferentiableEmbodiedEnv for APG`
   - `embodichain/lab/gym/envs/differentiable_env.py`
   - `tests/gym/envs/test_differentiable_env.py`
4. `feat(tasks): Franka reach APG example task`
   - `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py`
5. `docs(newton): update backend design doc + agent_context routing`
   - update `design/newton-backend-design.md` Done/Remaining lists
   - new `agent_context/` topic `differentiable-env`

## 8. Tests Summary

| File | Coverage |
|------|----------|
| `tests/sim/test_newton_multi_env.py` (new) | clone-at-finalize, batched body IDs, attribute mutation does not re-clone, topology change does, arena_index>0 spawn guard |
| `tests/sim/test_differentiable_stepper.py` (new) | manager delegator behavior on each backend; tape recording smoke |
| `tests/gym/envs/test_differentiable_env.py` (new) | construction validation, requires_grad on outputs, backward yields non-zero action.grad, finite-difference parity, one-iter loss reduction |
| `tests/sim/test_backend_parity.py` (existing) | unchanged — multi-env on Newton does not change capability flags |
| `tests/sim/test_newton_finalize_lifecycle.py` (existing) | extend with a multi-env case and an APG-config case |

## 9. Acceptance Criteria

This PR is ready to merge when:

- All targets 1–5 are reflected in code or in this design's deferred list.
- `pytest -q tests/sim tests/gym/envs/test_differentiable_env.py` is green.
- The Franka APG example task runs `python -m embodichain.lab.scripts.run_env
  --task=FrankaReachApg-v0 --num-envs=4 --steps=50` and loss decreases.
- `design/newton-backend-design.md` is updated to mark Targets 4 and 5
  Done and to point at this spec for the implementation rationale.
- The `wip` commits on the branch are squashed.
