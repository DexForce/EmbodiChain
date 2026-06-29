# Rigid Object Constraint — Design Spec

- **Date:** 2026-06-29
- **Status:** Approved
- **Topic:** Fixed constraint support between two `RigidObject`s, plus an on-demand event functor

## 1. Goal

Add the ability to attach two rigid objects via a fixed physics constraint and to remove that constraint again, both as a standalone simulation API (usable outside the gym) and as an on-demand event functor triggered from a task environment.

### Functional requirements

1. Two `RigidObject`s can be attached via a constraint.
2. The constraint between the two rigid objects can be removed.
3. A functor creates and removes the constraint inside a task environment.

### Non-goals (v1, deferred with extension points)

- Prismatic / revolute / spherical / d6 typed constraints (reserved `constraint_type` field).
- Per-constraint physics tuning (limits, drive, motion) — comes with typed constraints.
- Articulation ↔ rigid and articulation ↔ articulation constraints — `RigidObject` ↔ `RigidObject` only.
- Auto-detach-on-reset baked into the sim layer — kept as task policy.

## 2. Reference

The design mirrors the dexsim `fixed_constraint` example (`dexsim/examples/python/physics/rigidbody/fixed_constraint.py`) and lifts its API onto EmbodiChain's batched object/manager layer.

dexsim constraint API (bound on `Arena`, inherited by `Env`; C++ in `dexsim/cpp/pybind/environment/environment.cpp`):

- `arena.create_fixed_constraint(name, actor0, actor1, local_frame0=I, local_frame1=I) -> FixedConstraint | None`
- `arena.remove_constraint(name)`
- `arena.get_constraint(name) -> Constraint | None`
- `arena.get_all_constraints() -> list[Constraint]`
- Constraint handle exposes `get_name()`, `get_constraint_type()`, `is_valid()`, `get_local_pose(idx)`, `get_relative_transform()`.
- Typed variants exist (`create_prismatic_constraint`, `create_revolute_constraint`, `create_spherical_constraint`, `create_d6_constraint`) — reserved for later.

## 3. Architecture

Two layers:

```
SIM LAYER (standalone, usable without gym)
  SimulationManager                      RigidConstraint (new)
  ├─ create_rigid_constraint(...) ──►  batch wrapper over N arenas
  ├─ remove_rigid_constraint(name)     ├─ per-arena dexsim handles
  ├─ get_rigid_constraint(name)        ├─ obj_a / obj_b refs
  └─ self._constraints: dict           ├─ get_relative_transform()
       {name: RigidConstraint}          ├─ get_local_pose(idx)
                                       └─ destroy()
  delegates to dexsim Arena.create_fixed_constraint /
                Arena.remove_constraint  (per env_id / arena)

FUNCTOR LAYER (gym, on-demand)
  events.py
  ├─ create_rigid_constraint(env, env_ids, obj_a_cfg, obj_b_cfg, name, ...)
  └─ remove_rigid_constraint(env, env_ids, name)
  registered under EventCfg(mode=<custom>, ...);
  task triggers via env.event_manager.apply(mode="attach", env_ids)
```

### Principles

1. **One source of truth** — the sim layer owns the constraint registry and all dexsim calls. The functor is a thin adapter: resolve `SceneEntityCfg` → `RigidObject`, then call the sim API.
2. **Per-arena batch symmetry** — `RigidConstraint` mirrors `RigidObject`: N arenas → N dexsim constraint handles, so `env_id i` ↔ arena `i` ↔ handle `i`. Attach/detach can target a subset of `env_ids`.
3. **Fixed-first, extensible** — v1 wires only `create_fixed_constraint`; `RigidConstraintCfg.constraint_type` is reserved (`"fixed"` default).
4. **Local frames default to the current relative pose** — `local_frame_a` defaults to identity (object A's origin); `local_frame_b` defaults to `inv(pose_B) @ pose_A` (computed from the objects' current poses), so the constraint welds the objects where they are rather than pulling their origins together. Caller can pass explicit 4×4 or `(N,4,4)` matrices to define a specific joint frame.

### Why `SimulationManager`, not `RigidObject`, owns the API

The sim owns `self._arenas`, `self._env`, and every scene-mutation method (`add_rigid_object`, `remove_asset`, `draw_marker`). A constraint lives *between* two objects, so it belongs to the scene owner, not to either object. This keeps `RigidObject` focused on a single body and the constraint registry co-located with the rigid-object registry.

## 4. Sim-layer API

### New file: `embodichain/lab/sim/objects/constraint.py`

```python
@dataclass
class RigidConstraint:
    """Batch of fixed constraints linking two RigidObjects across all arenas.

    Each entry binds rigid_object_a's entity[i] to rigid_object_b's entity[i]
    within arena[i] via a dexsim FixedConstraint.
    """
    cfg: RigidConstraintCfg
    constraint_handles: list[Any]   # length == num_envs; None where inactive
    rigid_object_a: RigidObject
    rigid_object_b: RigidObject
    device: torch.device

    @property
    def num_envs(self) -> int: ...

    def get_relative_transform(self, env_ids=None) -> list[np.ndarray]: ...
    def get_local_pose(self, actor_index: int, env_ids=None) -> list[np.ndarray]: ...
    def get_name(self, env_id: int) -> str: ...
    def is_valid(self, env_ids=None) -> list[bool]: ...
    def destroy(self, env_ids: Sequence[int] | None = None) -> None: ...
```

`constraint_handles` is a list of length `num_envs` with `None` wherever the constraint is not active in that arena, so **arena index == list index** always holds.

### `RigidConstraintCfg` (in `embodichain/lab/sim/cfg.py`)

```python
@configclass
class RigidConstraintCfg:
    name: str = MISSING
    rigid_object_a_uid: str = MISSING
    rigid_object_b_uid: str = MISSING
    local_frame_a: np.ndarray | None = None   # None → identity; 4x4 or (N,4,4)
    local_frame_b: np.ndarray | None = None
    constraint_type: Literal["fixed"] = "fixed"   # reserved for typed constraints
```

### `SimulationManager` additions

```python
def create_rigid_constraint(
    self, cfg: RigidConstraintCfg, env_ids: Sequence[int] | None = None
) -> RigidConstraint:
    """Create a fixed constraint between two RigidObjects (env_ids-aware)."""

def remove_rigid_constraint(
    self, name: str, env_ids: Sequence[int] | None = None
) -> bool:
    """Remove by base name; idempotent. env_ids subset clears only those arenas."""

def get_rigid_constraint(self, name: str) -> RigidConstraint | None: ...
def get_rigid_constraint_uid_list(self) -> list[str]: ...
```

- New registry `self._constraints: Dict[str, RigidConstraint]`.
- `_deferred_destroy` severs `_constraints` like the other registries.

## 5. Functor layer

Two function-style event functors in `embodichain/lab/gym/envs/managers/events.py`, following the `add-functor` conventions: signature `(env, env_ids, ...) -> None`, `SceneEntityCfg` for entity refs, `from __future__ import annotations`, `TYPE_CHECKING` guard for `EmbodiedEnv`.

### `create_rigid_constraint` (attach)

```python
def create_rigid_constraint(
    env, env_ids,
    obj_a_cfg: SceneEntityCfg,
    obj_b_cfg: SceneEntityCfg,
    name: str,
    local_frame_a: np.ndarray | None = None,
    local_frame_b: np.ndarray | None = None,
) -> None:
    """Attach two rigid objects via a fixed constraint for the given env_ids."""
    obj_a = env.sim.get_asset(obj_a_cfg.uid)
    obj_b = env.sim.get_asset(obj_b_cfg.uid)
    # type-check both are RigidObject; else log_error
    env.sim.create_rigid_constraint(
        cfg=RigidConstraintCfg(
            name=name,
            rigid_object_a_uid=obj_a_cfg.uid,
            rigid_object_b_uid=obj_b_cfg.uid,
            local_frame_a=local_frame_a,
            local_frame_b=local_frame_b,
        ),
        env_ids=env_ids,
    )
```

### `remove_rigid_constraint` (detach)

```python
def remove_rigid_constraint(env, env_ids, name: str) -> None:
    """Remove the named constraint for the given env_ids. Idempotent."""
    env.sim.remove_rigid_constraint(name, env_ids=env_ids)
```

### Registration & triggering

```python
@configclass
class MyTaskEventsCfg:
    attach_objects: EventCfg = EventCfg(
        func=create_rigid_constraint, mode="attach",
        params={
            "obj_a_cfg": SceneEntityCfg(uid="cube"),
            "obj_b_cfg": SceneEntityCfg(uid="block"),
            "name": "cube_block_weld",
        },
    )
    detach_objects: EventCfg = EventCfg(
        func=remove_rigid_constraint, mode="detach",
        params={"name": "cube_block_weld"},
    )
```

```python
self.event_manager.apply(mode="attach", env_ids=gripping_env_ids)
self.event_manager.apply(mode="detach", env_ids=released_env_ids)
```

### Decisions

1. **Thin adapter** — functor does resolution + delegation only; no dexsim calls, no state.
2. **`env_ids` threading** — forwarded to `env.sim.create_rigid_constraint(..., env_ids=...)`.
3. **Custom modes** (`"attach"`/`"detach"`) — task-driven; supported by `EventManager` (arbitrary mode string, task wires `apply`).
4. **Detach takes only `name` + `env_ids`** — looked up by name in the sim registry.

## 6. Data flow

### Create (attach)

```
task → event_manager.apply(mode="attach", env_ids=gripping_env_ids)
  → EventManager iterates "attach" functors → func(env, env_ids, **params)
  → create_rigid_constraint(env, env_ids, obj_a_cfg, obj_b_cfg, name, frames)
     obj_a = env.sim.get_asset(obj_a_cfg.uid); obj_b = env.sim.get_asset(obj_b_cfg.uid)
     build RigidConstraintCfg(...)
  → env.sim.create_rigid_constraint(cfg, env_ids)
     resolve obj_a, obj_b from self._rigid_objects (raise if missing)
     target_env_ids = env_ids or range(num_envs)
     frames_a = broadcast(cfg.local_frame_a)            # None→I, 4x4→repeat, (N,4,4)→index
     frames_b = broadcast(cfg.local_frame_b) if cfg.local_frame_b is not None
               else inv(pose_B) @ pose_A per env         # default: weld at current relative pose
     for i in target_env_ids:
        arena = self.get_env(i)
        name_i = cfg.name if num_envs==1 else f"{cfg.name}_{i}"
        handle = arena.create_fixed_constraint(name_i, obj_a[i], obj_b[i], fa, fb)
        if handle is None: log_error(arena i)
     self._constraints[cfg.name] = RigidConstraint(...)
  → physics: obj_a[i] and obj_b[i] welded in arena[i]
```

### Remove (detach)

```
task → event_manager.apply(mode="detach", env_ids=released_env_ids)
  → remove_rigid_constraint(env, env_ids, name)
  → env.sim.remove_rigid_constraint(name, env_ids)
     constraint = self._constraints.pop(name, None)
     if None: log_warning; return False
     constraint.destroy(env_ids)   # arena.remove_constraint(name_i) per env
  → physics: obj_a[i] and obj_b[i] no longer welded
```

### Per-env selectivity — index alignment

`constraint_handles` is a list of length `num_envs`, `None` wherever inactive:

```
num_envs = 4, attach on env_ids=[0, 2]
constraint_handles = [handle_0, None, handle_2, None]
```

- `get_relative_transform(env_ids=[2])` reads `constraint_handles[2]`.
- `remove(env_ids=[0])` clears index 0 only; index 2 stays attached.
- Partial remove: `destroy(env_ids=[0])` → `arena.remove_constraint("name_0")` + `constraint_handles[0] = None`. When all handles become `None`, the wrapper is dropped from `self._constraints` (base name freed).

v1 lifecycle is `create` → `remove`. Re-attaching after a partial remove requires removing the whole constraint by name and creating it again (consistent with `add_rigid_object`'s "already exists" semantics — a duplicate base name on `create` is an error, even if only some envs are currently active).

At create time the `constraint_handles` list is pre-sized to `num_envs` filled with `None`, then only the `target_env_ids` entries are populated, so arena-index alignment always holds.

### Local-frame resolution (once, at create time)

`local_frame_a` is broadcast as below. `local_frame_b` is handled differently
when `None`: instead of identity, it is computed per env as
`inv(pose_B) @ pose_A` from the objects' current poses, so the default welds the
objects at their current relative pose (rather than pulling their origins
together). An explicit `local_frame_b` is broadcast like `local_frame_a`.

| Input (`local_frame_a`, or explicit `local_frame_b`) | Normalized per-env |
|-------|--------------------|
| `None` (`local_frame_a`) | `np.eye(4)` for all envs |
| `None` (`local_frame_b`) | `inv(pose_B) @ pose_A` per env (current relative pose) |
| `(4, 4)` | same matrix for all envs |
| `(N, 4, 4)` | `frames[i]` for env `i` (requires N == num_envs) |

### Interaction with reset

Constraints are **not** auto-reset by `reset_objects_state` (constraints aren't bodies). Default task policy: register a `reset`-mode `remove_rigid_constraint` (or call `sim.remove_rigid_constraint(name)` in the task's `_reset`) so stale constraints don't leak across episodes. The sim layer does not silently create/destroy on reset.

## 7. Error handling

| Condition | Layer | Behavior |
|-----------|-------|----------|
| Either RigidObject uid missing | sim | `log_error` (raises) |
| Duplicate base name in `_constraints` | sim | `log_error` |
| `create_fixed_constraint` returns `None` | sim | `log_error` with arena index |
| `(N,4,4)` frame N ≠ num_envs | sim | `log_error` |
| `remove` on unknown name | sim | `log_warning`, return `False` |
| `remove` with env_ids subset | sim | clear only those handles |
| Entity in functor not a `RigidObject` | functor | `log_error` |
| Remove a constraint already removed (handle `None`) | sim | no-op, success |
| Static + dynamic body combo | sim | allowed (weld-to-environment) |

## 8. Testing

**`tests/sim/objects/test_rigid_constraint.py`** (sim layer, mocks — `MockSim` exposing `create_fixed_constraint`/`remove_constraint` returning mock handles; `add_rigid_object` returning mock `RigidObject`s with `_entities`):

- `test_create_resolves_both_objects`
- `test_create_missing_object_raises`
- `test_create_subset_env_ids` — handles at 0,2 only; `None` elsewhere; alignment holds
- `test_local_frame_broadcasting` — None→identity, 4×4→repeat, (N,4,4)→indexed, bad N→error
- `test_remove_by_name_clears_handles`
- `test_remove_unknown_name_warns_false`
- `test_partial_remove_keeps_others`
- `test_all_removed_drops_from_registry`
- `test_get_relative_transform_skips_none_handles`

**`tests/gym/envs/managers/test_event_rigid_constraint.py`** (functor layer, mocks — `MockEnv` with `MockSim`, spied `create/remove_rigid_constraint`):

- `test_create_functor_delegates_to_sim` — params forwarded, `env_ids` threaded
- `test_create_functor_rejects_non_rigid_object` — `Articulation` uid → error
- `test_remove_functor_delegates` — forwards name + env_ids
- `test_custom_mode_apply_invokes_functor` — `EventCfg(mode="attach")` → `apply("attach", env_ids)` calls the spy once with those env_ids

**`tests/sim/test_rigid_constraint_integration.py`** (real-sim smoke, `@pytest.mark.gpu` / skipped without display):

- Mirror the dexsim `test_constraint.py` contract at the EmbodiChain layer: attach two dynamic cubes, step, assert relative transform stays constant; detach, step, assert they separate.

## 9. File layout

```
embodichain/lab/sim/objects/constraint.py              # RigidConstraint
embodichain/lab/sim/objects/__init__.py                # export RigidConstraint
embodichain/lab/sim/cfg.py                             # + RigidConstraintCfg
embodichain/lab/sim/sim_manager.py                     # + create/remove/get_rigid_constraint,
                                                        #   _constraints registry, asset_uids +
                                                        #   _deferred_destroy wiring
embodichain/lab/gym/envs/managers/events.py            # + create/remove_rigid_constraint functors
                                                        #   + __all__
tests/sim/objects/test_rigid_constraint.py             # sim-layer unit (mocks)
tests/gym/envs/managers/test_event_rigid_constraint.py # functor unit (mocks)
tests/sim/test_rigid_constraint_integration.py        # real-sim smoke (gpu-marked)
```

## 10. Out of scope / extension points

- Typed constraints (prismatic/revolute/spherical/d6) → `RigidConstraintCfg.constraint_type` reserved; typed factories land later without public API change.
- Per-constraint physics tuning (limits, drive, motion) → with typed constraints.
- Articulation constraints → `RigidObject`↔`RigidObject` only for v1.
- Auto-detach-on-reset → task policy, not sim policy.
- A dedicated `/add-...` skill → not needed; functors follow existing `add-functor` conventions.
