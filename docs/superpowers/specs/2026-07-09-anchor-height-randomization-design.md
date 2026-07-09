# Anchor Height Randomization Functor Design

**Date:** 2026-07-09  
**Status:** Approved for implementation  
**Author:** Claude (brainstorming session)  

## 1. Overview

This document specifies a new randomization functor, `randomize_anchor_height`, for EmbodiChain environments. The functor randomizes the Z position (height) of an anchor object — typically a table — and applies the same height delta to all other scene objects so that their relative placement is preserved. The functor is intended to run **after** all pose randomization functors in an episode reset sequence.

### Requirements

1. Execute after all pose randomization functors.
2. Move the anchor object by a sampled height delta.
3. Move all other included objects by the **same** height delta, preserving their XY position and rotation.
4. Do **not** move the robot base.
5. Support configurable object inclusion/exclusion.
6. Expose the sampled delta so downstream observations, rewards, or functors can use it.
7. Use lowercase-with-underscores naming consistent with existing class-style functors such as `planner_grid_cell_sampler`.

## 2. Architecture

### 2.1 Component placement

- **Implementation file:** `embodichain/lab/gym/envs/managers/randomization/spatial.py`
- **Config class:** `randomize_anchor_height_cfg` in the same file
- **Base class:** `Functor` from `embodichain.lab.gym.envs.managers.manager_base`
- **Registration:** Wired through `EventCfg` in task configs, e.g.:

```python
events:
  randomize_anchor_height:
    func: randomize_anchor_height
    mode: reset
    params:
      anchor_uid: "table"
      height_delta_range: [[-0.05], [0.05]]
      include_groups: null
      exclude_uids: ["floor"]
      store_key: "table_height_delta"
```

### 2.2 Ordering

The functor must be declared **after** all pose randomization events in the same mode. EmbodiChain’s `EventManager` executes functors in config-definition order within a mode, so placing `randomize_anchor_height` after pose randomizers guarantees it runs last.

## 3. Configuration Interface

```python
@configclass
class randomize_anchor_height_cfg(FunctorCfg):
    """Configuration for randomize_anchor_height."""

    anchor_uid: str = MISSING
    height_delta_range: tuple[list[float], list[float]] | None = None
    height_delta_candidates: list[float] | None = None
    include_groups: list[str] | None = None
    exclude_uids: list[str] = []
    mode: str = "reset"
    physics_update_step: int = 0
    store_key: str = "anchor_height_delta"
```

### Field semantics

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `anchor_uid` | `str` | `MISSING` | Exact UID of the anchor object (e.g., `"table"`). |
| `height_delta_range` | `tuple[list[float], list[float]] \| None` | `None` | Uniform sampling range `[min_z], [max_z]`. |
| `height_delta_candidates` | `list[float] \| None` | `None` | Discrete set of allowed delta values. |
| `include_groups` | `list[str] \| None` | `None` | Object groups to shift; `None` means all (`background`, `rigid_object`, `rigid_object_group`, `articulation`). The anchor UID is always excluded. |
| `exclude_uids` | `list[str]` | `[]` | Additional UIDs to skip beyond the anchor. |
| `mode` | `str` | `"reset"` | Event mode (`"reset"`, `"interval"`, etc.). |
| `physics_update_step` | `int` | `0` | Sim steps to run after moving objects. |
| `store_key` | `str` | `"anchor_height_delta"` | Attribute name on `env` where the sampled delta is stored. |

### Sampling mode selection

- If `height_delta_range` is provided, sample uniformly from it.
- Else if `height_delta_candidates` is provided, sample uniformly from the candidate list.
- If neither is provided, raise `ValueError` at init time.
- If both are provided, use `height_delta_range` and log a warning.

### Inclusion semantics

- `include_groups=None` → include `background`, `rigid_object`, `rigid_object_group`, and `articulation`. The anchor UID is always excluded, so other background objects are shifted while the anchor itself stays in its own handling path.
- Any explicit list restricts shifting to only those groups.
- Additional UIDs in `exclude_uids` are also skipped.

## 4. Runtime Behavior

### 4.1 `__init__(self, cfg, env)`

1. Validate that at least one sampling field is provided; if both are provided, prefer `height_delta_range` and log a warning.
2. Resolve `include_groups` to the default set if `None`.
3. Build the affected UID list:
   - Collect UIDs from the requested groups.
   - Remove `cfg.anchor_uid`.
   - Remove all UIDs in `cfg.exclude_uids`.
4. Confirm the anchor object exists in the scene; raise `ValueError` if not.
5. Cache the resolved target UIDs.

### 4.2 `__call__(self, env, env_ids)`

1. If `env_ids` is `None`, target all environments.
2. Sample `delta_z` per environment:
   - Range mode: `sample_uniform(low, high, size=(N,), device=env.device)`
   - Discrete mode: random choice from candidates, converted to a tensor of shape `(N,)`
3. Move the anchor object:
   - Read current pose.
   - Compute `anchor_target_z = anchor.cfg.init_pos[2] + delta_z`.
   - Write pose preserving XY/rotation.
4. For each affected object:
   - Read current pose.
   - Add `delta_z` to the Z component.
   - Write pose and call `clear_dynamics()`.
5. If `cfg.physics_update_step > 0`, call `env.sim.update(step=cfg.physics_update_step)`.
6. Store `delta_z` on `env` under `cfg.store_key`.

### 4.3 Pose representation

- Read poses via `rigid_object.get_local_pose()` → shape `(num_envs, 7)` = `(x, y, z, qw, qx, qy, qz)`.
- Modify only index `2` (Z).
- Write poses via `rigid_object.set_local_pose(pose, env_ids=env_ids)`.
- For articulations, use the equivalent root-state APIs.

## 5. Error Handling

### Init-time validation

- `anchor_uid` not found → `ValueError`.
- Neither `height_delta_range` nor `height_delta_candidates` provided → `ValueError`.
- `height_delta_candidates` empty → `ValueError`.
- Unknown group in `include_groups` → `ValueError` with valid options.

### Runtime guards

- Empty `env_ids` → return immediately.
- Affected object UID missing at call time → log warning and skip.
- Anchor object does not support pose get/set → `TypeError`.
- `store_key == ""` → skip storing the delta.
- `physics_update_step < 0` → treat as `0`.

## 6. Testing Plan

### Unit tests

1. **Config validation**
   - Missing `anchor_uid` raises.
   - Missing both sampling fields raises.
   - Empty `height_delta_candidates` raises.
   - Unknown `include_groups` raises.

2. **Sampling**
   - Range mode produces values within bounds.
   - Discrete mode produces only candidate values.

3. **Pose update**
   - Anchor Z equals `init_pos.z + delta`.
   - Affected objects’ Z increases by exactly `delta`.
   - XY and rotation are unchanged.
   - Excluded UIDs are not moved.

4. **State storage**
   - `env.store_key` exists after call and has shape `(N,)`.

### Integration tests

1. Add the functor to a minimal task config and run one reset. Verify objects remain above the anchor surface and the robot base does not move.
2. Verify execution order: when placed after pose randomizers, the functor preserves pose-randomization XY/rotation and only adjusts Z.

## 7. Open Questions / Future Work

- Should the functor support non-rigid anchors (e.g., soft objects)? Out of scope for the first version.
- Should the delta be logged to `env.extras` for dataset recording? Can be added later via `store_key` convention or explicit registration.

## 8. References

- `embodichain/lab/gym/envs/managers/randomization/spatial.py`
- `embodichain/lab/gym/envs/managers/randomization/__init__.py`
- `embodichain/lab/gym/envs/managers/manager_base.py`
- `embodichain/lab/gym/envs/managers/event_manager.py`
- `embodichain/lab/gym/envs/managers/events.py`
- `agent_context/topics/randomization/randomization.md`
- `agent_context/topics/manager-functor/manager-functor.md`
