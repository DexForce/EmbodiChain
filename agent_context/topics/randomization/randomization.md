# Randomization

## Entry Points

| What | Path |
|---|---|
| Randomization module root | `embodichain/lab/gym/envs/managers/randomization/` |
| Physics randomizers | `embodichain/lab/gym/envs/managers/randomization/physics.py` |
| Visual randomizers | `embodichain/lab/gym/envs/managers/randomization/visual.py` |
| Spatial randomizers | `embodichain/lab/gym/envs/managers/randomization/spatial.py` |
| Geometry randomizers | `embodichain/lab/gym/envs/managers/randomization/geometry.py` |
| `EventCfg` (how randomizers are wired) | `embodichain/lab/gym/envs/managers/cfg.py` |
| `EventManager` (runtime dispatch) | `embodichain/lab/gym/envs/managers/event_manager.py` |
| Event functors (non-randomization events) | `embodichain/lab/gym/envs/managers/events.py` |

## Overview

All randomization functions are implemented as **event functors**. They follow the standard functor signature `(env, env_ids, entity_cfg, **params) -> None` and are registered via `EventCfg` in a task's event config. The `EventManager` dispatches them at the configured mode (`startup`, `reset`, or `interval`).

The `__init__.py` of the randomization package re-exports everything via `from .physics import *` etc.

## Randomization Types

### Physics (`physics.py`)

| Function | Target | Key params |
|---|---|---|
| `randomize_rigid_object_mass` | `RigidObject` mass | `mass_range`, `relative` |
| `randomize_rigid_object_center_of_mass` | `RigidObject` CoM offset | `com_pos_offset_range` |
| `randomize_articulation_mass` | `Articulation` link masses | `mass_range` (uniform or per-link dict), `link_names` (regex), `relative` |

- `relative=True` adds sampled value to the initial/default mass instead of replacing.
- `randomize_articulation_mass` supports a `dict[str, tuple]` for per-link ranges; when used, `link_names` is ignored.
- Link names are resolved via `resolve_matching_names` (regex matching).

### Visual (`visual.py`)

| Function | Target | Key params |
|---|---|---|
| `set_rigid_object_visual_material` | Deterministic material set | `mat_cfg` (`VisualMaterialCfg` or dict) |
| `set_rigid_object_group_visual_material` | Group material set | `mat_cfg` |
| `randomize_visual_material` | Random material properties | (varies) |
| `randomize_camera_extrinsics` | Camera pose | `pos_range`, `euler_range` (attach mode) or `eye_range`, `target_range`, `up_range` (look-at mode) |
| `randomize_camera_intrinsics` | Camera intrinsic params | (varies) |
| `randomize_light` | Light pos/color/intensity/direction | `position_range`, `color_range`, `intensity_range`, `direction_range` |
| `randomize_emission_light` | Emission light props | (varies) |
| `randomize_indirect_lighting` | Indirect lighting | (varies) |

- Camera extrinsics auto-detect mode: if `extrinsics.parent` is set → attach mode (pos/euler perturbation via `set_local_pose`); if `extrinsics.eye` is set → look-at mode (eye/target/up perturbation via `look_at`).
- `set_rigid_object_visual_material` is deterministic (not random) but uses the same functor mechanism for fixed material assignment at reset.
- Light randomization applies the **same values across all envs** (documented limitation).
- ``position_range`` is ignored for global scene lights (``"sun"``, ``"direction"``). Use ``direction_range`` instead.
- ``direction_range`` is only applicable for directional light types (``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, ``"mesh"``).
- Global lights (``"sun"``, ``"direction"``) have ``num_instances == 1``; all other types are batched per environment.

### Spatial (`spatial.py`)

| Function | Target | Key params |
|---|---|---|
| `randomize_rigid_object_pose` | Object position/rotation | `position_range`, `rotation_range` (Euler degrees), `relative_position`, `relative_rotation`, `physics_update_step` |
| `randomize_robot_eef_pose` | Robot end-effector pose | `position_range`, `rotation_range` |
| `randomize_robot_qpos` | Robot joint positions | (varies) |
| `randomize_articulation_root_pose` | Articulation root | (varies) |
| `randomize_target_pose` | Target pose | (varies) |

- Helper: `get_random_pose(init_pos, init_rot, ...)` generates batched random 4×4 poses.
- Rotation ranges are in **degrees** (converted to radians internally).
- `relative_position=True` (default) adds offset to initial position; `relative_rotation=False` (default) replaces rotation.
- After setting pose, `clear_dynamics()` is called to zero out velocities.
- `physics_update_step > 0` triggers `env.sim.update(step=N)` to let physics settle after randomization.

### Geometry (`geometry.py`)

| Function | Target | Key params |
|---|---|---|
| `randomize_rigid_object_scale` | Single object body scale | `scale_factor_range`, `same_scale_all_axes` |
| `randomize_rigid_objects_scale` | Multiple objects | `entity_cfgs` (list), `shared_sample` |
| `randomize_rigid_object_body_scale` | *(deprecated)* | Redirects to `randomize_rigid_object_scale` |

- Scale is **multiplicative** (factor), not absolute size.
- `same_scale_all_axes=True` → single scalar sampled and replicated to x/y/z.
- `shared_sample=True` → one scale sample shared across all objects in the list.
- `_normalize_env_ids` helper: if `env_ids is None`, targets all environments.

## Randomization as Events

Randomizers are wired into tasks using `EventCfg`, which extends `FunctorCfg`:

```python
@configclass
class EventCfg(FunctorCfg):
    mode: Literal["startup", "interval", "reset"] = "reset"
    interval_step: int = 10
    is_global: bool = False
```

### Modes

| Mode | When applied | Use case |
|---|---|---|
| `startup` | Once when environment initializes | One-time scene setup (e.g., fixed material assignment) |
| `reset` | Every environment reset | Domain randomization per episode |
| `interval` | Every `interval_step` env steps | Continuous perturbation during episode |

### `is_global`

- `True` → same interval counter for all envs.
- `False` → per-env independent interval counters.

### Wiring example

```python
@configclass
class MyTaskEventCfg:
    randomize_obj_mass = EventCfg(
        func=randomize_rigid_object_mass,
        mode="reset",
        params={
            "entity_cfg": SceneEntityCfg(uid="target_object"),
            "mass_range": (0.1, 0.5),
            "relative": False,
        },
    )

    randomize_obj_pose = EventCfg(
        func=randomize_rigid_object_pose,
        mode="reset",
        params={
            "entity_cfg": SceneEntityCfg(uid="target_object"),
            "position_range": ([-0.05, -0.05, 0.0], [0.05, 0.05, 0.0]),
            "rotation_range": ([0, 0, -45], [0, 0, 45]),
        },
    )
```

Each `EventCfg` attribute in the config class becomes a named event functor managed by `EventManager`.

## How to Add a Randomizer

1. Use the `/add-functor` skill to scaffold the function with correct signature.
2. Place function-style randomizers in the appropriate file under `embodichain/lab/gym/envs/managers/randomization/` (physics, visual, spatial, or geometry).
3. Signature: `def randomize_*(env: EmbodiedEnv, env_ids: torch.Tensor | None, entity_cfg: SceneEntityCfg, **params) -> None`.
4. Add the function name to `__all__` in the source file.
5. The `__init__.py` uses wildcard imports, so `__all__` membership is sufficient for export.
6. Wire it in a task config via `EventCfg(func=your_function, mode="reset", params={...})`.

For class-style randomizers (stateful), inherit from `Functor` and implement `__init__(cfg, env)` + `__call__(env, env_ids, ...)`.

## Configuration

### `FunctorCfg` (base)

```python
@configclass
class FunctorCfg:
    func: Callable | Functor = MISSING     # function or callable class
    params: dict[str, Any] = dict()         # keyword args passed to func
    extra: dict[str, Any] = dict()          # metadata (e.g., output shape)
```

### `SceneEntityCfg`

Used in `params` to reference simulation objects by `uid`. The manager resolves the entity from `SimulationManager` at initialization.

### Range conventions

- Position ranges: `tuple[list[float], list[float]]` → `([x_min, y_min, z_min], [x_max, y_max, z_max])`
- Rotation ranges: same shape, values in **degrees**
- Mass ranges: `tuple[float, float]` → `(min, max)` or `dict[str, tuple]` for per-link
- Scale ranges: `tuple[list[float], list[float]]` → `([sx_min, sy_min, sz_min], [sx_max, sy_max, sz_max])` or `[[s_min], [s_max]]` when `same_scale_all_axes=True`

### Sampling

All randomizers use `embodichain.utils.math.sample_uniform(lower, upper, size)` for uniform sampling.

## Common Failure Modes

| Symptom | Likely cause |
|---|---|
| Randomizer silently does nothing | `entity_cfg.uid` not found in `sim.get_rigid_object_uid_list()` — all randomizers early-return on UID mismatch |
| `ValueError` on link name | `mass_range` dict key doesn't match any `articulation.link_names` |
| Camera randomization error | Extrinsics config has neither `parent` nor `eye` set — unsupported mode |
| Light randomization not per-env | By design: `randomize_light` applies same values across all envs |
| CoM randomization warning on static object | Object is `is_non_dynamic` — CoM cannot be randomized |
| Scale has no visible effect | `scale_factor_range` is `None` — function returns immediately |
| Deprecated warning on `randomize_rigid_object_body_scale` | Migrate to `randomize_rigid_object_scale` with `scale_factor_range` parameter |
| Pose randomization leaves residual velocity | `clear_dynamics()` is called, but if `physics_update_step` is not set, objects may still drift on next step |
| `env_ids` is `None` at `interval` mode | `_normalize_env_ids` converts `None` to `torch.arange(env.num_envs)` — this is safe |
