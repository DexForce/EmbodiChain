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
| `randomize_rigid_object_mass` | Dynamic `RigidObject` mass/inertia | `mass_range`, `relative`, `recompute_inertia`, `min_mass` |
| `randomize_rigid_object_center_of_mass` | `RigidObject` CoM offset | `com_pos_offset_range` |
| `randomize_articulation_mass` | `Articulation` link mass/inertia | `mass_range` (uniform or per-link dict), `link_names` (regex), `relative`, `recompute_inertia`, `min_mass` |

- `relative=True` adds the sampled value to the backend-resolved initial mass
  stored in the target object's `default_mass` snapshot; repeated calls
  therefore do not accumulate for either rigid objects or articulations.
- Rigid-object and articulation mass samples are clamped to positive
  `min_mass`. By default, inertia is recomputed from the corresponding
  initialization snapshot using the mass ratio; set `recompute_inertia=False`
  only when inertia is managed separately.
- Non-dynamic rigid objects are skipped with a warning.
- `randomize_articulation_mass` supports a `dict[str, tuple]` for per-link
  ranges; when used, `link_names` is ignored.
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

## Event scheduling and reproducibility

Wire randomizers with named `EventCfg` entries (`mode="startup"`, `"reset"` or
`"interval"`). `interval_step` controls the interval; `is_global=True` shares a
counter, while the default uses per-row counters. Configuration and callable
scaffolding are owned by [manager-functor](../manager-functor/manager-functor.md)
and `/add-functor`.

`BaseEnv` establishes the seed before scene construction. `EventManager` derives
a stable seed from `(seed, phase, mode, functor_name, invocation)` for each
constructor/call scope. `_scoped_random_seed()` temporarily seeds Python,
NumPy, Torch CPU and the selected CUDA device, then restores their previous
states, including on exceptions.

- Class-functor construction has its own phase, separate from event calls.
- `set_seed()` resets invocation and interval counters; a repeated seed rewinds
  streams. `None` disables scoped seeding and uses the ambient RNG state.
- Adding another named functor does not consume this functor's stream.
- Renaming a functor changes its stream. Invocation order and selected row
  batches still matter; this is not an independent RNG stream per environment row.

For reproducibility changes, inspect `event_manager.py` and the effective seed
propagation in `base_env.py` / `embodied_env.py`. Run
`tests/gym/envs/test_env_seed.py` and
`tests/gym/envs/managers/test_event_manager_seed.py`.

Workspace-aware spatial sampling uses
[robot workspace](../robot-workspace/robot-workspace.md); validate it with
`tests/gym/envs/managers/test_workspace_randomization.py`.

### Range conventions

- Position ranges: `tuple[list[float], list[float]]` → `([x_min, y_min, z_min], [x_max, y_max, z_max])`
- Rotation ranges: same shape, values in **degrees**
- Mass ranges: `tuple[float, float]` → `(min, max)` or `dict[str, tuple]` for per-link
- Scale ranges: `tuple[list[float], list[float]]` → `([sx_min, sy_min, sz_min], [sx_max, sy_max, sz_max])` or `[[s_min], [s_max]]` when `same_scale_all_axes=True`

### Sampling

Randomizers use `embodichain.utils.math.sample_uniform(...)` for uniform
sampling where applicable. Physics samples are allocated on the target object's
device, not assumed to share `env.device`.
## Common Failure Modes

| Symptom | Likely cause |
|---|---|
| Randomizer silently does nothing | `entity_cfg.uid` not found in `sim.get_rigid_object_uid_list()` — all randomizers early-return on UID mismatch |
| Rigid-object mass is clamped | The sampled absolute mass or relative result was below positive `min_mass` |
| `ValueError` on link name | `mass_range` dict key doesn't match any `articulation.link_names` |
| Camera randomization error | Extrinsics config has neither `parent` nor `eye` set — unsupported mode |
| Light randomization not per-env | By design: `randomize_light` applies same values across all envs |
| CoM randomization warning on static object | Object is `is_non_dynamic` — CoM cannot be randomized |
| Scale has no visible effect | `scale_factor_range` is `None` — function returns immediately |
| Deprecated warning on `randomize_rigid_object_body_scale` | Migrate to `randomize_rigid_object_scale` with `scale_factor_range` parameter |
| Pose randomization leaves residual velocity | `clear_dynamics()` is called, but if `physics_update_step` is not set, objects may still drift on next step |
| `env_ids` is `None` at `interval` mode | `_normalize_env_ids` converts `None` to `torch.arange(env.num_envs)` — this is safe |
