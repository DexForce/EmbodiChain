# Light Type Expansion — Design Spec

**Date:** 2026-07-05
**Status:** approved
**Scope:** Core pipeline — `LightCfg`, `Light` class, `SimulationManager.add_light`

## Motivation

EmbodiChain currently supports only `point` lights via `LightCfg(light_type="point")`. The dexsim rendering backend exposes 6 light types via Python bindings (`POINT`, `SUN`, `DIRECTION`, `SPOT`, `RECT`, `MESH`), each with distinct properties. The existing `LightCfg` has an explicit TODO: *"to be added more light type, such as spot, sun, etc."*

This spec covers adding support for all 6 backend light types to the core configuration and runtime API. Gym-layer config (`EnvLightCfg`) and light randomization are out of scope for this round.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Config structure | Single `LightCfg` with flat optional fields | Simpler, backward-compatible, `from_dict` works automatically |
| Runtime API | Single `Light` class with all setters | Warns on incompatible type; avoids class explosion |
| Parametric vs asset lights | Two-tier: 5 parametric types get tensor batching; `mesh` is string-set-only | Aligns with Isaac Lab's design philosophy — mesh lights are emissive geometry, not parametric primitives |
| Error handling | Warn on misconfiguration, error only on unknown type | Lets users iterate without crashes |

## Light Types

| Type | Category | Key Properties |
|---|---|---|
| `point` | Parametric | position, falloff (radius) |
| `sun` | Parametric | position, direction, angular_radius, halo_size, halo_falloff |
| `direction` | Parametric | direction only (infinite distance, no position) |
| `spot` | Parametric | position, direction, spot_angle_inner, spot_angle_outer |
| `rect` | Parametric | position, direction, rect_width, rect_height |
| `mesh` | Asset | position, direction, mesh_path (set-once, no tensor batching) |

## `LightCfg` — Config Fields

**File:** `embodichain/lab/sim/cfg.py`

All fields are flat on the existing `LightCfg` class. Every new field has a default — existing point-light code works unchanged.

```python
@configclass
class LightCfg(ObjectBaseCfg):
    light_type: Literal["point", "sun", "direction", "spot", "rect", "mesh"] = "point"

    # Universal
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 30.0
    enable_shadow: bool = True

    # Point light
    radius: float = 10.0

    # Directional (sun, direction, spot, rect, mesh)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Sun
    angular_radius: float = 0.5
    halo_size: float = 10.0
    halo_falloff: float = 3.0

    # Spot
    spot_angle_inner: float = 30.0
    spot_angle_outer: float = 45.0

    # Rect
    rect_width: float = 1.0
    rect_height: float = 1.0

    # Mesh
    mesh_path: str = ""
```

### Field Applicability Matrix

| Field | point | sun | direction | spot | rect | mesh |
|---|---|---|---|---|---|---|
| `color` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `intensity` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `enable_shadow` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `radius` | ✓ | — | — | — | — | — |
| `direction` | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| `angular_radius` | — | ✓ | — | — | — | — |
| `halo_size` | — | ✓ | — | — | — | — |
| `halo_falloff` | — | ✓ | — | — | — | — |
| `spot_angle_inner` | — | — | — | ✓ | — | — |
| `spot_angle_outer` | — | — | — | ✓ | — | — |
| `rect_width` | — | — | — | — | ✓ | — |
| `rect_height` | — | — | — | — | ✓ | — |
| `mesh_path` | — | — | — | — | — | ✓ |

## `Light` Class — Runtime API

**File:** `embodichain/lab/sim/objects/light.py`

### Existing Methods (unchanged signature)

- `set_color(colors, env_ids=None)` — vector3 broadcast
- `set_intensity(intensities, env_ids=None)` — scalar broadcast
- `set_falloff(falloffs, env_ids=None)` — scalar broadcast (point only, warns otherwise)
- `set_local_pose(pose, env_ids=None, to_matrix=False)` — updated for directional types
- `get_local_pose(to_matrix=False)` — unchanged
- `reset(env_ids=None)` — updated to apply type-specific properties

### New Methods

| Method | Tensor Shape | Broadcast Pattern | Applies To |
|---|---|---|---|
| `set_direction(directions, env_ids=None)` | (3,) or (M, 3) | `_apply_vector3` | sun, direction, spot, rect, mesh |
| `set_spot_angle(inner, outer, env_ids=None)` | scalar each | `_apply_scalar` × 2 | spot |
| `set_angular_radius(radii, env_ids=None)` | scalar or (M,) | `_apply_scalar` | sun |
| `set_halo_size(sizes, env_ids=None)` | scalar or (M,) | `_apply_scalar` | sun |
| `set_halo_falloff(falloffs, env_ids=None)` | scalar or (M,) | `_apply_scalar` | sun |
| `set_rect_size(widths, heights, env_ids=None)` | scalar each | `_apply_scalar` × 2 | rect |
| `set_mesh_path(path, env_ids=None)` | `str` | single string, no tensor | mesh |
| `enable_shadow(flags, env_ids=None)` | scalar or (M,) | `_apply_scalar` | all |

### Runtime Validation Pattern

Each setter checks `self.cfg.light_type` and warns + no-ops if the property doesn't apply:

```python
def set_spot_angle(self, inner, outer, env_ids=None):
    if self.cfg.light_type != "spot":
        logger.warning(
            f"set_spot_angle not applicable to light type '{self.cfg.light_type}', ignoring."
        )
        return
    self._apply_scalar(inner, env_ids, "set_spot_angle_inner")
    self._apply_scalar(outer, env_ids, "set_spot_angle_outer")
```

### `reset()` Behavior

`reset()` applies only the properties relevant to `self.cfg.light_type`:

- **point:** color, intensity, radius, init_pos, shadow
- **sun:** color, intensity, init_pos, direction, angular_radius, halo_size, halo_falloff, shadow
- **direction:** color, intensity, direction, shadow
- **spot:** color, intensity, init_pos, direction, spot_angle_inner, spot_angle_outer, shadow
- **rect:** color, intensity, init_pos, direction, rect_width, rect_height, shadow
- **mesh:** color, intensity, init_pos, direction, mesh_path, shadow

## `SimulationManager.add_light` — Backend Mapping

**File:** `embodichain/lab/sim/sim_manager.py`

### Type Mapping

```python
_LIGHT_TYPE_MAP: dict[str, LightType] = {
    "point":     LightType.POINT,
    "sun":       LightType.SUN,
    "direction": LightType.DIRECTION,
    "spot":      LightType.SPOT,
    "rect":      LightType.RECT,
    "mesh":      LightType.MESH,
}
```

### Creation Flow

1. Validate `cfg.light_type` against `_LIGHT_TYPE_MAP` — error on unknown type
2. Create one backend light per env via `env.create_light(f"{uid}_{i}", light_type)`
3. Apply initial properties from `cfg` based on light type (see `reset()` table above)
4. Wrap in `Light(cfg=cfg, entities=light_list)` and store in `self._lights[uid]`

### Validation Warnings (at creation time)

| Condition | Action |
|---|---|
| Unknown `light_type` | `logger.log_error`, return None |
| `mesh` with empty `mesh_path` | `logger.warning` |
| `rect` with zero `rect_width` or `rect_height` | `logger.warning` |

## Backward Compatibility

All existing call sites continue to work without modification:

- `LightCfg(uid="light", intensity=10.0, init_pos=(0, 0, 2.0))` — `light_type` defaults to `"point"`
- `LightCfg.from_dict({"light_type": "point", ...})` — flat fields map 1:1
- `sim.add_light(cfg=LightCfg(...))` — unchanged API
- `light.set_color(...)`, `light.set_intensity(...)`, `light.set_falloff(...)` — unchanged
- `light.set_local_pose(...)`, `light.get_local_pose(...)` — unchanged for point lights

## Out of Scope

- `EnvLightCfg` gym-layer changes
- Light randomization (`visual.py`) for new light types
- `indirect` lighting changes (IBL, emission light — already supported)
- USD import/export for new light types

## Testing

**File:** `tests/sim/objects/test_light.py` (extend)

| Test | Coverage |
|---|---|
| `test_point_light_backward_compat` | Existing tests pass unchanged |
| `test_create_all_light_types` | All 6 types created via `add_light` |
| `test_light_type_specific_properties` | Type-specific props applied at creation |
| `test_incompatible_setter_warns` | Wrong-type setter logs warning, no-ops |
| `test_unknown_light_type_errors` | Invalid type logs error |
| `test_directional_light_no_position` | `set_local_pose` on direction warns |
| `test_mesh_light_empty_path_warns` | Empty `mesh_path` warns at creation |
| `test_reset_applies_type_specific_props` | `reset()` restores only relevant props |
| `test_set_direction_broadcasting` | (3,) tensor broadcasts to all instances |
| `test_from_dict_new_types` | `from_dict` works with new fields |
| `test_set_rect_size` | (M, 2) tensor per-instance |
| `test_set_spot_angle` | Inner/outer tensor per-instance |

## Files Changed

| File | Change |
|---|---|
| `embodichain/lab/sim/cfg.py` | Expand `LightCfg` fields |
| `embodichain/lab/sim/objects/light.py` | Add setters, update `reset()` |
| `embodichain/lab/sim/sim_manager.py` | Expand type mapping in `add_light` |
| `tests/sim/objects/test_light.py` | Add tests for new types |
