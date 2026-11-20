# Light

```{currentmodule} embodichain.lab.sim
```

Use {class}`~objects.Light` to illuminate a natively rendered scene.
Lights work with headless offscreen rendering. The manager creates and
registers direct lights through `sim.add_light(LightCfg(...))`.

Environment emission and environment maps are configured through
`sim.set_emission_light()` and `sim.set_indirect_lighting()`. The
startup inventory's `Direct lights` row counts explicitly registered
light UIDs.

## Configuration and scope

{class}`~cfg.LightCfg` supports six types.

| `light_type` | Instance scope | Placement and shape |
| :--- | :--- | :--- |
| `"point"` | One per environment | Position. |
| `"sun"` | One global instance | Emitted `direction`. |
| `"direction"` | One global instance | Emitted `direction`. |
| `"spot"` | One per environment | Position, direction, `spot_angle_inner`, and `spot_angle_outer`. |
| `"rect"` | One per environment | Position, direction, `rect_width`, and `rect_height`. |
| `"mesh"` | One per environment | Emissive `MeshObject` assigned through `set_mesh()`. |

Common fields include `uid`, RGB `color`, `intensity`, and
`enable_shadow`. Defaults are white, intensity `30.0`, and shadows
enabled. Use `init_pos` for a positional light and `direction` for an
oriented light. Global sun and direction lights use direction for their
orientation. The sun-specific `angular_radius`, `halo_size`, and
`halo_falloff` fields are reserved for future native support.

## Setup and initialization

This example creates one global sun and one fill light replicated across
two environments. It changes only the second fill instance's intensity.

```python
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import LightCfg, RenderCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=2,
        render_cfg=RenderCfg(renderer="hybrid"),
        startup_summary="full",
    )
)
try:
    sun = sim.add_light(
        LightCfg(
            uid="global_sun",
            light_type="sun",
            direction=(0.0, -1.0, -1.0),
        )
    )
    fill = sim.add_light(
        LightCfg(
            uid="local_fill",
            light_type="point",
            init_pos=(0.0, 0.0, 2.0),
            intensity=20.0,
        )
    )
    sim.prepare()
    fill.set_intensity(torch.tensor([40.0]), env_ids=[1])
    intensities = fill.get_intensity()  # [20.0, 40.0]
    assert sun.num_instances == 1
    assert fill.num_instances == 2
    sim.update(step=1, render_final_step=False)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

## Runtime operations

Let `E = light.num_instances`. Select instances in row order with
`env_ids`. A global light has a single row shared by all environments.

| Operation | Shape or meaning |
| :--- | :--- |
| `get_color()` | RGB tensor `(E, 3)`. |
| `get_intensity()` | Intensity tensor `(E,)`. |
| `get_local_pose()` | Translation tensor `(E, 3)`. |
| `get_local_pose(to_matrix=True)` | Transforms `(E, 4, 4)`. |
| `set_color(colors, env_ids=...)` | Broadcast `(3,)` RGB or selected `(selected E, 3)` colors. |
| `set_intensity(values, env_ids=...)` | Broadcast scalar or selected intensity values. |
| `set_local_pose(position, env_ids=...)` | Positional lights accept translations; `to_matrix=True` extracts matrix translations. |
| `set_direction(directions, env_ids=...)` | Broadcast `(3,)` or selected `(selected E, 3)` directions for oriented types. |
| `enable_shadow(flags, env_ids=...)` | Scalar or selected shadow flags. |

Use `set_spot_angle()` for spot cones and `set_rect_wh()` for
rectangular dimensions. `apply_state(..., env_ids=...)` writes several
supported fields in one call. Omitted fields retain their current values.
Mesh-light geometry is assigned with `set_mesh()`.

`reset(env_ids=...)` restores configured light properties. Resetting a
global light restores its shared instance. Changing the global instance
affects every environment, so use positional per-environment lights for
independent lighting variants.

## Startup summary

The example reports `Direct lights = 2`, counting the two registered UIDs.
In `full` mode, the `Light details` section also lists
`global_sun · sun · global` and `local_fill · point · per env`.

## Related topics

- {doc}`sim_assets` for the complete scene inventory.
- {doc}`sim_manager/rendering/configuration` for native rendering options.
- {doc}`sim_manager/rendering/materials` for surface appearance.
- {doc}`sensors/camera` for capturing illuminated images.
