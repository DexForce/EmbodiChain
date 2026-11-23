# Visual Object

```{currentmodule} embodichain.lab.sim
```

Use {class}`~objects.VisualObject` for mesh scenery, primitive props, or
Gaussian splat appearance. The manager tracks its pose, visibility, and visual
materials. A {doc}`sim_rigid_object` supplies a physics body and colliders
when the asset needs physical interaction.

## Configuration

Pass {class}`~cfg.VisualObjectCfg` to `sim.add_visual_object(cfg)`.

| Field | Meaning |
| :--- | :--- |
| `uid` | Unique asset identifier in the manager. |
| `shape` | Procedural geometry or an ordinary mesh file through `MeshCfg.fpath`. |
| `source` | Select one authored USD object through `UsdSourceCfg`. Choose either `source` or `shape`. |
| `visuals` | Optional replacement appearance using `VisualElementCfg` entries or one `GaussianSplatCfg`. |
| `init_pos`, `init_rot`, `init_local_pose` | Initial arena-local placement. Euler angles use degrees, and an explicit transform takes precedence. |
| `body_scale` | Geometry scale. Procedural assets default to unit scale. Omission retains the authored scale for USD sources. |
| `visible` | Initial visibility, restored by `reset()`. Defaults to `True`. |

With `visuals=None`, source appearance is preserved. An empty list removes
the appearance. Multiple elements need stable, explicit IDs. For a physical
asset with custom appearance, configure `RigidObjectCfg.visuals` or
`ArticulationCfg.link_visuals`.

## Setup and initialization

This example creates one visual cube in each of two environments. It moves
and hides the first instance, then restores its configured state. Native
rendering also works with `headless=True`.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, VisualObjectCfg
from embodichain.lab.sim.shapes import CubeCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=2,
        render_cfg=RenderCfg(renderer="hybrid"),
    )
)
try:
    prop = sim.add_visual_object(
        VisualObjectCfg(
            uid="display_cube",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            init_pos=(0.0, 0.0, 0.3),
        )
    )
    sim.prepare()
    pose = prop.get_local_pose(env_ids=[0])
    pose[:, 2] += 0.1
    prop.set_local_pose(pose, env_ids=[0])
    prop.set_visible(False, env_ids=[0])
    prop.reset(env_ids=[0])
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

## Poses, visibility, and materials

Reads and writes use the order of the selected environment indices. In the
table below, `E` is the selected instance count.

| Operation | Shape or result |
| :--- | :--- |
| `get_local_pose(env_ids=...)` | Independent arena-local poses `(E, 7)`, ordered as `xyz + xyzw`. |
| `get_local_pose(to_matrix=True, env_ids=...)` | Rigid transforms `(E, 4, 4)`. |
| `set_local_pose(pose, env_ids=...)` | Accepts `(E, 7)` or `(E, 4, 4)` transforms. |
| `get_visibility(env_ids=...)` | Boolean tensor `(E,)`. |
| `set_visible(visible, env_ids=...)` | Applies one boolean to selected instances. |
| `get_body_scale(env_ids=...)` | Geometry scales `(E, 3)`, independent of the root pose. |
| `get_visual_material_inst(env_ids=...)` | One material wrapper or `None` per selected instance. |

Use `set_visual_material(material, env_ids=...)` with a material created by
the manager. Selected resets restore the initial pose, visibility, and
material assignments. See {doc}`sim_manager/rendering/materials` for material
creation and presets.

## USD and Gaussian appearance

`source=UsdSourceCfg(file_path=..., source_prim_path=...)` selects the
appearance of one USD asset. Whole-scene imports through `sim.add_usd()`
retain the source entities' physical types. See the USD import recipes in
{doc}`/api_reference/embodichain/embodichain.lab.sim.cfg`.

For explicit Gaussian appearance, configure
`render_cfg=RenderCfg.for_gaussian()` before constructing the manager and
supply `GaussianSplatCfg(file_path=...)` through `visuals`. Gaussian
resources support pinhole color and view-Z depth. Camera mask, normal,
position, and stereo combinations have additional restrictions.

USD-authored Gaussian references currently require Default physics and
Hybrid rendering for native display. Explicit visual replacement on such
references is unsupported. Check {doc}`sim_manager/rendering/configuration`
for renderer and Gaussian configuration before loading the source.

## Related topics

- {doc}`sim_assets` for asset selection and preparation.
- {doc}`sim_rigid_object` for collision geometry and physical state.
- {doc}`visualization/viser` for browser visualization support.
