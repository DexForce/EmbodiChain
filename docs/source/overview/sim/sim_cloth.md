# Cloth Object

```{currentmodule} embodichain.lab.sim
```

{class}`~objects.SurfaceDeformableObject` represents batched Newton cloth
particle sets. Cloth requires the
Newton backend on CUDA and a particle-capable solver.

See {doc}`newton_physics` for solver selection, mixed rigid/deformable scene
limitations, and the requirement to declare deformables before preparation.

## Configuration

Use {class}`~cfg.SurfaceDeformableObjectCfg` with physical parameters grouped
in `attrs`, following the rigid-object configuration convention.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `attrs.density` | `1.0` | Surface density in kg/m². |
| `attrs.surface_props.tri_ke/tri_ka/tri_kd` | `None` | Triangle elastic stiffness, area stiffness, and damping. |
| `attrs.surface_props.tri_drag/tri_lift` | `None` | Aerodynamic drag and lift. |
| `attrs.surface_props.edge_ke/edge_kd` | `None` | Bending stiffness and damping. |
| `attrs.add_springs` | `False` | Create explicit mesh-edge springs. |
| `attrs.spring_ke/spring_kd` | `None` | Spring stiffness and damping. |
| `particle_radius` | `None` | Particle radius; inherit solver default when omitted. |
| `shape` | `MeshCfg()` | Simulation source mesh and default render mesh. |
| `visual_shape` | `None` | Optional independently indexed render mesh. |

The seven `surface_props` coefficients use
{class}`~cfg.SurfaceElementPropertiesCfg`. `None` preserves Newton's
cloth defaults. Cloth uses its source triangle mesh directly and does not
require volume voxelization.

```python
from embodichain.lab.sim.cfg import (
    SurfaceElementPropertiesCfg,
    SurfaceDeformableObjectCfg,
    SurfaceDeformablePhysicsCfg,
)
from embodichain.lab.sim.shapes import MeshCfg

cfg = SurfaceDeformableObjectCfg(
    uid="cloth",
    shape=MeshCfg(fpath="cloth.obj"),
    attrs=SurfaceDeformablePhysicsCfg(
        density=0.2,
        surface_props=SurfaceElementPropertiesCfg(
            tri_ke=1000.0, tri_ka=1000.0, edge_ke=0.001,
        ),
    ),
)
```

Add this configuration through `sim.add_deformable_object(cfg)` and call
`sim.prepare()` before accessing `cloth.data`. For a runnable example, see
{doc}`Cloth Body Simulation </tutorial/create_cloth>`.

### Cloth Object Class
#### Nodal State

After `sim.prepare()`, read simulation nodes through `object.data`. Each state
property returns an independent tensor snapshot.

| Property | Shape | Description |
| :--- | :--- | :--- |
| `data.nodal_pos_w` | `(N, V, 3)` | Current simulation-node positions in world coordinates. |
| `data.nodal_vel_w` | `(N, V, 3)` | Current simulation-node velocities in world coordinates. |
| `data.default_nodal_state_w` | `(N, V, 6)` | Positions and velocities captured at Spawn binding. |

`N` is the number of instances; `V` is `data.n_nodes`. Render vertices are
separate and are read through `get_surface_vertices()`.

```python
# Example: Accessing vertex data
vert_position = cloth_object.data.nodal_pos_w
print(f"vertices positions: {vert_position}")

vert_velocity = cloth_object.data.nodal_vel_w
print(f"Vertex Velocities: {vert_velocity}")
```

#### Visual Material

When a cloth object is constructed, EmbodiChain wraps the first valid material already attached to each environment's render body. The original dexsim material instance is retained.

```python
materials = cloth_object.get_visual_material_inst()
if materials[0] is not None:
    materials[0].set_roughness(0.8)
```

`set_visual_material(mat, env_ids=None, shared=False)` assigns a replacement material, while `restore_visual_material(env_ids=None)` restores the original per-segment assignments. `get_visual_material_inst(env_ids=None)` returns one representative `VisualMaterialInst | None` per selected environment. `reset()` restores the original material before resetting the selected cloth bodies.

#### Pose Management
You can set the global pose of a cloth object (which transforms all its vertices), but getting a single "pose" from a deformed surface object is not supported.

| Method | Description |
| :--- | :--- |
| `set_local_pose(pose)` | Sets the pose of the object by transforming all vertices. |
| `get_local_pose()` | **Not Supported**. Raises `NotImplementedError` because a deformed object does not have a single rigid pose. |


```python
# Reset or Move the Cloth Object
target_pose = torch.tensor([[0, 0, 1.0, 0, 0, 0, 1]], device=device) # (x, y, z, qx, qy, qz, qw)
cloth_object.set_local_pose(target_pose)

# Important: Step simulation to apply changes
sim.update()
```
