# Soft Object

```{currentmodule} embodichain.lab.sim
```

{class}`~objects.VolumeDeformableObject` represents batched Newton volumetric
particle sets. Volume deformables
require the Newton backend on CUDA and a particle-capable solver.

See {doc}`newton_physics` for solver selection, mixed rigid/deformable scene
limitations, and the requirement to declare deformables before preparation.

## Configuration

{class}`~cfg.VolumeDeformableObjectCfg` separates physical parameters under
`attrs` from mesh generation under `meshing`.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `attrs.density` | `1000.0` | Volume density in kg/m³. |
| `attrs.youngs` | `1e6` | Young's modulus in Pa. |
| `attrs.poissons` | `0.45` | Poisson's ratio in (-1, 0.5). |
| `attrs.elasticity_damping` | `0.0` | Volumetric damping. |
| `attrs.surface_props` | Seven zero coefficients | Optional Newton surface forces. |
| `attrs.add_surface_edges` | `True` | Create surface bending-edge constraints. |
| `meshing.triangle_remesh_resolution` | `8` | Source surface remeshing resolution. |
| `meshing.triangle_simplify_target` | `0` | Target proxy face count; zero disables simplification. |
| `meshing.simulation_mesh_resolution` | `8` | Voxel resolution for the tetrahedral mesh. |
| `meshing.voxel_num_relaxation_iters` | `5` | Tetrahedral-mesh relaxation iterations. |
| `meshing.voxel_rel_min_tet_volume` | `0.05` | Relative minimum tetrahedral volume. |
| `meshing.voxel_surface_dist_ratio` | `0.2` | Surface distance as a voxel-size ratio. |
| `meshing.embedding_impl` | `dexsim_exact_cpu` | Render-to-volume binding implementation. |

`attrs.surface_props` uses the same
{class}`~cfg.SurfaceElementPropertiesCfg` as cloth. Volume defaults are
zero rather than cloth's `None`; explicitly omitted coefficients in a partial
surface group also resolve to zero for volume descriptors.

```python
from embodichain.lab.sim.cfg import (
    VolumeDeformableObjectCfg,
    VolumeDeformablePhysicsCfg,
    VolumeDeformableMeshingCfg,
)
from embodichain.lab.sim.shapes import MeshCfg

cfg = VolumeDeformableObjectCfg(
    uid="soft_body",
    shape=MeshCfg(fpath="body.obj"),
    attrs=VolumeDeformablePhysicsCfg(
        youngs=1e5, poissons=0.4, density=75.0,
    ),
    meshing=VolumeDeformableMeshingCfg(simulation_mesh_resolution=12),
)
```

Add this configuration through `sim.add_deformable_object(cfg)` and call
`sim.prepare()` before reading object data. For a runnable example, see
{doc}`Soft Body Simulation </tutorial/create_softbody>`.

### Soft Object Class
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
sim_verts = soft_object.data.nodal_pos_w
print(f"Simulation Vertices Shape: {sim_verts.shape}")

velocities = soft_object.data.nodal_vel_w
print(f"Vertex Velocities: {velocities}")
```

#### Visual Material

When a soft object is constructed, EmbodiChain wraps the first valid material already attached to each environment's render body. The original dexsim material instance is retained.

```python
materials = soft_object.get_visual_material_inst()
if materials[0] is not None:
    materials[0].set_base_color([0.8, 0.3, 0.2, 1.0])
```

`set_visual_material(mat, env_ids=None, shared=False)` assigns a replacement material, while `restore_visual_material(env_ids=None)` restores the original per-segment assignments. `get_visual_material_inst(env_ids=None)` returns one representative `VisualMaterialInst | None` per selected environment. `reset()` restores the original material before resetting the selected soft bodies.

#### Pose Management
You can set the global pose of a soft object (which transforms all its vertices), but getting a single "pose" from a deformed object is not supported.

| Method | Description |
| :--- | :--- |
| `set_local_pose(pose)` | Sets the pose of the object by transforming all vertices. |
| `get_local_pose()` | **Not Supported**. Raises `NotImplementedError` because a deformed object does not have a single rigid pose. |


```python
# Reset or Move the Soft Object
target_pose = torch.tensor([[0, 0, 1.0, 0, 0, 0, 1]], device=device) # (x, y, z, qx, qy, qz, qw)
soft_object.set_local_pose(target_pose)

# Important: Step simulation to apply changes
sim.update()
```
