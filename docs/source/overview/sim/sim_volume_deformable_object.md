(soft-object)=
# Volume Deformable Object

```{currentmodule} embodichain.lab.sim
```

{class}`~objects.VolumeDeformableObject` simulates a soft solid using a
tetrahedral mesh. Its simulation nodes carry world-frame positions and
velocities. Use a {doc}`surface deformable <sim_surface_deformable_object>` for
geometry simulated as a triangle surface.

Use Newton on CUDA with a particle-capable solver and gradients disabled.
Declare the soft object before `sim.prepare()`. See
{doc}`sim_manager/physics/newton/runtime_qualification` for supported solver
and scene combinations.

## Configuration

{class}`~cfg.VolumeDeformableObjectCfg` separates the volume topology from
the material and contact settings.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `shape` | `MeshCfg()` | Source surface mesh, or array-backed vertices for an explicit tetrahedral mesh. |
| `source` | `None` | Optional USD input selected with `UsdSourceCfg`. It supplies the authored simulation and render topology. |
| `meshing.simulation_mesh_resolution` | `8` | Voxel resolution when building a tetrahedral mesh from a source surface. |
| `meshing.embedding_impl` | `"dexsim_exact_cpu"` | Render-to-volume embedding implementation for generated meshes. |
| `tetrahedra` | `None` | Explicit `(M, 4)` vertex indices. Supply array-backed `shape.vertices` and optionally boundary triangles. |
| `particle_radius` | `None` | Contact radius per particle. Omission uses the solver default. |
| `attrs.density` | `1000.0` | Volume density in kg/m³. |
| `attrs.youngs` | `1e6` | Young's modulus in Pa. |
| `attrs.poissons` | `0.45` | Poisson's ratio, with a valid range of `(-1, 0.5)`. |
| `attrs.elasticity_damping` | `0.0` | Volumetric damping. |
| `attrs.surface_props`, `attrs.add_surface_edges` | Zero coefficients, `True` | Optional surface forces and surface bending-edge constraints. |

A generated tetrahedral mesh has its own node order, so a per-node array
must follow the generated topology. For stable authored indices, supply
`tetrahedra` and array-backed `shape.vertices`. The vertices become the
simulation nodes in that order. The boundary is derived from the tetrahedra
when `shape.triangles` is omitted.

Use {class}`~cfg.VolumeDeformableMeshingCfg` to tune voxelization, surface
remeshing, simplification, and embedding. `attrs.surface_props` uses
{class}`~cfg.SurfaceElementPropertiesCfg`, with omitted coefficients
resolved to zero for volume descriptors. `attrs.contact_material` can specify
a DexUni per-object contact material on SDK builds that support it.

## Setup & Initialization

This headless example uses an explicit tetrahedron to keep the four
simulation-node indices stable. The same configuration fields work with a
larger explicit mesh.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    VolumeDeformableObjectCfg,
    VolumeDeformablePhysicsCfg,
)
from embodichain.lab.sim.shapes import MeshCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        physics_cfg=NewtonPhysicsCfg(
            device="cuda:0", solver_cfg={"solver_type": "dexuni"}
        ),
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    soft = sim.add_deformable_object(
        VolumeDeformableObjectCfg(
            uid="soft_body",
            init_pos=(0.0, 0.0, 0.5),
            shape=MeshCfg(
                vertices=[(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)],
            ),
            tetrahedra=[(0, 1, 2, 3)],
            attrs=VolumeDeformablePhysicsCfg(youngs=1e5, poissons=0.4, density=75.0),
        )
    )
    sim.prepare()
    sim.update(step=1, render_final_step=False)
    state = soft.data.nodal_state_w  # (1, 4, 6)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

For a file-backed mesh, set `shape=MeshCfg(fpath="body.obj")` and configure
`meshing` to control the generated tetrahedra. Use a closed surface that
encloses the desired volume.

(soft-object-class)=
### Volume Deformable Object API

#### Nodal State

These properties return independent snapshots after preparation. `E` is the
instance count and `V = soft.data.n_nodes`.

| Property | Shape | Meaning |
| :--- | :--- | :--- |
| `data.nodal_pos_w`, `data.nodal_vel_w` | `(E, V, 3)` | Simulation-node world positions and velocities. |
| `data.nodal_state_w` | `(E, V, 6)` | Position followed by velocity at each node. |
| `data.default_nodal_state_w` | `(E, V, 6)` | Initial state captured at binding. |

Use `set_nodal_state(state, env_ids=...)` to initialize a deformed state.
For example, the following compresses the prepared object in environment
zero along its world Z axis and sets its initial velocity to zero.

```python
state = soft.data.default_nodal_state_w[0:1]
height = state[..., 2:3]
base_height = height.amin(dim=1, keepdim=True)
state[..., 2:3] = base_height + 0.9 * (height - base_height)
state[..., 3:] = 0
soft.set_nodal_state(state, env_ids=[0])
```

Use `get_surface_vertices()` and `get_surface_triangles()` for the rendered
surface. `get_collision_surface_triangles()` indexes the simulated body
nodes on its physical boundary.

#### Visual Material

With native rendering, `get_visual_material_inst()` reports one
representative material or `None` per selected environment. Use
`set_visual_material()` for a replacement, and
`restore_visual_material()` to restore the source assignments.

#### Pose Management

`set_local_pose(pose, env_ids=...)` places the captured rest shape at an
arena-local `xyz + xyzw` pose or 4×4 transform and zeros node velocities.
Use `set_nodal_state()` for a deformed initial shape. `reset()` restores
the configured placement and materials, while the nodal snapshots expose the
current shape.

## Related Topics

- {doc}`Soft Body Simulation </tutorial/create_softbody>` for a rendered
  tutorial using volume meshing.
- {doc}`sim_surface_deformable_object` for triangle-surface topology and nodal controls.
- {doc}`sim_assets` for the asset overview.
