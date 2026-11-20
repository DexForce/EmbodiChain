# Cloth Object

```{currentmodule} embodichain.lab.sim
```

{class}`~objects.SurfaceDeformableObject` simulates a triangle surface such
as cloth or a membrane. Its state is the world-frame position and velocity
of each simulation node. Choose a {doc}`sim_soft_object` for a solid that
needs an interior simulation mesh.

Use Newton on CUDA with a particle-capable solver and gradients disabled.
Declare the cloth before `sim.prepare()`. See
{doc}`sim_manager/physics/newton/runtime_qualification` for supported solver
and scene combinations.

## Configuration

Configure the asset with {class}`~cfg.SurfaceDeformableObjectCfg`.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `shape` | `MeshCfg()` | Triangle simulation mesh. It also provides the default render mesh. |
| `source` | `None` | Optional authored USD cloth selected with `UsdSourceCfg`. The source supplies the topology and physics. |
| `visual_shape` | `None` | Separate render mesh for features such as UV seams. |
| `visual_binding_mode` | `"auto"` | Surface embedding by default. Use `"nearest_vertex"` when the render mesh duplicates simulation vertices along texture seams. |
| `particle_radius` | `None` | Contact radius per particle. Omission uses the solver default. |
| `attrs.density` | `1.0` | Surface density in kg/m². |
| `attrs.surface_props` | Solver defaults | `tri_ke`, `tri_ka`, and `tri_kd` control triangle stiffness and damping. `tri_drag` and `tri_lift` configure aerodynamics. `edge_ke` and `edge_kd` control bending. |
| `attrs.add_springs` | `False` | Enable explicit mesh-edge springs with optional `spring_ke` and `spring_kd`. |

`attrs.surface_props` uses {class}`~cfg.SurfaceElementPropertiesCfg`.
`None` inherits the solver value for that coefficient. An explicit
`attrs.contact_material` configures DexUni contact stiffness, damping, and
dynamic friction on SDK builds that support per-object contact materials.
With a compatible DexSim build, `attrs.edge_rest_angle` sets a signed rest
dihedral angle in radians. Use `attrs.pneumatic` for DexUni cavity pressure
parameters on a closed surface.

Use an array-backed `MeshCfg` when a controller depends on stable node
indices. Its vertex order is the simulation-node order. File-based meshes
can be reordered by the importer. `particle_flags` and
`register_kinematic_nodal_trajectory()` address these simulation nodes.
Register nodal trajectories before preparing the scene and clear Newton's
`ACTIVE` flag for the driven nodes in the configuration.

## Setup & Initialization

This example declares a four-node cloth patch without a mesh file. The
prepared scene uses CUDA and the DexUni solver.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RenderCfg,
    SurfaceDeformableObjectCfg,
    SurfaceDeformablePhysicsCfg,
    SurfaceElementPropertiesCfg,
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
    cloth = sim.add_deformable_object(
        SurfaceDeformableObjectCfg(
            uid="cloth",
            init_pos=(0.0, 0.0, 0.5),
            shape=MeshCfg(
                vertices=[(0, 0, 0), (0.2, 0, 0), (0.2, 0.2, 0), (0, 0.2, 0)],
                triangles=[(0, 1, 2), (0, 2, 3)],
            ),
            attrs=SurfaceDeformablePhysicsCfg(
                density=0.2,
                surface_props=SurfaceElementPropertiesCfg(tri_ke=1000.0, tri_ka=1000.0),
            ),
        )
    )
    sim.prepare()
    sim.update(step=1, render_final_step=False)
    state = cloth.data.nodal_state_w  # (1, 4, 6)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

### Cloth Object Class

#### Nodal State

These properties return independent snapshots after preparation. `E` is the
instance count and `V = cloth.data.n_nodes`.

| Property | Shape | Meaning |
| :--- | :--- | :--- |
| `data.nodal_pos_w`, `data.nodal_vel_w` | `(E, V, 3)` | Simulation-node world positions and velocities. |
| `data.nodal_state_w` | `(E, V, 6)` | Position followed by velocity at each node. |
| `data.default_nodal_state_w` | `(E, V, 6)` | Initial state captured at binding. |

Use `set_nodal_state(state, env_ids=...)` to write positions and velocities
for selected environments. Input rows follow the selected order and use
world coordinates including the arena offsets. For example, translate the
current shape in a prepared scene by editing a snapshot and writing it back.

```python
state = cloth.data.nodal_state_w[0:1]
state[..., 2] += 0.05
cloth.set_nodal_state(state, env_ids=[0])
```

`get_surface_vertices()` and `get_surface_triangles()` describe the render
mesh. Use the nodal properties for physics observations and controls.

#### Visual Material

With native rendering, `get_visual_material_inst()` reports one
representative material or `None` per selected environment. Use
`set_visual_material()` to assign a replacement, and
`restore_visual_material()` to restore the authored assignments.

#### Pose Management

`set_local_pose(pose, env_ids=...)` places the captured rest shape at an
arena-local `xyz + xyzw` pose or 4×4 transform and zeros node velocities.
Use `set_nodal_state()` to preserve the current deformation while moving or
initializing the nodes. To restore the configured resting placement and
materials, call `reset(env_ids=...)`. Read the current shape through
`data.nodal_pos_w`.

## Related Topics

- {doc}`Cloth Body Simulation </tutorial/create_cloth>` for a rendered tutorial.
- {doc}`sim_soft_object` for particle sets with a tetrahedral interior.
- {doc}`sim_assets` for the asset overview and the cable segment-state interface.
