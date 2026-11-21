# Rigid Object

```{currentmodule} embodichain.lab.sim
```

Use {class}`~objects.RigidObject` for a single body with a pose, physical
properties, and collision geometry. A dynamic body responds to forces and
contacts. A kinematic body follows commanded poses, and a static body is used
for fixed geometry. Use {doc}`sim_rigid_object_group` when each piece needs an
independent body state.

## Configuration

Create a {class}`~cfg.RigidObjectCfg` and declare it with
`sim.add_rigid_object(cfg)`.

| Field | Use |
| :--- | :--- |
| `uid` | Unique identifier for the logical object in its manager. |
| `shape` | A primitive such as `CubeCfg`, a file- or array-backed `MeshCfg`, or a `CompoundShapeCfg` for several colliders on one body. |
| `source` | Select a rigid body from USD with `UsdSourceCfg`. Supply either `shape` or `source`. |
| `body_type`, `body_scale` | Procedural bodies default to `"dynamic"` and unit scale. USD preserve mode uses the authored actor type and scale. |
| `attrs` | Grouped mass, rigid-body, collision, and contact-material settings. |
| `init_pos`, `init_rot` | Arena-local placement for procedural replicas. Euler angles are in degrees. An explicit `init_local_pose` matrix takes precedence. |
| `visuals` | Appearance overrides independent of the physical shape. An optional `semantic_shape` supplies geometry for semantic queries. |

### Rigid Body Physics

{class}`~cfg.RigidBodyPhysicsCfg` groups optional overrides. Omitted values
use the applicable source or backend defaults.

| Group | Example fields |
| :--- | :--- |
| `mass_props` | `mass`, `density`, `inertia`, `com_position`, `com_quaternion` |
| `rigid_props` | `linear_damping`, `angular_damping`, `enable_ccd` |
| `collision_props` | `collision_enabled`, `contact_offset`, `rest_offset` |
| `material_props` | `dynamic_friction`, `static_friction`, `restitution` |

With three principal inertia moments and a COM quaternion in `xyzw` order,
EmbodiChain composes the inertia matrix as `R diag(I) Rᵀ`. A full 3×3 inertia
matrix is already expressed in body axes. Supply an explicit inertia whenever
you supply its orientation. See {doc}`sim_manager/physics/index` for
backend-specific properties.

## Setup & Initialization

This CPU example uses two environments and moves only the first cube. Declare
the initial scene, then call `prepare()` before reading or writing state.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    MassPropertiesCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=2,
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    cube = sim.add_rigid_object(
        RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            init_pos=(0.0, 0.0, 0.3),
            attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
        )
    )
    sim.prepare()
    poses = cube.get_local_pose()
    selected_pose = poses[0:1].clone()
    selected_pose[:, 2] += 0.1
    cube.set_local_pose(selected_pose, env_ids=[0])
    sim.update(step=1, render_final_step=False)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

### USD Import

Select the authored rigid body by prim path. Preserve mode uses the source
physics. Switch to overlay mode when you want to replace a configured field.
Declare the chosen configuration before preparing the scene.

```python
from embodichain.lab.sim.cfg import (
    MassPropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    UsdSourceCfg,
)

part_cfg = RigidObjectCfg(
    uid="part",
    source=UsdSourceCfg(file_path="workcell.usda", source_prim_path="/World/Part"),
)
heavy_part_cfg = part_cfg.replace(
    asset_physics_mode="overlay",
    attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=2.0)),
)
```

Use `sim.add_usd()` for the whole workcell. See
{doc}`sim_assets` for import recipes and lifecycle guidance.

### Per-environment variants

With Newton and a compatible SDK, `env_variants` can assign different
procedural shapes, placements, and physics to selected environment IDs under
the same logical UID. Prepare the scene after declaring the variants. Solver
support depends on collider topology. See {doc}`sim_manager/physics/newton/runtime_qualification`.

## Rigid Object Class — Common Methods & Attributes

### Pose & State

`E` is the number of environment instances. An arena-local translation is
relative to that environment's origin. Pose vectors use
`[x, y, z, qx, qy, qz, qw]`.

| Read | Shape | Meaning |
| :--- | :--- | :--- |
| `get_local_pose()` | `(E, 7)` | Arena-local object pose. `to_matrix=True` returns `(E, 4, 4)`. |
| `body_state` | `(E, 13)` | Pose, linear velocity, and angular velocity. Static bodies report zero velocities. |
| `body_data.lin_vel`, `body_data.ang_vel` | `(E, 3)` | Linear and angular velocity for bound body data. |
| `body_data.lin_acc`, `body_data.ang_acc` | `(E, 3)` | Linear and angular acceleration where the backend supports the read. |
| `body_data.com_pose` | `(E, 7)` | Local center-of-mass pose. |

`set_local_pose(pose, env_ids=...)` accepts one pose per selected environment
in the requested order. The write leaves simulation time unchanged.
Clone views from `body_data` when you need to keep a snapshot across reads.

### Dynamics Control

Use `set_velocity(lin_vel=..., ang_vel=..., env_ids=...)` for direct velocity
writes on dynamic bodies. Use `add_force_torque()` to apply a wrench at the
center of mass for a physics update. For a continuing load, apply it before
each step. `clear_dynamics(env_ids=...)` clears dynamics for selected bodies.

### Physical Properties

Use `set_attrs()` for supported grouped writes or `get_mass()`,
`set_mass()`, `get_friction()`, `set_friction()`, `get_damping()`, and
`set_damping()` for specific properties. `get_inertia()` and `set_inertia()`
use three inertia values per selected body. `set_com_pose()` uses one local
`xyz + xyzw` pose per selection. Runtime write support depends on the backend
and the prepared object's topology.

### Geometry & Body Type

Choose the actor type and geometry when declaring the asset. Actor type is
fixed after Spawn binding. Call `get_body_scale()` to read its scale.
Explicit composed visuals retain their
prepared scale. Use `get_vertices()` and `get_triangles()` for mesh queries,
and `get_collision_shapes()` to inspect the resolved collision geometry.
`is_static` and `is_non_dynamic` describe the body's runtime actor type.

### Collision & Filtering

Use `MeshCfg.collision` to choose the collision approximation. For a complex
mesh, use a simplified collider or bound the number of hulls with
`max_hulls` and `approximation="convex_decomposition"`. The backend capability
determines which cooked representations are supported. Use
`enable_collision()` and `set_collision_filter()` only on backends that
support the requested runtime filter changes.

### Visual & Appearance

With native rendering, `get_visual_material_inst()` returns one representative
instance per selected environment. For a render body with multiple mesh
segments, `get_existing_visual_material()` retains every original instance
and exposes a working instance for edits. `set_visual_material(mat)` creates
one assignment per selected environment by default. Pass `shared=True` to
share one assignment across the selection. See
{doc}`sim_manager/rendering/materials` for material presets.

### Utility & Identification

`sim.get_rigid_object(uid)` retrieves the logical batch. `get_user_ids()`
returns engine user IDs for its instances. `reset(env_ids=...)` restores the
selected initial poses and captured mass properties and materials, then clears
dynamics. Use `sim.remove_asset(uid)` or `sim.replace_rigid_object(cfg)` for
topology changes supported by the current scene. Use the replacement's returned
handle for subsequent reads and writes.

### Observation Shapes

Keep the leading environment dimension when selecting a single object or
collecting a batch. A two-environment `body_state` has shape `(2, 13)`
even when both environments use the same configuration.

## Example: Applying Force and Torque

In a prepared scene with a dynamic `cube`, one force row targets environment
zero. The following physics step advances every environment, with the load
targeted to the selected body.

```python
import torch

force = torch.tensor([[0.0, 0.0, 20.0]], device=sim.device)
cube.add_force_torque(force=force, env_ids=[0])
sim.update(step=1, render_final_step=False)
linear_velocity = cube.body_data.lin_vel.clone()
```

## Related Topics

- {doc}`Create Scene </tutorial/create_scene>` for a complete rendered setup.
- {doc}`sim_rigid_object_group` for batched collections with a member axis.
- {doc}`sensors/contact_sensor` for selecting contact targets and observing forces.
