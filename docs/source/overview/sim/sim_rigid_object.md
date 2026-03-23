# Rigid Object

```{currentmodule} embodichain.lab.sim
```

The `RigidObject` class represents non-deformable (rigid) physical objects in EmbodiChain. Rigid objects are characterized by a single pose (position + orientation), collision and visual shapes, and standard rigid-body physical properties.

## Configuration

Configured via the {class}`~cfg.RigidObjectCfg` class.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shape` | {class}`~shapes.ShapeCfg` | `ShapeCfg()` | Geometry configuration for visual and collision shapes. Use `MeshCfg` for mesh files or primitive cfgs (e.g., `CubeCfg`). |
| `body_type` | `Literal["dynamic","kinematic","static"]` | `"dynamic"` | Actor type for the rigid body. See `{class}`~cfg.RigidObjectCfg.to_dexsim_body_type` for conversion. |
| `attrs` | {class}`~cfg.RigidBodyAttributesCfg` | defaults in code | Physical attributes (mass, damping, friction, restitution, collision offsets, CCD, etc.). |
| `init_pos` | `Sequence[float]` | `(0,0,0)` | Initial root position (x, y, z). |
| `init_rot` | `Sequence[float]` | `(0,0,0)` (Euler degrees) | Initial root orientation (Euler angles in degrees) or provide `init_local_pose`. |
| `use_usd_properties` | `bool` | `False` | If True, use physical properties from USD file; if False, override with config values. Only effective for usd files. |
| `uid` | `str` | `None` | Optional unique identifier for the object; manager will assign one if omitted. |

### Rigid Body Attributes ({class}`~cfg.RigidBodyAttributesCfg`)

The full attribute set lives in `{class}`~cfg.RigidBodyAttributesCfg`. Common fields shown in code include:

| Parameter | Type | Default (from code) | Description |
| :--- | :--- | :---: | :--- |
| `mass` | `float` | `1.0` | Mass of the rigid body in kilograms (set to 0 to use density). |
| `density` | `float` | `1000.0` | Density used when mass is negative/zero. |
| `linear_damping` | `float` | `0.7` | Linear damping coefficient. |
| `angular_damping` | `float` | `0.7` | Angular damping coefficient. |
| `dynamic_friction` | `float` | `0.5` | Dynamic friction coefficient. |
| `static_friction` | `float` | `0.5` | Static friction coefficient. |
| `restitution` | `float` | `0.0` | Restitution (bounciness). |
| `contact_offset` | `float` | `0.002` | Contact offset for collision detection. |
| `rest_offset` | `float` | `0.001` | Rest offset for collision detection. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection. |

Use the `.attr()` helper to convert to `dexsim.PhysicalAttr` when interfacing with the engine.

## Setup & Initialization

```python
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg

# 1. Initialize Simulation
device = "cuda" if torch.cuda.is_available() else "cpu"
sim_cfg = SimulationManagerCfg(sim_device=device)
sim = SimulationManager(sim_cfg)

# 2. Configure a rigid object (cube)
physics_attrs = RigidBodyAttributesCfg(mass=1.0, dynamic_friction=0.5, static_friction=0.5, restitution=0.1)

cfg = RigidObjectCfg(
    uid="cube",
    shape=CubeCfg(size=[0.1, 0.1, 0.1]),
    body_type="dynamic",
    attrs=physics_attrs,
    init_pos=(0.0, 0.0, 1.0),
)

# 3. Spawn Rigid Object
cube: RigidObject = sim.add_rigid_object(cfg=cfg)

# 4. (Optional) Open window and run
if not sim.sim_config.headless:
    sim.open_window()
sim.update()
```

> Note: `scripts/tutorials/sim/create_scene.py` provides a minimal working example of adding a rigid cube and running the simulation loop.

### USD Import

You can import USD files (`.usd`, `.usda`, `.usdc`) as rigid objects:

```python
from embodichain.data import get_data_path

# Import USD with properties from file
usd_cfg = RigidObjectCfg(
    shape=MeshCfg(fpath=get_data_path("path/to/object.usd")),
    body_type="dynamic",
    use_usd_properties=True  # Keep USD properties
)
obj = sim.add_rigid_object(cfg=usd_cfg)

# Or override USD properties with config
usd_cfg_override = RigidObjectCfg(
    shape=MeshCfg(fpath=get_data_path("path/to/object.usd")),
    body_type="dynamic",
    use_usd_properties=False,  # Use config instead
    attrs=RigidBodyAttributesCfg(mass=2.0)
)
obj2 = sim.add_rigid_object(cfg=usd_cfg_override)
```

## Rigid Object Class — Common Methods & Attributes

Rigid objects are observed and controlled via single poses and linear/angular velocities. Key APIs include:

### Pose & State

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `get_local_pose(to_matrix=False)` | `(N, 7)` or `(N, 4, 4)` | Get object local pose as (x, y, z, qw, qx, qy, qz) or 4x4 matrix per environment. |
| `set_local_pose(pose, env_ids=None)` | `pose: (N, 7)` or `(N, 4, 4)` | Teleport object to given pose (requires calling `sim.update()` to apply). |
| `body_data.pose` | `(N, 7)` | Access object pose directly (for dynamic/kinematic bodies). |
| `body_data.lin_vel` | `(N, 3)` | Access linear velocity of object root (for dynamic/kinematic bodies). |
| `body_data.ang_vel` | `(N, 3)` | Access angular velocity of object root (for dynamic/kinematic bodies). |
| `body_data.vel` | `(N, 6)` | Concatenated linear and angular velocities. |
| `body_data.com_pose` | `(N, 7)` | Get center of mass pose of rigid bodies. |
| `body_data.default_com_pose` | `(N, 7)` | Default center of mass pose. |
| `body_state` | `(N, 13)` | Get full body state: [x, y, z, qw, qx, qy, qz, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]. |

### Dynamics Control

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `add_force_torque(force, torque, pos, env_ids)` | `force: (N, 3)`, `torque: (N, 3)` | Apply continuous force and/or torque to object. |
| `set_velocity(lin_vel, ang_vel, env_ids)` | `lin_vel: (N, 3)`, `ang_vel: (N, 3)` | Set linear and/or angular velocity directly. |
| `clear_dynamics(env_ids=None)` | - | Reset velocities and clear all forces/torques. |

### Physical Properties

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `set_attrs(attrs, env_ids=None)` | `attrs: RigidBodyAttributesCfg` | Set physical attributes (mass, friction, damping, etc.). |
| `set_mass(mass, env_ids=None)` | `mass: (N,)` | Set mass for rigid object. |
| `get_mass(env_ids=None)` | `(N,)` | Get mass for rigid object. |
| `set_friction(friction, env_ids=None)` | `friction: (N,)` | Set dynamic and static friction. |
| `get_friction(env_ids=None)` | `(N,)` | Get friction (dynamic friction value). |
| `set_damping(damping, env_ids=None)` | `damping: (N, 2)` | Set linear and angular damping. |
| `get_damping(env_ids=None)` | `(N, 2)` | Get linear and angular damping. |
| `set_inertia(inertia, env_ids=None)` | `inertia: (N, 3)` | Set inertia tensor diagonal values. |
| `get_inertia(env_ids=None)` | `(N, 3)` | Get inertia tensor diagonal values. |
| `set_com_pose(com_pose, env_ids=None)` | `com_pose: (N, 7)` | Set center of mass pose (dynamic/kinematic only). |

### Geometry & Body Type

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `get_vertices(env_ids=None)` | `(N, num_verts, 3)` | Get mesh vertices of the rigid objects. |
| `get_body_scale(env_ids=None)` | `(N, 3)` | Get the body scale. |
| `set_body_scale(scale, env_ids=None)` | `scale: (N, 3)` | Set scale of rigid body (CPU only). |
| `set_body_type(body_type)` | `body_type: str` | Change body type between 'dynamic' and 'kinematic'. |
| `is_static` | `bool` | Check if the rigid object is static. |
| `is_non_dynamic` | `bool` | Check if the rigid object is non-dynamic (static or kinematic). |

### Collision & Filtering

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `enable_collision(enable, env_ids=None)` | `enable: (N,)` | Enable/disable collision for specific instances. |
| `set_collision_filter(filter_data, env_ids=None)` | `filter_data: (N, 4)` | Set collision filter data (arena id, collision flag, ...). |

### Visual & Appearance

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `set_visual_material(mat, env_ids=None, shared=False)` | `mat: VisualMaterial` | Change visual appearance at runtime. |
| `get_visual_material_inst(env_ids=None)` | `List[VisualMaterialInst]` | Get material instances for the rigid object. |
| `share_visual_material_inst(mat_insts)` | `mat_insts: List[VisualMaterialInst]` | Share material instances between objects. |
| `set_visible(visible)` | `visible: bool` | Set visibility of the rigid object. |
| `set_physical_visible(visible, rgba=None)` | `visible: bool`, `rgba: (4,)` | Set collision body render visibility. |

### Utility & Identification

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `get_user_ids()` | `(N,)` | Get the user IDs of the rigid bodies. |
| `reset(env_ids=None)` | - | Reset objects to initial configuration. |
| `destroy()` | - | Destroy and remove the rigid object from simulation. |

### Observation Shapes

- Pose: `(N, 7)` per-object pose (position + quaternion).
- Velocities: `(N, 3)` for linear and angular velocities respectively.

N denotes the number of parallel environments when using vectorized simulation (`SimulationManagerCfg.num_envs`).

## Notes & Best Practices

- When moving objects programmatically via `set_local_pose`, call `sim.update()` (or step the sim) to ensure transforms and collision state are synchronized.
- Use `static` body type for fixed obstacles or environment pieces (they do not consume dynamic simulation resources).
- Use `kinematic` for objects whose pose is driven by code (teleporting or animation) but still interact with dynamic objects.
- For complex meshes, enabling convex decomposition (`RigidObjectCfg.max_convex_hull_num`) or providing a simplified collision mesh improves stability and performance.
- To use GPU physics, ensure `SimulationManagerCfg.sim_device` is set to `cuda` and call `sim.init_gpu_physics()` before large-batch simulations.

## Example: Applying Force and Torque

```python
import torch

# Apply force to the cube
force = torch.tensor([[0.0, 0.0, 100.0]], device=sim.device)  # (N, 3) upward force
cube.add_force_torque(force=force, torque=None)
sim.update()

# Apply torque to the cube
torque = torch.tensor([[0.0, 0.0, 10.0]], device=sim.device)  # (N, 3) torque around z-axis
cube.add_force_torque(force=None, torque=torque)
sim.update()

# Access velocity data
linear_vel = cube.body_data.lin_vel  # (N, 3)
angular_vel = cube.body_data.ang_vel  # (N, 3)
```

## Integration with Sensors & Scenes

Rigid objects integrate with sensors (cameras, contact sensors) and gizmos. You can attach sensors referencing object `uid`s or query contacts for collision-based events.

## Related Topics

- Soft bodies: see the Soft Object documentation for deformable object interfaces. ([Soft Body Simulation Tutorial](https://dexforce.github.io/EmbodiChain/tutorial/create_softbody.html))
- Examples: check `scripts/tutorials/sim/create_scene.py` and `examples/sim` for more usage patterns.


<!-- End of rigid object overview -->