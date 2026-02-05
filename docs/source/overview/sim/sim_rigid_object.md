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

## Rigid Object Class — Common Methods & Attributes

Rigid objects are observed and controlled via single poses and linear/angular velocities. Key APIs include:

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `get_local_pose(to_matrix=False)` | `(N, 7)` or `(N, 4, 4)` | Get object local pose as (x, y, z, qw, qx, qy, qz) or 4x4 matrix per environment. |
| `set_local_pose(pose, env_ids=None)` | `pose: (N, 7)` or `(N, 4, 4)` | Teleport object to given pose (requires calling `sim.update()` to apply). |
| `body_data.pose` | `(N, 7)` | Access object pose directly (for dynamic/kinematic bodies). |
| `body_data.lin_vel` | `(N, 3)` | Access linear velocity of object root (for dynamic/kinematic bodies). |
| `body_data.ang_vel` | `(N, 3)` | Access angular velocity of object root (for dynamic/kinematic bodies). |
| `body_state` | `(N, 13)` | Get full body state: [x, y, z, qw, qx, qy, qz, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]. |
| `add_force_torque(force, torque, pos, env_ids)` | `force: (N, 3)`, `torque: (N, 3)` | Apply continuous force and/or torque to the object. |
| `clear_dynamics(env_ids=None)` | - | Reset velocities and clear all forces/torques. |
| `set_visual_material(mat, env_ids=None)` | `mat: VisualMaterial` | Change visual appearance at runtime. |
| `enable_collision(flag, env_ids=None)` | `flag: torch.Tensor` | Enable/disable collision for specific instances. |
| `reset(env_ids=None)` | - | Reset objects to initial configuration. |

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