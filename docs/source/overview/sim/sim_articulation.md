# Articulation

```{currentmodule} embodichain.lab.sim
```

The {class}`~objects.Articulation` class represents the fundamental physics entity for articulated objects (e.g., robots, grippers, cabinets, doors) in EmbodiChain.

## Configuration

Articulations are configured using the {class}`~cfg.ArticulationCfg` dataclass.
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fpath` | `str` | `None` | Path to the asset file (URDF/USD). |
| `init_pos` | `tuple` | `(0,0,0)` | Initial root position `(x, y, z)`. |
| `init_rot` | `tuple` | `(0,0,0)` | Initial root rotation `(r, p, y)` in degrees. |
| `fix_base` | `bool` | `True` | Whether to fix the base of the articulation. |
| `use_usd_properties` | `bool` | `False` | If True, use physical properties from USD file; if False, override with config values. Only effective for usd files. |
| `init_qpos` | `List[float]` | `None` | Initial joint positions. |
| `body_scale` | `List[float]` | `[1.0, 1.0, 1.0]` | Scaling factors for the articulation links. |
| `disable_self_collisions` | `bool` | `True` | Whether to disable self-collisions. |
| `drive_props` | `JointDrivePropertiesCfg` | `...` | Default drive properties. |
| `attrs` | `RigidBodyAttributesCfg` | `...` | Rigid body attributes configuration. |


### Drive Configuration

The `drive_props` parameter controls the joint physics behavior. It is defined using the `JointDrivePropertiesCfg` class. For articulation object without internal drive force, like cabinet and drawer, better set `drive_type` to `"none"`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stiffness` | `float` / `Dict` | `1.0e4` | Stiffness (P-gain) of the joint drive. Unit: $N/m$ or $Nm/rad$. |
| `damping` | `float` / `Dict` | `1.0e3` | Damping (D-gain) of the joint drive. Unit: $Ns/m$ or $Nms/rad$. |
| `max_effort` | `float` / `Dict` | `1.0e10` | Maximum effort (force/torque) the joint can exert. |
| `max_velocity` | `float` / `Dict` | `1.0e10` | Maximum velocity allowed for the joint ($m/s$ or $rad/s$). |
| `friction` | `float` / `Dict` | `0.0` | Joint friction coefficient. |
| `drive_type` | `str` | `"none"` | Drive mode: `"force"`(driven by a force), `"acceleration"`(driven by an acceleration) or `none`(no force). |

### Setup & Initialization

```python
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Articulation, ArticulationCfg

# 1. Initialize Simulation
device = "cuda" if torch.cuda.is_available() else "cpu"
sim_cfg = SimulationManagerCfg(sim_device=device)
sim = SimulationManager(sim_config=sim_cfg)

# 2. Configure Articulation
art_cfg = ArticulationCfg(
    fpath="assets/robots/franka/franka.urdf",
    init_pos=(0, 0, 0.5),
    fix_base=True
)

# 3. Spawn Articulation
# Note: The method is 'add_articulation'
articulation: Articulation = sim.add_articulation(cfg=art_cfg)

# 4. Reset Simulation
# This performs a global reset of the simulation state
sim.reset_objects_state()
```

### USD Import

You can import USD files (`.usd`, `.usda`, `.usdc`) as articulations:

```python
from embodichain.data import get_data_path

# Import USD with properties from file
usd_art_cfg = ArticulationCfg(
    fpath=get_data_path("path/to/robot.usd"),
    init_pos=(0, 0, 0.5),
    use_usd_properties=True  # Keep USD drive/physics properties
)
usd_robot = sim.add_articulation(cfg=usd_art_cfg)

# Or override USD properties with config (URDF behavior)
usd_art_cfg_override = ArticulationCfg(
    fpath=get_data_path("path/to/robot.usd"),
    init_pos=(0, 0, 0.5),
    use_usd_properties=False,  # Use config instead
    drive_props=JointDrivePropertiesCfg(stiffness=5000, damping=500)
)
robot = sim.add_articulation(cfg=usd_art_cfg_override)
```

## Articulation Class

State data is accessed via getter methods that return batched tensors (`N` environments). Certain static properties are available as standard class properties.

| Property | Type | Description |
| :--- | :--- | :--- |
| `num_envs` | `int` | Number of simulation environments this articulation is instantiated in. |
| `dof` | `int` | Degrees of freedom (number of actuated joints). |
| `joint_names` | `List[str]` | Names of all movable joints. |
| `link_names` | `List[str]` | Names of all rigid links. |
| `mass` | `Tensor` | Total mass of the articulation per environment `(N, 1)`. |

| Method | Shape / Return Type | Description |
| :--- | :--- | :--- |
| `get_local_pose(to_matrix=False)` | `(N, 7)` or `(N, 4, 4)` | Root link pose `[x, y, z, qw, qx, qy, qz]` or a 4x4 matrix. |
| `get_link_pose(link_name, to_matrix=False)` | `(N, 7)` or `(N, 4, 4)` | Specific link pose `[x, y, z, qw, qx, qy, qz]` or a 4x4 matrix. |
| `get_qpos(target=False)` | `(N, dof)` | Current joint positions (or joint targets if `target=True`). |
| `get_qvel(target=False)` | `(N, dof)` | Current joint velocities (or velocity targets if `target=True`). |
| `get_joint_drive()` | `Tuple[Tensor, ...]` | Returns `(stiffness, damping, max_effort, max_velocity, friction)`, each shaped `(N, dof)`. |

```python
# Example: Accessing state
print(f"Degrees of freedom: {articulation.dof}")
print(f"Current Joint Positions: {articulation.get_qpos()}")
print(f"End Effector Pose: {articulation.get_link_pose('ee_link')}")
```

### Control & Dynamics
You can control the articulation by setting target states or directly applying forces.

### Joint Control
```python
# Set joint position targets (PD Control)
# Get current qpos to create a target tensor of correct shape
current_qpos = articulation.get_qpos()
target_qpos = torch.zeros_like(current_qpos)

# Set target position
# target=True: Sets the drive target. The physics engine applies forces to reach this position.
# target=False: Instantly resets/teleports joints to this position (ignoring physics).
articulation.set_qpos(target_qpos, target=True)

# Set target velocities
target_qvel = torch.zeros_like(current_qpos)
articulation.set_qvel(target_qvel, target=True)

# Apply forces directly
# Sets an external force tensor (N, dof) applied at the degree of freedom. 
target_qf = torch.ones_like(current_qpos) * 10.0
articulation.set_qf(target_qf)

# Important: Step simulation to apply control
sim.update()
```

### Pose Control
```python
# Teleport the articulation root to a new pose
# shape: (N, 7) formatted as [x, y, z, qw, qx, qy, qz]
new_root_pose = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=device).repeat(sim.num_envs, 1)
articulation.set_local_pose(new_root_pose)
```

### Drive Configuration
Dynamically adjust drive properties.

```python
# Set stiffness for all joints
articulation.set_joint_drive(
    stiffness=torch.tensor([100.0], device=device), 
    damping=torch.tensor([10.0], device=device)
)
```

### Kinematics
Supports differentiable Forward Kinematics (FK) and Jacobian computation.

```python
# Compute Forward Kinematics
# Note: Ensure `build_pk_chain=True` in cfg
if getattr(art_cfg, 'build_pk_chain', False):
    # Returns (batch_size, 4, 4) homogeneous transformation matrix
    ee_pose = articulation.compute_fk(
        qpos=articulation.get_qpos(),
        end_link_name="ee_link" # Replace with actual link name
    )
    
    # Or return a dictionary of multiple link transforms (pytorch_kinematics Transform3d objects)
    link_poses = articulation.compute_fk(
        qpos=articulation.get_qpos(),
        link_names=["link1", "link2"],
        to_dict=True
    )
```

### State Reset
Resetting an articulation returns it to its initial state properties.
```python
# Clear the physical dynamics and velocities
articulation.clear_dynamics()

# Reset the articulation entirely (resets pose, velocities, and root states to config defaults)
articulation.reset()
```
