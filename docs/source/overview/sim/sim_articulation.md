# Articulation

```{currentmodule} embodichain.lab.sim
```

The {class}`~objects.Articulation` class represents the fundamental physics entity for articulated objects (e.g., robots, grippers, cabinets, doors) in EmbodiChain.

## Configuration

Articulations are configured using the {class}`~cfg.ArticulationCfg` dataclass.
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fpath` | `str` | `None` | Path to the asset file (URDF/MJCF). |
| `init_pos` | `tuple` | `(0,0,0)` | Initial root position `(x, y, z)`. |
| `init_rot` | `tuple` | `(0,0,0)` | Initial root rotation `(r, p, y)` in degrees. |
| `fix_base` | `bool` | `True` | Whether to fix the base of the articulation. |
| `drive_props` | `JointDrivePropertiesCfg` | `...` | Default drive properties. |

### Drive Configuration

The `drive_props` parameter controls the joint physics behavior. It is defined using the `JointDrivePropertiesCfg` class.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `stiffness` | `float` / `Dict` | `1.0e4` | Stiffness (P-gain) of the joint drive. Unit: $N/m$ or $Nm/rad$. |
| `damping` | `float` / `Dict` | `1.0e3` | Damping (D-gain) of the joint drive. Unit: $Ns/m$ or $Nms/rad$. |
| `max_effort` | `float` / `Dict` | `1.0e10` | Maximum effort (force/torque) the joint can exert. |
| `max_velocity` | `float` / `Dict` | `1.0e10` | Maximum velocity allowed for the joint ($m/s$ or $rad/s$). |
| `friction` | `float` / `Dict` | `0.0` | Joint friction coefficient. |
| `drive_type` | `str` | `"force"` | Drive mode: `"force"` or `"acceleration"`. |

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

# 4. Initialize Physics
sim.reset_objects_state()
```
## Articulation Class
State Data (Observation)
State data is accessed via getter methods that return batched tensors.

| Property | Shape | Description |
| :--- | :--- | :--- |
| `get_local_pose` | `(N, 7)` | Root link pose `[x, y, z, qw, qx, qy, qz]`. |
| `get_qpos` | `(N, dof)` | Joint positions. |
| `get_qvel` | `(N, dof)` | Joint velocities. |



```python
# Example: Accessing state
# Note: Use methods (with brackets) instead of properties
print(f"Current Joint Positions: {articulation.get_qpos()}")
print(f"Root Pose: {articulation.get_local_pose()}")
```
### Control & Dynamics
You can control the articulation by setting joint targets.

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

# Important: Step simulation to apply control
sim.update()
```
### Drive Configuration
Dynamically adjust drive properties.

```python
# Set stiffness for all joints
articulation.set_drive(
    stiffness=torch.tensor([100.0], device=device), 
    damping=torch.tensor([10.0], device=device)
)
```
### Kinematics
Supports differentiable Forward Kinematics (FK) and Jacobian computation.
```python
# Compute Forward Kinematics
# Note: Ensure 'build_pk_chain=True' in cfg
if getattr(art_cfg, 'build_pk_chain', False):
    ee_pose = articulation.compute_fk(
        qpos=articulation.get_qpos(), # Use method call
        end_link_name="ee_link" # Replace with actual link name
    )
```
