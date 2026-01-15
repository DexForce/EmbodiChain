# Soft Object

The `SoftObject` class represents deformable entities (e.g., cloth, sponges, soft robotics) in EmbodiChain. Unlike rigid bodies, soft objects are defined by vertices and meshes rather than a single rigid pose.

## Configuration

Soft objects are configured using the `SoftObjectCfg` dataclass.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `fpath` | `str` | `None` | Path to the soft body asset file (e.g., `.msh`, `.vtk`). |
| `init_pos` | `tuple` | `(0,0,0)` | Initial position `(x, y, z)`. |
| `init_rot` | `tuple` | `(0,0,0)` | Initial rotation `(r, p, y)` in degrees. |

### Setup & Initialization

```python
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import SoftObject, SoftObjectCfg

# 1. Initialize Simulation
device = "cuda" if torch.cuda.is_available() else "cpu"
sim_cfg = SimulationManagerCfg(sim_device=device)
sim = SimulationManager(sim_config=sim_cfg)

# 2. Configure Soft Object
soft_cfg = SoftObjectCfg(
    fpath="assets/objects/sponge.msh", # Example asset path
    init_pos=(0, 0, 0.5),
    init_rot=(0, 0, 0)
)

# 3. Spawn Soft Object
# Note: Assuming the method in SimulationManager is 'add_soft_object'
soft_object: SoftObject = sim.add_soft_object(cfg=soft_cfg)

# 4. Initialize Physics
sim.reset_objects_state()
```
### Soft Object Class
#### Vertex Data (Observation)
For soft objects, the state is represented by the positions and velocities of its vertices, rather than a single root pose.

| Method | Return Shape | Description |
| :--- | :--- | :--- |
| `get_current_collision_vertices()` | `(N, V_col, 3)` | Current positions of collision mesh vertices. |
| `get_current_sim_vertices()` | `(N, V_sim, 3)` | Current positions of simulation mesh vertices (nodes). |
| `get_current_sim_vertex_velocities()` | `(N, V_sim, 3)` | Current velocities of simulation vertices. |
| `get_rest_collision_vertices()` | `(N, V_col, 3)` | Rest (initial) positions of collision vertices. |
| `get_rest_sim_vertices()` | `(N, V_sim, 3)` | Rest (initial) positions of simulation vertices. |

> Note: N is the number of environments/instances, V_col is the number of collision vertices, and V_sim is the number of simulation vertices.

```python
# Example: Accessing vertex data
sim_verts = soft_object.get_current_sim_vertices()
print(f"Simulation Vertices Shape: {sim_verts.shape}")

velocities = soft_object.get_current_sim_vertex_velocities()
print(f"Vertex Velocities: {velocities}")
```
#### Pose Management
You can set the global pose of a soft object (which transforms all its vertices), but getting a single "pose" from a deformed object is not supported.

| Method | Description |
| :--- | :--- |
| `set_local_pose(pose)` | Sets the pose of the object by transforming all vertices. |
| `get_local_pose()` | **Not Supported**. Raises `NotImplementedError` because a deformed object does not have a single rigid pose. |


```python
# Reset or Move the Soft Object
target_pose = torch.tensor([[0, 0, 1.0, 1, 0, 0, 0]], device=device) # (x, y, z, qw, qx, qy, qz)
soft_object.set_local_pose(target_pose)

# Important: Step simulation to apply changes
sim.update()
```