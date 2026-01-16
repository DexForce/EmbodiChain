# Soft Object

```{currentmodule} embodichain.lab.sim
```

The {class}`~objects.SoftObject` class represents deformable entities (e.g., cloth, sponges, soft robotics) in EmbodiChain. Unlike rigid bodies, soft objects are defined by vertices and meshes rather than a single rigid pose.

## Configuration

Configured via {class}`~cfg.SoftObjectCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `voxel_attr` | `SoftbodyVoxelAttributesCfg` | `...` | Voxelization attributes. |
| `physical_attr` | `SoftbodyPhysicalAttributesCfg` | `...` | Physical attributes. |
| `shape` | `MeshCfg` | `MeshCfg()` | Mesh configuration. |

### Soft Body Attributes

Soft bodies require both voxelization and physical attributes.

**Voxel Attributes ({class}`~cfg.SoftbodyVoxelAttributesCfg`)**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `triangle_remesh_resolution` | `int` | `8` | Resolution to remesh the softbody mesh before building physx collision mesh. |
| `triangle_simplify_target` | `int` | `0` | Simplify mesh faces to target value. |
| `simulation_mesh_resolution` | `int` | `8` | Resolution to build simulation voxelize textra mesh. |
| `simulation_mesh_output_obj` | `bool` | `False` | Whether to output the simulation mesh as an obj file for debugging. |

**Physical Attributes ({class}`~cfg.SoftbodyPhysicalAttributesCfg`)**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `youngs` | `float` | `1e6` | Young's modulus (higher = stiffer). |
| `poissons` | `float` | `0.45` | Poisson's ratio (higher = closer to incompressible). |
| `dynamic_friction` | `float` | `0.0` | Dynamic friction coefficient. |
| `elasticity_damping` | `float` | `0.0` | Elasticity damping factor. |
| `material_model` | `SoftBodyMaterialModel` | `CO_ROTATIONAL` | Material constitutive model. |
| `enable_kinematic` | `bool` | `False` | If True, (partially) kinematic behavior is enabled. |
| `enable_ccd` | `bool` | `False` | Enable continuous collision detection. |
| `enable_self_collision` | `bool` | `False` | Enable self-collision handling. |
| `mass` | `float` | `-1.0` | Total mass. If negative, density is used. |
| `density` | `float` | `1000.0` | Material density in kg/m^3. |

For Soft Object tutorial, please refer to the [Soft Body Simulation](https://dexforce.github.io/EmbodiChain/tutorial/create_softbody.html).


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