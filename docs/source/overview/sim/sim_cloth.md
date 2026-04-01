# Cloth Object

```{currentmodule} embodichain.lab.sim
```

The {class}`~objects.Cloth` class represents deformable surface entities in EmbodiChain. Unlike rigid bodies, cloth objects are defined by vertices and meshes rather than a single rigid pose.

## Configuration

Configured via {class}`~cfg.ClothObjectCfg`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `physical_attr` | `ClothPhysicalAttributesCfg` | `...` | Physical attributes. |
| `shape` | `MeshCfg` | `MeshCfg()` | Mesh configuration. |

### CLoth Body Attributes

Cloth bodies require both voxelization and physical attributes.

**Physical Attributes ({class}`~cfg.ClothPhysicalAttributesCfg`)**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `youngs` | `float` | `1e10` | Young's modulus (higher = stiffer). |
| `poissons` | `float` | `0.3` | Poisson's ratio. |
| `dynamic_friction` | `float` | `0.5` | Dynamic friction coefficient. |
| `elasticity_damping` | `float` | `0.0` | Elasticity damping factor. |
| `thickness` | `float` | `0.01` | Cloth thickness (m). |
| `bending_stiffness` | `float` | `0.001` | Bending stiffness. |
| `bending_damping` | `float` | `0.0` | Bending damping. |
| `enable_kinematic` | `bool` | `False` | If True, (partially) kinematic behavior is enabled. |
| `enable_ccd` | `bool` | `True` | Enable continuous collision detection (CCD). |
| `enable_self_collision` | `bool` | `False` | Enable self-collision handling. |
| `has_gravity` | `bool` | `True` | Whether the cloth is affected by gravity. |
| `self_collision_stress_tolerance` | `float` | `0.9` | Stress tolerance threshold for self-collision constraints. |
| `collision_mesh_simplification` | `bool` | `True` | Whether to simplify the collision mesh for self-collision. |
| `vertex_velocity_damping` | `float` | `0.005` | Per-vertex velocity damping. |
| `mass` | `float` | `-1.0` | Total mass of the cloth. If negative, density is used to compute mass. |
| `density` | `float` | `1.0` | Material density in kg/m^3. |
| `max_depenetration_velocity` | `float` | `1e6` | Maximum velocity used to resolve penetrations. |
| `max_velocity` | `float` | `100.0` | Clamp for linear (or vertex) velocity. |
| `self_collision_filter_distance` | `float` | `0.1` | Distance threshold for filtering self-collision vertex pairs. |
| `linear_damping` | `float` | `0.05` | Global linear damping applied to the cloth. |
| `sleep_threshold` | `float` | `0.05` | Velocity/energy threshold below which the cloth can go to sleep. |
| `settling_threshold` | `float` | `0.1` | Threshold used to decide convergence/settling state. |
| `settling_damping` | `float` | `10.0` | Additional damping applied during settling phase. |
| `min_position_iters` | `int` | `4` | Minimum solver iterations for position correction. |
| `min_velocity_iters` | `int` | `1` | Minimum solver iterations for velocity updates. |

For Cloth Object tutorial, please refer to the [Cloth Body Simulation](https://dexforce.github.io/EmbodiChain/tutorial/create_cloth.html).


### Setup & Initialization

```python
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import ClothObject, ClothObjectCfg


def create_2d_grid_mesh(width: float, height: float, nx: int = 1, ny: int = 1):
    """Create a flat rectangle in the XY plane centered at `origin`.

    The rectangle is subdivided into an `nx` by `ny` grid (cells) and
    triangulated. `nx=1, ny=1` yields the simple two-triangle rectangle.

    Returns an vertices and triangles.
    """
    w = float(width)
    h = float(height)
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")

    # Vectorized vertex positions using PyTorch
    x_lin = torch.linspace(-w / 2.0, w / 2.0, steps=nx + 1, dtype=torch.float64)
    y_lin = torch.linspace(-h / 2.0, h / 2.0, steps=ny + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(y_lin, x_lin)  # shapes: (ny+1, nx+1)
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)
    zz_flat = torch.full_like(xx_flat, 0, dtype=torch.float64)
    verts = torch.stack([xx_flat, yy_flat, zz_flat], dim=1)  # (Nverts, 3)

    # Vectorized triangle indices
    idx = torch.arange((nx + 1) * (ny + 1), dtype=torch.int64).reshape(ny + 1, nx + 1)
    v0 = idx[:-1, :-1].reshape(-1)
    v1 = idx[:-1, 1:].reshape(-1)
    v2 = idx[1:, :-1].reshape(-1)
    v3 = idx[1:, 1:].reshape(-1)
    tri1 = torch.stack([v0, v1, v3], dim=1)
    tri2 = torch.stack([v0, v3, v2], dim=1)
    faces = torch.cat([tri1, tri2], dim=0).to(torch.int32)
    return verts, faces

# 1. Initialize Simulation
device = "cuda" if torch.cuda.is_available() else "cpu"
sim_cfg = SimulationManagerCfg(sim_device=device)
sim = SimulationManager(sim_config=sim_cfg)

cloth_verts, cloth_faces = create_2d_grid_mesh(width=0.3, height=0.3, nx=12, ny=12)
cloth_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(cloth_verts.to("cpu").numpy()),
    triangles=o3d.utility.Vector3iVector(cloth_faces.to("cpu").numpy()),
)
cloth_save_path = os.path.join(tempfile.gettempdir(), "cloth_mesh.ply")
o3d.io.write_triangle_mesh(cloth_save_path, cloth_mesh)
# 2. Configure Cloth Object
cfg=ClothObjectCfg(
    uid="cloth_demo",
    shape=MeshCfg(fpath=cloth_save_path),
    init_pos=[0.5, 0.0, 0.3],
    init_rot=[0, 0, 0],
    physical_attr=ClothPhysicalAttributesCfg(
        mass=0.01,
        youngs=1e10,
        poissons=0.4,
        thickness=0.04,
        bending_stiffness=0.01,
        bending_damping=0.1,
        dynamic_friction=0.95,
        min_position_iters=30,
    ),
)

# 3. Spawn Cloth Object
# Note: Assuming the method in SimulationManager is 'add_cloth_object'
cloth_object: ClothObject = sim.add_cloth_object(cfg=cfg)

# 4. Initialize Physics
sim.reset_objects_state()
```
### Cloth Object Class
#### Vertex Data (Observation)
For cloth objects, the state is represented by the positions and velocities of its vertices, rather than a single root pose.

| Method | Return Shape | Description |
| :--- | :--- | :--- |
| `get_current_vertex_position()` | `(n_envs, n_vert, 3)` | Current positions of mesh vertices. |
| `get_current_vertex_velocity()` | `(n_envs, n_vert, 3)` | Current positions of  mesh vertices. |
| `get_rest_vertex_position()` | `(n_envs, n_vert, 3` | Rest (initial) positions of collision vertices. |

> Note: N is the number of environments/instances, V_col is the number of collision vertices, and V_sim is the number of simulation vertices.

```python
# Example: Accessing vertex data
vert_position = cloth_object.get_current_vertex_position()
print(f"vertices positions: {vert_position}")

vert_velocity = cloth_object.get_current_vertex_velocity()
print(f"Vertex Velocities: {vert_velocity}")
```
#### Pose Management
You can set the global pose of a cloth object (which transforms all its vertices), but getting a single "pose" from a deformed surface object is not supported.

| Method | Description |
| :--- | :--- |
| `set_local_pose(pose)` | Sets the pose of the object by transforming all vertices. |
| `get_local_pose()` | **Not Supported**. Raises `NotImplementedError` because a deformed object does not have a single rigid pose. |


```python
# Reset or Move the Cloth Object
target_pose = torch.tensor([[0, 0, 1.0, 1, 0, 0, 0]], device=device) # (x, y, z, qw, qx, qy, qz)
cloth_object.set_local_pose(target_pose)

# Important: Step simulation to apply changes
sim.update()
```