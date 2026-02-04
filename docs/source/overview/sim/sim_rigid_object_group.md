# Rigid Object Group

```{currentmodule} embodichain.lab.sim
```

The `RigidObjectGroup` class represents a logical collection of rigid objects that are created and managed together. It is useful for spawning multiple related rigid bodies (e.g., multi-part props, object sets) and performing batch operations such as resetting, applying transforms, or querying group-level observations.

## Configuration

Configured via the {class}`~cfg.RigidObjectGroupCfg` class.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `uid` | `str` | `None` | Unique identifier for the group. |
| `rigid_objects` | `Dict[str, RigidObjectCfg]` | `MISSING` | Mapping from member uid to its `RigidObjectCfg`. Each entry spawns one rigid object. When `folder_path` is provided, this acts as a template. |
| `body_type` | `Literal["dynamic","kinematic"]` | `"dynamic"` | Default body type applied to group members (can be overridden per-member). |
| `folder_path` | `str | None` | `None` | Optional folder to initialize many objects from mesh files; if specified, `rigid_objects` should contain a template `RigidObjectCfg`. |
| `max_num` | `int` | `1` | When `folder_path` is used, number of objects to sample/create from the folder. |
| `ext` | `str` | `".obj"` | File extension filter when loading assets from `folder_path`. |
| `init_pos` / `init_rot` | `Sequence` (optional) | group-level transform | Optional transform to apply as a base offset to all members. |

Refer to {class}`~cfg.RigidObjectCfg` and {class}`~cfg.RigidBodyAttributesCfg` for per-member configuration options (mass, friction, restitution, collision options, shapes, etc.).

### Folder-based initialization

If `RigidObjectGroupCfg.folder_path` is specified, the group can be auto-populated from files under that folder. The `from_dict` implementation will:

- list files with extension `ext` under `folder_path`,
- select up to `max_num` files (wrapping if necessary),
- copy the template `RigidObjectCfg` provided in `rigid_objects` and set each member's `shape.fpath` to the selected file path and give each member a generated uid.

This makes it convenient to spawn many similar objects from an asset directory.

## Setup & Initialization

```python
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import (
    RigidObjectGroup, RigidObjectGroupCfg, RigidObjectCfg
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg

# 1. Initialize Simulation
device = "cuda" if torch.cuda.is_available() else "cpu"
sim_cfg = SimulationManagerCfg(sim_device=device)
sim = SimulationManager(sim_cfg)

# 2. Define shared physics attributes
physics_attrs = RigidBodyAttributesCfg(
    mass=1.0,
    dynamic_friction=0.5,
    static_friction=0.5,
    restitution=0.1,
)

# 3. Create group config with multiple members
group_cfg = RigidObjectGroupCfg(
    uid="obj_group",
    rigid_objects={
        "cube_1": RigidObjectCfg(uid="cube_1", shape=CubeCfg(size=[0.1,0.1,0.1]), attrs=physics_attrs, init_pos=[0.0,0.0,1.0]),
        "cube_2": RigidObjectCfg(uid="cube_2", shape=CubeCfg(size=[0.2,0.2,0.2]), attrs=physics_attrs, init_pos=[0.5,0.0,1.0]),
        "cube_3": RigidObjectCfg(uid="cube_3", shape=CubeCfg(size=[0.3,0.3,0.3]), attrs=physics_attrs, init_pos=[-0.5,0.0,1.0]),
    }
)

# 4. Spawn the rigid object group
obj_group: RigidObjectGroup = sim.add_rigid_object_group(cfg=group_cfg)

# 5. Run or step simulation
sim.update()
```

The example `scripts/tutorials/sim/create_rigid_object_group.py` demonstrates creating and running a scene with a `RigidObjectGroup`.

## Rigid Object Group — Common Methods & Observations

A group provides batch operations on multiple rigid objects. Key APIs include:

| Method / Property | Return / Args | Description |
| :--- | :--- | :--- |
| `num_objects` | `int` | Number of objects in each group instance. |
| `body_data` | `RigidBodyGroupData` | Data manager providing `pose`, `lin_vel`, `ang_vel` properties. |
| `body_state` | `(N, M, 13)` | Full body state of all members: [x, y, z, qw, qx, qy, qz, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]. |
| `get_local_pose(to_matrix=False)` | `(N, M, 7)` or `(N, M, 4, 4)` | Poses of all members across N envs; M = number of members. |
| `set_local_pose(pose, env_ids=None, obj_ids=None)` | `pose: (N, M, 7)` or `(N, M, 4, 4)` | Set poses for specific environments and/or objects; requires `sim.update()` to apply. |
| `get_user_ids()` | `(N, M)` | Get user IDs tensor for all members in the group. |
| `clear_dynamics(env_ids=None)` | - | Reset velocities and clear all forces/torques for the group. |
| `set_visual_material(mat, env_ids=None)` | `mat: VisualMaterial` | Change visual appearance for all members. |
| `set_visible(visible=True)` | `visible: bool` | Set visibility for all members in the group. |
| `set_physical_visible(visible=True, rgba=None)` | - | Set collision body visibility for debugging. |
| `reset(env_ids=None)` | - | Reset all members to their initial configured transforms. |

### Observation Shapes

- Group member poses (`body_data.pose`): `(N, M, 7)` where N is number of environments and M is number of objects in the group.
- Member linear velocities (`body_data.lin_vel`): `(N, M, 3)`.
- Member angular velocities (`body_data.ang_vel`): `(N, M, 3)`.
- Full body state (`body_state`): `(N, M, 13)` containing [position (3), orientation (4), linear velocity (3), angular velocity (3)].

Use these shapes when collecting vectorized observations for multi-environment training.

## Best Practices

- Groups are convenient for batch operations: resetting, setting visibility, and applying transforms to multiple objects together.
- Use `obj_ids` parameter in `set_local_pose()` to control specific objects within the group rather than all members.
- Prefer providing simplified collision meshes or enabling convex decomposition (`max_convex_hull_num` > 1) for complex visual meshes to improve physics stability.
- `RigidObjectGroup` only supports `dynamic` and `kinematic` body types (not `static`).
- When teleporting many members, batch pose updates and call `sim.update()` once to avoid synchronization overhead.
- For GPU physics, set `SimulationManagerCfg.sim_device` to `cuda` and call `sim.init_gpu_physics()` before running simulations.
- Use `clear_dynamics()` to reset velocities without changing poses.

## Example: Working with Group Poses

```python
import torch

# Get current poses of all members
poses = obj_group.get_local_pose()  # (N, M, 7) where N=num_envs, M=num_objects

# Move specific objects in the group
# Example: Move only the first object (cube_1) in all environments
new_pose = torch.tensor([[[0.0, 1.0, 0.5, 1, 0, 0, 0]]], device=sim.device)  # (1, 1, 7)
obj_group.set_local_pose(new_pose, env_ids=[0], obj_ids=[0])
sim.update()

# Access velocity data for all members
linear_vels = obj_group.body_data.lin_vel  # (N, M, 3)
angular_vels = obj_group.body_data.ang_vel  # (N, M, 3)

# Reset all objects to initial configuration
obj_group.reset()
sim.update()
```

## Integration with Sensors & Gizmos

Members in a group behave like normal `RigidObject`s: they can be observed by cameras, attached to contact sensors, and controlled via gizmos. You can operate on individual members or treat the group as a single unit depending on your scenario.

## Related Topics

- Rigid objects: See the Rigid Object overview for single-body.
- Soft bodies: Deformable objects have different observation semantics (vertex-level data). ([Soft Body Simulation Tutorial](https://dexforce.github.io/EmbodiChain/tutorial/create_softbody.html))
- Examples: `scripts/tutorials/sim/create_rigid_object_group.py` 


<!-- End of rigid object group overview -->