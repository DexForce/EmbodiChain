# Rigid Object Group

```{currentmodule} embodichain.lab.sim
```

Use {class}`~objects.RigidObjectGroup` to declare several independent rigid
bodies under one scene UID. Each environment gets the same ordered group of
members. A group exposes both an environment axis and a member axis. For a
single body with several collider shapes, use `CompoundShapeCfg` with
{doc}`sim_rigid_object`.

## Configuration

Supply a {class}`~cfg.RigidObjectGroupCfg` to
`sim.add_rigid_object_group(cfg)`.

| Field | Meaning |
| :--- | :--- |
| `uid` | Unique scene identifier for the whole group. |
| `rigid_objects` | Ordered mapping of member names to `RigidObjectCfg`. Each member supplies its own shape, placement, and physical properties. |
| `body_type` | One actor type for every member. Use `"dynamic"` or `"kinematic"`. |
| `folder_path`, `max_num`, `ext` | Folder-backed member expansion via `RigidObjectGroupCfg.from_dict()`. |

The manager creates internal UIDs for group members. Use the order of
`rigid_objects` to address those members with `obj_ids`. Declare a separate
`RigidObject` when a body needs its own scene lookup or a different actor type.

### Folder-based initialization

`RigidObjectGroupCfg.from_dict()` can expand a directory of mesh files. Give
it a valid `folder_path`, an extension such as `".obj"`, a positive
`max_num`, and one entry in `rigid_objects` as the configuration template.
The loader requires at least one matching file and cycles the available files
if more members are requested. An explicit ordered mapping is the right
choice when member order must be reproducible.

## Setup & Initialization

This headless CPU example creates two bodies in each of two environments.
The pose write targets member zero in environment zero.

```python
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg, RigidObjectCfg, RigidObjectGroupCfg
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
    group = sim.add_rigid_object_group(
        RigidObjectGroupCfg(
            uid="blocks",
            rigid_objects={
                "small": RigidObjectCfg(
                    shape=CubeCfg(size=[0.08] * 3), init_pos=(-0.2, 0.0, 0.2)
                ),
                "large": RigidObjectCfg(
                    shape=CubeCfg(size=[0.12] * 3), init_pos=(0.2, 0.0, 0.2)
                ),
            },
        )
    )
    sim.prepare()
    poses = group.get_local_pose().clone()  # (2, 2, 7)
    # Shape (1, 1, 7) for one environment and one member, with xyzw rotation.
    target = torch.tensor([[[0.0, 0.0, 0.5, 0, 0, 0, 1]]], device=sim.device)
    group.set_local_pose(target, env_ids=[0], obj_ids=[0])
    sim.update(step=1, render_final_step=False)
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

## Rigid Object Group — Common Methods & Attributes

### Observation Shapes

`E = group.num_instances` and `M = group.num_objects`. Whole-group reads
preserve both dimensions.

| Read | Shape | Meaning |
| :--- | :--- | :--- |
| `get_local_pose()` | `(E, M, 7)` | Arena-local poses in `xyz + xyzw` order. `to_matrix=True` gives `(E, M, 4, 4)`. |
| `body_data.lin_vel`, `body_data.ang_vel` | `(E, M, 3)` | Velocity vectors per body. |
| `body_state` | `(E, M, 13)` | Pose, linear velocity, and angular velocity per body. |
| `get_user_ids()` | `(E, M)` | Engine user IDs per member. |

Clone arrays you plan to retain across subsequent state reads.

## Example: Working with Group Poses

For selected pose writes, the leading axes must match the number of requested
environment and member indices. The values are interpreted in the order
supplied. For example, the following moves the second member of environment
one up by 10 cm in a prepared two-environment scene.

```python
target = group.get_local_pose()[1:2, 1:2].clone()
target[..., 2] += 0.1
group.set_local_pose(target, env_ids=[1], obj_ids=[1])
```

Use `reset(env_ids=[1])` to restore the initial states of all members in
environment one. `clear_dynamics(env_ids=[1])` clears velocities and wrenches
for its dynamic members. `set_visual_material(mat, env_ids=[1])` assigns one
material instance to all members in that environment.

## Related Topics

- {doc}`sim_rigid_object` for per-member shape and physics options.
- {doc}`sensors/contact_sensor` and {doc}`sensors/backend_support` for
  contact-target selection and backend capabilities.
- [Rigid object group tutorial](https://github.com/DexForce/EmbodiChain/blob/main/scripts/tutorials/sim/create_rigid_object_group.py)
  for a rendered example.
