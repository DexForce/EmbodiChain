# Cable Object

```{currentmodule} embodichain.lab.sim
```

Use {class}`~objects.CableObject` for a rope, wire, or hose represented by
oriented capsule segments. Segment orientation carries material twist, and
each segment has both linear and angular velocity.

Cables require Newton with DexUni on CUDA and gradients disabled. Declare
the physical scene before `sim.prepare()`. The current cable reset path
requires a separate scene from fluid, cloth, or soft-body particles. See
{doc}`sim_manager/physics/newton/runtime_qualification` for solver support.

## Configuration

Pass {class}`~cfg.CableObjectCfg` to `sim.add_cable_object(cfg)`.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `uid` | Required | Unique manager-scoped asset identifier. |
| `points` | Required | Ordered centreline endpoints `(N, 3)` in local metres. An open cable has `N - 1` segments. |
| `radius` | Required | Positive segment radius in metres. |
| `segment_quaternions` | `None` | Optional material frames in `xyzw` order. Local +Z follows each segment. Omission uses parallel-transport frames. |
| `closed` | `False` | Request a closed cable through the native descriptor. |
| `fixed_segments` | `()` | Segment indices held fixed at their authored placement. |
| `collision_filter_span` | `1` | Native filtering span for neighboring cable segments. |
| `attrs` | `CablePhysicsCfg()` | Density, stretch, shear, bend, twist, damping, and contact properties. |
| `start_attachment`, `end_attachment` | `None` | Permanent endpoint attachments through `CableAttachmentCfg`. |
| `render_mode` | `"tube"` | Generated appearance, either `"tube"` or `"capsules"`. |
| `visual_material`, `visual_shape` | `None` | Generated-surface material or a custom visual mesh. |

`CablePhysicsCfg` uses stiffness and damping coefficients per connection.
Linear stiffness uses N/m and angular stiffness uses N m/rad. An omitted
coefficient inherits the solver value. Contact settings use
`attrs.collision_props` and `attrs.material_props`. Material changes and
physical topology changes require constructing a new simulation.

## Setup and initialization

This headless example creates a three-segment cable in two environments.
It reads the prepared segment state and applies a force to the final segment
in environment zero for one outer update.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import CableObjectCfg, NewtonPhysicsCfg, RenderCfg

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        num_envs=2,
        physics_cfg=NewtonPhysicsCfg(
            device="cuda:0",
            physics_dt=1 / 120,
            num_substeps=2,
            solver_cfg={"solver_type": "dexuni"},
        ),
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    cable = sim.add_cable_object(
        CableObjectCfg(
            uid="wire",
            points=[(0, 0, 0), (0, 0, 0.1), (0, 0, 0.2), (0, 0, 0.3)],
            radius=0.01,
            init_pos=(0.0, 0.0, 0.5),
        )
    )
    sim.prepare()
    state = cable.data.segment_state_w  # (2, 3, 13)
    cable.set_force([0.0, 0.1, 0.0], segment_index=-1, env_ids=[0])
    sim.update(step=1, render_final_step=False)
    sim.reset_objects_state(env_ids=[0])
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

## Segment state and loads

After preparation, `cable.data` returns independent snapshots. World
positions already include each environment's origin. Let `E` be the batch
instance count and `S = cable.data.num_segments`.

| Property | Shape | Meaning |
| :--- | :--- | :--- |
| `data.segment_pose_w` | `(E, S, 7)` | Segment COM position and `xyzw` orientation. |
| `data.segment_pos_w`, `data.segment_quat_w` | `(E, S, 3)`, `(E, S, 4)` | Position and material frame. |
| `data.segment_vel_w` | `(E, S, 6)` | World linear velocity followed by angular velocity. |
| `data.segment_state_w` | `(E, S, 13)` | Position, quaternion, linear velocity, and angular velocity in one snapshot. |
| `data.default_segment_state_w` | `(E, S, 13)` | Immutable creation state. |

`set_force()` applies world-frame COM force in newtons, and `set_torque()`
applies world-frame torque in N m. Both accept a three-vector or one vector
per selected environment. Negative segment indices count from the tip.
Loads clear automatically after the next outer update. Call
`clear_wrenches(env_ids=...)` to discard pending loads before that update.

## State writes and reset

To initialize a perturbed cable after a scene reset, edit the selected
segment-state snapshot and call
`write_segment_state_to_sim(state, env_ids=...)`. Input must be a finite,
contiguous float32 tensor `(selected E, S, 13)` on the cable device with
unit quaternions. Keep permanent attachment anchors consistent.

Segment writes clear solver history in the affected owning worlds,
including history belonging to other cables. Controls and pending loads
are preserved. Contact measurements become current after another physics
update.

`cable.reset(env_ids=...)` restores that cable's creation state and clears
its loads. For an episode reset including attached robots, other physical
assets, targets, and supported controls, call
`sim.reset_objects_state(env_ids=...)`. This path resets all physical
assets in the selected environments together. Runtime material changes are
outside its reset contract.

## Attachments and imported cables

`CableAttachmentCfg` names a target `uid`, a body-local `position`, and
an optional `link_name` for an articulation. The cable endpoint must
coincide with the target anchor at preparation. These attachments are
permanent welds authored before preparation. The current engine rejects
attachments to articulations with nonzero initial joint poses.

`sim.add_usd(UsdImportCfg(...))` also returns cable objects under their
source resource keys. Imported and procedural cables expose the same
state and reset methods. Keep simulation resources alive until manager
teardown, which releases the owning scene.

## Related topics

- {doc}`sim_assets` for asset selection and UID lookup.
- {doc}`sim_cloth` and {doc}`sim_soft_object` for node-based deformables.
- {doc}`visualization/viser` for cable segment visualization.
