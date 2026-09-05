# Interactive Gizmos

```{currentmodule} embodichain.lab.sim
```

A Gizmo is a transform control for manipulating a simulation target. Native
windows use DexSim's entity and robot IK controllers. Viser uses registered
controls whose callbacks enqueue requested poses for `SimulationManager` to
apply on the simulation thread.

Native entity interaction is enabled by default when the first native window
opens. Select an object and press **G** to attach its gizmo. Set
`SimulationManagerCfg(enable_entity_gizmo=False)` or call
`sim.disable_entity_gizmo()` to opt out; reopening preserves that choice.
Robot IK controls require explicit creation for a selected control part.
See {doc}`native window controls <window>` for native entity configuration.

## Supported Targets

| Target | Gizmo behavior |
|---|---|
| Robot | Moves the selected control part through native DexSim IK by default, or its configured EmbodiChain solver. |
| Rigid object | Sets the object's local arena pose. |
| Camera | Sets the camera's local pose. |

Registered Viser controls and robot IK controls currently require `num_envs=1`.
Robot targets require a valid `control_part`; an EmbodiChain solver configuration
is needed only when selecting `GizmoCfg(ik_solver="embodichain")`.

## Quick Start

Run the robot Gizmo tutorial in the native window:

```bash
python scripts/tutorials/sim/gizmo_robot.py
```

Use the same target through Viser:

```bash
python scripts/tutorials/sim/gizmo_robot.py --viser
```

For Viser, register controls through the manager:

```python
sim.enable_gizmo(
    uid="robot",
    control_part="arm",
)

if not sim.has_gizmo("robot", control_part="arm"):
    raise RuntimeError("Gizmo setup failed")
```

`SimulationManager.update()` drains pending Gizmo commands during a normal
manual-physics loop. An automatic-physics loop that does not call `update()`
must continue calling:

```python
sim.update_gizmos()
sim.capture_visualization_safely()  # Publish the authoritative pose to Viser.
```

Use `disable_gizmo()`, `set_gizmo_visibility()`, and
`toggle_gizmo_visibility()` for lifecycle and visibility changes.

## Frontend Behavior

- Native entity interaction starts with the first native window; pure headless
  and Viser runs do not automatically create native controllers. Native robot
  targets use `create_robot_ik_gizmo_controller()` explicitly.
- Viser exports browser-native transform controls when
  `VisualizationCfg.allow_commands=True`; otherwise it shows read-only frames.
- The standard `--viser` launcher enables registered commands for trusted
  clients. EmbodiChain does not add authentication to the Viser endpoint.
- A Viser Gizmo is owned by one browser client from drag start until drag end
  or disconnect. Other clients continue receiving authoritative poses.

For the complete robot setup and IK walkthrough, continue with the
{doc}`Gizmo tutorial </tutorial/gizmo>`. See
{doc}`Viser browser visualization </overview/sim/viser_visualization>` for
server configuration and remote-access guidance.
