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
Robot IK controls are discovered automatically from configured control parts
and their solver's root link, end link, and TCP.
See {doc}`native window controls <window>` for native entity configuration.

## Supported Targets

| Target | Gizmo behavior |
|---|---|
| Robot | Moves the selected control part through native DexSim IK by default, or its configured EmbodiChain solver. |
| Rigid object | Sets the object's local arena pose. |
| Camera | Sets the camera's local pose. |

Registered Viser controls and robot IK controls currently require `num_envs=1`.
Automatic discovery requires robot solver metadata for each supported control
part. IK computation still defaults to DexSim. Robots without this metadata
can use explicit `GizmoCfg` chain settings instead; no end link or TCP is guessed.

## Quick Start

Run the robot Gizmo tutorial in the native window:

```bash
python scripts/tutorials/sim/gizmo_robot.py
```

Use the same target through Viser:

```bash
python scripts/tutorials/sim/gizmo_robot.py --viser
```

The tutorial opts into immediate native activation with
`robot_ik_gizmo=GizmoCfg(ik_start_enabled=True)`. It sets the initial robot pose
before opening the window, then its ordinary manual-physics loop creates and
updates the controller:

```python
sim.open_window()  # Safely skipped when Viser is configured.
while True:
    sim.update(step=1)
```

By default, the first **I** press creates eligible native robot IK controllers.
With `ik_start_enabled=True`, the first update with an open window activates
them from the current robot pose; **I** then toggles their visibility. Opening a window alone
does not construct an IK solver or change existing drive targets. Viser displays
the TCP handles and constructs the solver on the first drag. Both paths update
through `SimulationManager.update()`.

The default `SimulationManagerCfg.robot_ik_gizmo` is `GizmoCfg()`. To use each
control part's configured solver (including Pink), or disable automatic setup:

```python
SimulationManagerCfg(robot_ik_gizmo=GizmoCfg(ik_solver="embodichain"))
SimulationManagerCfg(robot_ik_gizmo=None)
```

Gym JSON/YAML deployments accept the same `robot_ik_gizmo` mapping or `null`.
`sim.enable_gizmo(uid, control_part, gizmo_cfg)` provides an explicit per-part
override. `sim.disable_gizmo(uid, control_part)` prevents automatic recreation,
including across window reopen; omit the part to disable all parts of a robot.
Use `set_gizmo_visibility()` and `toggle_gizmo_visibility()` for visibility.
Activated native controls retain their IK and visibility state across window
close/reopen. Removing a robot or destroying the manager releases their input
handlers and native target nodes.

## Frontend Behavior

- Native entity interaction starts with the first native window; pure headless
  and Viser runs do not automatically create native controllers. Native robot
  targets activate on **I** without a script-level controller call.
- Viser exports browser-native transform controls when
  `VisualizationCfg.allow_commands=True`; otherwise it shows read-only frames.
- The standard `--viser` launcher enables registered commands for trusted
  clients. EmbodiChain does not add authentication to the Viser endpoint.
- A Viser Gizmo is owned by one browser client from drag start until drag end
  or disconnect. Other clients continue receiving authoritative poses.

Advanced callers may still use `create_robot_ik_gizmo_controller()` directly
and retain/update its returned objects. Automatic setup detects an existing
factory-created controller and does not create or update a duplicate.

For the complete robot setup and IK walkthrough, continue with the
{doc}`Gizmo tutorial </tutorial/gizmo>`. See
{doc}`Viser browser visualization </overview/sim/viser_visualization>` for
server configuration and remote-access guidance.
