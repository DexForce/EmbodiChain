# Interactive Gizmos

```{currentmodule} embodichain.lab.sim
```

A Gizmo is a registered transform control for manipulating a simulation target
from either the native DexSim window or a trusted Viser browser. The simulation
owns the authoritative state: UI callbacks enqueue requested poses, and
`SimulationManager` applies them on the simulation thread.

## Supported Targets

| Target | Gizmo behavior |
|---|---|
| Robot | Moves the selected control part through its configured FK/IK solver. |
| Rigid object | Sets the object's local arena pose. |
| Camera | Sets the camera's local pose. |

Gizmo interaction currently requires `num_envs=1`. Robot targets also require
a valid `control_part` and solver configuration.

## Quick Start

Run the robot Gizmo tutorial in the native window:

```bash
python scripts/tutorials/sim/gizmo_robot.py
```

Use the same target through Viser:

```bash
python scripts/tutorials/sim/gizmo_robot.py --viser
```

Register controls through the manager rather than constructing or destroying
Gizmo instances directly:

```python
sim.enable_gizmo(
    uid="robot",
    control_part="arm",
)

if not sim.has_gizmo("robot", control_part="arm"):
    raise RuntimeError("Gizmo setup failed")
```

`SimulationManager.update()` drains pending Gizmo commands, steps physics,
and publishes the resulting pose to Viser. Interactive applications call it
explicitly in their main loop:

```python
while True:
    sim.update(step=1)
```

The caller can pace this loop against wall time using the configured physics
timestep. For editing while physics is paused, call `update_gizmos()` followed
by `capture_visualization_safely()` without stepping the world.

Use `disable_gizmo()`, `set_gizmo_visibility()`, and
`toggle_gizmo_visibility()` for lifecycle and visibility changes.

## Frontend Behavior

- The native window uses the DexSim Gizmo controller and requires an open
  window.
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
