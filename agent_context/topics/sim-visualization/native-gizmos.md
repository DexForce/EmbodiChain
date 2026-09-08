# Native entity and robot gizmos

Read this when the request needs these details. [Topic overview](sim-visualization.md).

### Gizmo ownership

Native entity manipulation belongs to DexSim 0.5.0. The first successful native
window open enables its world-owned `EntityGizmoManipulator` by default, after
the scene and default plane are ready. The manager registers the default plane
as a static external target. `SimulationManagerCfg.enable_entity_gizmo=False`
opts out; Gym deployments accept the same top-level JSON/YAML field through
`gym.utils.gym_utils.config_to_cfg()`.

`sim.enable_entity_gizmo(config)` explicitly enables/configures the controller;
`sim.disable_entity_gizmo()` also cancels pending automatic enablement before
the first window. These explicit calls take precedence over the startup default.
Query through `sim.get_world().get_entity_gizmo()`. DexSim owns window
detach/reopen and controller state; reopening never reapplies the default or
overwrites an explicit native disable. Pure headless and Viser runs do not
automatically create a native entity controller.

`SimulationManagerCfg.robot_ik_gizmo` defaults to `GizmoCfg()`. During normal
updates the manager registers robot control parts with complete solver chain/TCP
metadata in single-environment interactive runs. Pure headless, read-only Viser, and
multi-environment runs do not register automatic controls. The first native I
press creates DexSim's `IKGizmoController` by default;
`GizmoCfg(ik_start_enabled=True)` opts into activation on the first update with
an open window. The startup attempt is consumed once, including on failure;
later key presses can retry. Viser constructs IK on its first drag.
Registration never writes drive targets. `Gizmo` owns managed native input and
target-node cleanup, detaches input on window close, and reattaches the same
controller on reopen. Robot removal releases all its managed controls.

Set `robot_ik_gizmo=None` to opt out or supply `GizmoCfg` overrides; Gym
JSON/YAML accepts the same mapping/null. `enable_gizmo()` can override one part,
and `disable_gizmo()` prevents automatic recreation (all parts when omitted).
The explicit `create_robot_ik_gizmo_controller()` factory still returns
caller-owned controllers; a weak registry prevents automatic duplicates.
Both robot paths
default to native Newton IK; `GizmoCfg(ik_solver="embodichain")` adapts the
control part's existing solver, such as PinkSolver. Both support one environment
and write only selected non-mimic joint drive targets through `Robot`.

`SimulationManagerCfg` owns window size, headless mode, rendering, GPU/CPU
selection, arena count and spacing, physics timestep, physics and GPU-memory
settings, recording, profiling, and browser visualization.

`EnvCfg` embeds `SimulationManagerCfg` and supplies the control-to-physics
step ratio. CLI and task config loaders may override runtime fields before
constructing the environment. Trace those overrides through the caller rather
than changing a default in the manager blindly.

Object-specific configuration belongs in `lab/sim/cfg.py` or the
corresponding robot/sensor module. Scene composition belongs in
`EmbodiedEnv` or a task config, not in `SimulationManagerCfg`.

For mesh collision decomposition, `MeshCfg.acd_method` defaults to `"visacd"`
with DexSim 0.5.0; it requires CUDA support. `"coacd"` and `"vhacd"` remain
supported explicit options.
