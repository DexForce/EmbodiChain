# Physics Backend Migration

Use this checklist when moving an EmbodiChain scene between the Default and
Newton physics backends. The shared object APIs reduce code changes, but they
do not make solver behavior, supported assets, contacts, or tuning identical.

## Select the backend at its owner

For Python construction, the type of `physics_cfg` selects the backend:

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg

default_cfg = SimulationManagerCfg(
    physics_cfg=DefaultPhysicsCfg(device="cuda:0", physics_dt=0.01)
)
newton_cfg = SimulationManagerCfg(
    physics_cfg=NewtonPhysicsCfg(
        device="cuda:0",
        physics_dt=0.01,
        num_substeps=10,
    )
)
```

For a runnable Gym YAML, `physics` and its optional `physics_config` are owned
by the inline environment or selected environment component:

```yaml
physics: newton
device: cuda:0
physics_config:
  physics_dt: 0.01
  num_substeps: 10
  gravity: [0.0, 0.0, -9.81]
```

A deployment cannot repeat or override fields owned by its environment
component. The launcher `--physics` option only confirms the file-owned value;
it does not switch backends. Keep a separate environment configuration for
each backend.

## Migrate the lifecycle

Declare the complete initial scene before calling
{meth}`~embodichain.lab.sim.SimulationManager.prepare`. Newton's automatic
solver selection inspects the finalized scene, so adding an articulation or
deformable later can change the required solver.

```python
sim.add_robot(robot_cfg)
sim.add_rigid_object(object_cfg)
sim.prepare()

while running:
    sim.update(physics_dt=0.01, step=1)
```

`init_gpu_physics()` and `finalize_newton_physics()` remain compatibility
aliases for `prepare()`. Physics is always caller-driven. The former
`set_manual_update()` switch has been removed; use `update()` to advance time.

Newton leaves `sync_to_renderer=None` by default so DexSim can synchronize an
initialized native window without forcing the cost on state-only headless
training. Set it to `True` when other code reads DexSim-side render transforms
after every step, or `False` to opt out explicitly. Camera rendering and the
post-reset observation path perform an on-demand synchronization regardless of
that per-step policy.

## Check capability differences

| Operation | Default | Newton migration action |
| :--- | :--- | :--- |
| Rigid objects, groups, articulations, robots | Supported | Supported; revalidate drives, contacts, and reset behavior with the chosen solver. |
| Volume/surface deformables | Unsupported | Use `VolumeDeformableObjectCfg` or `SurfaceDeformableObjectCfg` on CUDA with a compatible solver. |
| Manager-created rigid constraints | Supported | Unsupported; do not assume solver-internal articulation constraints replace this API. |
| Cameras | Supported | Supported; call `prepare()` and synchronize render state before reading immediately after a reset. |
| `ContactSensor` | Supported | Unsupported for MuJoCo-Warp on CPU and DexUni; other paths can be geometry-only. |
| Differentiable route | Unsupported | Explicit `semi_implicit`, CUDA Graph disabled, and only the documented kinematics bridge. |

Contact capability is checked again after `prepare()` because an automatic
solver is not concrete before scene finalization. Treat a successful static
configuration decode as necessary but not sufficient runtime qualification.

## Translate object configuration

Keep portable mass, rigid-body, collision, and material values in the shared
`RigidBodyPhysicsCfg` groups. Backend-native values use the concrete subtype in
that same slot; do not maintain parallel `default_props` and `newton_props`
blocks.

Use the current topology-specific deformable classes:

- `VolumeDeformableObjectCfg` for tetrahedral/volume soft bodies.
- `SurfaceDeformableObjectCfg` for triangle/surface cloth.

EmbodiChain public poses and COM quaternions use `xyzw` ordering. Do not pass a
native backend's `wxyz` ordering into public configuration or setters.

## Retune time and contacts

Preserve simulated duration and control frequency during the first comparison:

- physics step: `physics_dt`;
- Gym control period: `physics_dt * sim_steps_per_control`;
- Newton solver interval: `physics_dt / num_substeps`.

Then retune solver-specific iterations, substeps, friction, restitution,
contact margins, and drive gains. Compare penetration, settling, joint error,
contact continuity, and task success rather than expecting trajectory equality.
For contact data, inspect `sensor.contact_capabilities`; zero impulse fields can
mean a geometry-only query rather than no contact.

## Validate the migrated task

At minimum, exercise construction, one step, reset, observation readback, and
teardown for the exact solver/device pair. For camera tasks, compare a frame
before stepping and the first frame immediately after reset. For training,
also run a short rollout that crosses at least one episode boundary and check
that observations, targets, velocities, forces, and episode statistics restart
consistently.

Record the resolved solver, device, substeps, CUDA Graph status, renderer, and
dependency versions with performance results. This checkout declares
`dexsim_engine==0.5.0`; the locally qualified stack when this guide was updated
used Newton `1.4.0` and Warp `1.15.0`. Patch-level DexSim builds can include
additional fixes, so retain their full build identifier in bug reports.

See {doc}`default_physics` and {doc}`newton_physics` for backend-specific
configuration and solver details.
