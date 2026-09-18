# Newton Runtime Qualification

```{currentmodule} embodichain.lab.sim
```

Newton runtime qualification is more specific than the backend name.
`physics: newton` selects the integration, but the resolved solver, device,
finalized scene contents, and requested operation determine what is actually
supported.

## Current solver/device matrix

This matrix describes the current EmbodiChain integration. An upstream solver
feature does not imply that every asset or operation exposes it here.

| Solver/device path | Rigid/articulation stepping | Deformables | `ContactSensor` | Qualification boundary |
| :--- | :--- | :--- | :--- | :--- |
| MuJoCo-Warp, CUDA | Supported | No | Geometry + impulse | Primary articulated-rigid path. |
| MuJoCo-Warp, CPU | Supported | No | Unsupported | CPU contact buffers are not exposed to the sensor. |
| XPBD, CUDA | Supported | Scene-dependent | Geometry-only where available | Validate the exact joints, assets, and task. |
| VBD, CUDA | Limited by scene | Volume/surface | Geometry-only where available | Deformable-focused path. |
| DexUni, CUDA | Coupled articulation/deformable stepping | Volume/surface | Unsupported | No rigid-rigid or rigid-ground solve/publication on the current path. |
| Semi-implicit, CUDA, gradients | Not stepped by `DifferentiableEnv` | Unsupported | Unsupported | Current integration is a kinematics bridge only. |

The matrix is an admission guide, not a substitute for a task-level rollout.
Joint types, source assets, contact geometry, deformable topology, and runtime
controls can narrow support further.

## Resolve before qualifying

AutoSolver is unresolved until the complete Spawn scene is finalized. Query the
concrete value only after {meth}`SimulationManager.prepare`:

```python
sim.prepare()
print(sim.physics.solver_type)
print(sim.device)
```

The startup summary also reports the resolved solver, device, substep count,
CUDA Graph state, and renderer. Record these values with test or benchmark
results. A configuration that merely decodes successfully has not yet passed
runtime qualification.

For contact-dependent code, qualify the query itself:

```python
capabilities = contact_sensor.contact_capabilities
if not capabilities.geometry:
    raise RuntimeError("This runtime does not expose contact geometry.")
```

Geometry and impulse availability are distinct. Zero-valued impulse fields on
a geometry-only path do not mean that no contact candidate exists. See
{doc}`../../../sensors/contact_sensor` for filtering, frames, batching, and
capacity behavior.

## Deformable admission checks

Volume soft bodies and surface cloth use `VolumeDeformableObjectCfg` and
`SurfaceDeformableObjectCfg`. The current integration requires:

- CUDA execution;
- `auto`, `dexuni`, `xpbd`, `semi_implicit`, or `vbd` as the configured solver;
- `requires_grad=False`;
- every deformable declaration before the first `prepare()`/scene finalization.

This check does not claim identical material, self-collision, articulation, or
contact behavior across the admitted solvers. Qualify the complete scene. For a
robot plus deformable, start with {doc}`dexuni` and account for its rigid-contact
boundary.

## Collision and contact qualification

The selected solver determines which collision path is active:

- MuJoCo-Warp can use its internal contacts or Newton-generated rigid contacts,
  controlled by its solver configuration.
- XPBD and VBD consume solver-specific subsets of Newton collision/contact
  data; verify geometry and impulse capabilities at runtime.
- DexUni owns particle-shape collision generation. An external
  `collision_cfg` does not add rigid contacts and should be `None` when DexUni
  is selected explicitly.

Do not tune external pipeline capacities or margins until the active solver is
known to consume that pipeline. When reducing collision refresh frequency,
measure penetration and contact continuity at the same physics/control timing.

## Differentiable route

The supported differentiable configuration is explicit:

```python
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

physics_cfg = NewtonPhysicsCfg(
    device="cuda:0",
    solver_cfg={"solver_type": "semi_implicit"},
    requires_grad=True,
    use_cuda_graph=False,
)
```

This enables EmbodiChain's current differentiable **kinematics bridge**. It
does not step the configured semi-implicit solver, run collision detection, or
make an ordinary Gym rollout and arbitrary state mutation differentiable. The
Franka reach APG example differentiates task-defined FK, action, and reward
kernels recorded on a Warp tape; it is not qualification evidence for
differentiable contact dynamics. See {doc}`/overview/rl/algorithm`.

## CUDA Graph and rendering status

For ordinary CUDA simulation, `use_cuda_graph=True` is a request rather than
proof that capture completed. Distinguish pending, captured, and disabled
states in the startup summary. Model changes and particle contact-material
schedules can invalidate or disable replay.

`sync_to_renderer=None` preserves DexSim's consumer-aware policy. Camera
rendering and the post-reset observation path synchronize on demand, so camera
qualification does not require forcing every-step publication. Test the first
frame after `prepare()` and immediately after reset for camera-dependent tasks.

## Minimum task-level check

For the exact solver/device pair, exercise construction, `prepare()`, one
physics step, reset, observation/state readback, and teardown. For training,
run across at least one episode boundary and check that observations, targets,
velocities, forces, and statistics restart consistently. Compare backends by
task success, penetration, settling, joint tracking, and contact continuity—not
by expecting identical trajectories.

Use the [Newton Solver Guide](https://newton-physics.github.io/newton/stable/solvers/index.html)
for native feature and joint-support tables, and the
[Solver Tuning Reference](https://newton-physics.github.io/newton/stable/concepts/simulation_tuning_solvers.html)
for upstream tuning guidance. This page remains limited to EmbodiChain's
integration qualification.
