# Physics

```{currentmodule} embodichain.lab.sim
```

## Backend selection

EmbodiChain exposes two physics backends, selected by the configuration type:

| Backend | Python configuration | Execution and solver model | Start here |
| :--- | :--- | :--- | :--- |
| `default` | `DefaultPhysicsCfg` | CPU or Direct GPU execution; TGS is the established constraint-solver default. | {doc}`default` |
| `newton` | `NewtonPhysicsCfg` | Newton through DexSim; scene-aware automatic selection or an explicit solver. | {doc}`newton` |

Both are integrated through DexSim's runtime and Spawn SDK. `default` and
`newton` are the only public backend identifiers. The renderer is selected
separately by `render_cfg`; native-window and browser visualization settings
control how the scene is displayed. Selecting Newton does not select a different
camera renderer. `headless=True` closes the native window while permitting
configured offscreen camera rendering.

## Supported capabilities

This table describes the current EmbodiChain integration. A feature in an
upstream solver does not imply that every asset or operation exposes it here.

| Capability | Default | Newton |
| :--- | :--- | :--- |
| Rigid objects, rigid-object groups, articulations, robots | Supported | Supported; joint features depend on the solver. |
| Volume and surface deformables | Unsupported | Supported on CUDA with a particle-capable solver; see {doc}`newton`. |
| Native rigid constraints through the manager | Supported | Unsupported; this is distinct from solver-internal articulation constraints. |
| Camera and stereo camera | Supported | Supported through the shared rendering integration. |
| `ContactSensor` | Supported on CPU and Direct GPU | Supported; geometry and impulse availability depend on solver/device. See {doc}`../../sensors/contact_sensor`. |
| Differentiable simulation | Unsupported | Explicit `semi_implicit` solver only; see {doc}`newton`. |

Use Default for workflows needing its native rigid constraints or established
TGS behavior. Use Newton for particle-based cloth/soft bodies, solver-specific
experiments, or the supported differentiable path. For rigid robot tasks that
can run on either backend, validate the target task with each configuration;
shared APIs do not guarantee identical trajectories or contact responses.

Newton runtime qualification is more specific than the backend name:

| Solver/device path | Rigid/articulation stepping | Deformables | `ContactSensor` | Status boundary |
| :--- | :--- | :--- | :--- | :--- |
| MuJoCo-Warp, CUDA | Supported | No | Geometry + impulse | Primary articulated-rigid path. |
| MuJoCo-Warp, CPU | Supported | No | Unsupported | CPU contact buffers are not exposed to the sensor. |
| XPBD, CUDA | Supported | Scene-dependent | Geometry-only where available | Validate the exact task. |
| VBD, CUDA | Limited by scene | Volume/surface | Geometry-only where available | Deformable-focused path. |
| DexUni, CUDA | Coupled articulation/deformable stepping | Volume/surface | Unsupported | No rigid-rigid or rigid-ground contact publication/current solve path. |
| Semi-implicit, CUDA, gradients | Not stepped by `DifferentiableEnv` | Unsupported | Unsupported | Kinematics bridge only. |

AutoSolver chooses from finalized scene contents, so query
`sim.physics.solver_type` and runtime capabilities after `prepare()` rather
than treating `physics: newton` as a complete support claim.

## Common configuration and devices

All backends inherit these parameters from {class}`~cfg.PhysicsBackendCfg`:

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `physics_dt` | `0.01` | Duration of one EmbodiChain physics step, in seconds. |
| `device` | `"cpu"` for Default; `"cuda:0"` for Newton | Compute device for physics and environment tensors. Solver/device restrictions still apply. |
| `gravity` | `[0.0, 0.0, -9.81]` | World-frame acceleration in m/s². |

`physics_cfg.device` owns the backend default. An explicit
`SimulationManagerCfg(device=...)` overrides it; the legacy `sim_device`
argument is an alias. `gpu_id` selects the rendering GPU and supplies the index
for an unindexed `"cuda"` device. Keep explicit compute and render GPU indices
consistent with the intended deployment. Omitting Gym's `--device` preserves
the configured value or backend default; an explicit `--device cpu` also applies
to Newton, subject to the selected solver and asset restrictions.

Python chooses a backend through `physics_cfg`. Gym files declare `physics`
and a matching `physics_config`; an environment component owns both fields and
its deployment cannot override either. `--physics` can confirm the file's
backend but cannot switch it. See {doc}`/guides/configuration` for paired
configuration examples and {doc}`migration` for migration checks.

## Physics, control, and solver time

| Interval | Definition | Example |
| :--- | :--- | :--- |
| Physics step | `physics_dt` | `0.01 s` (100 Hz) |
| Gym control step | `physics_dt * sim_steps_per_control` | `0.01 * 4 = 0.04 s` (25 Hz) |
| Newton solver substep | `physics_dt / num_substeps` | `0.01 / 10 = 0.001 s` (1 kHz) |

In this example, one control action spans four physics steps and forty Newton
solver substeps. Increasing Newton's `num_substeps` refines integration without
changing the control period. Reducing `physics_dt` also changes the control
period unless `sim_steps_per_control` is adjusted. Rendering and visualization
publication have their own cadence and do not define the control frequency.

```{toctree}
:maxdepth: 1

default
newton
migration
```
