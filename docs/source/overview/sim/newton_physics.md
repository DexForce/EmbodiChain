# Newton Physics Backend

```{currentmodule} embodichain.lab.sim
```

Use {class}`~cfg.NewtonPhysicsCfg` to enable the Newton backend. Its
`solver_cfg` defaults to `None` intentionally: EmbodiChain leaves the
`solver_cfg` argument unset when it creates DexSim's `NewtonCfg`, preserving
DexSim's `AutoSolverCfg` default.

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

sim_config = SimulationManagerCfg(
    physics_cfg=NewtonPhysicsCfg(
        device="cuda:0",
        physics_dt=0.01,
        num_substeps=10,
    )
)
```

AutoSolver is resolved when DexSim finalizes the complete Spawn scene during
{meth}`SimulationManager.prepare`. Add all initial robots and objects before
calling `prepare()` so the selection sees the complete scene. An explicit
`{"solver_type": "auto"}` or `{"class_type": "AutoSolverCfg"}` mapping has
the same effect as leaving `solver_cfg` unset.

:::{important}
This integration requires a DexSim build that exports `AutoSolverCfg`.
EmbodiChain does not fall back to a hard-coded concrete solver when that API is
unavailable.
:::

## Configuration reference

Scene parameters belong to `NewtonPhysicsCfg`; per-object collision, material,
and joint properties belong to asset configurations. Default-backend fields
such as `gpu_memory`, `bounce_threshold`, and `enable_ccd` do not belong here.

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `physics_dt` | `0.01` | One EmbodiChain physics step in seconds. |
| `device` | `"cuda:0"` | Compute device; explicit CPU execution remains subject to solver/asset support. |
| `gravity` | `[0.0, 0.0, -9.81]` | World-frame acceleration in m/s². |
| `num_substeps` | `10` | Solver substeps per physics step; solver interval is `physics_dt / num_substeps`. |
| `solver_cfg` | `None` | Preserve DexSim AutoSolver; a mapping or native solver config fixes an explicit choice. |
| `collision_cfg` | `NewtonCollisionPipelineCfg()` | External Newton collision pipeline; consumption depends on the selected solver. `None` disables this external pipeline. |
| `requires_grad` | `False` | Differentiable model; requires explicit `semi_implicit` and disables CUDA Graph capture. |
| `use_cuda_graph` | `True` | Request graph capture when supported; unavailable on CPU and disabled in gradient mode. |
| `debug_mode` | `False` | Additional runtime diagnostics. |
| `suppress_warp_kernel_logs` | `True` | Suppress Warp startup/kernel compilation messages; warnings and errors remain visible. |
| `visualizer_enabled` | `False` | DexSim Newton diagnostic visualizer; separate from the ordinary camera renderer and Viser. |
| `broad_phase` | `None` | Deprecated shortcut; use `collision_cfg.broad_phase`. |

Use the minimal falling-cube loop in {doc}`default_physics` with this
`physics_cfg` to compare the same scene. Keep the asset, initial state, physics
duration, and environment count fixed; tune solver-specific behavior separately.

## Automatic solver selection

The following scene types are exposed by EmbodiChain. Independent rigid objects
and articulation links are classified separately.

| Finalized scene contents | Selected configuration | Solver type | Active collision path |
| :--- | :--- | :--- | :--- |
| Empty scene or independent rigid bodies only | `XPBDSolverCfg` | `xpbd` | Newton collision pipeline |
| Articulations, with or without independent rigid bodies | `MJWarpSolverCfg` | `mujoco_warp` | MuJoCo Warp collision pipeline |
| Cloth or soft bodies, optionally with rigid bodies | `VBDSolverCfg` | `vbd` | Newton collision pipeline; VBD may handle deformable self-contact |
| Cloth or soft bodies with articulations, optionally with rigid bodies | `MJVBDSolverCfg` | `mjvbd` | Newton particle-shape soft contacts; MuJoCo rigid collision is disabled |

The current MJVBD path does not generate rigid-rigid or rigid-ground contacts.
MuJoCo Warp still advances rigid bodies and articulations, while Newton's soft
contact kernels handle deformable particle-shape contacts.

### Additional upstream resolver rules

DexSim also recognizes the following particle families. EmbodiChain does not
currently expose public fluid or MPM asset APIs; these rows describe the
upstream resolver only.

| Finalized scene contents | Selected configuration | Solver type | Active collision path |
| :--- | :--- | :--- | :--- |
| Fluid particles, optionally with rigid SDF boundaries | `SPHSolverCfg` | `sph` | SPH one-way SDF boundary handling; rigid contacts are not consumed |
| MPM particles, optionally with rigid colliders | `ImplicitMPMSolverCfg` | `implicit_mpm` | Implicit-MPM collider projection; the rigid collision pipeline is not stepped |

AutoSolver rejects multiple particle families in one scene, fluid particles
combined with articulations, and MPM particles combined with articulations.

Selection is based on scene contents, not the configured device. DexSim reports
device incompatibility after resolution; cloth, soft-body, fluid, and MPM
solvers currently require CUDA. The selected type is also written to the
DexSim log, for example `Newton AutoSolver selected 'mujoco_warp'.`

## Explicit solver selection

Pass a concrete solver configuration when an algorithm or solver-specific
parameter must be fixed:

```python
sim_config = SimulationManagerCfg(
    physics_cfg=NewtonPhysicsCfg(
        device="cpu",
        solver_cfg={
            "solver_type": "xpbd",
            "iterations": 8,
        },
    )
)
```

EmbodiChain mapping configs recognize `auto`, `mujoco_warp` (or `mjwarp`),
`mjvbd`, `xpbd`, `semi_implicit`, `featherstone`, and `vbd`. A DexSim
`NewtonSolverCfg` object may also be assigned directly when another explicit
solver class is required. AutoSolver never selects `DFSPHSolverCfg`,
`FeatherstoneSolverCfg`, or `SemiImplicitSolverCfg`. In particular,
`requires_grad=True` requires an explicit `semi_implicit` configuration;
automatic selection is rejected for differentiable simulation.

## Deformable support

Volume soft bodies and surface cloth are supported through
`VolumeDeformableObjectCfg` and `SurfaceDeformableObjectCfg`. The integration
requires all of the following:

- CUDA execution.
- `auto`, `xpbd`, `semi_implicit`, `vbd`, or `mjvbd` as the configured solver.
- `requires_grad=False`.
- All deformable declarations before the first `prepare()`/scene finalization.

This list is the integration's admission check, not a claim that every solver
has identical material, self-collision, or joint behavior. Choose a solver for
the complete scene, including any articulated robot. In particular, the MJVBD
rigid-contact limitation above matters when a robot or object must also collide
with the ground or other rigid bodies. A solver accepted by upstream Newton
may still lack an EmbodiChain deformable adapter or operation.

Read physical nodes through `data.nodal_pos_w` and `data.nodal_vel_w` after
preparation. Render vertices can have different topology and indexing. See
{doc}`sim_soft_object` and {doc}`sim_cloth` for units, configuration, and runnable
tutorials.

## Collision pipelines and contact data

With `mujoco_warp`, `solver_cfg.use_mujoco_contacts` selects whether the solver
uses its internal collision path or Newton-generated rigid contacts. An
external `collision_cfg` does not by itself make the solver consume those
contacts. Confirm the active path before tuning its capacities or margins.
MJVBD uses particle-shape soft contacts and does not publish rigid contacts.

| `collision_cfg` field | Default | Meaning |
| :--- | :--- | :--- |
| `reduce_contacts` | `True` | Reduce dense mesh contacts. |
| `rigid_contact_max` | `None` | Use model capacity or scene estimation for rigid contacts. |
| `max_triangle_pairs` | `4_000_000` | Narrow-phase triangle-pair candidate capacity. |
| `soft_contact_max` | `None` | Derive particle/soft-contact capacity from the scene. |
| `soft_contact_margin` | `0.01` | Particle/soft-contact generation margin in metres. |
| `broad_phase` | `None` | Preserve upstream default; explicit modes include `nxn`, `sap`, and `explicit`. |
| `update_interval` | `1` | Refresh at every solver substep; integer `k` refreshes at substeps `0, k, 2k, ...`. `None` refreshes only at the first substep of each physics step. |

`shape_pairs_filtered`, `narrow_phase`, and `sdf_hydroelastic_config` accept
expert pipeline objects; their detailed contracts are in
{class}`~cfg.NewtonCollisionPipelineCfg`. Per-shape values belong to
`NewtonCollisionPropertiesCfg`. Reducing collision refresh frequency can miss
changes between updates, so assess penetration and contact continuity before
using it as a performance optimization.

`ContactSensor` supports Newton. It creates a backend-neutral contact query and
reports the query's capabilities at runtime:

| Execution path | Geometry | Impulse data |
| :--- | :--- | :--- |
| MuJoCo-Warp on CUDA | Available | Normal and friction impulse data available. |
| Other supported Newton rigid solvers | Available | Geometry-only paths use zero-valued impulse fields. |
| MuJoCo CPU mode | Device contact buffers unavailable | Unsupported by this sensor path. |
| MJVBD | Rigid contacts not published through `ContactQuery` | Unsupported by this sensor path. |

Check `sensor.contact_capabilities` before interpreting measurements. Zero
impulse on a geometry-only solver does not mean absence of a contact candidate.
See {doc}`sim_sensor` for frames, filtering, arena batching, and capacity limits.

## Differentiable simulation and CUDA Graphs

For the supported differentiable route, choose the solver explicitly:

```python
physics_cfg = NewtonPhysicsCfg(
    device="cuda:0",
    solver_cfg={"solver_type": "semi_implicit"},
    requires_grad=True,
    use_cuda_graph=False,
)
```

This enables differentiable model state; it does not turn an ordinary Gym
rollout, arbitrary state mutation, or every asset into a differentiable
computation. Use the learning integration described in
{doc}`/overview/rl/algorithm`. AutoSolver and deformable state mutation are
rejected in this mode. Upstream support for other differentiable solvers does
not expand the current EmbodiChain contract.

For ordinary CUDA simulation, `use_cuda_graph=True` is a request, not evidence
that capture has completed. The startup summary distinguishes pending,
captured, and disabled states. Preparation and initial camera rendering do not
advance the simulation merely to capture a graph. Model changes can invalidate
captured execution; author fixed-root placements and initial topology before
the first update. Particle contact-material schedules disable graph replay.

Measure compilation and first-step capture separately from steady-state
stepping. Compare runs at the same simulated duration and control period, and
record the resolved solver, substeps, graph status, device, and rendering load.

## Upstream references

The [Newton solver guide](https://newton-physics.github.io/newton/stable/solvers/index.html)
provides native solver feature tables. DexSim's AutoSolver and coupled solver
configuration names describe this integration and should not be confused with
native Newton Python class names. See {doc}`physics_migration` for the reference
versions and a practical migration workflow.
