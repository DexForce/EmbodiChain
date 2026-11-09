# Newton Physics Backend

```{currentmodule} embodichain.lab.sim
```

The Newton backend integrates Newton through DexSim while preserving
EmbodiChain's shared scene, object, sensor, and stepping APIs. Select it with
{class}`~cfg.NewtonPhysicsCfg`, or with `physics: newton` in a Gym environment
configuration.

```{important}
For scenes that combine an articulated robot with cloth or a volume soft body,
start with **DexUni**. It is the DexSim coupled solver configuration for this
mixed scene class and is the main Newton path documented by EmbodiChain.
```

## DexSim Unified Coupled Solver For Embodied AI

DexUni supports scenes that combine articulated robots with cloth or volume
soft bodies. It is selected automatically when a finalized scene contains both
an articulation and a supported deformable, or explicitly with
`solver_type="dexuni"`.

```python
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

sim_config = SimulationManagerCfg(
    physics_cfg=NewtonPhysicsCfg(
        device="cuda:0",
        physics_dt=0.01,
        num_substeps=10,
        solver_cfg={
            "solver_type": "dexuni",
            "iterations": 8,
        },
        # DexUni owns particle-shape collision generation.
        collision_cfg=None,
    )
)
```

DexUni is a DexSim integration configuration, not another native solver class
in Newton's public solver list. Its capabilities and current EmbodiChain
limitations are documented in {doc}`newton/dexuni`.

## Automatic solver selection

Leaving `solver_cfg=None` preserves DexSim's scene-aware `DexUniSolverCfg`.
Add the initial robots and objects before {meth}`SimulationManager.prepare`
so DexUni can resolve solver ownership and contact handling from the complete
Spawn scene. After preparation, `sim.physics.solver_type` reports the configured
solver family. See {doc}`newton/dexuni` for DexUni dispatch and options.

## Explicit solver selection

Choose a concrete solver when an algorithm or solver-specific parameter must be
fixed. EmbodiChain mapping configs recognize `dexuni`, `mujoco_warp`
(or `mjwarp`), `xpbd`, `semi_implicit`, `featherstone`, and `vbd`. A DexSim
`NewtonSolverCfg` object can also be assigned directly.

`requires_grad=True` requires an explicit `semi_implicit` configuration. See
{doc}`newton/runtime_qualification` before treating a successfully decoded
configuration as a supported runtime path.

## Configuration reference

Scene parameters belong to `NewtonPhysicsCfg`; per-object collision, material,
and joint properties belong to asset configurations. Default-backend fields
such as `gpu_memory`, `bounce_threshold`, and `enable_ccd` do not belong here.

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `physics_dt` | `0.01` | One EmbodiChain physics step in seconds. |
| `device` | `"cuda:0"` | Compute device; explicit CPU execution remains solver/asset dependent. |
| `gravity` | `[0.0, 0.0, -9.81]` | World-frame acceleration in m/s². |
| `num_substeps` | `10` | Solver substeps per physics step; interval is `physics_dt / num_substeps`. |
| `solver_cfg` | `None` | Preserve DexSim AutoSolver; a mapping or native config fixes a solver. |
| `collision_cfg` | `NewtonCollisionPipelineCfg()` | External Newton collision pipeline. Set `None` when the selected path owns collision generation, including DexUni. |
| `requires_grad` | `False` | Enable the current differentiable integration; requires explicit `semi_implicit`. |
| `use_cuda_graph` | `True` | Request graph capture when supported; unavailable on CPU and disabled in gradient mode. |
| `sync_to_renderer` | `None` | Preserve DexSim's automatic render-sync policy; cameras still synchronize on demand. |
| `debug_mode` | `False` | Enable additional runtime diagnostics. |
| `suppress_warp_kernel_logs` | `True` | Suppress Warp startup and kernel compilation chatter, but not warnings or errors. |

The external `collision_cfg` only matters when the resolved solver consumes
that pipeline. It does not force a solver to use those contacts. Tune collision
capacity, refresh intervals, and margins only after confirming the active
collision path.

## What to read next

- {doc}`newton/dexuni` — DexUni capabilities, configuration, and current
  limitations.
- {doc}`newton/runtime_qualification` — solver/device capability matrix and
  post-`prepare()` qualification workflow.
- {doc}`../../sim_soft_object` and {doc}`../../sim_cloth` — EmbodiChain
  deformable configuration and runtime state.

For Newton-native solver internals and tuning, refer directly to the
[official solver documentation](https://newton-physics.github.io/newton/stable/solvers/index.html)
and [solver tuning reference](https://newton-physics.github.io/newton/stable/concepts/simulation_tuning_solvers.html).

```{toctree}
:maxdepth: 1
:caption: Newton topics

newton/dexuni
newton/runtime_qualification
```

## Mesh query acceleration

`NewtonPhysicsCfg.mesh_bvh_constructor` selects the mesh BVH constructor used
for mesh and heightfield ray queries. Its default `None` preserves the engine
configuration. Large CUDA terrain scans can select `"cubql"`. The constructor
changes the acceleration structure while preserving the mesh geometry and
physical contact parameters.
