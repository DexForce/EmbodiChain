# Fluid Object

```{currentmodule} embodichain.lab.sim
```

Use {class}`~objects.fluid.FluidObject` for a fixed initial sample of liquid
particles. The manager samples the source at declaration and binds the
particle state during `sim.prepare()`. The particle count stays fixed
throughout the scene lifetime.

Fluids require Newton on CUDA with DexUni's nested DFSPH component and
gradients disabled. All fluid objects in one scene share density and
particle spacing. Use separate scenes for fluids and cable, cloth, or
soft-body assets. See {doc}`sim_manager/physics/newton/dexuni` for solver
configuration.

## Configuration

Pass {class}`~cfg.fluid.FluidObjectCfg` to `sim.add_fluid_object(cfg)`.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `uid` | Required | Unique identifier for one logical fluid batch. |
| `density` | `1000.0` | Rest density in kg/m³, shared by scene fluids. |
| `particle_spacing` | `0.0015` | Initial particle lattice spacing in metres. |
| `particle_radius` | `None` | Optional particle radius passed to the native descriptor. |
| `init_pos`, `init_rot`, `init_local_pose` | Common placement defaults | Initial placement relative to each environment. |
| `spawn` | `FluidSpawnCfg()` | Local particle positions or a volume sampling recipe. |
| `render` | `FluidRenderCfg()` | Particle or reconstructed-surface appearance. |

### Initial particle sources

Configure {class}`~cfg.fluid.FluidSpawnCfg` with one source kind.

| `kind` | Source fields |
| :--- | :--- |
| `"points"` | Supply exactly one of `positions` with shape `(N, 3)` or an NPZ `file_path` containing the `positions` array. |
| `"box"` | `size=(x, y, z)` in metres. |
| `"sphere"` | `radius` in metres. |
| `"cylinder"` | `radius` and `height` in metres. |
| `"revolved_volume"` | Ordered `profile` pairs `(local Z, radius)`, capped at both ends. |
| `"mesh_volume"` | `file_path` naming a closed mesh of the liquid's fill volume. |

Volume sources support `fill_level` in `(0, 1]` and an inward
`clearance`. `fill_level` clips the complete source Z extent before
sampling. `max_particles` bounds the sampling allocation.
`spawn.initial_velocity` supplies the local initial velocity.

`spawn.reference_uid` can name an already declared rigid object. The
manager composes that object's initial pose with the fluid placement once.
After preparation the fluid moves according to its own particle state.

## Setup and initialization

This example samples a small box of water in two environments using
physics-only rendering. It reads the particle state and restores
environment zero after one update.

```python
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    FluidObjectCfg,
    FluidRenderCfg,
    FluidSpawnCfg,
    NewtonPhysicsCfg,
    RenderCfg,
)

sim = SimulationManager(
    SimulationManagerCfg(
        headless=True,
        num_envs=2,
        physics_cfg=NewtonPhysicsCfg(
            device="cuda:0",
            physics_dt=0.001,
            num_substeps=2,
            solver_cfg={"solver_type": "dexuni"},
        ),
        render_cfg=RenderCfg(renderer="no-render"),
    )
)
try:
    water = sim.add_fluid_object(
        FluidObjectCfg(
            uid="water",
            particle_spacing=0.01,
            spawn=FluidSpawnCfg(kind="box", size=(0.04, 0.04, 0.04)),
            init_pos=(0.0, 0.0, 0.5),
            render=FluidRenderCfg(mode="none"),
        )
    )
    sim.prepare()
    sim.update(step=1, render_final_step=False)
    state = water.data.particle_state_w  # (2, water.num_particles, 6)
    water.reset(env_ids=[0])
finally:
    sim.destroy(exit_process=False)
    SimulationManager.flush_cleanup_queue()
```

## Particle state and reset

State properties return independent world-frame snapshots after
preparation. They already include environment origins. Let
`E = water.num_instances` and `N = water.num_particles`.

| Property | Shape or unit | Meaning |
| :--- | :--- | :--- |
| `data.particle_pos_w` | `(E, N, 3)` | Current positions in metres. |
| `data.particle_vel_w` | `(E, N, 3)` | Current velocities in m/s. |
| `data.particle_state_w` | `(E, N, 6)` | Position followed by velocity. |
| `data.default_particle_state_w` | `(E, N, 6)` | Initial state retained at binding. |
| `particle_mass` | kg | `density * particle_spacing**3`. |
| `nominal_volume` | m³ | `N * particle_spacing**3`. |

Use `write_particle_state_to_sim(state, env_ids=...)` for a finite
floating tensor `(selected E, N, 6)`. Rows follow the selected environment
order. Writes invalidate captured graphs and supported solver history.
Density diagnostics refresh on the next physical update.

`reset(env_ids=...)` restores initial particle positions and velocities.
`reset_to_pose(pose_w, env_ids=...)` places the saved local sample at
selected world-frame `xyz + xyzw` poses. Both operations preserve the
particle count and leave the simulation clock at its current value.

## Rendering and coupling

Use `RenderCfg(renderer="no-render")` together with
`FluidRenderCfg(mode="none")` for physics-only runs. Native fluid display
requires `renderer="fast-rt"` or `"rt"`, selected before the first World
in the process. Display modes `"particles"` and `"mesh_surface"` use the
same physical particle state. Optional `render.clip_below_z` affects the
display copy.

The current DFSPH component receives one-way interaction from rigid
geometry. Rigid bodies in these scenes use DexUni's pure MuJoCo rigid path.
Viscosity, surface tension, and iteration settings belong under
`NewtonPhysicsCfg.solver_cfg.sph_cfg`.

## Related topics

- {doc}`sim_assets` for preparation and asset lookup.
- {doc}`sim_rigid_object` for container colliders and particle SDF boundaries.
- [Fluid tutorial](https://github.com/DexForce/EmbodiChain/blob/main/scripts/tutorials/sim/create_fluid.py)
  for a container, native rendering, diagnostics, and reset.
