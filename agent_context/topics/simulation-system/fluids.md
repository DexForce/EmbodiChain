# Initial-fill fluids

`cfg/fluid.py` owns source, object and render configuration. Numerical settings
remain DexSim `SPHSolverCfg`, accepted directly or through
`NewtonPhysicsCfg.solver_cfg` with `solver_type: sph`.
`SimulationManager.add_fluid_object()` declares through SpawnScene and binds
`objects/fluid.py` in place at `prepare()`.

## Runtime requirements

The foundation supports the published `dexsim_engine==0.5.1rc1` artifact
(`56c12415053617cb0ffad6881ed57a45541a6da6`) with Newton CUDA, no gradients,
one environment and fixed particle topology. Its SPH solver has no world
isolation or selected-world solver reset. Requests for multiple environments
fail before declaration unless the loaded solver explicitly advertises both
capabilities. Arena spacing alone does not establish isolation.

The release provides SPH/DFSPH and particle/surface rendering. It does not
provide `SPHSolverCfg.rigid_solver_cfg`, `rigid_solver_substeps`, or DexUni's
`sph_cfg` prototype. Online robot pouring, batch fluid tasks, Gym declarations
and replay belong to the dependent task layer and require their respective
DexSim development implementations. Basic fluid use does not require them.

## State and sampling

Fluid state uses world-frame `(instances, particles, 3|6)` Torch snapshots.
Writes use Scene particle batches, invalidate captured graphs and reset
supported solver history. The legacy single-world solver rebaselines optional
SDF boundary pose history. Reset restores the immutable initial state or
transforms a saved local template without stepping the world. Explicit state
writes do not refresh density diagnostics until the next SPH step.

A spawn reference resolves a declared rigid body's initial pose once; it is
not a persistent attachment. Sources include explicit points/NPZ, primitives,
capped Z-axis revolved profiles and closed fill-volume meshes. Profile points
are ordered `(local Z, radius)` pairs. `fill_level` clips the full source Z
extent before sampling; clearance does not inset that free surface. Mesh
sources describe closed cavities, rather than container walls. Point imports
reject volume-only profile/fill-level controls.

Sampling math belongs to `compute/geometry/particles.py`, independent of lab.
The manager samples once at declaration and retains the local template;
binding and reset do not reopen source files. Particle count is fixed after
preparation, and all fluids share density and spacing. Adding, replacing or
removing rigid assets after preparing a particle scene requires a new scene.

## Rendering and validation

Native fluids require FastRT (`fast-rt`) or OfflineRT (`rt`). Hybrid's Vulkan
primary surface lacks the OptiX liquid geometry, so the manager rejects Hybrid
before sampling or declaration, including an already initialized Hybrid
process. Use `no-render` for physics only; fluid mode `none` requires it.
Display clipping changes only a render copy. Viser consumes detached particle
point clouds. NoRender needs no optical material.

`scripts/tutorials/sim/create_fluid.py` demonstrates procedural box sampling,
five primitive container walls, prepare, step, state reads and reset with one
environment. It needs no robot, task configuration or downloaded assets:

```bash
python scripts/tutorials/sim/create_fluid.py --headless --renderer no-render
```

Focused checks are `tests/compute/test_particles.py`,
`tests/sim/test_fluid_cfg.py`, `tests/sim/objects/test_fluid_object.py` and
`tests/visualization/test_scene_exporter.py`. Serial CUDA lifecycle coverage is
`tests/sim/objects/test_fluid_runtime.py`; `test_fluid_visibility.py` checks
camera pixels after stepping and direct state writes. Tutorial checks remain
syntax/CLI smoke checks rather than tutorial import tests.

## Online task and replay layer

Online pouring is built on the foundation above. `SPHSolverCfg.rigid_solver_cfg`
composes MuJoCo rigid dynamics with SPH on one model and clock; microsteps refine
contact integration without advancing a second clock. This requires the DexSim
implementation from MR !1473 (`codex/fluid-sph-worlds`), including world isolation,
selected reset and live MuJoCo/SPH composition. `dexsim_engine==0.5.1rc1` and an
unqualified `dev` checkout are insufficient. The DexUni tutorial additionally
needs MR !1475 (`feat/dexuni-dfsph-prototype`) and its `sph_cfg` prototype options;
that path remains single-world. Coupling is one-way rigid-to-fluid, with no fluid
reaction force on the cup or robot.

`RigidObjectCfg.particle_collision_shape` places a particle-only SDF boundary on
the same body as the grasp/contact hulls, contributing neither mass nor rigid
contacts. Native support for `NewtonCollisionDesc.contributes_to_mass` is part
of that development dependency. Independent `visual_shape` and optical material
settings preserve the render mesh. `MeshCollisionCfg.force_sdf=False` requests
an eagerly baked texture SDF when explicit grid settings are needed.

Replay may write `RigidObject.set_kinematic_state()` or register a manager-owned
kinematic rigid trajectory before preparation. These host controls disable graph
replay and rewind selected clocks on object reset; they do not qualify online
grasping. Normal Gym reset restores or refills selected fluid rows without
stepping peer worlds. Online action execution supplies physical settling.

The two robot tutorials, task assets, Gym declarations, observations, recording,
Task Program runner and replay live with the pouring task layer. See
[fluid Gym integration](../env-framework/fluid-replay.md) and the task's
`embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/README.md`.
Relevant development-runtime checks are `test_fluid_batch_runtime.py` and
`test_fluid_online_runtime.py` under `tests/sim/objects/`, alongside the Gym
integration/replay tests. Earlier pouring qualification is historical evidence;
the split preserves the implementation rather than rerunning a full pour.
