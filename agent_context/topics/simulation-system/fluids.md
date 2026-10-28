# Initial-fill fluids

`cfg/fluid.py` owns source, object and render configuration. Fluids use DexUni,
which dispatches DFSPH when the scene contains fluid particles. Optional
numerical settings belong to DexSim `SPHSolverCfg`, nested under
`NewtonPhysicsCfg.solver_cfg.sph_cfg`; omitting them uses native DFSPH defaults.
`SimulationManager.add_fluid_object()` declares through SpawnScene and binds
`objects/fluid.py` in place at `prepare()`.

The fluid declaration owns density and spacing. Default native SPH values for
`rest_density` and `particle_diameter` are placeholders, equally for mappings
and native config objects; non-default solver values must match the fluid.
DexSim derives the effective solver values from the fluid descriptors.

## Runtime requirements

Fluid simulation requires Newton CUDA, a DexSim build exposing
`DexUniSolverCfg.sph_cfg`, no gradients and fixed particle topology.
Multiple environments require a DexSim SPH solver that explicitly advertises
both world isolation and selected-world reset; arena spacing alone does not
establish isolation. The development build
`0.5.1+0b8422cc6f6717058bd15371e61bd35cdbbe3c0c` supports both. The published
`0.5.1rc1` artifact does not provide the DexUni/DFSPH composition and cannot
run this fluid path.

The current native DexUni/DFSPH implementation is a prototype: DFSPH follows
the rigid step with one-way rigid-to-fluid interaction. When rigid bodies are
present, it requires the pure MuJoCo rigid path; static collision shapes do not
require a rigid-body solver. It cannot mix fluids with cloth or soft-body particles.
Online robot pouring, Gym declarations and replay belong to the dependent task
layer; two-way fluid feedback is not implied by DexUni's rigid/soft coupling.

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

`FluidObject(cfg, device)` creates an unregistered facade without sampling or
state allocation. The manager resolves the source recipe and reference pose,
then privately initializes the facade's local particle template, initial
rotation and write-invalidation hook. Spawn receives the same sampled points
and resolved pose. Derived caches never become additional config fields;
binding and reset reuse them while the input config remains a source recipe.

## Rendering and validation

Native fluids require FastRT (`fast-rt`) or OfflineRT (`rt`). Hybrid's Vulkan
primary surface lacks the OptiX liquid geometry, so the manager rejects Hybrid
before sampling or declaration, including an already initialized Hybrid
process. Use `no-render` for physics only; fluid mode `none` requires it.
Display clipping changes only a render copy. Viser consumes detached particle
point clouds. NoRender needs no optical material. Advanced optical parameters
follow the renderer requirements of the [material system](materials.md).

`scripts/tutorials/sim/create_fluid.py` demonstrates procedural box sampling,
five primitive container walls, prepare, step, state reads and reset. Pass
`--num-envs` to use multiple independent worlds on a capable SPH build. The
fixed container uses static SDF collision shapes, so DFSPH can collide with it
without running a rigid-body solver. It needs no robot, task configuration or downloaded assets:

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

The pouring task uses DexUni with nested DFSPH, a shared simulation clock and
one-way rigid-to-fluid coupling. When no joint ownership was explicitly authored,
the fluid setup assigns every finalized joint to MuJoCo before solver construction,
which covers a robot and a free cup across replicated worlds. Explicit ownership
selectors remain caller-owned. The current native prototype requires the pure
MuJoCo rigid path when bodies exist and does not support fluid reaction forces.

`RigidObjectCfg.particle_collision_shape` puts a particle-only SDF boundary on
the same body as its rigid contact hulls. Where DexSim exposes
`NewtonCollisionDesc.contributes_to_mass`, the boundary explicitly excludes
itself from mass. Older builds need explicit mass and inertia for dynamic bodies;
the pouring cup authors its COM and body-frame inertia from its contact hulls.
Static/kinematic boundaries retain their respective motion semantics. Independent
`visual_shape` settings preserve the render mesh.

Replay may write `RigidObject.set_kinematic_state()` or register a manager-owned
kinematic rigid trajectory before preparation. These host controls disable graph
replay and rewind selected clocks on object reset; they do not qualify online
grasping. Normal Gym reset restores or refills selected fluid rows without
stepping peer worlds. Online action execution supplies physical settling.

The robot demonstrations, task assets, Gym declarations, observations, recording,
Task Program runner and replay belong to the pouring task layer. See
[fluid Gym integration](../env-framework/fluid-replay.md) and the task's
`extensions/embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/README.md`.
