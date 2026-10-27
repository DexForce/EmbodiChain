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
the rigid step with one-way rigid-to-fluid interaction. It requires the pure
MuJoCo rigid path and cannot mix fluids with cloth or soft-body particles.
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
`--num-envs` to use multiple independent worlds on a capable SPH build. It
disables MuJoCo rigid-rigid contacts for its fixed container while retaining
DFSPH boundary collision. It needs no robot, task configuration or downloaded assets:

```bash
python scripts/tutorials/sim/create_fluid.py --headless --renderer no-render
```

Focused checks are `tests/compute/test_particles.py`,
`tests/sim/test_fluid_cfg.py`, `tests/sim/objects/test_fluid_object.py` and
`tests/visualization/test_scene_exporter.py`. Serial CUDA lifecycle coverage is
`tests/sim/objects/test_fluid_runtime.py`; `test_fluid_visibility.py` checks
camera pixels after stepping and direct state writes. Tutorial checks remain
syntax/CLI smoke checks rather than tutorial import tests.
