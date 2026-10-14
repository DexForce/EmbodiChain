# Initial-fill fluids

`cfg/fluid.py` owns source, object and render configuration. Numerical solver
configuration remains DexSim `SPHSolverCfg`, accepted directly or through the
existing `NewtonPhysicsCfg.solver_cfg` mapping with `solver_type: sph`.
`SimulationManager.add_fluid_object()` declares through SpawnScene and binds
`objects/fluid.py` in place at `prepare()`.

Fluids require Newton CUDA without gradients. Multiple environments require the
native SPH world-isolation and selected-reset capability flags; older runtimes
remain restricted to one environment. Fixed particle counts and shared
density/spacing are required.
The prepared particle scene has fixed topology: adding, replacing or removing
rigid assets also requires a new scene. Arena spacing does not prove isolation:
native kernels filter neighbors and boundaries by world and keep convergence
statistics per world. State writes clear only selected solver caches through
canonical reset masks, preserving shared boundary history.

Fluid state uses world-frame `(instances, particles, 3|6)` Torch snapshots.
Writes update Scene particle batches; resetting transforms a saved local
particle template or restores the immutable initial state. Render topology is
reconstructed and is not inherited from volume/surface deformables.

Kinematic rigid boundaries can be written with `set_kinematic_state()` or
driven before each solver substep using `register_kinematic_rigid_trajectory()`.
Registered host controls disable graph replay and rewind selected clocks on
object reset.

A spawn reference is a declared rigid body's initial pose, not a persistent
attachment. The source may be an explicit point import, a primitive, a capped
Z-axis revolved profile, or a closed fill-volume mesh. Profiles contain ordered
`(local Z, radius)` pairs. `fill_level` clips the source's full Z extent before
sampling; solid-wall clearance does not inset that free-surface plane. Mesh
volume sources must describe closed cavities, not cup walls.

Sampling math belongs to `compute/geometry/particles.py`, independent of lab.
The manager samples once at declaration and retains the local template in
memory; replica binding and selected reset never reopen or regenerate a source.
Changing spacing or fill level before construction regenerates the point count;
prepared topology remains fixed. Points/NPZ imports already contain fixed
sampling and reject volume-only profile/fill-level controls. The default pour
scene uses a profile source with no particle file or disk cache. Its existing
stationary replay hold provides physical settling after direct initialization.

Native rendering is published once through the existing Newton render bridge.
Display clipping acts only on copied positions; Viser consumes detached point
clouds. NoRender fluid mode does not require optical materials. Ordinary rigid
objects may supply a separate `visual_shape`; SDF cooking stays on the physical
shape. Explicit `force_sdf=False` permits pre-baked mesh SDFs.

Focused checks are `tests/sim/test_fluid_cfg.py`,
`tests/sim/objects/test_fluid_object.py` and the serial CUDA lifecycle test
`tests/sim/objects/test_fluid_runtime.py` and `test_fluid_batch_runtime.py`.
For YAML declarations, refill observations and Task Program replay, see
[fluid Gym integration](../env-framework/fluid-replay.md). The pouring tutorial is an end-to-end
qualification workload, not a unit-test import target.
