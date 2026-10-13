# Initial-fill fluids

`cfg/fluid.py` owns source, object and render configuration. Numerical solver
configuration remains DexSim `SPHSolverCfg`, accepted directly or through the
existing `NewtonPhysicsCfg.solver_cfg` mapping with `solver_type: sph`.
`SimulationManager.add_fluid_object()` declares through SpawnScene and binds
`objects/fluid.py` in place at `prepare()`.

The current qualification boundary is one Newton CUDA environment without
gradients. Fixed particle counts and shared density/spacing are required.
The prepared particle scene has fixed topology: adding, replacing or removing
rigid assets also requires a new scene. Batch Gym integration remains gated on demo parity and native SPH world
isolation and selected-world reset; arena spacing does not prove isolation.

Fluid state uses world-frame `(instances, particles, 3|6)` Torch snapshots.
Writes update Scene particle batches; resetting transforms a saved local
particle template or restores the immutable initial state. Render topology is
reconstructed and is not inherited from volume/surface deformables.

Kinematic rigid boundaries can be written with `set_kinematic_state()` or
driven before each solver substep using `register_kinematic_rigid_trajectory()`.
Registered host controls disable graph replay and rewind selected clocks on
object reset.

A spawn reference is a declared rigid body's initial pose, not a persistent
attachment. Mesh volume sources must describe closed cavities, not cup walls.
Sampling math belongs to `compute/geometry/particles.py`.

Native rendering is published once through the existing Newton render bridge.
Display clipping acts only on copied positions; Viser consumes detached point
clouds. NoRender fluid mode does not require optical materials. Ordinary rigid
objects may supply a separate `visual_shape`; SDF cooking stays on the physical
shape. Explicit `force_sdf=False` permits pre-baked mesh SDFs.

Focused checks are `tests/sim/test_fluid_cfg.py`,
`tests/sim/objects/test_fluid_object.py` and the serial CUDA lifecycle test
`tests/sim/objects/test_fluid_runtime.py`. The pouring tutorial is an end-to-end
qualification workload, not a unit-test import target.
