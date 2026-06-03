# EmbodiChain Newton Backend Integration Design

This document records the current EmbodiChain integration state for the DexSim
Newton physics backend and the remaining work needed to complete it.

Use these EmbodiChain backend names consistently:

- `default`: the existing DexSim default physics backend.
- `newton`: the DexSim Newton physics backend.

Avoid exposing lower-level DexSim implementation names in EmbodiChain-facing
configuration, docs, and conditionals.

## Current State

### Configuration

Backend selection is currently inferred from `SimulationManagerCfg.physics_cfg`:

- `DefaultPhysicsCfg` selects the `default` backend.
- `NewtonPhysicsCfg` selects the `newton` backend.
- `physics_cfg_for_backend("default" | "newton")` returns the matching config.
- `physics_backend_from_cfg(...)` maps a config instance to its backend name.

`DefaultPhysicsCfg` owns default-backend PhysX settings and GPU-memory settings.
`NewtonPhysicsCfg` owns Newton settings: `physics_dt`, `device`, `num_substeps`,
`requires_grad`, `use_cuda_graph`, `debug_mode`, `solver_type`, `broad_phase`,
and `visualizer_enabled`.

`NewtonPhysicsCfg.to_dexsim_cfg(...)` creates a DexSim `NewtonCfg`, uses
`physics_dt` for `NewtonCfg.dt`, disables CUDA graph when gradient mode is
enabled, and requires `solver_type="semi_implicit"` for gradient mode.

### SimulationManager

`SimulationManager` now tracks the active backend with:

- `physics_backend`
- `is_default_backend`
- `is_newton_backend`
- `newton_manager`

For the `default` backend, manager initialization keeps the existing DexSim
behavior:

- apply `DefaultPhysicsCfg.to_dexsim_args()`
- apply default-backend GPU-memory config
- enable default GPU simulation only when the selected device is CUDA

For the `newton` backend, manager initialization:

- imports DexSim Newton lazily during world-config conversion
- sets `world_config.newton_cfg`
- obtains the per-world Newton manager through `get_newton_manager(self._world)`
- avoids default-backend GPU flags and default GPU memory APIs

Newton finalization is separate from default-backend GPU initialization:

- `finalize_newton_physics()` prepares or rebuilds the Newton model until the
  manager reaches `READY`.
- `update(...)` finalizes Newton before stepping.
- `init_gpu_physics()` delegates to `finalize_newton_physics()` when Newton is
  active.
- `set_manual_update(False)` is ignored for Newton because the backend does not
  support switching to automatic update.

Scene mutation invalidates Newton finalization with `_invalidate_newton_physics()`.
After finalization, `_reset_newton_entities_after_finalize()` reapplies rigid
object reset state. Rigid object groups are not yet supported on Newton.

### Object Backend Adapters

Rigid-body data access is routed through:

```text
embodichain/lab/sim/objects/backends/
    base.py
    default.py
    newton.py
```

`RigidBodyViewBase` defines the backend-neutral rigid-body API. The default
adapter handles existing CPU/default-GPU paths. The Newton adapter uses DexSim
Newton batch APIs for body data and collision filters.

EmbodiChain public rigid-body tensor convention is:

```text
(x, y, z, qx, qy, qz, qw)
```

Current Newton rigid-object support includes:

- dynamic and kinematic single `RigidObject` creation
- static single `RigidObject` creation
- local pose get/set
- body state get
- linear/angular velocity get/set
- linear/angular acceleration get
- force and torque at center of mass
- clear dynamics
- reset
- center-of-mass local pose get/set for dynamic rigid objects
- mass get/set
- friction get/set
- inertia diagonal get/set
- collision filter set for dynamic, kinematic, static, and pre-finalize bodies
- visual material, visibility, geometry, scale, and user-id APIs through the
  existing MeshObject paths

Static Newton bodies do not have `RigidBodyData`; static collision-filter writes
therefore use DexSim's per-entity metadata hook when a Newton body ID is not
available yet.

### Currently Unsupported Newton APIs

`SimulationManager` explicitly rejects these asset types on Newton:

- `add_soft_object(...)`
- `add_cloth_object(...)`
- `add_rigid_object_group(...)`
- `add_articulation(...)`
- `add_robot(...)`

`RigidObject` still does not support these runtime updates on Newton:

- `set_attrs(...)`
- `set_body_type(...)`
- `set_damping(...)`

`RigidObject.add_force_torque(pos=...)` ignores `pos` and applies force/torque at
the center of mass.

Newton kinematic pose locking is not complete. The rigid-object test suite keeps
a Newton-specific allowance for kinematic bodies changing after stepping.

Newton SDF rigid mesh support is not validated in EmbodiChain. The SDF rigid
object test is skipped for Newton.

### Verified Tests

The current rigid-object test file passes after the latest Newton integration
fixes:

```bash
pytest -q tests/sim/objects/test_rigid_object.py
```

Observed result:

```text
62 passed, 1 skipped, 41 warnings
```

## Improvements To Make

### API Clarity

- Add explicit capability checks for backend-specific support instead of relying
  on scattered `is_newton_scene(...)` checks.
- Make unsupported Newton APIs fail consistently with either `NotImplementedError`
  or a documented warning/no-op policy.
- Separate `is_use_gpu_physics` into clearer concepts:
  - selected tensor/device location
  - default-backend GPU API availability
  - Newton GPU execution

### Newton Lifecycle

- Keep `finalize_newton_physics()` as the single Newton preparation API.
- Do not add a separate non-stepping synchronization method until DexSim exposes
  a real Newton synchronization API.
- Track dirty scene/model state more explicitly so mutations after finalization
  can choose between live batch updates and model rebuilds.
- Avoid global Newton teardown while another world may still use monkey-patched
  DexSim classes.

### RigidObject

- Implement Newton `set_attrs(...)` by decomposing supported fields into batch
  property updates and rejecting unsupported fields explicitly.
- Implement Newton damping get/set through DexSim Newton if a runtime API exists;
  otherwise keep it metadata-only before finalization and document that runtime
  damping changes require rebuild.
- Implement `set_body_type(...)` for Newton or keep a hard unsupported error if
  DexSim cannot safely switch dynamic/kinematic/static bodies at runtime.
- Implement force-at-position when DexSim Newton exposes the needed API.
- Validate SDF rigid mesh creation and collision behavior on Newton.
- Fix or document kinematic pose-lock semantics.

### Object Groups, Articulations, Robots, Soft, Cloth

- Add Newton rigid-object-group support after single-object support is stable.
- Keep articulations and robots fail-fast until DexSim Newton articulation APIs
  are ready and tested.
- Keep soft and cloth fail-fast until there is an explicit Newton design and
  test coverage for those object types.

### Gym Env Integration

Use backend-specific initialization in env setup:

```python
if self.sim.is_default_backend and self.sim.is_use_gpu_physics:
    self.sim.init_gpu_physics()
elif self.sim.is_newton_backend:
    self.sim.finalize_newton_physics()
```

For stepping, keep the existing high-level flow:

```python
self._preprocess_action(action)
self._step_action(action)
self.sim.update(self.sim_cfg.physics_dt, self.cfg.sim_steps_per_control)
```

For reset, call object/manager reset methods and finalize Newton before reading
observations when the backend is Newton. Do not rely on a separate sync API.

## Completion Plan

1. Stabilize the single-rigid-object Newton API and keep
   `tests/sim/objects/test_rigid_object.py` green.
2. Add backend capability declarations and use them in public object APIs.
3. Finish Newton `RigidObject` parity for attributes, damping, body type,
   force-at-position, SDF meshes, and kinematic pose semantics.
4. Add tests for Newton lifecycle rebuild after scene mutation and runtime
   property mutation after finalization.
5. Implement and test Newton `RigidObjectGroup`.
6. Update gym env initialization/reset paths to use `finalize_newton_physics()`
   directly.
7. Add rigid-only Newton gym smoke tests.
8. Add gradient rollout wrapper and a minimal differentiable Newton smoke test.
9. Add articulation and robot support only after DexSim Newton exposes stable
   articulation APIs.
10. Add soft/cloth support only after a dedicated Newton object design and tests.

## Tests To Maintain

Configuration:

- `SimulationManagerCfg(physics_cfg=DefaultPhysicsCfg())` preserves current
  default-backend behavior.
- `SimulationManagerCfg(physics_cfg=NewtonPhysicsCfg())` creates a Newton world.
- `physics_cfg_for_backend(...)` and `physics_backend_from_cfg(...)` return the
  expected backend mapping.

Simulation:

- Newton world can be created, finalized, stepped, destroyed, and recreated.
- Default-backend GPU initialization does not run for Newton.
- Newton finalization does not call default-backend GPU fetch/apply APIs.
- Destroying a Newton simulation does not break subsequent default-backend
  simulation creation.

Rigid object:

- Dynamic rigid bodies fall under Newton.
- Static and kinematic rigid bodies can be created under Newton.
- Pose, velocity, acceleration, force/torque, reset, COM pose, mass, friction,
  inertia, collision filters, and geometry APIs behave consistently with the
  documented support matrix.
- Unsupported APIs produce the documented warning or exception.

Gym:

- Rigid-only Newton env initializes, steps, resets, and reads observations.
- Robot/articulation env under Newton raises the expected unsupported error.

Gradient:

- `requires_grad=True` plus `solver_type="semi_implicit"` can create a gradient
  rollout.
- A simple loss can backpropagate through a rollout without CPU/NumPy observation
  paths.

## Known Risks

- DexSim Newton monkey-patches global classes. Global teardown can affect other
  worlds if used at the wrong time.
- Public body/articulation ID mapping APIs may still need DexSim improvements.
- Newton gravity and contact configuration may not yet match every default-backend
  setting.
- Some object constructors still contain default-backend assumptions such as
  warmup updates; keep Newton guarded from those paths.
- Runtime shape/property mutations may require model rebuilds rather than live
  updates.
