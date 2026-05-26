# EmbodiChain Newton Backend Integration Design

This memory records the intended design for adding DexSim Newton physics backend support to EmbodiChain.
Use `default` to refer to the existing DexSim physics backend everywhere in new EmbodiChain code and docs. Low-level DexSim implementation details should not leak into EmbodiChain-facing backend names.

## Scope

Primary files to update:

- `/root/sources/EmbodiChain/embodichain/lab/sim/cfg.py`
- `/root/sources/EmbodiChain/embodichain/lab/sim/sim_manager.py`
- `/root/sources/EmbodiChain/embodichain/lab/sim/objects/`
- `/root/sources/EmbodiChain/embodichain/lab/gym/envs/`

Relevant DexSim Newton files:

- `/root/sources/dexsim/python/dexsim/engine/newton_physics/__init__.py`
- `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_cfg.py`
- `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_physics_scene.py`
- `/root/sources/dexsim/python/dexsim/engine/newton_physics/gradient_rollout.py`

Reference design from IsaacLab:

- `/root/sources/IsaacLab/source/isaaclab/isaaclab/physics/physics_manager.py`
- `/root/sources/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py`
- `/root/sources/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`
- `/root/sources/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py`

## Backend Names

EmbodiChain backend names:

- `"default"`: the existing DexSim backend and current behavior.
- `"newton"`: DexSim Newton backend.

Do not introduce older backend-specific names into user-facing EmbodiChain config, docs, or conditionals. If a local variable must refer to a low-level DexSim GPU API, use a narrow name such as `is_default_gpu_backend`.

## Configuration Design

Group the original physics-related configuration under a default-backend config, then add a new Newton config to `SimulationManagerCfg`.

Recommended structure in `embodichain/lab/sim/cfg.py`:

```python
@configclass
class DefaultPhysicsCfg:
    # Move or alias the existing PhysicsCfg fields here.
    # Keep backwards compatibility by preserving PhysicsCfg as an alias or subclass during transition.
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    bounce_threshold_velocity: float = 0.2
    enable_pcm: bool = True
    enable_tgs: bool = True
    enable_ccd: bool = False
    enable_enhanced_determinism: bool = False
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025
    length_tolerance: float = 1.0
    speed_tolerance: float = 1.0

    def to_dexsim_args(self) -> dict:
        ...


# Transitional compatibility option:
PhysicsCfg = DefaultPhysicsCfg
```

Add:

```python
@configclass
class NewtonPhysicsCfg:
    num_substeps: int = 10
    device: str | None = None
    require_grad: bool = False
    use_cuda_graph: bool = True
    debug_mode: bool = False
    solver_type: str = "mjwarp"  # allowed: mjwarp, xpbd, semi_implicit, featherstone
    broad_phase: str = "sap"     # allowed: nxn, sap, explicit
    visualizer_enabled: bool = False

    def to_dexsim_cfg(self, physics_dt: float, device: str, gpu_id: int):
        # Import dexsim.engine.newton_physics lazily so default backend users do not pay import/setup cost.
        ...
```

Update `SimulationManagerCfg`:

```python
@configclass
class SimulationManagerCfg:
    physics_backend: Literal["default", "newton"] = "default"
    default_physics_cfg: DefaultPhysicsCfg = DefaultPhysicsCfg()
    newton_physics_cfg: NewtonPhysicsCfg = NewtonPhysicsCfg()
    gpu_memory_config: GPUMemoryCfg = GPUMemoryCfg()
    ...
```

`gpu_memory_config` is only meaningful for the default backend. It should be ignored or warned about under Newton.

`NewtonPhysicsCfg.to_dexsim_cfg(...)` should set `NewtonCfg.dt` from `SimulationManagerCfg.physics_dt`. Avoid duplicating `dt` in both configs unless an explicit override is required later.

For gradient mode:

- `require_grad=True`
- `solver_type="semi_implicit"`
- CUDA graph should be disabled by DexSim Newton or by the config conversion when needed.

## SimulationManager Design

In `embodichain/lab/sim/sim_manager.py`, route world creation through the backend name.

For `physics_backend == "default"`:

- Keep current behavior.
- Set `world_config.enable_gpu_sim` and `world_config.direct_gpu_api` when `device` is CUDA.
- Call `dexsim.set_physics_config(**cfg.default_physics_cfg.to_dexsim_args())`.
- Call `dexsim.set_physics_gpu_memory_config(**cfg.gpu_memory_config.to_dict())`.

For `physics_backend == "newton"`:

- Lazily import `dexsim.engine.newton_physics`.
- Set `world_config.newton_cfg = cfg.newton_physics_cfg.to_dexsim_cfg(...)` before creating `dexsim.World`.
- Do not set `world_config.enable_gpu_sim` or `world_config.direct_gpu_api`; those are default-backend GPU API flags.
- Do not call `dexsim.set_physics_gpu_memory_config(...)`.
- Avoid default-backend-only GPU APIs such as `gpu_fetch_rigid_body_data` and `gpu_apply_rigid_body_data`.
- Obtain the manager through `dexsim.engine.newton_physics.get_newton_manager(self._world)`.

Add properties:

```python
@property
def is_default_backend(self) -> bool: ...

@property
def is_newton_backend(self) -> bool: ...

@property
def is_default_gpu_backend(self) -> bool: ...

@property
def is_newton_gpu_backend(self) -> bool: ...

@property
def newton_manager(self): ...

@property
def newton_scene(self): ...
```

Replace direct calls to `init_gpu_physics()` in higher-level code with a backend-neutral method:

```python
def prepare_physics(self):
    if self.is_default_gpu_backend:
        self.init_gpu_physics()
    elif self.is_newton_backend:
        self._world.update(0.0)  # forces lazy Newton model finalization if needed
```

`SimulationManager.update(...)` should:

- Call `init_gpu_physics()` only for `is_default_gpu_backend`.
- For Newton, simply call `self._world.update(physics_dt)` for each step; DexSim Newton handles lazy finalize, rebuild, stepping, and render synchronization.

Destroy/cleanup:

- Be careful with `dexsim.engine.newton_physics.teardown_newton_physics()` because DexSim Newton currently monkey-patches classes globally.
- Do not call global teardown while another world may still be using Newton.
- Prefer a per-world manager clear API if DexSim exposes one later.

## Object Layer Design

Keep the public EmbodiChain object classes stable, but route backend-specific data access through adapters.

Recommended package:

```text
embodichain/lab/sim/objects/backends/
    __init__.py
    base.py
    default.py
    newton.py
```

The public classes stay in place:

- `RigidObject`
- `RigidObjectGroup`
- `Articulation`
- `Robot`

For now, implement Newton support only for rigid objects and rigid object groups.

Newton articulation support in DexSim is still under development. Do not implement EmbodiChain Newton `Articulation` or `Robot` support yet. Add an explicit fail-fast error if a user attempts to create an articulation or robot with `physics_backend == "newton"`:

```python
raise NotImplementedError(
    "Newton articulation support is under development in DexSim and is not enabled in EmbodiChain yet."
)
```

Rigid object Newton adapter:

- Map each DexSim `MeshObject` to Newton body IDs.
- Prefer a public DexSim API if available, such as `manager.get_body_id(mesh_object)`.
- If no public API exists yet, request one from DexSim rather than relying permanently on private mappings.

Use `manager.newton_scene` APIs:

- `fetch_pose(body_ids, out)`
- `apply_pose(body_ids, data)`
- `fetch_vec3(body_ids, data_type, out)`
- `apply_vec3(body_ids, data_type, data)`
- `fetch_force(body_ids, force_type, out)`
- `apply_force(body_ids, force_type, data)`

Pose format conversion:

- Newton scene pose: `(qx, qy, qz, qw, x, y, z)`
- EmbodiChain pose: `(x, y, z, qw, qx, qy, qz)`

Runtime behavior:

- Before Newton model finalization, either use DexSim object setters or call `sim.prepare_physics()` before data access.
- After finalization, prefer direct `newton_scene` reads/writes to avoid default-backend GPU APIs.
- Runtime changes to shape, mass, COM, or collision settings may mark the Newton model stale and trigger a rebuild on the next update. Prefer doing these changes before finalization or during reset.

Default plane:

- The current default plane is implemented as a visual plane plus hidden collision cube.
- For Newton, prefer a true static plane or explicit static box if DexSim Newton supports it cleanly.

## Gym Env Integration

In `embodichain/lab/gym/envs/base_env.py`, replace CUDA-based backend initialization:

```python
if self.device.type == "cuda":
    self.sim.init_gpu_physics()
```

with:

```python
self.sim.prepare_physics()
```

This lets `SimulationManager` decide whether to initialize default-backend GPU buffers or finalize Newton.

In `BaseEnv.step(...)`, keep the current high-level flow, but leave room for a backend-neutral write hook:

```python
self._preprocess_action(action)
self._step_action(action)
self.sim.write_data_to_physics()  # no-op initially; useful later
self.sim.update(self.sim_cfg.physics_dt, self.cfg.sim_steps_per_control)
```

In `BaseEnv.reset(...)`, after resetting object state and initializing the episode, refresh Newton state before reading observations:

```python
if self.sim.is_newton_backend:
    self.sim.forward_physics()
```

`forward_physics()` can initially call into DexSim Newton manager full forward kinematics/state sync if available. It can be optimized later with dirty masks.

Because articulation is skipped for now, gym environments that require `Robot` or `Articulation` should fail fast under Newton with a clear message.

## Gradient Mode

Expose gradient mode only through Newton.

Recommended API:

```python
rollout = sim.newton_manager.create_gradient_rollout(record_steps=...)
```

or a higher-level wrapper:

```python
rollout = env.create_gradient_rollout(record_steps, loss_fn, optimizer_step)
```

Constraints:

- `newton_physics_cfg.require_grad` must be true.
- `newton_physics_cfg.solver_type` must be `semi_implicit`.
- Observations and rewards used for differentiable training must avoid CPU getters, NumPy conversion, and detached tensors.
- Rendering and randomization should be disabled inside differentiable rollout unless explicitly made gradient-safe.

## IsaacLab-Inspired Improvements

Apply these IsaacLab ideas in EmbodiChain:

- Add a small backend manager abstraction instead of scattering backend checks everywhere.
- Use lifecycle events or hooks such as `MODEL_INIT`, `PHYSICS_READY`, and `STOP`.
- Replace object-constructor warmup calls like `world.update(0.001)` with a single `sim.prepare_physics()` after scene construction.
- Add backend-specific object data adapters.
- Add task/backend presets later, because Newton often needs different `physics_dt`, substeps, solver, and contact settings from the default backend.
- Add mask/index write APIs for vectorized envs and CUDA graph safety.
- Track dirty FK/render state instead of synchronizing every write.

## Implementation Milestones

1. Add `physics_backend`, `DefaultPhysicsCfg`, and `NewtonPhysicsCfg`.
2. Update `SimulationManager` world creation and backend properties.
3. Add `prepare_physics()` and update gym env initialization to use it.
4. Add Newton rigid object adapter.
5. Add Newton rigid object group adapter.
6. Add clear fail-fast errors for Newton articulation/robot creation.
7. Add rigid-object Newton smoke tests.
8. Add gym smoke tests for rigid-only Newton environments.
9. Add gradient rollout wrapper and a minimal gradient smoke test.
10. Add articulation/robot support later after DexSim Newton articulation API is ready.

## Tests To Add

Configuration:

- `SimulationManagerCfg(physics_backend="default")` preserves current behavior.
- `SimulationManagerCfg(physics_backend="newton")` creates a DexSim world with Newton manager.
- Newton config conversion sets `dt` from `physics_dt`.

Simulation:

- Newton world can be created and stepped headlessly.
- `prepare_physics()` finalizes Newton without calling default-backend GPU APIs.
- Destroying a Newton simulation does not break subsequent default-backend simulation creation.

Rigid object:

- Dynamic cube falls under Newton.
- Pose and velocity tensors have the same EmbodiChain layout as default backend.
- `set_local_pose`, `set_velocity`, `add_force_torque`, and `clear_dynamics` work.
- Multi-env rigid object group fetch/write reshapes correctly.

Gym:

- BaseEnv with Newton and no robot initializes, steps, and resets.
- Robot/articulation env under Newton raises the expected `NotImplementedError`.

Gradient:

- `require_grad=True` plus `solver_type="semi_implicit"` can create a gradient rollout.
- A simple loss can backpropagate through the rollout without CPU/NumPy observation paths.

## Known Risks

- DexSim Newton monkey-patches global classes. Avoid global teardown while other worlds exist.
- DexSim Newton gravity handling may need a full gravity-vector API to match EmbodiChain's existing default config.
- Public body/articulation ID mapping APIs may be needed in DexSim.
- The current `is_use_gpu_physics` concept conflates CUDA device with default-backend GPU APIs and should be replaced.
- Current object constructors may finalize physics too early by calling `world.update(0.001)`; avoid this under Newton.
- Newton articulation is intentionally skipped until DexSim support is ready.
