# Simulation System

## Entry Points

| What | Path |
|------|------|
| Public simulation package | `embodichain/lab/sim/__init__.py` |
| World and scene owner | `embodichain/lab/sim/sim_manager.py` → `SimulationManager` |
| Global simulation config | `embodichain/lab/sim/sim_manager.py` → `SimulationManagerCfg` |
| Spawn lifecycle coordinator | `embodichain/lab/sim/spawn/scene.py` → `SpawnScene` |
| EmbodiChain-to-Spawn translation | `embodichain/lab/sim/spawn/descriptors.py` |
| Object and physics configs | `embodichain/lab/sim/cfg.py` |
| Gym lifecycle integration | `embodichain/lab/gym/envs/base_env.py` |
| Task scene construction | `embodichain/lab/gym/envs/embodied_env.py` |

`embodichain.lab.sim` exports the manager, its config, shared material
types, `BatchEntity`, and the simulation profiler. Import a specialized
object, sensor, solver, planner, or atomic-action API from its own subpackage.

## Ownership

`SimulationManager` owns one DexSim `World`, a `SpawnScene`, and the Python
registries for scene resources. DexSim's `SceneBuilder` and `SpawnResult` own
descriptor revisions, native materialization, replicated arenas, and backend
runtime readiness. EmbodiChain registry objects are stable facades: `add_*()`
returns a declared facade and `prepare()` binds that same object in place.

The registries cover:

- rigid objects and rigid-object groups;
- soft and cloth objects;
- articulations and robots;
- rigid constraints, sensors, lights, gizmos, and markers;
- visual materials and texture caches;
- the optional browser-visualization runtime.

The manager owns simulation resources and physics stepping. `BaseEnv` owns
the Gym reset/step contract. `EmbodiedEnv` translates task configuration
into robots, sensors, lights, backgrounds, articulations, and interactive
objects added through the manager.

## Lifecycle

The environment-owned lifecycle is:

```text
EnvCfg.sim_cfg
  → BaseEnv._setup_scene()
  → SimulationManager(SimulationManagerCfg)
  → create World and a replicated Spawn scene declaration
  → EmbodiedEnv declares robot, objects, lights, and physical sensors
       → Default/PhysX may materialize native handles eagerly
       → Newton keeps physical descriptors deferred
  → SimulationManager.prepare()
       → finalize/rebuild pending Spawn descriptors
       → prepare the active backend runtime
       → bind declared EmbodiChain facades in place
       → attach sensors whose parents are now materialized
  → initialize metadata-dependent robot, action, and render-only resources
  → BaseEnv.step()
       → preprocess/apply action
       → SimulationManager.update(physics_dt, sim_steps_per_control)
       → update task state, observations, rewards, and termination
  → BaseEnv.reset()
       → SimulationManager.reset_objects_state(env_ids)
       → task episode-initialization hook
  → BaseEnv.close()
       → profiler report
       → SimulationManager.destroy()
```

`BaseEnv._setup_scene()` temporarily constructs the manager headlessly so
the scene can be assembled before a native window is opened. It sets
`SimulationManagerCfg.num_envs` from `EnvCfg.num_envs`.

`SimulationManager` enables physics, selects manual physics updates, prepares
the configured Arena layout, and owns a thin Spawn scene coordinator. With the
Default backend, preparing the Arena layout lets `add_*` materialize native
entities immediately, so articulation metadata and render nodes are available
before finalization. Newton builds its immutable model at `prepare()`. A Viser
backend forces `headless=True`; Viser and the native DexSim window are mutually
exclusive.

The default ground plane authors its repeated texture coordinates in the Spawn
render descriptor before materialization, so native and offscreen render paths
receive identical UV data on their first GPU upload.

`SimulationManager.prepare()` is the backend-neutral readiness boundary for
Default CPU, Direct GPU, and Newton. It is idempotent. Topology is committed
only when dirty, while runtime preparation, facade binding, and sensor
attachment are retried on every call. A failed binding therefore leaves that
declaration unbound and retryable; already completed declarations are not
rebound. `init_gpu_physics()` and `finalize_newton_physics()` remain compatibility
aliases, but new code should call `prepare()`.

Standalone callers must call `prepare()` after their last `add_*()` and before
reading link/joint metadata, object state, or advancing physics. `BaseEnv`
provides this boundary automatically between `_setup_scene()` and
metadata-dependent setup. `SimulationManager.update()` still calls the
readiness path defensively before advancing the requested physics steps.

## Module Boundaries

| Area | Owner | Routed topic |
|------|-------|--------------|
| World, arenas, asset registries, physics update, cleanup | `sim_manager.py` | `simulation-system` |
| Spawn declaration, commit/rebuild, and facade binding | `spawn/scene.py`, `spawn/descriptors.py` | `simulation-system` |
| Backend-neutral batched state/property access | `objects/backends/spawn.py` | `simulation-system` |
| Shared object, render, physics, drive, and URDF configs | `cfg.py` | `configclass-pattern` for config mechanics |
| Rigid, deformable, articulation, robot, light, constraint, gizmo | `objects/` | `robot-system` for robots |
| Camera, stereo camera, contact sensor | `sensors/` | `sensor-system` |
| Robot-specific configuration | `robots/` | `robot-system` |
| Inverse kinematics | `solvers/` | `ik-solvers` |
| Trajectory and motion generation | `planners/` | `motion-planning` |
| Typed action planning and execution | `atomic_actions/` | `atomic-actions` |
| Semantic scene and robot skill bindings | `skills/` | `atomic-actions` |
| Reachability analysis and runtime workspace queries | `workspace/` | `robot-system` |
| Browser scene export and Viser runtime | `embodichain/lab/visualization/` | `sim-visualization` |

Use the narrow topic when a request names one of these subsystems. Use
`simulation-system` for the overall `lab/sim` architecture, manager
lifecycle, scene ownership, or cross-module flow.

## Configuration Flow

`SimulationManagerCfg` owns window size, headless mode, rendering, GPU/CPU
selection, arena count and spacing, physics timestep, physics and GPU-memory
settings, recording, profiling, and browser visualization. Newton-specific
settings include `suppress_warp_kernel_logs=True`, which suppresses Warp module
compile/load chatter only during Newton build and facade initialization and
restores the process-wide Warp logging configuration afterward.

`EnvCfg` embeds `SimulationManagerCfg` and supplies the control-to-physics
step ratio. CLI and task config loaders may override runtime fields before
constructing the environment. Trace those overrides through the caller rather
than changing a default in the manager blindly.

Object-specific configuration belongs in `lab/sim/cfg.py` or the
corresponding robot/sensor module. Scene composition belongs in
`EmbodiedEnv` or a task config, not in `SimulationManagerCfg`.

## Where to Make Changes

| Change | Primary location |
|--------|------------------|
| Global world, renderer, device, arena, or physics lifecycle | `sim_manager.py` |
| Spawn source translation or typed link/joint overrides | `spawn/descriptors.py` plus the DexSim Spawn descriptor/adapter boundary |
| Declaration-to-result binding or retry behavior | `spawn/scene.py` and the object's `bind_spawn()` |
| Batched row/DOF selection or backend property parity | `objects/backends/spawn.py` and the DexSim Spawn batch facade |
| Shared object or physics config type | `cfg.py` |
| Add/get/remove behavior for a scene entity | `sim_manager.py` plus its `objects/` implementation |
| Task scene composition | `embodied_env.py` or the task config |
| Environment timing, reset, or control-step behavior | `base_env.py` and `env-framework` |
| Robot, sensor, solver, planner, or atomic action | Follow the corresponding routed topic and add-* skill |
| Browser visualization | `embodichain/lab/visualization/` and `sim-visualization` |

## Invariants

- Configure `num_envs`, device, renderer, and physics settings before
  constructing `SimulationManager`.
- Treat `add_*()` as declaration. Call `prepare()` before consuming native
  handles, link/joint metadata, batched state, or physics results.
- Keep `prepare()` convergent and retryable: do not mark a declaration bound
  until its full facade construction succeeds.
- Treat resource UIDs as registry identities; retrieve and mutate resources
  through the manager instead of maintaining a parallel scene registry.
- Keep batched object and sensor state aligned with the manager's arena count.
- Add the initial physical scene before `prepare()`. Calls to the legacy
  `init_gpu_physics()` and `finalize_newton_physics()` aliases are equivalent to
  `prepare()` and do not cause a second build.
- Delegate environment and DOF selections to DexSim Spawn batches instead of
  full-batch read/modify/write loops in object facades.
- Newton descriptor or topology mutations that cannot update the immutable
  runtime model live remain pending until the next `prepare()` rebuild.
- Manual update is the default; normal environment stepping must advance
  physics through `SimulationManager.update()`.
- Reset only the requested environment rows and honor
  `excluded_uids` for resources detached from automatic reset.
- `destroy()` queues deferred cleanup. Tests and non-exiting standalone
  callers that use `exit_process=False` must call
  `SimulationManager.flush_cleanup_queue()`.

## Common Failure Modes

| Symptom | Likely cause |
|---------|--------------|
| Scene resource cannot be found or the wrong object is returned | UID mismatch or code bypassed the manager registry |
| Link/joint metadata is empty or state access fails after `add_*()` | The declared facade has not crossed `SimulationManager.prepare()` yet |
| CUDA/Newton physics data is stale after a topology or descriptor mutation | Call `prepare()` so the dirty Spawn result can rebuild and rebind runtime views |
| Warp module compile/load lines appear during Newton initialization | `NewtonPhysicsCfg.suppress_warp_kernel_logs` was explicitly disabled, or compilation happened outside the managed preparation scope |
| Native window does not open | `headless=True`, often forced by the Viser backend |
| Device and renderer use the wrong GPU | `sim_device` and `gpu_id` disagree; the device index takes precedence for CUDA simulation |
| Simulation advances at the wrong control rate | `physics_dt` and `sim_steps_per_control` were configured inconsistently; see `env-framework` |
| A test leaks a DexSim world | `destroy(exit_process=False)` was called without flushing the cleanup queue |
| Python exits during cleanup | `destroy()` used its process-exit default; pass `exit_process=False` for embedded or test lifecycles |
