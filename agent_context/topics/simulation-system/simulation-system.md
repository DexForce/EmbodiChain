# Simulation System

## Entry Points

| What | Path |
|------|------|
| Public simulation package | `embodichain/lab/sim/__init__.py` |
| World and scene owner | `embodichain/lab/sim/sim_manager.py` → `SimulationManager` |
| Global simulation config | `embodichain/lab/sim/sim_manager.py` → `SimulationManagerCfg` |
| Object and physics configs | `embodichain/lab/sim/cfg.py` |
| Gym lifecycle integration | `embodichain/lab/gym/envs/base_env.py` |
| Task scene construction | `embodichain/lab/gym/envs/embodied_env.py` |

`embodichain.lab.sim` exports the manager, its config, shared material
types, `BatchEntity`, and the simulation profiler. Import a specialized
object, sensor, solver, planner, or atomic-action API from its own subpackage.

## Ownership

`SimulationManager` owns one DexSim `World`, its global environment,
parallel arenas, and the Python registries for scene resources:

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
  → create World, global environment, defaults, and N arenas
  → EmbodiedEnv adds robot, objects, lights, and sensors
  → initialize GPU physics after scene construction when using CUDA
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

`SimulationManager` enables physics, selects manual physics updates, creates
the configured arenas, installs default plane/background/lighting resources,
and starts configured visualization during initialization. A Viser backend
forces `headless=True`; Viser and the native DexSim window are mutually
exclusive.

`SimulationManager.update()` initializes GPU physics lazily if needed and
then advances the world for the requested number of physics steps. Each
environment control step normally calls it with
`sim_steps_per_control`.

## Module Boundaries

| Area | Owner | Routed topic |
|------|-------|--------------|
| World, arenas, asset registries, physics update, cleanup | `sim_manager.py` | `simulation-system` |
| Shared object, render, physics, drive, and URDF configs | `cfg.py` | `configclass-pattern` for config mechanics |
| Rigid, deformable, articulation, robot, light, constraint, gizmo | `objects/` | `robot-system` for robots |
| Camera, stereo camera, contact sensor | `sensors/` | `sensor-system` |
| Robot-specific configuration | `robots/` | `robot-system` |
| Inverse kinematics | `solvers/` | `ik-solvers` |
| Trajectory and motion generation | `planners/` | `motion-planning` |
| Typed action planning and execution | `atomic_actions/` | `atomic-actions` |
| Task Program Semantic Calls and robot profiles | `embodichain/lab/task_program/semantics/` | `task-programs` |
| Reachability analysis and runtime workspace queries | `workspace/` | `robot-system` |
| Browser scene export and Viser runtime | `embodichain/lab/visualization/` | `sim-visualization` |

Use the narrow topic when a request names one of these subsystems. Use
`simulation-system` for the overall `lab/sim` architecture, manager
lifecycle, scene ownership, or cross-module flow.

`Articulation.get_parent_joint_chain(link_name)` is the public topology query
for integrations that need link ancestry. It returns immediate-parent-first
`ArticulationJointKinematics` values containing copied names, joint type,
origin, axis, and optional limits. Consumers must not reach into
`BatchEntity._entities` or retain backend-native joint-info objects.

`Articulation` also exposes deterministic link meshes through
`get_link_vert_face()` and named-state FK through `compute_fk()` with
`qpos_joint_names`. Stochastic surface sampling and Atomic Action geometry keys
do not belong to the simulation object; use
`atomic_actions.sample_initial_articulation_geometry()` for that adaptation.

## Configuration Flow

`SimulationManagerCfg` owns window size, headless mode, rendering, GPU/CPU
selection, arena count and spacing, physics timestep, physics and GPU-memory
settings, recording, profiling, and browser visualization.

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
| Shared object or physics config type | `cfg.py` |
| Add/get/remove behavior for a scene entity | `sim_manager.py` plus its `objects/` implementation |
| Task scene composition | `embodied_env.py` or the task config |
| Environment timing, reset, or control-step behavior | `base_env.py` and `env-framework` |
| Robot, sensor, solver, planner, or atomic action | Follow the corresponding routed topic and add-* skill |
| Browser visualization | `embodichain/lab/visualization/` and `sim-visualization` |

## Invariants

- Configure `num_envs`, device, renderer, and physics settings before
  constructing `SimulationManager`.
- Treat resource UIDs as registry identities; retrieve and mutate resources
  through the manager instead of maintaining a parallel scene registry.
- Keep batched object and sensor state aligned with the manager's arena count.
- Build scene assets before explicitly initializing GPU physics. The manager
  will warn and initialize lazily on the first update if this was missed.
- Manual update is the default; normal environment stepping must advance
  physics through `SimulationManager.update()`.
- Reset only the requested environment rows and honor
  `excluded_uids` for resources detached from automatic reset.
- `destroy()` queues deferred cleanup. Tests and non-exiting standalone
  callers that use `exit_process=False` must call
  `SimulationManager.flush_cleanup_queue()`.
- Resolve articulation ancestry through `get_parent_joint_chain()`; keep
  DexSim topology access encapsulated by `Articulation`.
- Keep articulation mesh access, FK, and topology domain-neutral. Perform
  affordance sampling and semantic-key conversion in the Atomic Action adapter.

## Common Failure Modes

| Symptom | Likely cause |
|---------|--------------|
| Scene resource cannot be found or the wrong object is returned | UID mismatch or code bypassed the manager registry |
| CUDA physics data is stale on the first step | GPU physics was initialized before all assets were added, or not initialized explicitly |
| Native window does not open | `headless=True`, often forced by the Viser backend |
| Device and renderer use the wrong GPU | `sim_device` and `gpu_id` disagree; the device index takes precedence for CUDA simulation |
| Simulation advances at the wrong control rate | `physics_dt` and `sim_steps_per_control` were configured inconsistently; see `env-framework` |
| A test leaks a DexSim world | `destroy(exit_process=False)` was called without flushing the cleanup queue |
| Python exits during cleanup | `destroy()` used its process-exit default; pass `exit_process=False` for embedded or test lifecycles |
