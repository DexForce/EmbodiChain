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
handles. `SimulationManager` owns the readiness boundary for each committed
Spawn topology revision. EmbodiChain registry objects are stable facades:
`add_*()` returns a declared facade and `prepare()` binds that same object in
place.

The registries cover:

- rigid objects and rigid-object groups;
- volume and surface deformables in one deformable-object registry;
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
       → for Newton, resolve source metadata and configure exact-name overlays
       → finalize/rebuild pending Spawn descriptors once
       → for Default, apply pending source overlays to materialized handles
       → prepare manager-owned runtime buffers for the committed revision
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

After backend materialization, dynamic `RigidObject`, `Articulation`, and
`RigidObjectGroup` facades capture their resolved mass, inertia diagonal, and
local center-of-mass pose in their data objects. The layouts are `[env]` in
`RigidBodyData`, `[env, link]` in `ArticulationData`, and `[env, object]` in
`RigidBodyGroupData`. Each data object exposes current `mass`, `inertia`, and
`com_pose` values plus immutable `default_*` initialization snapshots. Runtime
property writes do not change these snapshots. During reset, only the selected
environment rows are restored before dynamics are cleared and the configured
pose is reapplied; reset-mode event functors then run from this clean physical
baseline in the episode-initialization hook.

Deformables use the same public hierarchy for both topologies:
`DeformableObjectCfg` is specialized by `VolumeDeformableObjectCfg` and
`SurfaceDeformableObjectCfg`; `SoftObjectCfg` and `ClothObjectCfg` remain
compatibility subclasses. `objects/deformable/` owns the common
`DeformableObject`/`DeformableObjectData` contract and the DexSim volume and
surface implementations. Consumers should use `data.nodal_pos_w`,
`data.nodal_vel_w`, `data.nodal_state_w`, `get_surface_vertices()`, and
`get_surface_triangles()`. Legacy soft/cloth methods delegate to that contract.

`SimulationManager` stores both topologies once in `_deformable_objects` and
exposes `add/get_deformable_object()` plus filtered legacy soft/cloth APIs.
Only the Default DexSim backend is registered today and still requires CUDA.
Backend capability flags and `_DEFORMABLE_BACKEND_IMPLEMENTATIONS` reserve the
Newton integration boundary; Newton volume/surface support must remain disabled
until native object and data adapters are implemented and validated.

`BaseEnv._setup_scene()` temporarily constructs the manager headlessly so
the scene can be assembled before a native window is opened. It sets
`SimulationManagerCfg.num_envs` from `EnvCfg.num_envs`.

`SimulationManager` enables physics, selects manual physics updates, prepares
the configured Arena layout, and owns a thin Spawn scene coordinator. With the
Default backend, preparing the Arena layout lets `add_*` materialize native
entities immediately, so articulation metadata and render nodes are available
before finalization. A source-backed articulation added to an eager Default
result is loaded first and then receives its exact-name typed properties on the
live native articulation. Newton defers physical materialization until
`prepare()`: EmbodiChain first reads exact URDF metadata through a disposable
render-only skeleton, applies the source-name overlays, and then builds the
immutable Newton model once. A Viser backend forces `headless=True`; Viser and
the native DexSim window are mutually exclusive.

`SpawnScene` always requests DexSim replication with
`collision_policy="isolated"`. Consequently, when `num_envs > 1`, all
per-environment dynamic, kinematic, and static rigid shapes and every
articulation link shape collide only with entities in the same Arena. Global
`per_env=False` physics resources still collide with every Arena. EmbodiChain
owns this policy choice; DexSim's `ReplicatePlan` and backend adapters own the
effective Default filter data and Newton collision groups. Do not duplicate
the backend-specific group calculation in object facades or task configs.

The default ground plane authors its repeated texture coordinates in the Spawn
render descriptor before materialization, so native and offscreen render paths
receive identical UV data on their first GPU upload.

`SimulationManager.prepare()` is the backend-neutral readiness boundary for
Default CPU, Direct GPU, and Newton. It is idempotent. Topology is committed
only when dirty. Newton source resolution and exact-name configuration precede
the first commit; Default source configuration follows native materialization.
A failed resolver or configurator remains pending and retryable. Runtime
preparation is recorded by committed topology revision: Default CUDA calls
`World.init_gpu_physics()`, while Default CPU and Newton need no additional
manager call after Spawn commit. Facade binding and sensor attachment are
retried on every call; already completed declarations are not reconfigured or
rebound. `init_gpu_physics()` and
`finalize_newton_physics()` remain compatibility aliases, but new code should
call `prepare()`.

Standalone callers must call `prepare()` after their last `add_*()` and before
reading link/joint metadata, object state, or advancing physics. `BaseEnv`
provides this boundary automatically between `_setup_scene()` and
metadata-dependent setup. `SimulationManager.update()` still calls the
readiness path defensively before advancing the requested physics steps.

## Module Boundaries

| Area | Owner | Routed topic |
|------|-------|--------------|
| World, arenas, asset registries, physics update, cleanup | `sim_manager.py` | `simulation-system` |
| Spawn declaration, source resolution, commit/rebuild, and facade binding | `spawn/scene.py`, `spawn/source.py`, `spawn/descriptors.py` | `simulation-system` |
| Backend-neutral batched state/property access | `objects/backends/spawn.py` | `simulation-system` |
| Shared object, render, physics, drive, and URDF configs | `cfg.py` | `configclass-pattern` for config mechanics |
| Rigid, articulation, robot, light, constraint, gizmo | `objects/` | `robot-system` for robots |
| Common deformable contract and DexSim volume/surface adapters | `objects/deformable/` | `sim-visualization` for export |
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

`Articulation.get_parent_joint_chain(link_name)` is the public topology query
for integrations that need link ancestry. It returns immediate-parent-first
`ArticulationJointKinematics` values containing copied names, joint type,
origin, axis, and optional limits. Consumers must not reach into
`BatchEntity._entities` or retain backend-native joint-info objects.

## Configuration Flow

`SimulationManagerCfg.physics_cfg` is the backend selector as well as the
backend config. `PhysicsBackendCfg` owns common timing, device, and gravity;
`DefaultPhysicsCfg`/the compatibility name `PhysicsCfg` add default-backend
scene settings, while `NewtonPhysicsCfg` adds the Newton solver, substeps,
gradient/CUDA-graph behavior, and a grouped `NewtonCollisionPipelineCfg`.
Do not add a second backend string that can disagree with the config type.
Newton's `suppress_warp_kernel_logs=True` suppresses Warp's one-time runtime
banner plus module compile/load chatter during manager startup, build, facade
initialization, and physics updates, then restores the process-wide setting.
It does not suppress DexSim native startup output or genuine Warp/Newton
warnings and errors.

EmbodiChain-authored Newton collision shapes use a default margin and gap of
`0.001 m` each unless an object-specific Newton collision config overrides them.

`EnvCfg` embeds `SimulationManagerCfg` and supplies the control-to-physics
step ratio. CLI and task config loaders may override runtime fields before
constructing the environment. Trace those overrides through the caller rather
than changing a default in the manager blindly.

Object-specific configuration belongs in `lab/sim/cfg.py` or the
corresponding robot/sensor module. Scene composition belongs in
`EmbodiedEnv` or a task config, not in `SimulationManagerCfg`.

Deformable configs use an explicit `deformable_type: volume|surface`
discriminator. Common source mesh and pose fields stay on
`DeformableObjectCfg`; tetrahedral voxelization/soft-body attributes stay on
the volume subclass, and cloth attributes stay on the surface subclass. Do not
add backend conditionals to one monolithic deformable config. Add a backend
implementation at the manager dispatch boundary when its runtime exists.

New rigid-body configs use `RigidBodyPhysicsCfg`, with one slot per physical
concept:

- `mass_props`: `MassPropertiesCfg` (`mass`, `density`, inertia, and COM);
- `rigid_props`: the common `RigidBodyPropertiesCfg` root or a
  `DexsimRigidBodyPropertiesCfg` / `NewtonRigidBodyPropertiesCfg` subclass;
- `collision_props`: the common collision-enable root or a backend subclass;
- `material_props`: common friction/restitution or a backend material subclass.

This follows the IsaacLab property-group/base-subclass pattern while matching
DexSim Spawn's actual ownership. A common quantity is defined once; backend
classes add only native fields. `NewtonRigidBodyPropertiesCfg` is intentionally
empty until DexSim Spawn exposes a Newton-only body property. Every grouped
field defaults to `None`, meaning “do not author this field”; source USD/URDF
values and backend defaults therefore survive partial overlays. Dynamic and
kinematic mass priority is explicit inertia with positive mass, then mass,
then density; static descriptors omit mass properties.

Python callers select a backend by constructing its subclass. Dict/YAML input
uses a local `backend: common|dexsim|newton` discriminator inside the property
group (the unique native fields can also infer it). `to_dict()` emits this
discriminator so typed configs round-trip. Do not mix the deprecated flat
`RigidBodyAttributesCfg` fields with grouped fields in one config or override.

File-backed rigid objects and articulations share one source-independent
physics policy: `asset_physics_mode="preserve"` keeps properties resolved from
the asset, while `asset_physics_mode="overlay"` applies only non-`None`
EmbodiChain fields after DexSim has translated the real materialized source.
This policy applies equally to USD rigid objects and USD/URDF articulations.
Generic `RigidObjectCfg` and `ArticulationCfg` default to `preserve`; `RobotCfg`
defaults to `overlay` to retain its established configured-drive behavior.
`use_usd_properties` remains only as a deprecated compatibility alias (`True`
maps to `preserve`, `False` to `overlay`) and must not be used by new callers.
Import concerns that the source format does not author, such as URDF root
fixation and body scale, remain controlled by their dedicated fields.

`ArticulationRootPropertiesCfg` groups fixed-base and self-collision intent;
its backend subclasses are extension points. `JointDrivePropertiesCfg` owns
portable gains, limits, friction, and armature. Every drive field is optional;
`None` means source-owned, which permits sparse overlays without resetting the
asset's drive mode or unrelated limits. Use the
`NewtonJointDrivePropertiesCfg` subclass only when a Newton `target_mode` is
needed; common effort/velocity/armature values stay on `JointDesc` instead of
being duplicated in both backend blocks. `link_attrs` accepts the same grouped
rigid-body schema for partial per-link overrides.

For articulations, `SimulationManager._declare_spawn_articulation()` supplies
`configure_articulation_desc()` as the source-configuration callback. Preserve
mode leaves source descriptors untouched, while overlay mode applies
exact-name link/joint fields. Default obtains those names from its loaded
native articulation and applies the typed properties live. Newton resolves the
same metadata first and consumes the configured descriptor during its initial
immutable-model build, so initial source configuration must not be implemented
as finalize-then-rebuild. Do not duplicate these writes in
`Articulation._apply_spawn_config()`.

Rigid USD objects follow the same overlay rule: parsed source descriptors are
updated field-by-field, never replaced wholesale by a partial config. The
legacy flat `RigidBodyAttributesCfg` and `RigidBodyAttributesOverrideCfg` live
together in private `_legacy_cfg.py` and are temporarily re-exported by
`cfg.py` so existing imports keep working. They are accepted by the Default
backend only, expose no nested Newton config, and Newton Spawn rejects them
with a grouped-config migration message. New code should use the grouped
schema so “unset” is distinguishable from an authored default and the entire
legacy layer can eventually be removed as one unit.

## Where to Make Changes

| Change | Primary location |
|--------|------------------|
| Global world, renderer, device, arena, or physics lifecycle | `sim_manager.py` |
| Spawn source translation or typed link/joint overrides | `spawn/descriptors.py` plus the DexSim Spawn descriptor/adapter boundary |
| Declaration-to-result binding or retry behavior | `spawn/scene.py` and the object's `bind_spawn()` |
| Batched row/DOF selection or backend property parity | `objects/backends/spawn.py` and the DexSim Spawn batch facade |
| Shared object or physics config type | `cfg.py` |
| Deformable nodal/surface contract or topology-specific buffers | `objects/deformable/` |
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
- Apply Newton collision and articulation-joint configuration to the
  source-translated Spawn descriptors before the first model build; post-bind
  object initialization is only for state and supported live batch properties.
- Manual update is the default; normal environment stepping must advance
  physics through `SimulationManager.update()`.
- Reset only the requested environment rows and honor
  `excluded_uids` for resources detached from automatic reset.
- Keep the `default_mass`, `default_inertia`, and `default_com_pose` values in
  `RigidBodyData`, `ArticulationData`, and `RigidBodyGroupData` as immutable
  initialization snapshots; runtime setters and randomizers must not mutate
  them.
- `destroy()` queues deferred cleanup. Tests and non-exiting standalone
  callers that use `exit_process=False` must call
  `SimulationManager.flush_cleanup_queue()`.
- Resolve articulation ancestry through `get_parent_joint_chain()`; keep
  DexSim topology access encapsulated by `Articulation`.

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
