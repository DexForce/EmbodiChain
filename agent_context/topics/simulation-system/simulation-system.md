# Simulation System

## Entry Points

| What | Path |
|------|------|
| Public simulation package | `embodichain/lab/sim/__init__.py` |
| World and scene owner | `embodichain/lab/sim/sim_manager.py` → `SimulationManager` |
| Global simulation config | `embodichain/lab/sim/sim_manager.py` → `SimulationManagerCfg` |
| Spawn lifecycle coordinator | `embodichain/lab/sim/spawn/scene.py` → `SpawnScene` |
| EmbodiChain-to-Spawn translation | `embodichain/lab/sim/spawn/descriptors.py` |
| Object and physics configs | `embodichain/lab/sim/cfg/` (public facade: `cfg/__init__.py`) |
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
       → Default may materialize native handles eagerly
       → Newton keeps physical descriptors deferred
  → SimulationManager.prepare()
       → for Newton, resolve source metadata and configure exact-name overlays
       → finalize/rebuild pending Spawn descriptors once
       → for Default, apply pending source overlays to materialized handles
       → apply Default articulation-root runtime properties to native handles
       → prepare manager-owned runtime buffers for the committed revision
       → bind declared EmbodiChain facades in place
       → publish bound state through the backend render-sync hook
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

`destroy(exit_process=False)` queues native cleanup; callers flush that queue
only after their scene/object locals have unwound. During
`SimulationManager._deferred_destroy()`, the manager stops recording and the
native window, invokes `PhysicsBackend.prepare_for_teardown()`, then runs GC
before closing the Spawn result, environment, and World. Default backends use
the no-op hook. Newton synchronizes its resolved Warp CUDA device and clears
its render bridge while Spawn still owns the parent skeletons, so cached link
views cannot be destructed after their native parents.

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

## Quaternion and pose convention

All EmbodiChain-owned public and runtime quaternion tensors use
`(x, y, z, w)` (`xyzw`). A 7D pose or state therefore uses
`(px, py, pz, qx, qy, qz, qw)` (`xyz + xyzw`), and the identity quaternion is
`(0, 0, 0, 1)`. This includes object/root/link/COM state, robot FK and IK,
sensor offsets, manager observations/actions, semantic poses, and task
configuration. `embodichain.utils.math` follows the same convention.

Backend and library adapters must preserve the external API's native order and
convert exactly once at that boundary. DexSim/Spawn rigid and articulation pose
buffers are native `xyzw + xyz`, so their adapters only permute pose layout.
DexSim mass-property and COM descriptors are native `wxyz`, so those adapters
use `convert_quat()` explicitly. Newton/Warp transforms expose position plus an
`xyzw` quaternion and therefore need no component-order conversion. Use a
non-symmetric rotation when testing an adapter; an identity or 180-degree
single-axis rotation can hide an incorrect order.

Deformables use the same public hierarchy for both topologies:
`DeformableObjectCfg` is specialized by `VolumeDeformableObjectCfg` and
`SurfaceDeformableObjectCfg`; `SoftObjectCfg` and `ClothObjectCfg` remain
compatibility subclasses. `objects/deformable/` owns the common
`DeformableObject`/`DeformableObjectData` contract and the DexSim volume and
surface implementations. Consumers should use `data.nodal_pos_w`,
`data.nodal_vel_w`, `data.nodal_state_w`, `get_surface_vertices()`, and
`get_surface_triangles()`. Legacy soft/cloth methods delegate to that contract.
At the Spawn boundary, volume and surface configs translate to DexSim's typed
`SoftBodyDesc` and `ClothDesc` particle-set descriptors. Their Default-native
attributes are carried by `DexsimSoftBodyPhysicsDesc` and
`DexsimClothPhysicsDesc`; volume voxel settings use `SoftBodyMeshingDesc`.

`SimulationManager` stores both topologies once in `_deformable_objects` and
exposes `add/get_deformable_object()` plus filtered legacy soft/cloth APIs.
Only the Default backend is registered today and still requires CUDA.
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
manager call after Spawn commit. After facade binding/reset, the active physics
backend publishes current state to render resources once per committed topology
revision. This is a no-op for Default and invokes Newton's render bridge without
advancing simulation time. Facade binding, render publication, and sensor
attachment remain retryable; already completed declarations are not
reconfigured or rebound. `init_gpu_physics()` and
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
| Backend activation and configured/resolved solver state | `physics/` | `simulation-system` |
| Spawn declaration, source resolution, commit/rebuild, and facade binding | `spawn/scene.py`, `spawn/source.py`, `spawn/descriptors.py` | `simulation-system` |
| Backend-neutral batched state/property access | `objects/backends/spawn.py` | `simulation-system` |
| Shared object, render, physics, drive, and URDF configs | `cfg/` domain modules; `cfg/__init__.py` preserves the public import surface | `configclass-pattern` for config mechanics |
| Rigid, articulation, robot, light, constraint, gizmo | `objects/` | `robot-system` for robots |
| Common deformable contract and DexSim volume/surface adapters | `objects/deformable/` | `sim-visualization` for export |
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

## Configuration Flow

`SimulationManagerCfg.physics_cfg` is the backend selector as well as the
backend config. `PhysicsBackendCfg` owns common timing, device, and gravity;
`DefaultPhysicsCfg` adds default-backend scene settings, while
`NewtonPhysicsCfg` adds the Newton solver, substeps, gradient/CUDA-graph
behavior, and a grouped `NewtonCollisionPipelineCfg`.
Do not add a second backend string that can disagree with the config type.
Leaving `NewtonPhysicsCfg.solver_cfg=None` preserves DexSim's
`AutoSolverCfg` default. A DexSim build exporting `AutoSolverCfg` is required;
EmbodiChain does not substitute a concrete solver. DexSim resolves that
placeholder from the complete Spawn scene during finalization: rigid-only
scenes select XPBD, scenes with an articulation select MuJoCo Warp, and
supported particle families select their matching particle/deformable solver.
A mapping with `solver_type: auto` or
`class_type: AutoSolverCfg` is the explicit equivalent. Gradient mode must
still select `semi_implicit` explicitly because AutoSolver does not choose a
differentiable solver. Before finalization, EmbodiChain treats `auto` as
unresolved; after finalization, `NewtonPhysicsBackend.solver_type` reads the
concrete type from DexSim's World-owned backend.
MuJoCo-Warp mappings may set `enable_multiccd: true`; EmbodiChain forwards it
to DexSim's `MJWarpSolverCfg`, which passes it to Newton `SolverMuJoCo`.
Enabling it changes contact generation (up to four contacts per geometry pair)
without changing the collision geometry authored by EmbodiChain. DexSim must
export an `MJWarpSolverCfg` version that declares the field.
The `open_drawer.py` tutorial combines this option with 20 Newton substeps per
10 ms control step, while keeping its authored robot gains, collision geometry,
pull trajectory, success criteria, and push trajectory identical to Default.
Atomic-action tutorials configure their shared Newton simulation in
`scripts/tutorials/atomic_action/tutorial_utils.py`: they retain 20 substeps
while following Newton's brick-stacking contact profile (`solver=newton`,
`integrator=implicitfast`, 15 solver iterations, 100 line-search iterations,
an elliptic friction cone, `impratio=50`, and the Newton collision pipeline
with contact reduction and an `nxn` broad phase). The shared factory leaves
the Default backend configuration unchanged.
The package dependency must identify the exact DexSim dev build containing
this API; a base `==0.4.3` requirement also accepts older local-version wheels
that do not export `AutoSolverCfg` and is therefore insufficient.
Newton's `suppress_warp_kernel_logs=True` suppresses Warp's one-time runtime
banner plus module compile/load chatter during manager startup, build, facade
initialization, and physics updates, then restores the process-wide setting.
It does not suppress DexSim native startup output or genuine Warp/Newton
warnings and errors.

EmbodiChain-authored Newton collision shapes use a default margin and gap of
`0.001 m` each only when no portable or Newton-native envelope is authored.
`CollisionPropertiesCfg.contact_offset/rest_offset` are portable: Default uses
them directly, while the Spawn compiler maps `rest_offset → margin` and
`contact_offset - rest_offset → gap`. Both values must be present to derive a
Newton gap; an active Newton configuration rejects an ambiguous standalone
`contact_offset` unless a native margin or gap completes the intent. Explicit
`NewtonCollisionPropertiesCfg.margin/gap` values take precedence over this
translation.

`EnvCfg` embeds `SimulationManagerCfg` and supplies the control-to-physics
step ratio. CLI and task config loaders may override runtime fields before
constructing the environment. Trace those overrides through the caller rather
than changing a default in the manager blindly.

Object-specific configuration belongs in the matching `lab/sim/cfg/` domain
module or the corresponding robot/sensor module. Scene composition belongs in
`EmbodiedEnv` or a task config, not in `SimulationManagerCfg`.

Deformable configs use an explicit `deformable_type: volume|surface`
discriminator. Common source mesh and pose fields stay on
`DeformableObjectCfg`; tetrahedral voxelization/soft-body attributes stay on
the volume subclass, and cloth attributes stay on the surface subclass. Do not
add backend conditionals to one monolithic deformable config. Add a backend
implementation at the manager dispatch boundary when its runtime exists.

New rigid-body configs use `RigidBodyPhysicsCfg`. Portable intent is organized
by physical concept:

- `mass_props`: `MassPropertiesCfg` (`mass`, `density`, inertia, COM, and the
  source-inertia recomputation policy);
- `rigid_props`: `DefaultRigidBodyPropertiesCfg`; Newton currently exposes no
  additional body-level property group beyond common mass properties;
- `collision_props`: common collision enablement and the portable
  `contact_offset/rest_offset` envelope, optionally specialized by
  `DefaultCollisionPropertiesCfg` or `NewtonCollisionPropertiesCfg`;
- `material_props`: common friction/restitution or a backend material subclass.

Each concept has exactly one slot. Backend-native fields are represented by the
slot's concrete subclass or its local `backend: default|newton` discriminator;
`default_props` and `newton_props` were removed. Every grouped field defaults to
`None`, meaning “do not author this field”; source USD/URDF values and backend
defaults therefore survive partial overlays. Dynamic and kinematic mass
priority is explicit inertia with positive mass, then mass, then density;
static descriptors omit mass properties.

Mesh collision construction is geometry-owned. `MeshCfg.collision` contains a
`MeshCollisionCfg` with an explicit `convex_hull`, `convex_decomposition`,
`triangle_mesh`, or `sdf` approximation. Strategy-specific fields are validated
when the config is constructed; numerical values never infer the strategy in
the canonical schema. Newton SDF and hydroelastic mesh settings share this
single owner. `RigidBodyPhysicsCfg` and articulation link overlays do not carry
mesh cooking. An imported articulation retains its source mesh approximation
until a named source-shape overlay API is introduced.

`MassPropertiesCfg.recompute_inertia=True` discards source-authored inertia so
the backend derives it from collision geometry and the effective mass or
density. The default `None` inherits an outer per-body overlay and otherwise
preserves source inertia. Explicit inertia and recomputation are mutually
exclusive. The policy lives with mass properties so global articulation,
per-link articulation, and rigid USD overlays share the same behavior;
`LinkPhysicsOverrideCfg` only selects links and carries their partial `attrs`.

Polymorphic collision and material slots use a local
`backend: common|default|newton` discriminator; a unique native field may infer
the subtype. `rigid_props` currently accepts only `backend: default`.
`MeshCfg.from_dict()` temporarily normalizes the deprecated flat
`max_convex_hull_num`, `acd_method`, and `sdf_resolution` inputs to
`MeshCfg.collision` with a deprecation warning. `RigidObjectCfg.from_dict()`
also migrates the former `attrs.mesh_collision_props` input when it has the
owning mesh shape. Serialization emits only the new nested geometry form.

`RigidBodyPhysicsCfg` is the only user-facing rigid-body physics schema.
Flat `attrs` keys such as `mass`, `dynamic_friction`, and `enable_collision`
are rejected at the parsing boundary; place them in `mass_props`,
`material_props`, or `collision_props` instead. `LinkPhysicsOverrideCfg.attrs`
uses the same partial schema, so global and per-link overlays share one model.
COM quaternions in every EmbodiChain config and public runtime API are `xyzw`.
The Spawn/Default adapter alone converts them to DexSim's native `wxyz` order.

Robot configs normally keep these portable values on one ordinary `RobotCfg`.
For a genuine backend-specific asset or actuator difference, subclass
`RobotPresetCfg` and declare complete `default`, `newton`, or
`newton_<solver>` alternatives.
`SimulationManager.add_robot()` derives the
selection from its existing `physics_cfg`, deep-copies the selected complete
robot config, and never merges alternatives. While AutoSolver is unresolved,
only the generic `newton` and `default` alternatives are eligible; do not guess
a solver-specific preset before DexSim has inspected the complete scene. This
is the only robot preset selection boundary; do not add a second backend
selector to robot configs.

File-backed rigid objects and articulations share one source-independent
physics policy: `asset_physics_mode="preserve"` keeps properties resolved from
the asset, while `asset_physics_mode="overlay"` applies only non-`None`
EmbodiChain fields after DexSim has translated the real materialized source.
This policy applies equally to USD rigid objects and USD/URDF articulations.
Generic `RigidObjectCfg` and `ArticulationCfg` default to `preserve`; `RobotCfg`
defaults to `overlay` to retain its established configured-drive behavior.
If an articulation in preserve mode contains explicit `attrs`, `link_attrs`,
`joint_drive_props`, or `qpos_limits`, configuration emits a warning
naming the ignored overlay fields instead of silently discarding them.
Import concerns that the source format does not author, such as URDF root
fixation and body scale, remain controlled by their dedicated fields. An
articulation defaults to `root_props.fixed_base=True` and
`root_props.self_collision_enabled=False`, so both URDF and USD assets are
fixed to the world with self-collision disabled unless configured otherwise.
Setting either field explicitly to `None` preserves the corresponding USD
property and selects the established URDF import default.

`ArticulationRootPropertiesCfg` is the single root-property definition. Spawn
consumes its portable fixed-base and self-collision intent through common
articulation descriptor fields. Its `sleep_threshold`, `min_position_iters`,
and `min_velocity_iters` fields are Default-only: EmbodiChain applies them to
the materialized native articulation before Direct GPU initialization and the
first reset, while Newton ignores them. PhysX Direct GPU runtime setup captures
the articulation solver iteration counts; applying them only during facade
binding leaves the active GPU solver at its source/default values and can make
mimic constraints much softer than CPU. The preparation is idempotent per
Spawn topology revision. The two iteration counts must be configured together
because the Default native API exposes one atomic setter. This remains distinct
from `DefaultRigidBodyPropertiesCfg`, whose same-named values configure
individual rigid bodies or articulation links. `root_props` is the only
root-property interface; `fix_base`, `disable_self_collision`, and the former
flat root solver fields are removed. `JointDrivePropertiesCfg` keeps the
original `drive_type` (`force`, Default-only `acceleration`, or `none`) and adds
the portable actuator `target_mode` (`none`, `position`, `velocity`,
`position_velocity`, or `effort`), stiffness/damping gains, effort/velocity
limits, passive friction, and armature. `ArticulationCfg.joint_drive_props` is
the single joint-property entry point. Every field is optional; `None` means source-owned,
which permits sparse overlays without resetting unrelated source values. If
`target_mode` is unset,
`drive_type="force"` or `"acceleration"` defaults it to `position_velocity`,
while `drive_type="none"` defaults it to `none`.
`NewtonJointDrivePropertiesCfg` is only a serialized configuration
compatibility subtype; new robot definitions use the common class. Common
effort/velocity/armature values stay on `JointDesc` instead of being duplicated
in both backend blocks. `link_attrs` accepts the same grouped rigid-body schema
for partial per-link overrides.

Spawn resolves drive intent per source-resolved joint before lowering it.
Default selects its force/acceleration enum and masks inactive gains; Newton
authors `JointTargetMode` values 0 through 4. `none` and `effort` always clear
both target gains, and `velocity` clears the position gain, so Newton solvers
that ignore `joint_target_mode` still receive deterministic passive,
effort-only, and velocity-only behavior. Non-MuJoCo Newton position mode is an
explicit gain-based emulation that assumes a zero velocity target. An active
`drive_type="acceleration"` is rejected for Newton because it has no exact
equivalent.

For articulations, `SimulationManager._declare_spawn_articulation()` supplies
`configure_articulation_desc()` as the source-configuration callback. Preserve
mode leaves source descriptors untouched, while overlay mode applies
exact-name link/joint fields. Both regex dictionaries and flattened
`(num_dofs, 2)` arrays in `qpos_limits` are compiled into the resolved
joint descriptors before either backend builds. Default obtains those names
from its loaded native articulation and applies the typed properties live.
Newton resolves the same metadata first and consumes the configured descriptor
during its initial immutable-model build, so initial source configuration must
not be implemented as finalize-then-rebuild. Do not duplicate these link/joint
descriptor writes in `Articulation._apply_spawn_config()`; that hook is
reserved for Default-native root setters, finalized Newton runtime adaptation,
and render work requiring finalized resources.

Spawn articulation state IDs follow the final batch `qpos`/`qvel` layout, which
can differ from Newton's source-articulation traversal order. Initial `qpos`,
mimic child/parent metadata, control groups, and every batch mutation must be
mapped by joint name into that state layout. Public `joint_names` uses this
same state-buffer order; use the Spawn handle's source-name query only when
resolving source topology. Newton solvers without configured mimic compliance
project reset positions onto the authored relation before the first step. The
MuJoCo-Warp compliance path preserves the authored current position, matching
Default's initial hand state.
`SpawnArticulationView` filters Newton root-pose rows that already match the
requested translation and rotation before calling the Spawn batch write. This
keeps ordinary fixed-root resets from invalidating a captured CUDA graph while
still forwarding genuine root-pose changes, which refresh Newton solver
constants and recapture the graph as required.
Initialization code that intentionally changes fixed-root poses should do so
after `prepare()` but before the first `update()`, allowing the first Newton
CUDA graph to capture the final anchors instead of immediately invalidating a
graph captured from transient poses.

MuJoCo-Warp lowers URDF mimic joints to native joint equality constraints, but
its default equality solver reference is underdamped compared with Default's
PhysX mimic. During
`Articulation._apply_spawn_config()`,
`_configure_newton_mimic_compliance()` in `objects/backends/newton.py` resolves
only that articulation's constraint rows and approximates Default's natural
frequency/damping ratio with MuJoCo's positive, effective-mass-scaled
`(timeconst, dampratio)` `solref`; the time constant observes MuJoCo's
two-solver-timestep safety floor. The native rows remain enabled, preserving
contact force coupling between follower and leader joints. A very weak
follower drive (one percent of its leader's target gains; `ke=1`, `kd=0.1`
for the W1 hand) stabilizes the equality between solver updates; target
`set_qpos()` and `set_qvel()` writes propagate the authored leader relation to
that drive. Never copy measured follower state or disable
the native equality: doing either turns mimic into an independent servo and
loses the Default backend's mechanical coupling. Other Newton solvers,
gradient mode, and Default retain native behavior. Keep private Newton
runtime/solver access inside this backend helper; the generic `Articulation`
owns state-order metadata, reset behavior, and target propagation only.

DexSim 0.4.3's Newton `RigidBodyBatch.apply_pose()` writes maximal `body_q`
state but does not update the standalone body's reduced FREE-joint state read
by MuJoCo-Warp on the next step. `SpawnRigidBodyView` therefore caches a
`StandaloneRigidStateSync` for its stable batch and projects both Newton state
buffers after pose writes. Invalidate that cache on a Spawn topology revision;
remove the compatibility path once DexSim's public batch operation guarantees
the same synchronization.

The `grasp_cup_to_caffe.py` comparison demo seeds its XY perturbations after
`prepare()` (default seed `0`). This placement makes the scene independent of
random numbers consumed by backend initialization. Pass a negative `--seed`
to restore non-deterministic perturbations.

Rigid USD objects follow the same overlay rule: parsed source descriptors are
updated field-by-field, never replaced wholesale by a partial config. The
former flat `RigidBodyAttributesCfg` and `RigidBodyAttributesOverrideCfg`
types have been removed. New and migrated definitions use the grouped schema,
where `None` means “leave the source/backend value unchanged.”

## Where to Make Changes

| Change | Primary location |
|--------|------------------|
| Global world, renderer, device, arena, or physics lifecycle | `sim_manager.py` |
| Spawn source translation or typed link/joint overrides | `spawn/descriptors.py` plus the DexSim Spawn descriptor/adapter boundary |
| Declaration-to-result binding or retry behavior | `spawn/scene.py` and the object's `bind_spawn()` |
| Batched row/DOF selection or backend property parity | `objects/backends/spawn.py` and the DexSim Spawn batch facade |
| Newton object/runtime adaptation | `objects/backends/newton.py` |
| Shared object or physics config type | Matching domain module under `cfg/`, then re-export from `cfg/__init__.py` |
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
- Keep backend render-state publication free of physics steps. Newton's initial
  state sync must not advance its simulation step or time.
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
