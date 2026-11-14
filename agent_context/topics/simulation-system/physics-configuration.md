# Physics configuration and source overlays

Read this for backend resolution and portable-versus-native property ownership.
Return to the [simulation overview](simulation-system.md) for lifecycle entry.

## Backend, solver and device resolution

`SimulationManagerCfg.physics_cfg` selects the backend by config type.
`cfg/physics.py` owns `PhysicsBackendCfg`, `DefaultPhysicsCfg` and
`NewtonPhysicsCfg`; do not add another backend string to the manager config.
Gym's file-owned selector and composition rules belong to
[environment configuration](../env-framework/configuration.md).

An explicit manager `device` (or legacy `sim_device`) overrides the physics
config device. Omission preserves the backend default. Environment tensors
use `sim.device`; `gpu_id` selects the render GPU and fills an unindexed CUDA
device, not a separate environment tensor device.

`DefaultPhysicsCfg.cache_material` controls native material reuse.
`SimulationManagerCfg.scene_node_capacity` sets World capacity before creation;
`None` retains the native default. Gym accepts the former in `physics_config`
and the latter at the top level through `config_to_cfg()`.

Newton `solver_cfg=None` preserves DexSim's scene-aware `DexUniSolverCfg`.
DexUni resolves its ownership and contact paths from the complete Spawn scene;
the configured solver type is exposed through `NewtonPhysicsBackend.solver_type`.
Do not guess a solver-specific robot preset while scene ownership is unresolved.
Explicit selection uses `dexuni`; `contact_mode="auto"` retains the native
contact-path policy. Expert settings use the native option groups
(`vbd_options`, `mujoco_options`, `collision_options`); unknown solver names
are rejected for mappings and native config objects, and flat DexUni options
are rejected by native validation.
Fluid settings belong under [DexUni's nested DFSPH configuration](fluids.md);
there is no separate fluid solver family. Robot presets select
`newton_dexuni` before the MuJoCo aliases and portable fallbacks. Gradient mode
requires an explicit `semi_implicit` solver or a direct/named provider supporting
the requested gradient mode. Functional Scene calls additionally require its
functional dynamics service. Python callers pass DexSim's
`ProviderNewtonSolverCfg` with a provider class/zero-argument factory directly
through `solver_cfg`; this path does not inspect installed metadata. DexSim
creates fresh instances per model build. Installed `dexsim.newton_solvers`
names pass through to DexSim's `RegisteredNewtonSolverCfg`; unregistered names
remain errors. Read exact solver options/defaults in `cfg/physics.py` and the
installed DexSim config types; do not substitute a
hardcoded fallback when an expected DexSim API is absent.
`mesh_bvh_constructor` selects the mesh and heightfield query acceleration
structure. `None` keeps the native default.

The spring-slider provider lives only in
`scripts/tutorials/sim/simple_newton_solver/`; see its
[tutorial](../../../docs/source/tutorial/simple_newton_solver.rst) for functional
Scene stepping. It is not a packaged solver or registered entry point.

Articulation initialization also calls `reset(clear_dynamics=False)` without a
later coordinated clear. Preserve its position hold targets and mimic target
propagation when the solver accepts them. Query the finalized public
`Scene.dynamics_capabilities().supported_control_fields`; only an explicit
exclusion of `joint_target_q` suppresses that command. Released SDKs predating
extensions retain their unrestricted target behavior. Extension SDKs missing
the query require an upgrade instead of guessing support from solver names.

`collision_cfg` owns the external Newton contact pipeline. Its
`update_interval=None` updates at the first local solver substep; positive `k`
updates at `0, k, 2k, ...`, reusing contacts between updates. `collision_cfg=None`
omits that pipeline. Solvers using internal contacts (including DexUni and
MuJoCo-Warp with `use_mujoco_contacts=True`) need their own configuration;
external-pipeline tuning does not configure internal contact generation.
The manual differentiable trajectory requires the external pipeline; see
[differentiable environment](../differentiable-env/differentiable-env.md).

DexUni preserves its one-way contact policy unless `coupling="two_way"` is
explicit. Two-way feedback requires dynamically integrated MuJoCo joints and
supports full-surface rigid-soft feedback in current DexSim builds. Native
validation owns compatible contact modes and option combinations. Prescribed kinematic trajectories
cannot respond to contact reactions. Forward solver expert options through
`NewtonPhysicsCfg.solver_cfg`; DexSim owns their validation and dispatch.
The current [DFSPH composition](fluids.md) remains one-way even when DexUni's
rigid/soft coupling options are set to two-way.
Articulation gravity compensation and passive damping are separate creation
settings; see [articulation adapters](articulation-adapters.md).
`async_sync`, `rigid_transform_sync_mode`, `capture_acquire_timeout` and
`capture_wait_log_interval` pass straight to DexSim `NewtonCfg`. Solver
diagnostics (graph status, effective VBD options, contact capacity) are read
from `sim.spawn_result.get_solver_diagnostics()` after `prepare()`.

Capability checks remain explicit: Default owns native rigid constraints and
Newton deformables are separate. Contact support is rechecked after Newton
solver resolution. DexUni requires per-row `ContactBuffer` quantity validity;
its one-way CPU MuJoCo path can expose geometry-only contacts, while CPU
MuJoCo force reports are unavailable. See `physics/` and
[sensors](../sensor-system/sensor-system.md).

## Sparse source physics

`asset_physics_mode="preserve"` retains source physics;
`"overlay"` applies non-`None` config fields after source translation. The
same policy covers USD rigid objects and USD/URDF articulations. Generic rigid
objects/articulations preserve by default; Robot configs overlay their authored
drives. Preserve mode keeps the source actor and scale even when saved config
fields contain legacy defaults; explicit actor/scale changes require overlay.
Preserve-mode articulation configs warn about ignored overlay fields.
Root import intent is handled separately in
[articulation adapters](articulation-adapters.md).

Unset USD actor type, scale, root flags and placement retain parsed source
values. An explicit singleton `source_pose_mode` replaces, preserves or offsets
the source transform. Singleton translation resolves a runtime-owned config
copy, preserving caller placement intent for repeated imports and Gym
replacement comparisons. Returned facade configs store resolved absolute poses
in `config` mode, so copying them for replacement retains source offsets and
scene placement without applying them again. Sparse pose keys preserve explicit
zero values; only omitted pose fields retain source placement in `auto` mode.
Non-USD rigid and URDF articulation imports keep their
dynamic/unit-scale and fixed-root defaults respectively. Whole-scene
`add_usd(UsdImportCfg(...))` applies sparse dictionary fields before Spawn
declaration. Source subtree and variants belong to its `source`; namespace and
scene placement belong to the import recipe.

`RigidBodyPhysicsCfg` is the single grouped schema. `mass_props`,
`rigid_props`, `collision_props` and `material_props` each own one physical
concept; concrete subclasses/local backend discriminators carry native fields.
`None` means leave source/backend intent unchanged. `link_attrs` carries the
same partial schema. Never replace a parsed source descriptor wholesale with
a sparse config, and never write link physics for a joint-only overlay.

`MassPropertiesCfg.recompute_inertia` explicitly discards source inertia so
geometry plus effective mass/density can derive it. Explicit inertia and
recomputation are mutually exclusive. Positive mass can preserve a valid
source tensor; density requires recomputation. Invalid source inertia with
collision geometry takes the shared geometry-derived fallback. Default source
capture must preserve replication collision filters while filling mass/COM.
COM adapters follow the [quaternion boundary](simulation-system.md#quaternion-and-pose-convention).

## Geometry and contact policy

`MeshCfg.collision` owns collision approximation and cooking, including Newton
SDF/hydroelastic settings. The strategy is explicit, not inferred from numeric
parameters. Rigid physics and articulation link overlays do not own cooking;
imported articulation geometry retains its source approximation. Parser
compatibility and current algorithm defaults stay in `shapes.py`,
`cfg/rigid_object.py` and `spawn/descriptors.py`, not duplicated here.

Portable `contact_offset/rest_offset` becomes Newton margin/gap:
`rest_offset → margin`, `contact_offset - effective_margin → gap`. Native
margin/gap take precedence; an ambiguous standalone contact offset is rejected
for Newton. Source overlays remain sparse. The compiler separately fills the
application's missing imported-shape gap after source resolution, even in
preserve mode, without replacing explicit zero, margins or materials. Inspect
`spawn/descriptors.py` for these application defaults rather than altering
DexSim global defaults.

Newton `condim`, `contype` and `conaffinity` belong in collision descriptors
before preparation; friction coefficients and raw MuJoCo `solref`, `solimp` and
`solmix` belong in material properties. Explicit `solref` retains its native
positive or direct-mode meaning. New typed declarations use only
`NewtonRigidBodyMaterialCfg`; it owns validation. Legacy dictionary input with
collision-owned raw contact keys migrates once in `RigidBodyPhysicsCfg.from_dict`
with a deprecation warning. Explicit non-None material values win in that block;
after migration, link-local values override global material values normally.
Use per-link overlays for contact intent, not
finalized solver-array mutations. `has_particle_collision` is
per-shape participation; Arena isolation remains scene-owned.

Rigid USD import rejects a parsed disabled body with authored child colliders
but no retained collision pieces. DexSim 0.6.0rc1 has this parser limitation;
use direct compound construction until the dependency retains static-body
collider ownership. Do not change a static fixture into a moving body to mask it.

`DefaultRigidBodyPropertiesCfg.has_gravity` is Default-native configuration,
not a portable cross-solver gravity feature; source assets need overlay mode
to apply it. Runtime `Articulation.set_gravity(enable, env_ids)` currently
delegates to native entities through `objects/backends/controls.py`. It does
not expose per-link selection or a rigid-object equivalent. Do not infer a
Newton capability or automatic reset baseline from this native helper.

## Change sites and validation

`cfg/rigid.py`, `cfg/rigid_object.py`, `cfg/articulation.py` and `cfg/asset.py`
own physics schema and parsing; `shapes.py` owns mesh configuration.
`spawn/descriptors.py` owns translation/merging and
`spawn/source.py` resolves imported metadata. Use `physics/` for manager-level
backend policy. Robot preset selection belongs to `SimulationManager.add_robot()`
and [robot context](../robot-system/robot-system.md).

Focused coverage: `tests/sim/test_rigid_physics_cfg.py`,
`tests/sim/test_sim_manager_cfg.py`, `tests/sim/test_cfg.py`,
`tests/sim/spawn/test_descriptors.py`, `tests/sim/spawn/test_source.py`, and
`tests/sim/objects/test_backend_physics_helpers.py`. Select tests for the affected
property, particularly sparse overlays, inertia fallback and backend rejection.
