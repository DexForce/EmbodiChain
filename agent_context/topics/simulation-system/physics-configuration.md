# Physics configuration and source overlays

Read this for backend resolution and portable-versus-native property ownership.
Return to the [simulation overview](simulation-system.md) for lifecycle entry.

## Backend, solver and device resolution

`SimulationManagerCfg.physics_cfg` selects the backend by config type.
`cfg/simulation.py` owns `PhysicsBackendCfg`, `DefaultPhysicsCfg` and
`NewtonPhysicsCfg`; do not add another backend string to the manager config.
Gym's file-owned selector and composition rules belong to
[environment configuration](../env-framework/configuration.md).

An explicit manager `device` (or legacy `sim_device`) overrides the physics
config device. Omission preserves the backend default. Environment tensors
use `sim.device`; `gpu_id` selects the render GPU and fills an unindexed CUDA
device, not a separate environment tensor device.

Newton `solver_cfg=None` preserves DexSim's scene-aware `AutoSolverCfg`.
The concrete solver is resolved from the complete Spawn scene, then exposed
through `NewtonPhysicsBackend.solver_type`. Do not guess a solver-specific robot
preset while it is unresolved. Gradient mode requires an explicit
`semi_implicit` solver. Read exact solver options/defaults in
`cfg/simulation.py` and the installed DexSim config types; do not substitute a
hardcoded fallback when an expected DexSim API is absent.

`collision_cfg` owns the external Newton contact pipeline. Its
`update_interval=None` updates at the first local solver substep; positive `k`
updates at `0, k, 2k, ...`, reusing contacts between updates. `collision_cfg=None`
omits that pipeline. Solvers using internal contacts (including DexUni and
MuJoCo-Warp with `use_mujoco_contacts=True`) need their own configuration;
external-pipeline tuning does not configure internal contact generation.
The manual differentiable trajectory requires the external pipeline; see
[differentiable environment](../differentiable-env/differentiable-env.md).

Capability checks remain explicit: Default owns native rigid constraints and
Newton deformables are separate. Contact support is rechecked after Newton
solver resolution; DexUni and MuJoCo-Warp on CPU do not expose the supported
ContactQuery path. See `physics/` and [sensors](../sensor-system/sensor-system.md).

## Sparse source physics

`asset_physics_mode="preserve"` retains source physics;
`"overlay"` applies non-`None` config fields after source translation. The
same policy covers USD rigid objects and USD/URDF articulations. Generic rigid
objects/articulations preserve by default; Robot configs overlay their authored
drives. Preserve-mode articulation configs warn about ignored overlay fields.
Root import intent is handled separately in
[articulation adapters](articulation-adapters.md).

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

Newton `condim` belongs in collision descriptors before preparation; friction
coefficients belong in material properties. Use per-link overlays for contact
intent, not finalized solver-array mutations. `has_particle_collision` is
per-shape participation; Arena isolation remains scene-owned.

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
