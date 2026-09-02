# EmbodiChain Newton Backend Integration Design

> **Status:** Supplementary implementation record, not a normative API
> specification. For current contracts use
> `agent_context/topics/simulation-system/simulation-system.md` and the Sphinx
> simulation guides; code and tests remain the source of truth when this record
> lags them. Dated files under `docs/superpowers/` are historical plans/specs.

This document summarizes the EmbodiChain integration state for the DexSim
Newton physics backend and records remaining work.

Use these EmbodiChain backend names consistently:

- `default`: the existing DexSim default physics backend.
- `newton`: the DexSim Newton physics backend.

Avoid exposing lower-level DexSim implementation names in EmbodiChain-facing
configuration, docs, and conditionals.

## Current State

### Configuration

Backend selection is inferred from `SimulationManagerCfg.physics_cfg`:

- `DefaultPhysicsCfg` selects the `default` backend.
- `NewtonPhysicsCfg` selects the `newton` backend.
- `physics_cfg_for_backend("default" | "newton")` returns the matching config.
- `physics_backend_from_cfg(...)` maps a config instance to its backend name.

`DefaultPhysicsCfg` owns default-backend settings and GPU-memory settings.
`NewtonPhysicsCfg` owns Newton settings: `physics_dt`, `device`, `num_substeps`,
`requires_grad`, `use_cuda_graph`, `debug_mode`, `solver_cfg` (mapping or
`NewtonSolverCfg` selecting `mujoco_warp` / `xpbd` / `semi_implicit` /
`featherstone` / `vbd`), `broad_phase`, and `visualizer_enabled`.
`NewtonPhysicsCfg.to_dexsim_cfg(...)` builds a DexSim `NewtonCfg`, disables
CUDA graph when gradient mode is enabled, and requires
`solver_type="semi_implicit"` for gradient mode.

The typed physics config owns its device default (`cpu` for Default and
`cuda:0` for Newton). `SimulationManagerCfg(device=...)` and legacy
`sim_device=...` are explicit, backend-neutral overrides; omission preserves
the typed config value. Config-backed CLI launchers likewise preserve the
file/backend device unless `--device` is supplied, including honoring an
explicit Newton CPU selection.

### PhysicsBackend abstraction

`SimulationManager` delegates backend-specific behavior to a
`PhysicsBackend` instance held as `self.physics` (selected by `physics_cfg`
type via `physics_backend_from_cfg`). The backend package lives at
`embodichain/lab/sim/physics/`:

```text
embodichain/lab/sim/physics/
    __init__.py    # registry + make_physics_backend(physics_cfg, manager)
    base.py        # PhysicsBackend ABC
    default.py     # DefaultPhysicsBackend  (name = "default")
    newton.py      # NewtonPhysicsBackend   (name = "newton")
```

`PhysicsBackend` is constructed with a back-reference to its owning
`SimulationManager` (an instance member, not a class singleton — this preserves
EmbodiChain's multiton, which IsaacLab's class-singleton approach would break).
The manager delegates through `self.physics.*` instead of branching on a backend
name for operational decisions:

- `configure_world(world_config, sim_config)` applies backend-specific
  `WorldConfig` fields (default tolerances/GPU flags, or `world_config.newton_cfg`).
- `activate(sim_config)` runs post-world-creation setup (default
  `set_physics_config` / GPU-memory config; Newton registration already comes
  from `WorldConfig.newton_cfg`).
- `prepare_spawn_runtime(result)` performs backend runtime work once per
  committed topology revision. Default initializes Direct GPU buffers on CUDA;
  the base implementation is a no-op used by Default CPU and Newton.
- `sync_render_state(result)` publishes bound state without stepping. Newton
  syncs its World-owned runtime to render resources; Default is a no-op.
- `prepare_for_teardown()` releases backend-owned views before Spawn releases
  their native parents.
- `get_scene()` returns the active physics scene.
- `solver_type` and `differentiable_runtime` expose optional runtime services
  without manager-side type checks.

Capability predicates drive the `add_*` guards (see Parity Matrix below):
`supports_robot`, deformable topology flags and their soft/cloth compatibility
aliases, `supports_rigid_object_group`, `supports_rigid_constraints`,
`supports_contact_sensor`, and `can_disable_manual_update`.

`SimulationManager.prepare()` owns the convergent readiness sequence: commit or
rebuild the dirty Spawn scene, apply runtime config, call the backend runtime
hook, bind stable facades, publish render state, and attach deferred sensors.
The legacy `init_gpu_physics()` and `finalize_newton_physics()` methods both
delegate to this same backend-neutral boundary.

Public `SimulationManager` accessors are preserved as thin delegators for
back-compat: `physics_backend`, `is_default_backend`, `is_newton_backend`,
`newton_manager`, `init_gpu_physics()`, `finalize_newton_physics()`,
`get_physics_scene()`.

`newton_manager` is retained only as a compatibility diagnostic: Newton raises
an actionable error because Spawn owns the World-level backend and no independent
`NewtonManager` exists. Scene dirtiness and topology revisions belong to
`SceneBuilder`/finalized `Scene`, not to a second backend lifecycle state machine.

### Object Backend Adapters

Rigid-body and articulation data access is routed through:

```text
embodichain/lab/sim/objects/backends/
    base.py     # Stable RigidBodyViewBase / ArticulationViewBase contracts
    scene.py    # Backend-neutral Scene rigid-body/articulation batch adapters
    newton.py   # Newton-only state synchronization and mimic hooks
```

Normal `SimulationManager` construction binds both Default and Newton facades
to the same Scene batch views. `RigidBodyData` and `ArticulationData` require a
finalized Scene; the raw `PhysicsScene`/native-entity adapter path has been
removed. Scene and its batches own backend dispatch, stable selection
rebinding, and topology revisions. `Scene*View.from_entities()` centralizes the
two public batch-factory calls instead of repeating them in each object facade.

EmbodiChain public rigid-body tensor convention is `(x, y, z, qx, qy, qz, qw)`;
the Scene adapters convert to/from DexSim's `(qx,qy,qz,qw,x,y,z)` batch layout.

Newton rigid-object support includes dynamic/kinematic/static creation, local
pose, body state, linear/angular velocity+acceleration, force/torque at COM,
clear dynamics, reset, COM local pose, mass/friction/inertia-diagonal/
restitution/contact-offset get+set, dynamic/kinematic collision filters, and
visual material/visibility/geometry/scale/user-id APIs. The common Scene view
implements `RigidBodyViewBase`, including contact-offset access.

Static Newton bodies do not have `RigidBodyData`; runtime collision-filter
writes are therefore unavailable and must be configured before materialization.

### Grouped Newton and Default physics attributes

`RigidBodyPhysicsCfg` is the single public schema for rigid-object and
articulation link physics. It separates portable values into `mass_props`,
`rigid_props`, `collision_props`, and `material_props`. Each concept has one
slot, and backend-native values use that slot's concrete subtype; parallel
`default_props`/`newton_props` blocks are not supported. Every field is
optional, so source-authored values survive sparse USD/URDF overlays. The same
partial schema is used by `LinkPhysicsOverrideCfg`, eliminating the former flat
compatibility/override type pair.

Mesh collision construction is owned by `MeshCfg.collision`, whose explicit
approximation selects convex hull, convex decomposition, triangle mesh, or SDF.
Newton SDF/hydroelastic cooking fields live there rather than in rigid-body
physics. Imported articulation links keep their source mesh approximation until
a named source-shape overlay is available.

Spawn compiles these groups into its backend-neutral rigid-body and shape
descriptors, then projects Default- or Newton-specific values at the selected
backend boundary. The remaining raw Default path uses a private
`PhysicalAttr` adapter only at that boundary. User-facing COM quaternions stay
in `xyzw` order; adapters convert to DexSim's `wxyz` order when writing native
attributes and convert back on reads.

### Runtime attribute mutation on Newton

`RigidObject.set_attrs`/`set_damping`/`set_body_type` are no longer warn-and-skip:

- `set_attrs`: when finalized, applies the Newton-supported subset (mass,
  dynamic_friction, restitution, contact_offset) via the batch view and mirrors
  all fields to the attr meta; before finalization, mirrors only.
- `set_damping`: documented runtime no-op that mirrors to meta (Newton does not
  model per-body damping) so `get_damping`/rebuild stay consistent.
- `set_body_type`: no-op with a clearer message — body type is fixed at
  registration on Newton and cannot change at runtime without a rebuild.

`set_mass`/`set_friction`/`set_inertia` use the batch view when finalized; their
not-ready `else` paths mirror the single field to meta on Newton (the default-bound
`get_physical_body().set_*` are not Newton-patched). `Articulation.set_link_physical_attr`
pushes per-link **mass** live on Newton via `set_link_mass` (mirroring the
dedicated `set_mass`); friction/restitution/contact_offset remain rebuild-time-
only for articulation links.

### add_robot / add_articulation on Newton

Robots are URDF articulations; the Newton `load_urdf` patch builds a
`NewtonArticulation`. `add_robot` and `add_articulation` are now **supported** on
Newton (`supports_robot = True`). This required an upstream dexsim fix
(`NewtonArticulation._joint_metas_from_ids`): explicit `joint_ids` were
raw-dict-indexed (including fixed joints) instead of active-joint-indexed,
conflicting with `get_dof()`/`get_actived_joint_names()` and breaking
mimic-jointed robots (dexforce_w1) at spawn. The fix indexes into active joints;
the `joint_ids=None` path is unchanged so existing callers are unaffected. The
dexsim fix lives on dexsim branch `yueci/adapt-embodichain` (commit `d0e86bb02`)
— `add_robot`-on-Newton depends on it being present.

### Backend capability parity matrix

`tests/sim/test_backend_parity.py` is the single source of truth for which
features each backend supports (`BACKEND_CAPABILITIES` table). It pins that each
backend's `supports_*`/`can_disable_manual_update` flags match the table, every
manager feature guards raise `NotImplementedError` iff their capability is
false, and the matrix covers every flag and backend. Current matrix:

| feature                  | default | newton |
|--------------------------|---------|--------|
| robot                    | yes     | yes    |
| soft_bodies              | yes     | no     |
| cloth                    | yes     | no     |
| rigid_object_group       | yes     | yes    |
| rigid_constraints        | yes     | no     |
| contact_sensor           | yes     | no     |
| can_disable_manual_update| yes     | no     |

### Currently Unsupported Newton APIs

`SimulationManager` explicitly rejects these asset types on Newton (per the
parity matrix):

- `add_soft_object(...)`
- `add_cloth_object(...)`
- `create_rigid_constraint(...)`
- `add_sensor(ContactSensorCfg(...))`

Newton does support `RigidObjectGroup`; it is an env-major view over the same
Scene rigid-body batch used by individual objects.

`RigidObject.add_force_torque(pos=...)` ignores `pos` and applies force/torque at
the center of mass.

Newton kinematic pose locking is not complete. The rigid-object test suite keeps
a Newton-specific allowance for kinematic bodies changing after stepping.

Newton SDF rigid mesh support is not validated in EmbodiChain. The SDF rigid
object test is skipped for Newton. Procedural SDF and CoACD geometry is compiled
from `MeshCfg.collision` through the Spawn descriptor path.

Articulation Newton-native **per-link** contact/shape params (`ke`/`kd`/`margin`/
...) are accepted in config but not applied (dexsim `NewtonArticulation` exposes
no per-link contact-material setter); a warning fires at spawn. Common fields are
applied.

### Verified Tests

Newton integration is covered across headless and GPU suites:

```bash
pytest -q tests/sim/objects/test_rigid_object.py
pytest -q tests/sim/objects/test_rigid_object_group.py
pytest -q tests/sim/objects/test_articulation.py::TestArticulationNewton
pytest -q tests/sim/objects/test_robot.py::TestRobotNewton
pytest -q tests/sim/test_physics_attrs.py tests/sim/test_backend_parity.py
pytest -q tests/sim/test_sim_manager.py tests/sim/test_sim_manager_cfg.py
```

Do not copy historical pass counts into this document; report results from the
current checkout and dependency build.

## Improvements To Make

### API Clarity

- Manager-level operational selection is routed through `PhysicsBackend` hooks,
  runtime properties, and capability flags. Backend names remain only for
  diagnostics, explicit implementation registries, and compatibility
  predicates. Object-view `is_newton_backend` checks are adapter-level storage
  and lifecycle distinctions; move them only when a backend-neutral view
  operation can express the same contract without hiding behavior.
- `is_use_gpu_physics` still conflates selected tensor/device location,
  default-backend GPU API availability, and Newton GPU execution; consider
  splitting when a consumer needs to distinguish them.

### Newton Lifecycle

- `SimulationManager.prepare()` is the single readiness API for Default CPU,
  Default CUDA, and Newton. Compatibility aliases delegate to it.
- Track dirty scene/model state more explicitly so mutations after finalization
  can choose between live batch updates and model rebuilds.
- Keep teardown World-owned and deferred; release backend views through
  `prepare_for_teardown()` before closing Spawn/native parents.

### RigidObject

- Implement force-at-position when DexSim Newton exposes the needed API.
- Validate SDF rigid mesh creation and collision behavior on Newton.
- Fix or document kinematic pose-lock semantics.

### Object Groups, Soft, Cloth

- Maintain Newton rigid-object-group parity through the Scene rigid-body batch.
- Keep soft and cloth fail-fast until there is an explicit Newton design and
  test coverage. dexsim exposes `SoftBodyObject`/`add_softbody`/`add_clothbody`
  (requires the VBD solver) — feasible but substantial.

### Articulation / Robot

- Apply Newton-native per-link contact/shape params once dexsim exposes a
  `NewtonArticulation` per-link shape-material setter.
- Add runtime `Articulation.set_link_physical_attr` Newton live push for
  friction/restitution/contact_offset once a live per-link API exists (mass is
  already live).

### Gym Env Integration

Use the backend-neutral readiness boundary after declaring the complete scene:

```python
self.sim.prepare()
```

For stepping, keep the existing high-level flow:

```python
self._preprocess_action(action)
self._step_action(action)
self.sim.update(self.sim_cfg.physics_dt, self.cfg.sim_steps_per_control)
```

For reset, call object/manager reset methods through the normal BaseEnv flow;
do not introduce a backend-specific second initialization path.

## Completion Plan

Done:

1. Single-rigid-object Newton API stabilized; `test_rigid_object.py` green.
2. Backend capability declarations (`PhysicsBackend.supports_*`) drive `add_*`
   guards, pinned by `test_backend_parity.py`.
3. Newton `RigidObject` parity for attributes, damping, body type — implemented
   (`set_attrs` live subset + meta-mirror, `set_damping` no-op+meta,
   `set_body_type` documented no-op).
4. Tests for Newton lifecycle rebuild and runtime property mutation after
   finalization — present (`test_sim_manager.py`, `spawn/test_scene.py`, and
   `test_rigid_object.py::TestRigidObjectNewton`).
5. Newton `RigidObjectGroup` support uses the Scene rigid-body batch and is
   covered by `test_rigid_object_group.py::TestRigidObjectGroupNewton`.
6. Gym environment construction uses the unified `SimulationManager.prepare()`
   boundary after scene declaration.
9. Articulation and robot support on Newton — implemented (incl. upstream
   dexsim joint-active-indexing fix); `TestArticulationNewton` and
   `TestRobotNewton` green.
13. Multi-env parallel simulation on Newton — already complete via the
    spawn-time prototype+clone path (`spawn_rigid_object_entities` /
    `spawn_articulation_entities` → dexsim's `clone_actor_to`,
    Newton-patched). Newton object views accept multi-entity lists and
    resolve one body ID per env. Covered by `TestRigidObjectNewton`
    (`NUM_ARENAS=2`, `test_spawn_clones_distinct_entities`),
    `TestArticulationNewton` (`num_envs=2`), `TestRobotNewton`
    (`num_envs=10`). Implementation plan:
    `docs/superpowers/plans/2026-06-22-newton-backend-pr.md`.
14. Differentiable env for APG — implemented.
    `embodichain.lab.sim.diff` provides `NewtonStepFunc`
    (`torch.autograd.Function`) bridging a `wp.Tape` around
    `DifferentiableStepper` into PyTorch autograd, plus `tape_context`
    and `differentiable_step` helpers. `SimulationManager` gains
    `create_differentiable_stepper` / `create_gradient_rollout`
    delegators. `DifferentiableEmbodiedEnv` validates
    `NewtonPhysicsCfg(requires_grad=True, solver_type="semi_implicit")`
    and overrides `step()` to call `NewtonStepFunc.apply`. The Franka
    FR3 reach APG example (`franka_reach_apg.py`) exercises the bridge
    end-to-end with a Warp action kernel and a Warp reward kernel
    computed inside the tape; `test_franka_apg_smoke_backward` and
    `test_franka_apg_one_iter_loss_reduces` are green. Agent context:
    `agent_context/topics/differentiable-env/`.

    .. note::
        The Franka task uses an FK-bypass step function
        (``newton.eval_fk``) because the ``semi_implicit`` solver does
        not propagate gradient through ``joint_target_pos`` to
        ``body_q``. The default ``_make_step_fn`` still uses the
        differentiable stepper for envs that want the dynamics-grad
        path; see the differentiable-env topic for details.

Remaining:

7. Add rigid-only Newton gym smoke tests.
10. Add soft/cloth support after a dedicated Newton object design and tests.
11. Newton-native per-link contact params for articulations (after dexsim
    exposes a per-link shape-material setter).
12. Full migration off legacy `PhysicalAttr` to dexsim's spawn descriptors
    (Phase 3 follow-up `3b`) — defer until a third backend appears or dexsim's
    attr-path deletion lands.

## Tests To Maintain

Configuration:

- `SimulationManagerCfg(physics_cfg=DefaultPhysicsCfg())` preserves current
  default-backend behavior.
- `SimulationManagerCfg(physics_cfg=NewtonPhysicsCfg())` creates a Newton world.
- `physics_cfg_for_backend(...)` and `physics_backend_from_cfg(...)` return the
  expected backend mapping.

PhysicsBackend abstraction:

- `PhysicsBackend` ABC contract enforced (abstract methods; concrete backends
  implement them). `test_backend_parity.py` pins the capability matrix and the
  `add_*` guard mapping.
- `SimulationManager.prepare()` delegates backend runtime preparation and render
  publication once per topology revision (`test_sim_manager.py`), while
  `SpawnScene`/DexSim own commit and rebuild state (`spawn/test_scene.py`).

Simulation:

- Newton world can be created, finalized, stepped, destroyed, and recreated.
- Default-backend GPU initialization does not run for Newton.
- Newton finalization does not call default-backend GPU fetch/apply APIs.
- Destroying a Newton simulation does not break subsequent default-backend
  simulation creation.

Newton-native attributes (`test_physics_attrs.py`, headless):

- `from_dict` parses local property-slot discriminators; the Spawn compiler
  projects common and Newton-native fields; per-solver warnings (`xpbd` ignores
  `ke`/`kd`; `mujoco_warp` ignores `restitution`) fire correctly.

Rigid object:

- Dynamic/static/kinematic rigid bodies under Newton.
- Pose, velocity, acceleration, force/torque, reset, COM pose, mass, friction,
  inertia, restitution, contact offset, collision filters, geometry APIs behave
  consistently with the documented support matrix.
- Single-slot physics properties and `MeshCfg.collision` spawn through the
  descriptor path; the body registers with the Newton manager after finalize;
  common fields round-trip via the batch view.
- `set_attrs`/`set_damping`/`set_body_type` produce the documented behavior
  (live subset / meta no-op / no-op).

Articulation / Robot:

- `TestArticulationNewton`: control API, setters, drive, per-link mass live via
  `set_link_physical_attr`, remove.
- `TestRobotNewton`: spawn (URDF assembly), finalize, control-part resolution,
  qpos round-trip via the Newton articulation view.

Gym:

- Rigid-only Newton env initializes, steps, resets, and reads observations.

Gradient:

- `requires_grad=True` plus `solver_type="semi_implicit"` can create a gradient
  rollout.
- A simple loss can backpropagate through a rollout without CPU/NumPy observation
  paths.

## Known Risks

- The `add_robot`-on-Newton path depends on the upstream dexsim fix
  (`_joint_metas_from_ids` active-joint indexing, dexsim
  `yueci/adapt-embodichain` `d0e86bb02`). If dexsim is rebuilt from a different
  ref, `supports_robot` would need re-gating.
- dexsim's Newton path hardcodes `density=0.0` in its desc resolver;
  EmbodiChain's Spawn compiler authors a positive configured density on the
  rigid-body descriptor to avoid the mass gap for dynamic bodies without an
  explicit mass and inertia. Watch for dexsim changing this.
- DexSim Newton monkey-patches global classes. Global teardown can affect other
  worlds if used at the wrong time.
- Public body/articulation ID mapping APIs may still need DexSim improvements.
- Newton gravity and contact configuration may not yet match every default-backend
  setting.
- Some object constructors still contain default-backend assumptions such as
  warmup updates; Newton is guarded from those paths.
- Runtime shape/property mutations may require model rebuilds rather than live
  updates; Newton-native per-link contact params are build-time only.
- Newton `RigidObjectGroup` partial reset does not currently restore the
  initialization-time inertia diagonal after the same row's COM orientation was
  mutated at runtime. The focused
  `TestRigidObjectGroupNewton::test_reset_restores_default_mass_properties`
  regression remains open at the DexSim Newton mass-property boundary.
- Standalone and embedded callers should use `destroy(exit_process=False)` plus
  `SimulationManager.flush_cleanup_queue()` after local scene/object references
  unwind; the manager's deferred teardown releases backend views before Spawn
  closes their native parents.
