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

Backend selection is inferred from `SimulationManagerCfg.physics_cfg`:

- `DefaultPhysicsCfg` selects the `default` backend.
- `NewtonPhysicsCfg` selects the `newton` backend.
- `physics_cfg_for_backend("default" | "newton")` returns the matching config.
- `physics_backend_from_cfg(...)` maps a config instance to its backend name.

`DefaultPhysicsCfg` owns default-backend PhysX settings and GPU-memory settings.
`NewtonPhysicsCfg` owns Newton settings: `physics_dt`, `device`, `num_substeps`,
`requires_grad`, `use_cuda_graph`, `debug_mode`, `solver_cfg` (mapping or
`NewtonSolverCfg` selecting `mujoco_warp` / `xpbd` / `semi_implicit` /
`featherstone` / `vbd`), `broad_phase`, and `visualizer_enabled`.
`NewtonPhysicsCfg.to_dexsim_cfg(...)` builds a DexSim `NewtonCfg`, disables
CUDA graph when gradient mode is enabled, and requires
`solver_type="semi_implicit"` for gradient mode.

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
name:

- `configure_world(world_config, sim_config)` applies backend-specific
  `WorldConfig` fields (default tolerances/GPU flags, or `world_config.newton_cfg`).
- `activate(sim_config)` runs post-world-creation setup (default
  `set_physics_config` / GPU-memory config, or `get_newton_manager(self._world)`).
- `prepare()` is the unified "force the backend ready-to-step" entry point.
  `SimulationManager.init_gpu_physics()` and `finalize_newton_physics()` both
  delegate to it — Newton's "GPU init" is a finalize; the default's "finalize"
  is a GPU init. Idempotent; after `invalidate()` it re-prepares (rebuilds).
- `ensure_initialized()` is the lazy `update()`-time wrapper (default: lazy GPU
  init; Newton: finalize/rebuild if invalidated).
- `invalidate()` marks the scene dirty after mutation (no-op for default).
- `get_scene()` returns the active physics scene.
- `newton_manager` returns the Newton manager or `None`.

Capability predicates drive the `add_*` guards (see Parity Matrix below):
`supports_robot`, `supports_soft_bodies`, `supports_cloth`,
`supports_rigid_object_group`, `can_disable_manual_update`.

Public `SimulationManager` accessors are preserved as thin delegators for
back-compat: `physics_backend`, `is_default_backend`, `is_newton_backend`,
`newton_manager`, `init_gpu_physics()`, `finalize_newton_physics()`,
`get_physics_scene()`.

Scene mutation invalidates Newton finalization via `_invalidate_newton_physics()`
(delegates to `self.physics.invalidate()`). After finalization,
`_reset_entities_after_finalize()` resets rigid objects, articulations, and
robots so deferred initial state is applied once Newton runtime data is ready.
Rigid object groups are not yet supported on Newton.

### Object Backend Adapters

Rigid-body and articulation data access is routed through:

```text
embodichain/lab/sim/objects/backends/
    base.py     # RigidBodyViewBase, ArticulationViewBase (ABCs)
    default.py  # DefaultRigidBodyView, DefaultArticulationView (PhysX/DexSim-GPU)
    newton.py   # NewtonRigidBodyView,  NewtonArticulationView  (Warp)
```

`*Data` selects the view at construction via `is_newton_scene(ps)` (a duck-type
check). The views implement lazy body-id resolution and a BUILDER-state
entity-level fallback before the Newton model is finalized.

EmbodiChain public rigid-body tensor convention is `(x, y, z, qx, qy, qz, qw)`;
the default adapter converts to/from DexSim's `(qx,qy,qz,qw,x,y,z)`, Newton
needs no conversion.

Newton rigid-object support includes dynamic/kinematic/static creation, local
pose, body state, linear/angular velocity+acceleration, force/torque at COM,
clear dynamics, reset, COM local pose, mass/friction/inertia-diagonal/
restitution/contact-offset get+set, collision filter (dynamic/kinematic/static/
pre-finalize), and visual material/visibility/geometry/scale/user-id APIs.
`apply_contact_offset`/`fetch_contact_offset` were added to
`RigidBodyViewBase` and the Newton view.

Static Newton bodies do not have `RigidBodyData`; static collision-filter writes
use DexSim's per-entity metadata hook when a Newton body ID is not available.

### Newton-native physics attributes (Phase 3)

`RigidBodyAttributesCfg` previously flattened to the legacy PhysX-oriented
`PhysicalAttr` via `.attr()`, so on Newton: Newton-native contact/shape params
(`ke`/`kd`/`margin`/`gap`/`mu_torsional`/...) were not representable, PhysX-only
fields were silently ignored, and `density`/`enable_collision` were dropped.
This is now fixed by adopting dexsim's spawn-descriptor pattern at the EmbodiChain
config layer.

- `cfg.py`: `NewtonCollisionAttributesCfg` (20 fields mirroring
  `dexsim.spawn.descs.NewtonCollisionDesc`, all `Optional`, `None` = keep backend
  default) + a `newton` sub-config on `RigidBodyAttributesCfg` and
  `RigidBodyAttributesOverrideCfg`. `from_dict` parses nested `"newton"`.
  `RigidBodyAttributesOverrideCfg.merged_cfg(base)` returns a merged
  `RigidBodyAttributesCfg` preserving the `newton` sub-config (override non-None
  wins, else base); `merge_with()` keeps its legacy `PhysicalAttr` return for the
  default path. `.attr()` is unchanged (no default-backend regression).
- `physics_attrs.py` (new resolver): `ResolvedNewtonShape(NewtonCollisionDesc)`
  + `resolve_newton_shape` (projects common `dynamic_friction→mu`,
  `restitution`, `enable_collision→has_shape_collision`, `density` — positive so
  dexsim computes a positive body mass) + `resolve_newton_body`
  (`RigidBodyPhysicsDesc.dynamic/static/kinematic`) +
  `resolve_rigid_body_attributes` (dispatch by backend). Re-exports dexsim's
  `NEWTON_CONTACT_SOLVER_FIELDS` / `NEWTON_CONTACT_FIELDS` and ports
  `warn_ignored_contact_fields` (per-solver) + `warn_backend_mismatched_fields`
  (PhysX-only fields on Newton).
- RigidObject spawn (`sim_utils.py`): **opt-in desc-native path** — when
  `is_newton and cfg.attrs.newton is not None`, route box/sphere/CONVEX-mesh
  through `register_mesh_object_to_newton_patch(newton_shape=, newton_body=)`
  (populating the `mgr.dexsim_meta` scaffolding registration/rebuild read),
  bypassing legacy `PhysicalAttr` so Newton-native contact/shape params reach the
  model. SDF and CoACD keep the legacy path this phase. When `attrs.newton` is
  `None`, the legacy `add_rigidbody(attr=)` path is unchanged.
- Articulation: common fields apply via the legacy `set_physical_attr` path on
  BUILDER skeletons; `set_dexsim_articulation_cfg` warns when Newton-native
  per-link fields are set (dexsim's `NewtonArticulation` has no per-link
  contact-material API — see Deferred).

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
not-ready `else` paths mirror the single field to meta on Newton (the PhysX-bound
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
`add_robot/add_soft_object/add_cloth_object/add_rigid_object_group` guard raises
`NotImplementedError` iff its flag is False, and the matrix covers every flag and
backend. Current matrix:

| feature                  | default | newton |
|--------------------------|---------|--------|
| robot                    | yes     | yes    |
| soft_bodies              | yes     | no     |
| cloth                    | yes     | no     |
| rigid_object_group       | yes     | no     |
| can_disable_manual_update| yes     | no     |

### Currently Unsupported Newton APIs

`SimulationManager` explicitly rejects these asset types on Newton (per the
parity matrix):

- `add_soft_object(...)`
- `add_cloth_object(...)`
- `add_rigid_object_group(...)`

`RigidObject.add_force_torque(pos=...)` ignores `pos` and applies force/torque at
the center of mass.

Newton kinematic pose locking is not complete. The rigid-object test suite keeps
a Newton-specific allowance for kinematic bodies changing after stepping.

Newton SDF rigid mesh support is not validated in EmbodiChain. The SDF rigid
object test is skipped for Newton. CoACD-decomposed meshes keep the legacy attr
path on Newton (no `attrs.newton` desc-native routing yet).

Articulation Newton-native **per-link** contact/shape params (`ke`/`kd`/`margin`/
...) are accepted in config but not applied (dexsim `NewtonArticulation` exposes
no per-link contact-material setter); a warning fires at spawn. Common fields are
applied.

### Verified Tests

Newton integration is covered across headless and GPU suites:

```bash
pytest -q tests/sim/objects/test_rigid_object.py
pytest -q tests/sim/objects/test_articulation.py::TestArticulationNewton
pytest -q tests/sim/objects/test_robot.py::TestRobotNewton
pytest -q tests/sim/test_physics_attrs.py tests/sim/test_backend_parity.py
pytest -q tests/sim/test_newton_finalize_lifecycle.py tests/sim/test_sim_manager_cfg.py
```

Recently observed results: Newton rigid (physical_attributes + desc-native
spawn), Newton articulation (incl. per-link mass-live), `TestRobotNewton`
(spawn/finalize/control smoke), 14 headless `physics_attrs` tests, 22 headless
`backend_parity` tests, 5 Newton lifecycle tests, 6 cfg tests — all green. The
default-backend rigid suite (CPU+CUDA) passes with no regression.

## Improvements To Make

### API Clarity

- The `is_newton_scene` sweep is largely complete: backend selection is via the
  `PhysicsBackend` ABC and the `add_*` capability guards; the remaining
  `is_newton_scene` branches in `rigid_object.py`/`articulation.py` are
  legitimate lifecycle fallbacks (BUILDER-state entity dynamics, not-ready meta
  reads, static-object paths) that don't map to the batch-oriented view ABC
  without extending its semantics.
- `is_use_gpu_physics` still conflates selected tensor/device location,
  default-backend GPU API availability, and Newton GPU execution; consider
  splitting when a consumer needs to distinguish them.

### Newton Lifecycle

- `finalize_newton_physics()` (`self.physics.prepare()`) is the single Newton
  preparation API.
- Track dirty scene/model state more explicitly so mutations after finalization
  can choose between live batch updates and model rebuilds.
- Avoid global Newton teardown while another world may still use monkey-patched
  DexSim classes.

### RigidObject

- Implement force-at-position when DexSim Newton exposes the needed API.
- Validate SDF rigid mesh creation and collision behavior on Newton; route SDF
  and CoACD through the desc-native path when `attrs.newton` is set.
- Fix or document kinematic pose-lock semantics.

### Object Groups, Soft, Cloth

- Add Newton rigid-object-group support after a design decision (dexsim has no
  first-class group API).
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
observations when the backend is Newton.

## Completion Plan

Done:

1. Single-rigid-object Newton API stabilized; `test_rigid_object.py` green.
2. Backend capability declarations (`PhysicsBackend.supports_*`) drive `add_*`
   guards, pinned by `test_backend_parity.py`.
3. Newton `RigidObject` parity for attributes, damping, body type — implemented
   (`set_attrs` live subset + meta-mirror, `set_damping` no-op+meta,
   `set_body_type` documented no-op).
4. Tests for Newton lifecycle rebuild and runtime property mutation after
   finalization — present (`test_newton_finalize_lifecycle.py`,
   `test_rigid_object.py::TestRigidObjectNewton`).
6. Gym env init/reset uses `init_gpu_physics()` / `finalize_newton_physics()`
   (already wired via the `base_env.py` pattern).
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

5. Implement and test Newton `RigidObjectGroup` (after a design decision).
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
- The Newton finalize/invalidate lifecycle is owned by `NewtonPhysicsBackend`
  (`test_newton_finalize_lifecycle.py` — headless, patches the rebuild entry
  point).

Simulation:

- Newton world can be created, finalized, stepped, destroyed, and recreated.
- Default-backend GPU initialization does not run for Newton.
- Newton finalization does not call default-backend GPU fetch/apply APIs.
- Destroying a Newton simulation does not break subsequent default-backend
  simulation creation.

Newton-native attributes (`test_physics_attrs.py`, headless):

- `from_dict` parses nested `newton`; `resolve_newton_shape` projects common
  fields (`friction→mu`, `restitution`, `enable_collision→has_shape_collision`,
  `density`); `merged_cfg` propagates `newton`; per-solver warnings
  (`xpbd` ignores `ke`/`kd`; `mujoco_warp` ignores `restitution`) and
  backend-mismatch warnings fire correctly.

Rigid object:

- Dynamic/static/kinematic rigid bodies under Newton.
- Pose, velocity, acceleration, force/torque, reset, COM pose, mass, friction,
  inertia, restitution, contact offset, collision filters, geometry APIs behave
  consistently with the documented support matrix.
- `attrs.newton` set spawns via the desc-native path; body registers with the
  Newton manager after finalize; common fields round-trip via the batch view.
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
- dexsim's Newton path hardcodes `density=0.0` in its desc resolver; EmbodiChain's
  `resolve_newton_shape` sets `density` from the cfg (positive) to avoid the
  desc-path mass gap where dynamic bodies without explicit mass+inertia fail to
  compute a positive body mass. Watch for dexsim changing this.
- DexSim Newton monkey-patches global classes. Global teardown can affect other
  worlds if used at the wrong time.
- Public body/articulation ID mapping APIs may still need DexSim improvements.
- Newton gravity and contact configuration may not yet match every default-backend
  setting.
- Some object constructors still contain default-backend assumptions such as
  warmup updates; Newton is guarded from those paths.
- Runtime shape/property mutations may require model rebuilds rather than live
  updates; Newton-native per-link contact params are build-time only.
- Standalone Newton scripts can segfault during teardown (`sim.destroy()` +
  `teardown_newton_physics()`); pytest's `flush_cleanup_queue` teardown path is
  stable — use the pytest pattern, not bare scripts.
