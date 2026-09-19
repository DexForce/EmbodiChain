# Simulation lifecycle and runtime controls

Read this for manager/Spawn ordering, mutation, reset and native lifetime.
Return to the [simulation overview](simulation-system.md) for ownership.

## Readiness and topology

`SimulationManager.prepare()` in `embodichain/lab/sim/sim_manager.py` owns
readiness; `spawn/scene.py` owns declaration coordination. The ordering is:

1. If unfinalized or dirty, detach parented cameras before rebuilding their
   native parents. Resolve/configure sources and commit Spawn topology.
2. Apply Default articulation root properties before Direct GPU buffers capture
   them. Delegate runtime setup to `PhysicsBackend.prepare_spawn_runtime()`.
3. Bind declared facades in place; publish their state through
   `PhysicsBackend.sync_render_state()` without a physics step.
4. Attach parented cameras and publish the ready topology revision only after
   every stage succeeds.

Newton source metadata/overlays precede its immutable model build; Default
configures materialized sources. Failed source resolution/configuration stays
pending. A later preparation retries incomplete runtime, facade, render or
camera work without unnecessarily recommitting topology. Revision markers
advance after success; never mark a partially constructed facade bound.
`init_gpu_physics()` and `finalize_newton_physics()` are aliases for `prepare()`.

`BaseEnv._setup_scene()` assembles the scene headlessly and propagates the
environment count; the environment prepares before metadata-dependent setup.
Standalone callers declare their entire initial scene before preparing. Viser
may start an empty server during construction, but that must not finalize it.

## Runtime mutations

Runtime rigid-object addition/removal prepares immediately. For same-UID
replacement use `replace_rigid_object()` or `replace_rigid_objects()` rather
than separate removal/addition: configurations are translated before mutation,
then replacements are declared and prepared once. Batch UIDs must be distinct
and already present; an empty batch does not prepare. Returned facades follow
input order. Discard replaced handles; replacements receive new physical
baselines while other assets retain runtime state. Native preparation failure
propagates without rollback. Randomization requiring the replacements runs
after this boundary.

Pending Newton topology/descriptor changes requiring a new immutable model
become usable only after preparation. Keep model rebuilding in Spawn; object
facades must not create a second runtime ownership path.

## Reset and native lifetime

`RigidBodyData`, `ArticulationData` and `RigidBodyGroupData` capture resolved
mass, inertia and COM baselines after materialization. Their leading layouts
are environment, environment/link and environment/object respectively.
`default_mass`, `default_inertia` and `default_com_pose` are initialization
snapshots: setters/randomizers must not mutate them. Reset restores selected
rows before episode initialization events run.

Manager reset restores all selected Newton articulations before one complete
Scene batch clears drive targets, dynamics, warm-start state, kinematics and
external wrenches. MuJoCo-Warp cannot safely clear a selected world's solver
state with only some of its articulations; excluding a subset is rejected.
Other resource exclusions and environment selections remain honored.

`destroy(exit_process=False)` queues native cleanup. Flush with
`SimulationManager.flush_cleanup_queue()` after caller scene/object locals
unwind. `_deferred_destroy()` stops recording/window rendering, calls
`PhysicsBackend.prepare_for_teardown()`, and runs GC while parents still live.
Newton synchronizes its resolved Warp CUDA device and releases render-link
views before Spawn closes skeletons. Cameras and result-scoped facade/batch
references are released before Scene, Arena and World teardown. Instance IDs
must remain unique among live managers; global native-world-count waiting is
safe only once no registered managers survive.

## Newton controls before preparation

Register through the manager; it expands logical UIDs into Arena paths. Tasks
must not reach into `_spawn_scene.builder`.

| Manager API | Cross-module contract |
|---|---|
| `register_kinematic_joint_trajectory()` | Declare articulation first; positions `[env, frames, dof]` use public qpos order; optional root poses `[env, frames, 4, 4]` |
| `register_kinematic_nodal_trajectory()` | Declare deformable with selected nodes inactive; offsets `[env, samples, selected_nodes, 3]` are relative to finalized initial world positions |
| `register_contact_material_schedule()` | Declared rigid object/robot/articulation; piecewise-constant shape friction/stiffness/damping tracks |
| `register_particle_contact_material_schedule()` | Scene-wide particle-versus-rigid contact properties |

All registrations reject finalized scenes. Joint and shape-material controls
remain graph-compatible; nodal trajectories and particle-material schedules
perform host writes and disable CUDA Graph replay. Nodal `fps` enables linear
substep interpolation; without it each substep consumes a sample. Particle
indexing rules are in [deformables](deformables.md).

## Change sites and validation

Use `sim_manager.py` for lifecycle sequencing, `spawn/scene.py` for commit/bind
coordination, `physics/base.py` and backend implementations for runtime hooks,
and `_runtime_controls.py` for nodal controls. `BaseEnv` owns Gym ordering.

`tests/sim/test_sim_manager.py` covers retry without recommit, render sync,
camera attachment, replacement, control registration, reset and teardown.
`tests/sim/spawn/test_scene.py` covers declaration/source/binding state;
`tests/sim/test_runtime_controls.py` covers nodal runtime behavior. Physical
baseline tests live with the corresponding objects under `tests/sim/objects/`.
