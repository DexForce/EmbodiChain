# Collision-world integration

Read when changing planner obstacle identity, scene conversion or live pose
binding. Return to [motion planning](motion-planning.md) for other entry points.

## Owners

| Boundary | Source |
|---|---|
| Generic world declaration | `embodichain/lab/sim/motion/planners/base_planner.py` → `CollisionWorldInfo` |
| Option copying and per-attempt binding | `embodichain/lab/sim/motion/motion_generator.py` |
| cuRobo world config, cache and updates | `embodichain/lab/sim/motion/planners/curobo/curobo_planner.py` |
| Collision-shape descriptors to native scene | `embodichain/lab/sim/motion/planners/curobo/curobo_yaml.py` |
| Owned observed query/payload values and scoped motion factory | `embodichain/lab/sim/motion/planners/curobo/payload.py` |
| Complete observed collision roster and fresh cuRobo scope | `motion/planners/curobo/observed.py`, `scene/collision_sources.py` |
| Discrete B1 input-mesh robot/scene queries | `motion/mesh_collision.py`, `scene/_collision_query_inputs.py` |
| Registry/provider agreement | `embodichain/lab/task_program/semantics/scene.py` → `SceneRegistry` |

## Logical identity survives geometry expansion

For registry-backed planning, pass `SceneRegistry.collision_geometry_by_id()`
into `CuroboWorldCfg.rigid_objects`. Mapping keys are canonical obstacle IDs,
used for cache identity, full-world declarations, generated-name prefixes and
dynamic updates. Do not infer IDs from simulator UIDs. The sequence form is a
direct-core option that infers names and is not the registry integration path.

A logical source may expand into several physical collision shapes. Read those
through `RigidObject.get_collision_shapes()`; preserve the source ID while
fanning pose updates through each shape's local transform. Expanded physical
names are not registry IDs. A registered source lacking collision geometry
fails instead of disappearing from the world.

`curobo_yaml.py` keeps supported primitives analytic and converts mesh-backed
shapes to convex-hull ESDF voxels. Direct native mesh entries are unsupported.
Dense allocations obey `max_voxel_count`, and representation changes require
cache-version invalidation. Compatibility-only options must not silently regain
old representation-selection behavior; inspect their config definitions.

Generated robot YAML carries cuRobo self-collision exemptions and per-link
padding. Default exemptions join fixed rigid components and their immediate
moving-joint neighbors; a complete authored mapping overrides that topology.
No sampled collision or task/model name creates an exemption. Generation and
runtime completion share `_self_collision.py`, preserve authored maps, and
invalidate older generated caches. `CuroboPlannerCfg.self_collision_check`
enables the native cost by default in planning and exact-sample validation;
False is an explicit opt-out. The validator makes each sample contiguous and
uses the same configured cuRobo joint-bound, self-collision and world costs.
It passes `KinematicsState` to the world cost to avoid the incompatible 0.8
`validate()` wrapper. Reuse an existing backend with the same control part and
batch size; only cold validation creates a joint backend.

Shared pose conventions belong to [simulation](../simulation-system/simulation-system.md).
cuRobo goal/scene adapters alone convert `xyzw` to native `wxyz`, exactly once;
homogeneous dynamic-pose matrices need no quaternion conversion until then.

## Observed query snapshots

`CuroboQuerySnapshot` owns full observed joint positions and an explicit
`world_to_base` transform. Its optional `CuroboPayload` owns object identity,
control part, local sphere envelope, explicit `touch_links` permissions and
same-observation world pose. Unlisted robot links retain self-collision checks. These are
runtime observations, not serialized configuration. All world providers must be
`CollisionObjectSnapshot` values projected with the same transform; the planner
does not reread live object or robot-base poses in this mode.

The opt-in supports one control part, disabled CUDA graphs, a static shared
world and preserved native samples. Without payload it supports the complete
robot environment batch in canonical row order through the existing native
BatchMotionPlanner; subsets, reordered environments and per-env worlds remain
unsupported. An omitted start uses the captured selected joint rows. The motion facade reports the world as immutable. Full qpos and world_to_base carry matching batch axes. Passive
joint locks must be identical across rows and use the observed values in both
planning and the no-payload exact-sample checker. The shared collision source
helper verifies that all supplied row poses project to the same planning-frame
world; its shape descriptors declare geometry shared by all rows. Payloads
remain B=1 and attach only to distinct buffers owned by the fresh backend's IK/optimization/
metric models, with alias deduplication and cleanup on failure/close. Arbitrary
trajectory validation with payload is unavailable because its separate checker
does not own those attachments; direct calls fail closed. Native self-collision defaults apply to both observed and ordinary planners.

`curobo_motion_generator()` provides a context-managed standard facade and makes
no phase decisions. Callers must choose explicit backend velocity/acceleration
options when switching strategies; a fixed-grid policy's absent acceleration
budget is not a substitute for a selected backend budget. Returned trajectories
retain ownership after closing. Native cuRobo optimizer and graph-planner
callbacks can keep device tensors in Python reference cycles after close.
Hosts requiring a measured memory boundary must first leave the query scope,
release native handles, then collect cyclic garbage and synchronize the owning
device. Close does not run a process-wide GC or empty the CUDA allocator cache;
reserved memory is not the same as live tensor allocation. See the observed-query
lifecycle regression in `tests/sim/motion/planners/test_curobo_integration.py`.
Input mesh envelopes and planner spheres do not establish native cooked
equivalence, CCD or physical task acceptance.

## Shared versus per-environment worlds

`multi_env` controls collision-world replication, not batched robot goals.
Compare obstacle poses after rebasing into each environment's robot base:
identical layouts can share one world even when arena world offsets differ.
Different robot-relative layouts require independent worlds.

Enabling `multi_env` clones the cached env-0 scene; it does not automatically
sample distinct initial poses for other environments. Objects differing by row
must be declared dynamic and supplied current `(B, 4, 4)` world poses. Keep the
cached raw scene representation until the backend's multi-env constructor has
materialized it; inspect `_materialize_multi_env_scene_model()` for that boundary.

## Bind and validate before execution

`CollisionWorldInfo` declares unique complete IDs, their dynamic subset, batch
mode and update capability. `BasePlanner.with_collision_world()` is the generic
hook; `MotionGenerator.bind_collision_world()` copies options before invoking
it. cuRobo clones incoming pose tensors into its typed options. Atomic Skills
bind snapshots through the facade on each planning attempt rather than building
backend-specific obstacle options inside individual skills.

For the canonical registry path, derive dynamic names from the registry and call
`SceneRegistry.make_planning_scene_provider()` before execution. Validation
requires exact agreement of the complete `STATIC ∪ DYNAMIC` world, dynamic
registry/provider/planner subsets, update capability and batch mode. `NONE`
registrations are excluded. External perception/hardware providers use
`validate_collision_integration(..., scene_provider=...)` at the same boundary.

A single environment may infer shared mode. Multi-environment registries with
dynamic entities must choose shared/per-env explicitly. Canonical aliases resolve
before planner construction; duplicate or undeclared dynamic IDs fail early.
This validation prevents apparently successful plans against stale/incomplete
worlds, which cannot be repaired by retrying with different motion parameters.

## Focused validation

Use `tests/sim/motion/planners/test_curobo_planner.py` for config, conversion,
cache and live-update behavior; `test_curobo_integration.py` in that directory
covers the backend integration. Registry/provider contracts are exercised under
`tests/lab/task_program/semantics/`. Include logical-ID/compound-shape and
shared/per-env cases when changing geometry or dynamic binding.

## Explicit B1 source-compatible query scopes

`ObservedCuroboQueryFactory` owns one fresh static-world backend and verifies the
explicit generated-URDF root identity. Its geometry reader keeps the complete
ordered roster, including collision-only links; missing selected poses reject.
Payload exclusions are explicit and snapshot outputs survive backend close.
The existing batched no-payload snapshot path remains separate and unchanged.

`SimulationFreeMotionQuery` captures current input geometry per discrete query;
`RobotMeshCollisionSnapshot` and `RobotArticulationCollisionSnapshot` own narrower
scopes. Precontact reports explicitly identify their excluded rigid owner.
`robot_collision_query_scope` reuses only the exact current robot/model identity
and restores nested scopes. These APIs retain B1 limits and source pair order.
These queries use the shared Scene `NewtonCollisionChecker` and isolated
`newton.CollisionPipeline`. Newton must be available before robot/scene capture.
Single-arm input trajectories keep other robot branches at their observed joints;
they cannot certify an uninspected moving arm. Convex inputs use the native
convex route. Explicit closed triangle inputs preserve nonconvex topology and
use triangle/BVH pairs for the whole query, including containment. Open solids,
implicit triangle/primitive mixtures, precision or buffer exhaustion are
unsupported. Motion queries use the robot device, with CUDA graph replay on CUDA.
Free queries recapture model, shape and pose inputs each call; only equal
geometry/filter declarations reuse native buffers. Snapshots build geometry once;
only owned transforms change, and close releases native resources. No physics step or backend fallback occurs.
