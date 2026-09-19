# Collision-world integration

Read when changing planner obstacle identity, scene conversion or live pose
binding. Return to [motion planning](motion-planning.md) for other entry points.

## Owners

| Boundary | Source |
|---|---|
| Generic world declaration | `embodichain/lab/sim/motion/planners/base_planner.py` → `CollisionWorldInfo` |
| Option copying and per-attempt binding | `embodichain/lab/sim/motion/motion_generator.py` |
| cuRobo world config, cache and updates | `embodichain/lab/sim/motion/planners/curobo/curobo_planner.py` |
| Physical collision descriptors to native scene | `embodichain/lab/sim/motion/planners/curobo/curobo_yaml.py` |
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

Shared pose conventions belong to [simulation](../simulation-system/simulation-system.md).
cuRobo goal/scene adapters alone convert `xyzw` to native `wxyz`, exactly once;
homogeneous dynamic-pose matrices need no quaternion conversion until then.

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
