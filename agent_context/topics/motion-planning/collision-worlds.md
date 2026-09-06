# Curobo collision worlds

Read this when the request needs these details. [Topic overview](motion-planning.md).

### CuroboPlanner collision worlds

`CuroboWorldCfg.rigid_objects` accepts either a mapping or a sequence. Use
`Mapping[registry_id, RigidObject]` for a registry-backed integration. The
mapping key is the authoritative logical/source obstacle ID used by the
content-cache key, `collision_world_entity_ids`, and registry validation. For
`cuboid` and `mesh`, it is also the physical YAML obstacle name and dynamic
update key. For `sphere`, one static source expands to physical YAML names such
as `registry_id_0`; dynamic sphere configuration is rejected, while cache and
full-world identity remain keyed by `registry_id`. A registry mapping whose
source lacks mesh geometry required by the selected representation fails fast
instead of silently dropping the source. The sequence form is an advanced
direct-core path that derives names from each object's `uid` or an
`obstacle_<index>` fallback.

`CuroboWorldCfg.multi_env` controls collision-world batching, not whether robot
states or goals are batched:

- `multi_env=False` (default) shares one world. Use it when obstacle poses are
  equal after each simulator-world pose is rebased into its environment's robot
  base. Replicated arenas may have different world-frame offsets and still
  safely share a world when their robot-relative layouts are identical.
- `multi_env=True` allocates one world per batch row. Use it when obstacles have
  different poses relative to their local robot bases, such as per-env pose
  randomization.

The multi-env scene is cloned from the YAML generated using env 0; enabling the
flag does not load distinct initial simulator poses for other rows. Per-env
differences require `"cuboid"` or `"mesh"` representation, registration in
`dynamic_obstacle_names`, and current `(B, 4, 4)` world poses in
`CuroboPlanOptions.dynamic_obstacle_poses`. Independent worlds replicate scene
data and collision caches, so retain the shared default for identical rebased
layouts.

`BasePlanner.collision_world_info` and
`with_collision_world(options, obstacle_poses=...)` form the generic per-plan
dynamic-world bridge. The base property returns `None` and the base hook leaves
options unchanged. `CuroboPlanner` returns an immutable `CollisionWorldInfo`
with updates enabled, clones the supplied pose tensors, and merges them into
`CuroboPlanOptions.dynamic_obstacle_poses`.
`MotionGenerator.supports_dynamic_collision_world` exposes the capability and
`MotionGenerator.bind_collision_world()` owns option copying before forwarding
to the backend hook. Atomic actions use that facade from their framework-owned
`plan()` template when a `SceneSnapshot` declares collision entities;
individual skills must not construct backend obstacle options themselves.
`CollisionWorldInfo` carries the complete canonical world, its dynamic subset,
the `"shared"` / `"per_env"` mode, and update capability as one validated
contract. It requires unique canonical IDs and requires the dynamic subset to
belong to the complete world. `MotionGenerator.collision_world_info` forwards
that contract and retains derived ID/mode properties for callers. For cuRobo,
the complete set is every mapping key (or inferred sequence name), while the
dynamic set is exactly `CuroboWorldCfg.dynamic_obstacle_names`. Sphere-expanded
physical YAML names are not part of either logical ID declaration.
`CuroboWorldCfg` rejects duplicate obstacle names and requires every
`dynamic_obstacle_name` to match an object registered in `rigid_objects`, so a
planner-local mismatch fails before backend construction.

For the canonical path, pass `SceneRegistry.collision_geometry_by_id()` into
`CuroboWorldCfg.rigid_objects`, derive dynamic names from the registry, and call
`SceneRegistry.make_planning_scene_provider(motion_generator, batch_size=...)`
before execution. The geometry mapping excludes `NONE` registrations. The
factory first requires the registry's complete `STATIC ∪ DYNAMIC` set to
equal `MotionGenerator.collision_world_entity_ids`, then requires exact
registry/derived-provider/planner dynamic-subset agreement. It also checks
update capability for a non-empty dynamic set and the same collision-world
batch mode. An external perception/hardware provider instead uses
`validate_collision_integration(..., scene_provider=provider)`.

One environment may infer `SHARED`; a multi-environment registry with dynamic
entities must explicitly choose `SHARED` or `PER_ENV`. Alias normalization
happens before planner construction, so planner IDs must never be simulator
UIDs unless that string is also the chosen canonical registry ID.

`MotionGenerator.resolve_plan_options()` is the corresponding option-ownership
boundary. It copies caller-supplied typed options, otherwise obtains backend
defaults; for TOPPRA it maps the requested sample count and generic
velocity/acceleration limits into `ToppraPlanOptions`. Atomic actions do not
import or branch on concrete planner option types.
