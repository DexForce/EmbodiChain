(scene-registry)=

# Scene registry

```{currentmodule} embodichain.lab.task_program.semantics
```

`SceneRegistry` is the canonical integration boundary between semantic scene
identity, atomic-action snapshots, and planner collision worlds. Register an
entity once under an authoritative ID, resolve external names at that boundary,
then use only the canonical ID in semantic calls, snapshots, dependencies, and
dynamic-obstacle configuration.

The registry is an immutable catalog. A {class}`RegistrySceneProvider` created
from it owns changing observation state, publication baselines, and revisions.
This separation lets multiple runtimes share one catalog without sharing their
revision counters.

The registry contract is lab-level. Simulation-specific constructors and
collision checks below are adapters around that contract; perception or
hardware integrations can provide the same state and geometry protocols.

## What the registry owns

Each {class}`SceneEntityRegistration` contains static integration metadata:

- a typed canonical reference;
- aliases for simulator, perception, or hardware names;
- an explicit pose/confidence provider;
- optional parent and backend-local name;
- dynamics and planner collision role;
- optional geometry, semantic type, and affordance data.

A `SceneSnapshot` contains only versioned dynamic pose/confidence values and
collision-world revisions. Snapshot construction copies every entity state, and
each public entity lookup returns a defensive copy. Mutating an original tensor
or a previously returned value therefore cannot change a published snapshot.

References use one flat, globally unique namespace:

```text
SceneEntityRef
+-- SceneObjectRef
+-- SceneArticulationRef
+-- SceneLinkRef
`-- SceneAffordanceRef
```

Do not encode hierarchy into link or affordance IDs. Store ancestry in
`SceneEntityRegistration.parent` and the backend-local member name in
`native_name`. A link parent must be an articulation; an affordance parent may
be an object, articulation, or link. The registry rejects duplicate canonical
IDs, ambiguous aliases, aliases that collide with another canonical ID,
unregistered parents, and typed-reference mismatches. Within one reference
type, a `(parent, native_name)` pair identifies one physical source and cannot
be registered under multiple canonical IDs. The same local name may still be
used under different parents or by different reference types.

String lookups may use an alias and are normalized once:

```python
cube = registry.resolve("sim_cube", expected_type=SceneObjectRef)
assert cube.entity_id == "cube"
```

An already typed reference is expected to contain a canonical ID. It cannot use
an alias or silently change entity kind.

## Author configured scene hierarchy

In `task_program/integration.yaml`, declare an affordance directly beneath its
owning scene entity:

```yaml
scene:
  registry_id: task_program_repeated_pick_place
  rigid_objects:
    - entity_id: cube
      dynamics: dynamic
      semantic_type: cube
      affordances:
        - entity_id: cube_grasp
          kind: antipodal_grasp
```

The two IDs have distinct roles. `cube` is the canonical physical object used
by calls such as `Pick(object="cube")`; `cube_grasp` is the globally unique
canonical ID of one semantic child that can be passed as an explicit grasp or
named by `default_grasp_affordance`. The child ID remains necessary because one
entity may expose multiple affordances and every affordance remains directly
addressable in the flat Scene Registry.

The configured affordance kinds are:

| `kind` | Allowed owner | Kind-specific fields |
|---|---|---|
| `antipodal_grasp` | rigid object | optional `native_name`, `revision`, `relative_pose`, `mesh_env_id`, and `internal_axis` |
| `support_surface` | rigid object, articulation, or link | required `native_name`; optional `object_target_pose`, `minimum_confidence`, and `is_default` |
| `container` | rigid object, articulation, or link | required `native_name`; optional `object_target_pose`, `minimum_confidence`, and `is_default` |

All kinds may also declare `aliases`. Ownership is structural: do not repeat it
with `object_id` or `parent_id`, and do not declare scene-level affordance
collections. The strict integration decoder rejects those forms and derives the
internal parent reference from the containing entity before building the flat,
globally indexed registry.

## Explicit simulation opt-in

Use {meth}`SceneRegistry.from_simulation` to select simulator entities
explicitly. Mapping keys are authoritative registry IDs and values are existing
simulation UIDs. The UIDs are installed as legacy aliases; unlisted simulation
entities are not scanned or imported.

```python
from embodichain.lab.task_program.semantics import SceneObjectRef, SceneRegistry

registry = SceneRegistry.from_simulation(
    sim,
    rigid_objects={
        "cube": "sim_cube",
        "tray": "task_tray_0",
    },
    articulations={"drawer": "cabinet_articulation"},
)

cube = registry.resolve("cube", expected_type=SceneObjectRef)
assert registry.resolve("sim_cube", expected_type=SceneObjectRef) == cube
```

For perception or hardware, construct registrations with an implementation of
{class}`SceneEntityStateProvider` instead. Collision registrations also require
a {class}`SceneGeometryProvider`; the geometry belongs to the catalog even
though the snapshot contains only its current pose and confidence.

## Semantic affordance capabilities

An affordance is a registered direct child of an object, articulation, or link.
Its {class}`SceneAffordanceRef` has its own canonical ID, while `parent` and
`native_name` describe topology and the backend-local member. Semantic calls
select affordances by open, namespaced capabilities rather than by payload class
or declaration order. The built-in capabilities are:

- {data}`GRASP_AFFORDANCE_CAPABILITY` (`affordance.grasp`);
- {data}`PLACE_ON_AFFORDANCE_CAPABILITY` (`affordance.place.on`);
- {data}`PLACE_IN_AFFORDANCE_CAPABILITY` (`affordance.place.in`).

Register a capability-bearing grasp affordance and a scoped parent default as
follows. `antipodal_affordance` is an existing
`AntipodalAffordance` value, for example one produced by the grasp annotation
pipeline:

```python
from dataclasses import replace

import torch

from embodichain.lab.task_program.semantics import (
    GRASP_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
)

object_ref = SceneObjectRef("workpiece")
grasp_ref = SceneAffordanceRef("workpiece.grasp.antipodal")

simulation_registry = SceneRegistry.from_simulation(
    sim,
    rigid_objects={"workpiece": "cube"},
)
object_registration = replace(
    simulation_registry.lookup(object_ref),
    semantic_type="cube",
    default_affordances={GRASP_AFFORDANCE_CAPABILITY: grasp_ref},
)
registry = SceneRegistry(
    (
        object_registration,
        SceneEntityRegistration(
            ref=grasp_ref,
            parent=object_ref,
            native_name="antipodal_grasp",
            affordance=antipodal_affordance,
            affordance_capabilities=frozenset(
                {GRASP_AFFORDANCE_CAPABILITY}
            ),
            affordance_revision="antipodal-v1",
            relative_pose=torch.eye(4),
        ),
    )
)
```

The registry enforces these rules:

- capabilities belong only to an affordance registration;
- a capability-bearing affordance declares an explicit
  `affordance_revision`;
- `affordance.grasp` requires an `AntipodalAffordance` payload;
- `default_affordances` belongs to the parent and maps each capability to one
  compatible direct child;
- an affordance has either a live `state_provider` or a parent-relative
  `relative_pose`, never neither.

{meth}`SceneRegistry.affordances` lists compatible direct children in canonical
ID order without selecting one. {meth}`SceneRegistry.resolve_affordance` applies
one strict selection rule:

1. validate and use the explicit affordance, when supplied;
2. otherwise use the only compatible child;
3. otherwise use the parent's capability-scoped default;
4. otherwise raise {class}`AmbiguousSceneAffordanceError`.

No compatible child, a parent mismatch, or a capability mismatch raises
{class}`UnsupportedSceneAffordanceError`. There is no declaration-order
fallback. Once selected, {meth}`SceneRegistry.object_semantics` creates an owned
atomic-action `ObjectSemantics` value using the canonical object ID and a copied
affordance payload.

{attr}`SceneRegistry.entity_metadata` projects provider-free
{class}`SceneEntityMetadata` values. {class}`SceneManifest` uses the same value
model, so semantic integration can validate IDs, aliases, topology, capability
sets, payload types and revisions, relative affordance poses, and collision mode
before observing the scene. Changing any of that metadata requires rebuilding
and rebinding the semantic integration. Relation grounders dispatch by the exact
capability, payload type, and revision tuple; revisions therefore form part of
the integration contract rather than runtime pose state.

## Publish canonical snapshots

For an atomic-action planning runtime, create the provider through
{meth}`SceneRegistry.make_planning_scene_provider` and pass it to
`SimulationExecutionAdapter`:

```python
provider = registry.make_planning_scene_provider(
    motion_generator,
    batch_size=robot.num_instances,
)
adapter = SimulationExecutionAdapter(
    sim,
    robot,
    scene_provider=provider,
)
```

This factory constructs a fresh provider and eagerly validates the complete
registry/provider/planner collision contract. Use
{meth}`SceneRegistry.make_scene_provider` only for perception and advanced
direct-core consumers that do not need planner agreement. Every factory call
returns an independent provider. Its snapshots contain canonical registry IDs
only; aliases never leak into `SceneSnapshot.entities` or
`collision_entity_ids`.

The provider observes entities in the supplied `env_ids` order. Those IDs must
remain stable and ordered for the provider lifetime, and timestamps must be
monotonic. Translation and rotation thresholds are measured from the last
materially published pose per entity and environment, so repeated
sub-threshold motion eventually publishes a new scene version. Dynamic
collision entities additionally advance per-environment collision revisions.

Parent-relative affordances are derived from the parent pose inside the same
observation. Their static relative transforms remain registry metadata.

## Validate a simulation collision world

Collision setup has one canonical namespace. For a registry-backed
cuRobo world, derive both the explicit `registry_id -> RigidObject` mapping and
the dynamic-obstacle ID list from the registry:

```python
from embodichain.lab.sim.planners import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
    MotionGenCfg,
    MotionGenerator,
)
from embodichain.lab.task_program.semantics import (
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneRegistry,
)

registry = SceneRegistry.from_simulation(
    sim,
    rigid_objects={"cube": "sim_cube"},
    collision_roles={"cube": SceneCollisionRole.DYNAMIC},
    collision_world_mode=SceneCollisionWorldMode.PER_ENV,
)

world = CuroboWorldCfg(
    rigid_objects=registry.collision_geometry_by_id(),
    obstacle_representation="cuboid",
    dynamic_obstacle_names=list(registry.dynamic_collision_entity_ids),
    multi_env=True,
)
motion_generator = MotionGenerator(
    MotionGenCfg(
        planner_cfg=CuroboPlannerCfg(
            robot_uid=robot.uid,
            planner_type="curobo",
            world=world,
        )
    )
)

provider = registry.make_planning_scene_provider(
    motion_generator,
    batch_size=robot.num_instances,
)
```

`collision_geometry_by_id()` derives the cuRobo mapping from the catalog. By
default it includes only `STATIC` and `DYNAMIC` registrations and excludes
`NONE`; an optional exact role filter is available when a backend needs one
subset. `from_simulation()` automatically exposes a selected live rigid object
as its geometry source. Articulations and manually constructed collision
registrations still need an appropriate explicit geometry provider.

The registry validator checks two nested identity contracts before execution:

1. The registry's complete `STATIC ∪ DYNAMIC` collision ID set exactly equals
   `MotionGenerator.collision_world_entity_ids`. This rejects a missing static
   obstacle as well as planner geometry not owned by the registry.
2. The registry's `DYNAMIC` subset exactly equals both the provider's
   `collision_entity_ids` and
   `MotionGenerator.dynamic_collision_entity_ids`.
3. Every ID in that complete collision world has materialized registered
   geometry.
4. The planner supports dynamic collision-world updates when that subset is
   non-empty.
5. The planner's shared/per-environment mode equals the registry mode.

Every collision registration has already proved that geometry exists. Planner
IDs are canonical logical/source IDs, not aliases. For cuRobo `cuboid` and
`mesh` worlds, each mapping key is also the physical YAML obstacle key and the
runtime pose-update key. A `sphere` world instead expands one canonical source
ID into physical YAML names such as `cube_0`, `cube_1`, and so on. Those derived
names are backend details: cache identity and the full-world contract remain
keyed by the canonical source ID, and dynamic sphere obstacles are rejected.

A registry-backed mapping also fails fast when a selected collision source has
no mesh geometry required by its representation. It never silently omits that
canonical ID from generated planner geometry.

When an external perception or hardware provider supplies snapshots, validate
that provider's dynamic subset explicitly instead of constructing a
registry-derived one. The complete registry/planner world check still applies:

```python
registry.validate_collision_integration(
    motion_generator,
    batch_size=batch_size,
    scene_provider=external_scene_provider,
)
```

{class}`SceneCollisionWorldMode` follows this rule:

| Batch and collision setup | Required registry choice | cuRobo setting |
|---|---|---|
| No dynamic collision entities | No mode required | Planner-specific |
| One environment | Omitted mode resolves to `SHARED`; explicit mode also allowed | Match the effective mode |
| Multiple environments | Explicit `SHARED` or `PER_ENV` is required | `multi_env=False` or `True`, respectively |

Choose `SHARED` only when obstacle poses are equal after rebasing every
environment into its robot-base frame. Choose `PER_ENV` for independently
randomized robot-relative layouts.

## Advanced direct-core paths

`RigidObjectSceneProvider` and a list-valued `CuroboWorldCfg.rigid_objects`
remain available to advanced callers that intentionally assemble the atomic
core by hand. The list form derives obstacle names from each object's `uid` (or
an `obstacle_<index>` fallback). It is not the registry-backed path and does not
provide alias normalization or registry/provider/planner construction checks.

See {doc}`index` for manifest and semantic-call integration,
{doc}`../sim/atomic_actions/index` for snapshot grounding and recovery
semantics, and {doc}`../sim/planners/curobo_planner` for cuRobo world
representation and frame details.
