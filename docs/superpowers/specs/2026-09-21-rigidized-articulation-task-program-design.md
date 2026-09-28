# Rigidized Articulation Task Program Integration

## Objective

Extend PR #632 so a configuration-defined Task Program can use the built-in
`Pick` and `Place` Semantic Calls on an articulation whose joints are locked at
a declared configuration. Add a runnable Rubik's-cube example derived from
`embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/`.

The program remains provider- and embodiment-independent:

```yaml
call:
  kind: pick
  object: rubiks_cube
```

Articulation-specific facts such as the native UID, grasp link, and locked
joint positions remain in the trusted integration. Physical construction,
including the floating root and joint lock, remains in the environment.

## Scope

This change will:

- repair the FK incompatibility introduced by PR #632 after merging current
  `main`;
- move reusable rigidized-articulation geometry construction to the existing
  `atomic_actions/articulation_geometry.py` owner;
- add a typed simulation binding for an articulation exposed semantically as
  a `SceneObjectRef`;
- extend configured Task Program decoding and physical-UID validation;
- reuse the existing built-in `Pick` and `Place` lowering without adding a new
  Semantic Call;
- add a `rubiks_cube_pick_place` configured task based on
  `repeated_pick_place`;
- register the published `demo/RubiksCube.zip` asset bundle;
- add focused asset, configuration, registry, lowering, and geometry tests.

This change will not add cube twisting, dynamic articulation geometry, a new
Task Program language node, or a new Semantic Call. The example resolves
`RubiksCube/rubiks_cube_001.usdc` through the standard EmbodiChain data registry,
matching PR #632's tutorial asset contract.

## Semantic Model

A locked articulation is exposed to Task Program as one semantic object:

- the articulation root UID owns identity and observed pose;
- one selected link supplies link-local grasp mesh geometry;
- the integration transforms that mesh into the articulation-root frame at the
  declared locked configuration;
- the resulting affordance is a direct child of a `SceneObjectRef`;
- `Pick`, held-object state, `Place`, and symbolic effects all use the same root
  frame and object ID.

The physical entity remains an articulation. The semantic type deliberately
does not expose `SceneArticulationRef`, because the built-in object lifecycle
is the desired contract and the movable joint is disabled for this deployment.

## Configuration Contract

Add `rigidized_articulations` to `integration.yaml.scene_binding`:

```yaml
scene_binding:
  contract_id: rubiks_cube_pick_place_scene_v1
  registry_id: task_program_rubiks_cube_pick_place
  rigidized_articulations:
    - entity_id: rubiks_cube
      simulation_uid: rubiks_cube
      locked_qpos:
        top_turn: 0.0
      dynamics: dynamic
      semantic_type: rubiks_cube
      affordances:
        - entity_id: rubiks_cube_grasp
          kind: antipodal_grasp
          grasp_link: lower_two_layers
```

`locked_qpos` belongs to the root binding because it defines the immutable
compound-object configuration. `grasp_link` belongs to the affordance because
different affordances may source different locked links.

The configured decoder remains closed:

- `rigidized_articulations` accepts the ordinary root metadata plus required
  `locked_qpos`;
- `antipodal_grasp` under a rigidized articulation requires `grasp_link` and
  rejects rigid-object-only mesh selection fields;
- `antipodal_grasp` under `rigid_objects` retains its current schema;
- articulation-backed semantic objects must reference a physical
  `simulation.articulation` UID;
- entity and affordance IDs remain globally unique.

## Physical Lock Contract

The Rubik's-cube example uses coincident joint limits for `top_turn` at zero.
This is the first supported Task Program lock mode because it removes the
degree of freedom rather than approximating a lock with an arbitrary positive
stiffness. The root must be floating (`fixed_base: false`).

The reusable adapter will continue to validate:

- every native movable joint appears in `locked_qpos`;
- current qpos matches the declared value in every arena;
- the joint is pinned by coincident limits, or is explicitly held by a
  position drive where the direct Python compatibility path permits it;
- all arenas share one root-to-link transform;
- poses, readings, limits, tolerances, vertices, and triangles are finite and
  structurally valid.

Configured Task Program bindings additionally reject a fixed root. A future
change may admit drive-locked Task Program objects with continuous drift
monitoring; this example does not claim that stronger runtime guarantee.

## Atomic Action Geometry

Add a reusable builder in
`embodichain/lab/sim/atomic_actions/articulation_geometry.py`:

```python
create_rigidized_articulation_antipodal_affordance(
    articulation,
    *,
    grasp_link,
    locked_qpos,
    joint_position_tolerance=1e-3,
    link_transform_tolerance=1e-5,
) -> AntipodalAffordance
```

For an available PK chain, call full-tree named FK with `link_names` and
`qpos_joint_names`; do not also pass `root_link_name`. For a provider without a
chain, derive `T_root_link` from live root and link poses after validating the
locked configuration across all arenas.

`sim_adapter.py` retains the public convenience wrapper that creates
`ObjectSemantics`, but delegates mesh construction and lock validation to the
geometry module. This keeps direct Atomic Skill callers compatible while
giving Task Program access to the affordance payload without constructing an
intermediate `ObjectSemantics` value.

## Simulation Binding and Registry Assembly

Add two immutable binding values:

- `SimulationRigidizedArticulationObjectBinding` for the semantic root;
- `RigidizedArticulationAntipodalGraspBinding` for its link-backed affordance.

Extend `SimulationSceneBinding` with a `rigidized_articulations` collection and
the matching affordance collection. Its provider-free declaration emits:

- `SceneObjectRef(entity_id)` for each rigidized articulation;
- `SceneAffordanceRef` parented by that object;
- the existing grasp capability and `AntipodalAffordance` payload type.

At live build time it:

1. resolves the native entity with `simulation.get_articulation()`;
2. rejects a fixed root;
3. observes the articulation root pose through an object state provider;
4. constructs the root-frame antipodal affordance from the selected link;
5. registers the result as an ordinary semantic object and grasp affordance.

`SceneRegistry.from_simulation()` will accept an explicit mapping for
articulation-backed objects so its existing private simulation state-provider
implementation remains the single owner of root-pose observation. Ordinary
articulations continue to produce `SceneArticulationRef` and joint state.

## Example Task

Add:

```text
embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place/
  env.yaml
  task.ur5.yaml
  task_program/program.yaml
  task_program/integration.yaml
```

The example reuses:

- `components/embodiments/ur5_dh_pgi_140_80.yaml`;
- `components/execution_policies/trajectory_open_loop.yaml`;
- the repeated-pick-place program structure, target cycling, profile defaults,
  action options, and task registration pattern.

`env.yaml` replaces the rigid cube with the PR's Rubik's-cube USDC
articulation, uses `fixed_base: false`, declares the initial `top_turn` value,
and pins its qpos limits at zero. The Scene Engine bottom-center compensation
remains a physical spawn-position concern in `env.yaml`.

The runnable ID will be `TaskProgramRubiksCubePickPlace-v1`. The program picks
the semantic object `rubiks_cube` and places it at configured cyclic targets.

## Failure Behavior

Configuration errors fail before simulator construction where possible:

- a rigidized-articulation UID absent from the physical articulation list;
- missing or malformed `locked_qpos`;
- an antipodal affordance without `grasp_link`;
- unsupported or repeated fields;
- duplicate scene IDs.

Live assembly fails before Task Program execution for:

- an unknown native articulation or link;
- a fixed root;
- incomplete, displaced, unheld, or non-finite joint state;
- invalid mesh topology;
- arena-dependent root-to-link transforms.

Errors identify the semantic entity, native link or joint, and arena where
applicable. No invalid affordance or partial registry is published.

## Validation

Focused tests will cover:

1. geometry with a non-null PK chain and a non-identity root-to-link transform;
2. strict finite tolerance and state validation;
3. configured decoding of `rigidized_articulations` and its negative cases;
4. physical UID validation against `simulation.articulation`;
5. provider-free declaration as `SceneObjectRef`;
6. live registry construction through `get_articulation()`;
7. built-in `Pick` lowering to `GraspGoal` with root-frame geometry;
8. the Rubik's-cube deployment through the read-only deployment inspector;
9. package-data coverage for the new YAML files.

When DexSim is available, physical qualification must also
measure positive cube lift, bounded `top_turn` displacement, successful Place,
and final Task Program acceptance using the automatically downloaded asset.
Static validation does not substitute for that run.

Configured tolerances may only tighten the safe maxima (`1e-3` for joint-state
agreement and `1e-5` for cross-arena root-to-link transforms). Native
joint-limit coincidence is an invariant checked with a fixed internal epsilon;
it is not weakened by either authoring tolerance.

## Compatibility

Existing rigid-object, articulation, link, and affordance configuration stays
unchanged. Built-in Semantic Call schemas stay unchanged. The direct PR #632
Python API remains available through the delegating wrapper. The new YAML field
is additive and closed, so older valid integrations decode identically.

Public binding and geometry exports will be added to the corresponding static
`__all__` lists and API documentation. Project context should be revised only
where the new configured binding materially extends the current Task Program
scene-composition contract.
