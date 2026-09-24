# Rigidized Articulation Task Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a joint-locked floating articulation available as an ordinary Task Program object, then ship a configured UR5 example that repeatedly picks and places the Rubik's-cube articulation from PR #632.

**Architecture:** The atomic-action geometry owner validates the physical lock and converts one link-local mesh into articulation-root coordinates. A new typed simulation binding exposes that native articulation as `SceneObjectRef`, while configured decoding and component composition keep the YAML closed and validate that the selected UID belongs to the physical articulation collection. Existing built-in `Pick` and `Place` lowering then operate unchanged on the root-frame affordance.

**Tech Stack:** Python 3.10+, PyTorch, immutable dataclasses, EmbodiChain Task Program semantics and simulation integration, strict YAML component decoding, pytest, Black 26.3.1, Sphinx API coverage checks.

**Spec:** `docs/superpowers/specs/2026-09-21-rigidized-articulation-task-program-design.md`

## Global Constraints

- Preserve the direct PR #632 Python API `create_rigidized_articulation_antipodal_semantics`; it must delegate to the reusable geometry builder.
- Keep `program.yaml` provider-independent and use the existing built-in `pick` and `place` calls with `rubiks_cube` as a `SceneObjectRef`.
- The configured Task Program lock mode requires a floating articulation root and coincident joint limits at every declared `locked_qpos` value.
- Configured tolerances may only tighten the safe maxima (`1e-3` for joint-state agreement and `1e-5` for cross-arena transforms); joint-limit coincidence uses a fixed internal epsilon.
- `locked_qpos` is owned by the rigidized-articulation root binding; `grasp_link` is owned by its nested antipodal affordance.
- The Rubik's-cube asset contract remains `RubiksCube/rubiks_cube_001.usdc`; register the published `demo/RubiksCube.zip` bundle so standard config resolution downloads it into the EmbodiChain data root.
- Do not add cube twisting, dynamic articulation geometry, a new Task Program language node, or a new Semantic Call.
- Preserve existing rigid-object, articulation, link, placement-affordance, and direct Atomic Skill behavior.
- New source files require the DexForce 2021-2026 Apache 2.0 header, `from __future__ import annotations`, public type annotations, static `__all__`, and Google-style docstrings.
- Run `black==26.3.1` with `black .` before every commit.

## Review Focus

1. A real articulation with a non-null PK chain must use full-tree named FK and must not pass `root_link_name`; Task 1 pins this with a spy provider that rejects that argument.
2. NaN or infinite tolerances, joint readings, joint limits, poses, vertices, and triangle topology must fail before an affordance is published; Tasks 1 and 3 pin the relevant layers.
3. A rigidized-articulation semantic UID must resolve only from `simulation.articulation`, never from `background` or `rigid_object`; Task 4 pins the cross-category failure.
4. A fixed root, incomplete lock, displaced joint, or arena-dependent root-to-link transform must fail during live binding assembly; Tasks 1 and 3 pin these failures.
5. The example must remain discoverable from an installed wheel, resolve the registered Rubik's-cube bundle, and apply its configured coincident joint limits through `asset_physics_mode="overlay"`.

---

### Task 1: Extract and Harden Rigidized-Articulation Geometry

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/articulation_geometry.py`
- Modify: `embodichain/lab/sim/atomic_actions/sim_adapter.py`
- Modify: `embodichain/lab/sim/atomic_actions/__init__.py`
- Modify: `tests/sim/atomic_actions/test_rigidized_articulation_semantics.py`

**Interfaces:**
- Consumes: `Articulation`, `AntipodalAffordance`, and the existing lock/mesh behavior introduced by PR #632.
- Produces: `create_rigidized_articulation_antipodal_affordance(articulation: Articulation, *, grasp_link: str, locked_qpos: Mapping[str, float], joint_position_tolerance: float = 1e-3, link_transform_tolerance: float = 1e-5) -> AntipodalAffordance`.
- Preserves: `create_rigidized_articulation_antipodal_semantics(...) -> ObjectSemantics` with its current public signature and result fields.

- [ ] **Step 1: Add a failing non-null-PK-chain regression test**

Add a PK-enabled subclass of the existing `_StubArticulation`. Its `compute_fk` signature intentionally omits `root_link_name`, so the regression fails if that incompatible keyword is supplied:

```python
class _PkArticulation(_StubArticulation):
    pk_chain = object()

    def compute_fk(
        self,
        qpos: torch.Tensor,
        *,
        link_names: tuple[str, ...],
        qpos_joint_names: tuple[str, ...],
    ) -> torch.Tensor:
        self.requested_link_names = tuple(link_names)
        self.requested_qpos_joint_names = tuple(qpos_joint_names)
        assert qpos.shape == (1, 1)
        return torch.stack(self._root_to_link).unsqueeze(1)


def test_rigidized_articulation_uses_full_tree_named_fk() -> None:
    root_to_link = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    articulation = _PkArticulation(
        root_to_link=root_to_link,
        qpos={"top_turn": 0.0},
    )

    semantics = create_rigidized_articulation_antipodal_semantics(
        articulation,
        grasp_link="top_layer",
        locked_qpos={"top_turn": 0.0},
    )

    assert articulation.requested_link_names == ("top_layer",)
    assert articulation.requested_qpos_joint_names == ("top_turn",)
    homogeneous = torch.cat(
        (
            LINK_VERTICES.to(torch.float64),
            torch.ones(len(LINK_VERTICES), 1, dtype=torch.float64),
        ),
        dim=1,
    )
    expected = (homogeneous @ root_to_link.transpose(0, 1))[:, :3]
    assert torch.allclose(
        semantics.grasp_affordance.mesh_vertices,
        expected,
    )
```

- [ ] **Step 2: Run the PK-chain regression and confirm the current incompatibility**

Run: `pytest -q tests/sim/atomic_actions/test_rigidized_articulation_semantics.py::test_rigidized_articulation_uses_full_tree_named_fk`

Expected: FAIL because the PR implementation combines `root_link_name` with `qpos_joint_names`, which `Articulation.compute_fk` rejects.

- [ ] **Step 3: Add the reusable affordance builder and delegate from the compatibility wrapper**

Move `_joint_readings`, root-to-link calculation, lock validation, mesh validation, and link-mesh transformation from `sim_adapter.py` into `articulation_geometry.py`. Add the exact public function below and use full-tree named FK when `pk_chain` is available:

```python
def create_rigidized_articulation_antipodal_affordance(
    articulation: Articulation,
    *,
    grasp_link: str,
    locked_qpos: Mapping[str, float],
    joint_position_tolerance: float = 1.0e-3,
    link_transform_tolerance: float = 1.0e-5,
) -> AntipodalAffordance:
    """Build root-frame antipodal geometry for a rigidized articulation link."""
    joint_tolerance = _positive_finite_tolerance(
        joint_position_tolerance,
        field_name="joint_position_tolerance",
    )
    transform_tolerance = _positive_finite_tolerance(
        link_transform_tolerance,
        field_name="link_transform_tolerance",
    )
    locked = _validated_locked_qpos(articulation, locked_qpos)
    _assert_link_is_rigid_to_root(
        articulation,
        grasp_link=grasp_link,
        locked_qpos=locked,
        joint_position_tolerance=joint_tolerance,
    )
    root_to_link = _articulation_root_to_link(
        articulation,
        grasp_link=grasp_link,
        locked_qpos=locked,
        link_transform_tolerance=transform_tolerance,
    )
    vertices, triangles = articulation.get_link_vert_face(grasp_link)
    return AntipodalAffordance(
        mesh_vertices=_transform_points(root_to_link, vertices),
        mesh_triangles=_validated_triangles(triangles, vertex_count=vertices.shape[0]),
    )
```

For the FK path, call:

```python
poses = articulation.compute_fk(
    qpos,
    link_names=(grasp_link,),
    qpos_joint_names=tuple(locked_qpos),
)
root_to_link = poses[:, 0]
```

Do not supply `root_link_name`. Keep live-pose fallback only when `articulation.pk_chain is None`. Rewrite the existing public wrapper as:

```python
affordance = create_rigidized_articulation_antipodal_affordance(
    articulation,
    grasp_link=grasp_link,
    locked_qpos=locked_qpos,
    joint_position_tolerance=joint_position_tolerance,
    link_transform_tolerance=link_transform_tolerance,
)
return ObjectSemantics(
    geometry=geometry,
    grasp_affordance=affordance,
)
```

Import the builder into `sim_adapter.py`, remove its duplicate private helpers, and export it from `atomic_actions/__init__.py` and `articulation_geometry.__all__`.

- [ ] **Step 4: Add strict finite-value and transform-consistency tests**

Add these parameterized tests to `tests/sim/atomic_actions/test_rigidized_articulation_semantics.py` using its existing `_StubArticulation`:

```python
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field_name",
    ["joint_position_tolerance", "link_transform_tolerance"],
)
def test_rigidized_affordance_rejects_non_finite_tolerance(
    field_name: str,
    value: float,
) -> None:
    kwargs = {
        "grasp_link": "lower_two_layers",
        "locked_qpos": {"top_turn": 0.0},
        field_name: value,
    }
    with pytest.raises(ValueError, match=field_name):
        create_rigidized_articulation_antipodal_affordance(
            _StubArticulation(
                root_to_link=torch.eye(4),
                qpos={"top_turn": 0.0},
            ),
            **kwargs,
        )


def test_rigidized_affordance_rejects_arena_dependent_link_transform() -> None:
    first = torch.eye(4)
    second = torch.eye(4)
    second[0, 3] = 0.01
    articulation = _StubArticulation(
        root_to_link=[first, second],
        qpos=[{"top_turn": 0.0}, {"top_turn": 0.0}],
    )

    with pytest.raises(ValueError, match="root-to-link.*arena"):
        create_rigidized_articulation_antipodal_affordance(
            articulation,
            grasp_link="lower_two_layers",
            locked_qpos={"top_turn": 0.0},
        )
```

Extend the existing PR tests so non-finite qpos, non-finite limits, non-finite poses, non-finite vertices, floating triangles, and out-of-range triangle indices each raise a field-specific `TypeError` or `ValueError`. Preserve tests for incomplete `locked_qpos`, displaced joints, and non-pinned limits/drives.

- [ ] **Step 5: Run the focused atomic-action suite**

Run: `pytest -q tests/sim/atomic_actions/test_articulation_geometry.py tests/sim/atomic_actions/test_rigidized_articulation_semantics.py`

Expected: PASS, including both the non-null-chain and live-pose fallback paths.

- [ ] **Step 6: Format and commit the geometry unit**

Run: `black .`

Run: `git diff --check`

Commit:

```bash
git add embodichain/lab/sim/atomic_actions/articulation_geometry.py embodichain/lab/sim/atomic_actions/sim_adapter.py embodichain/lab/sim/atomic_actions/__init__.py tests/sim/atomic_actions/test_rigidized_articulation_semantics.py
git commit -m "fix(atomic): support rigidized articulation root geometry"
```

### Task 2: Register Articulations as Semantic Objects

**Files:**
- Modify: `embodichain/lab/task_program/semantics/scene.py`
- Modify: `tests/lab/task_program/semantics/test_scene.py`

**Interfaces:**
- Consumes: a `SimulationManager` that provides `get_articulation(uid)` and native articulation handles that provide `get_local_pose(to_matrix=True)`.
- Produces: `SceneRegistry.from_simulation(..., articulation_objects: Mapping[str, str] | None = None, ...) -> SceneRegistry`.
- Guarantees: articulation-backed objects receive `SceneObjectRef`, root-pose observation, optional supplied geometry, no articulation joint-state provider, and the same alias/collision-role handling as rigid objects.

- [ ] **Step 1: Add failing registry tests for articulation-backed objects**

Add the following cases to `tests/lab/task_program/semantics/test_scene.py`:

```python
def test_from_simulation_registers_articulation_as_object() -> None:
    native = _SimulationEntityStub(uid="native_cube")
    simulation = _SimulationStub(articulations={"native_cube": native})

    registry = SceneRegistry.from_simulation(
        simulation,
        articulation_objects={"rubiks_cube": "native_cube"},
    )

    registration = registry.lookup(SceneObjectRef("rubiks_cube"))
    assert registration.aliases == ("native_cube",)
    assert registration.joint_state_provider is None
    assert simulation.articulation_lookups == ["native_cube"]
    assert simulation.rigid_object_lookups == []


def test_from_simulation_rejects_duplicate_articulation_object_id() -> None:
    simulation = _SimulationStub()
    with pytest.raises(ValueError, match="globally unique"):
        SceneRegistry.from_simulation(
            simulation,
            rigid_objects={"cube": "rigid_cube"},
            articulation_objects={"cube": "articulated_cube"},
        )
```

Also cover duplicate IDs against ordinary `articulations`, unknown collision/geometry keys, non-mapping input, missing `get_articulation`, and `get_articulation()` returning `None`.

- [ ] **Step 2: Run the new registry tests and confirm the missing keyword**

Run: `pytest -q tests/lab/task_program/semantics/test_scene.py -k articulation_as_object`

Expected: FAIL with `TypeError` because `articulation_objects` is not yet accepted.

- [ ] **Step 3: Extend `SceneRegistry.from_simulation` with an explicit mapping**

Add the keyword-only parameter and normalize it independently:

```python
@classmethod
def from_simulation(
    cls,
    simulation: SimulationManager,
    *,
    rigid_objects: Mapping[str, str] | None = None,
    articulation_objects: Mapping[str, str] | None = None,
    articulations: Mapping[str, str] | None = None,
    collision_roles: Mapping[str, SceneCollisionRole] | None = None,
    geometry_providers: Mapping[str, SceneGeometryProvider] | None = None,
    collision_world_mode: SceneCollisionWorldMode | None = None,
) -> SceneRegistry:
```

Normalize all three root maps, reject pairwise duplicate canonical IDs, include all three groups in the allowed role/geometry key set, and register `articulation_objects` using:

```python
entity = cls._get_simulation_entity(
    simulation,
    getter_name="get_articulation",
    registry_id=registry_id,
    uid=uid,
)
registrations.append(
    SceneEntityRegistration(
        ref=SceneObjectRef(registry_id),
        state_provider=_SimulationEntityStateProvider(entity),
        aliases=(() if uid == registry_id else (uid,)),
        geometry_provider=geometry.get(registry_id),
        collision_role=roles.get(registry_id, SceneCollisionRole.NONE),
    )
)
```

Do not attach `_SimulationArticulationJointStateProvider`; joint state is deliberately hidden by this semantic-object projection.

- [ ] **Step 4: Run the full scene-registry module**

Run: `pytest -q tests/lab/task_program/semantics/test_scene.py`

Expected: PASS with existing rigid-object and articulation behavior unchanged.

- [ ] **Step 5: Format and commit the registry unit**

Run: `black .`

Commit:

```bash
git add embodichain/lab/task_program/semantics/scene.py tests/lab/task_program/semantics/test_scene.py
git commit -m "feat(task-program): register articulation-backed objects"
```

### Task 3: Add Typed Simulation Bindings and Live Assembly

**Files:**
- Modify: `embodichain/lab/task_program/integrations/simulation/bindings.py`
- Modify: `embodichain/lab/task_program/integrations/simulation/__init__.py`
- Modify: `embodichain/lab/task_program/integrations/__init__.py`
- Modify: `tests/gym/envs/task_program/test_simulation.py`
- Modify: `tests/lab/task_program/test_semantic_compiler.py`

**Interfaces:**
- Consumes: Task 1's `create_rigidized_articulation_antipodal_affordance(...)` and Task 2's `articulation_objects` mapping.
- Produces: `SimulationRigidizedArticulationObjectBinding` and `RigidizedArticulationAntipodalGraspBinding` immutable public values.
- Extends: `SimulationSceneBinding.rigidized_articulations` and `SimulationSceneBinding.rigidized_articulation_grasps`.

- [ ] **Step 1: Add failing declaration tests for the new binding types**

In `tests/gym/envs/task_program/test_simulation.py`, construct:

```python
binding = SimulationSceneBinding(
    registry_id="rubiks_scene",
    rigidized_articulations=(
        SimulationRigidizedArticulationObjectBinding(
            entity_id="rubiks_cube",
            simulation_uid="cube_articulation",
            locked_qpos={"top_turn": 0.0},
            dynamics=SceneDynamics.DYNAMIC,
            semantic_type="rubiks_cube",
        ),
    ),
    rigidized_articulation_grasps=(
        RigidizedArticulationAntipodalGraspBinding(
            entity_id="rubiks_cube_grasp",
            object_id="rubiks_cube",
            grasp_link="lower_two_layers",
            native_name="lower_two_layers",
            revision="1",
        ),
    ),
)
manifest = binding.declare()
object_entry = manifest.lookup(SceneObjectRef("rubiks_cube"))
grasp_entry = manifest.lookup(SceneAffordanceRef("rubiks_cube_grasp"))
assert object_entry.semantic_type == "rubiks_cube"
assert grasp_entry.parent == SceneObjectRef("rubiks_cube")
assert grasp_entry.affordance_payload_type is AntipodalAffordance
```

Add constructor failures for empty `locked_qpos`, booleans/non-finite lock values, non-positive/non-finite tolerances, missing parent object, duplicate IDs across all binding collections, and an ordinary `AntipodalGraspAffordanceBinding` attempting to target a rigidized articulation.

- [ ] **Step 2: Run declaration tests and confirm the public types are absent**

Run: `pytest -q tests/gym/envs/task_program/test_simulation.py -k rigidized_articulation`

Expected: collection FAIL because the two binding classes are not exported.

- [ ] **Step 3: Implement immutable root and affordance bindings**

Add these dataclasses beside the existing rigid-object and antipodal binding values:

```python
@dataclass(frozen=True, slots=True)
class SimulationRigidizedArticulationObjectBinding:
    entity_id: str
    simulation_uid: str
    locked_qpos: Mapping[str, float]
    aliases: tuple[str, ...] = ()
    dynamics: SceneDynamics = SceneDynamics.UNKNOWN
    collision_role: SceneCollisionRole = SceneCollisionRole.NONE
    semantic_type: str | None = None
    default_grasp_affordance: str | None = None
    geometry_provider: SceneGeometryProvider | None = None
    joint_position_tolerance: float = 1.0e-3
    link_transform_tolerance: float = 1.0e-5


@dataclass(frozen=True, slots=True)
class RigidizedArticulationAntipodalGraspBinding:
    entity_id: str
    object_id: str
    grasp_link: str
    native_name: str
    revision: str
    aliases: tuple[str, ...] = ()
    relative_pose: tuple[float, ...] = _IDENTITY_POSE
```

Validate identifiers with existing helpers, normalize `locked_qpos` to an immutable `MappingProxyType`, reject bool and non-finite values, and validate both tolerances as positive finite reals. Validate the grasp binding's relative pose with `_pose_tuple`.

- [ ] **Step 4: Extend provider-free declaration and live build**

Add both collections to `SimulationSceneBinding`, its exact-type map, and the global duplicate-ID check. In `declare()`, emit rigidized roots with `SceneObjectRef` and emit each rigidized grasp as `SceneAffordanceRef` parented by that object.

Add a binding-local `_require_coincident_locked_limits` check that reads `get_qpos_limits()`, resolves every native joint by exact `joint_names`, and requires both limit endpoints to be finite and equal to the declared locked value within `joint_position_tolerance`. This check is deliberately stricter than Task 1's direct-Python compatibility path: a position-drive-only lock must raise `ValueError` for configured Task Program assembly.

In `build()`:

```python
base = SceneRegistry.from_simulation(
    simulation,
    rigid_objects={item.entity_id: item.simulation_uid for item in self.rigid_objects},
    articulation_objects={
        item.entity_id: item.simulation_uid
        for item in self.rigidized_articulations
    },
    articulations={item.entity_id: item.simulation_uid for item in self.articulations},
    collision_roles=roles,
    geometry_providers=geometry,
    collision_world_mode=self.collision_world_mode,
)
```

Resolve every rigidized affordance via `simulation.get_articulation()`, reject `articulation.cfg.root_props.fixed_base is not False`, and append:

```python
SceneEntityRegistration(
    ref=SceneAffordanceRef(binding.entity_id),
    aliases=binding.aliases,
    parent=SceneObjectRef(binding.object_id),
    native_name=binding.native_name,
    affordance=create_rigidized_articulation_antipodal_affordance(
        articulation,
        grasp_link=binding.grasp_link,
        locked_qpos=object_binding.locked_qpos,
        joint_position_tolerance=object_binding.joint_position_tolerance,
        link_transform_tolerance=object_binding.link_transform_tolerance,
    ),
    affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
    affordance_revision=binding.revision,
    relative_pose=_pose_tensor(binding.relative_pose),
)
```

Include rigidized roots in placement-parent lookup so support-surface/container affordances can remain additive. Export both new classes from the simulation and parent integration packages.

- [ ] **Step 5: Add live-build and lowering vertical-slice tests**

Add a live simulation stub test that asserts `get_articulation("cube_articulation")` is used, `get_rigid_object` is not used, a fixed root raises `ValueError("floating")`, a drive-held joint with non-coincident limits raises `ValueError("coincident")`, and no registry is returned when geometry validation fails.

In `tests/lab/task_program/test_semantic_compiler.py`, add a focused semantic regression using its existing `_PoseProvider`, `_compiler`, and `_context` helpers. The simulation-binding test already proves that this payload is produced from `get_articulation()`; this test proves built-in Pick preserves the resulting root-frame mesh:

```python
def test_pick_preserves_rigidized_articulation_root_frame_geometry() -> None:
    object_ref = SceneObjectRef("rubiks_cube")
    grasp_ref = SceneAffordanceRef("rubiks_cube_grasp")
    root_frame_vertices = torch.tensor(
        ((0.1, -0.2, 0.3), (0.2, -0.2, 0.3), (0.1, -0.1, 0.3)),
        dtype=torch.float32,
    )
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=object_ref,
                state_provider=_PoseProvider(torch.eye(4).repeat(2, 1, 1)),
                default_affordances={GRASP_AFFORDANCE_CAPABILITY: grasp_ref},
            ),
            SceneEntityRegistration(
                ref=grasp_ref,
                parent=object_ref,
                native_name="lower_two_layers",
                affordance=AntipodalAffordance(
                    mesh_vertices=root_frame_vertices,
                    mesh_triangles=torch.tensor(((0, 1, 2),), dtype=torch.int64),
                ),
                affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                affordance_revision="1",
                relative_pose=torch.eye(4),
            ),
        )
    )
    compiler, _ = _compiler(registry)
    workflow = compiler.analyze(
        (Pick(object=SceneObjectRef("rubiks_cube")),),
        workflow_id="pick_rubiks_cube",
    )
    grounded = compiler.ground(workflow, 0, _context(registry))
    goal = grounded.invocation.goal
    assert isinstance(goal, GraspGoal)
    assert goal.semantics.entity_id == "rubiks_cube"
    assert isinstance(goal.semantics.affordance, AntipodalAffordance)
    assert torch.allclose(
        goal.semantics.affordance.mesh_vertices,
        root_frame_vertices,
    )
```

- [ ] **Step 6: Run binding and vertical-slice tests**

Run: `pytest -q tests/gym/envs/task_program/test_simulation.py tests/lab/task_program/test_semantic_compiler.py -k 'rigidized_articulation or pick_rubiks_cube'`

Expected: PASS, including fixed-root and failed-geometry cases.

- [ ] **Step 7: Format and commit the simulation-binding unit**

Run: `black .`

Commit:

```bash
git add embodichain/lab/task_program/integrations/simulation/bindings.py embodichain/lab/task_program/integrations/simulation/__init__.py embodichain/lab/task_program/integrations/__init__.py tests/gym/envs/task_program/test_simulation.py tests/lab/task_program/test_semantic_compiler.py
git commit -m "feat(task-program): bind rigidized articulations as objects"
```

### Task 4: Decode and Validate Configured Rigidized Articulations

**Files:**
- Modify: `embodichain/lab/task_program/integrations/configured.py`
- Modify: `embodichain/lab/task_program/integrations/_configured_composition.py`
- Modify: `embodichain/lab/gym/utils/_component_composition.py`
- Modify: `tests/gym/envs/task_program/test_configured_integration.py`
- Modify: `tests/gym/utils/test_gym_utils.py`

**Interfaces:**
- Consumes: Task 3's `SimulationRigidizedArticulationObjectBinding` and `RigidizedArticulationAntipodalGraspBinding`.
- Produces: closed `integration.scene.rigidized_articulations` decoding with nested `antipodal_grasp` plus pre-runtime physical articulation UID validation.

- [ ] **Step 1: Add failing configured-decoder tests**

Add a scene payload with:

```python
scene = {
    "registry_id": "rubiks_scene",
    "rigidized_articulations": [
        {
            "entity_id": "rubiks_cube",
            "simulation_uid": "cube_articulation",
            "locked_qpos": {"top_turn": 0.0},
            "dynamics": "dynamic",
            "semantic_type": "rubiks_cube",
            "affordances": [
                {
                    "kind": "antipodal_grasp",
                    "entity_id": "rubiks_cube_grasp",
                    "grasp_link": "lower_two_layers",
                }
            ],
        }
    ],
}
binding = _decode_scene(scene)
assert binding.rigidized_articulations[0].locked_qpos == {"top_turn": 0.0}
assert binding.rigidized_articulation_grasps[0].grasp_link == "lower_two_layers"
```

Add parameterized failures for missing/empty/malformed/non-finite `locked_qpos`; missing `grasp_link`; `mesh_env_id` or `internal_axis` under a rigidized articulation; `grasp_link` under a rigid object; placement affordances remaining accepted under a rigidized object; scene-level `rigidized_articulation_grasps`; and unknown root fields.

- [ ] **Step 2: Run decoder tests and confirm the new scene key is rejected**

Run: `pytest -q tests/gym/envs/task_program/test_configured_integration.py -k rigidized_articulation`

Expected: FAIL with `unsupported fields` for `rigidized_articulations`.

- [ ] **Step 3: Implement category-specific strict decoding**

Add `_decode_rigidized_articulation` with required `entity_id` and `locked_qpos`, ordinary root metadata, optional default/tolerance fields, and nested `affordances`. Decode lock values through `_real` so booleans and non-finite values fail.

Add `_decode_rigidized_articulation_antipodal_grasp` with required `kind`, `entity_id`, and `grasp_link`; optional `native_name`, `revision`, `aliases`, and `relative_pose`; and defaults:

```python
return RigidizedArticulationAntipodalGraspBinding(
    entity_id=entity_id,
    object_id=object_id,
    grasp_link=_identifier(config["grasp_link"], path=f"{path}.grasp_link"),
    native_name=_identifier(
        config.get("native_name", config["grasp_link"]),
        path=f"{path}.native_name",
    ),
    revision=_identifier(
        config.get("revision", _CONFIGURED_ANTIPODAL_GRASP_REVISION),
        path=f"{path}.revision",
    ),
    aliases=_identifier_tuple(config.get("aliases", ()), path=f"{path}.aliases"),
    relative_pose=(
        _finite_tuple(config["relative_pose"], path=f"{path}.relative_pose", expected_length=16)
        if "relative_pose" in config
        else identity_pose
    ),
)
```

Teach `_decode_entity_affordance` to branch by `parent_category`: rigid objects use the existing mesh binding, rigidized articulations use the link-backed binding, and ordinary articulations/links continue rejecting antipodal grasps. Extend `_decode_scene` and `_configured_composition.py` closed-field sets with only `rigidized_articulations`; keep all affordance collections nested.

- [ ] **Step 4: Add pre-runtime physical-category validation tests**

In `tests/gym/utils/test_gym_utils.py`, add:

```python
def test_rigidized_articulation_binding_requires_physical_articulation(
    tmp_path: Path,
) -> None:
    deployment = _configured_task_program_deployment(tmp_path)
    integration = deployment.integration
    integration["scene_binding"] = {
        "registry_id": "rubiks_scene",
        "rigidized_articulations": [
            {
                "entity_id": "rubiks_cube",
                "simulation_uid": "cube",
                "locked_qpos": {"top_turn": 0.0},
            }
        ],
    }
    deployment.environment["simulation"]["rigid_object"] = [{"uid": "cube"}]

    with pytest.raises(ValueError, match="rigidized_articulations.*cube.*physical"):
        _compose_deployment(deployment)
```

Add the passing counterpart with `simulation.articulation`, and a failure where the UID exists only in `background`.

- [ ] **Step 5: Extend component composition validation**

Map both `articulations` and `rigidized_articulations` to physical `simulation.articulation` UIDs in `_validate_scene_binding_targets`; do not add rigidized articulations to the rigid-object UID set.

- [ ] **Step 6: Run configured integration and component composition suites**

Run: `pytest -q tests/gym/envs/task_program/test_configured_integration.py tests/gym/utils/test_gym_utils.py`

Expected: PASS with old nested-affordance rejection behavior unchanged.

- [ ] **Step 7: Format and commit the configured-binding unit**

Run: `black .`

Commit:

```bash
git add embodichain/lab/task_program/integrations/configured.py embodichain/lab/task_program/integrations/_configured_composition.py embodichain/lab/gym/utils/_component_composition.py tests/gym/envs/task_program/test_configured_integration.py tests/gym/utils/test_gym_utils.py
git commit -m "feat(task-program): decode rigidized articulation bindings"
```

### Task 5: Add the Rubik's-Cube Pick-and-Place Example

**Files:**
- Create: `embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place/env.yaml`
- Create: `embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place/task.ur5.yaml`
- Create: `embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place/task_program/program.yaml`
- Create: `embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place/task_program/integration.yaml`
- Modify: `tests/gym/envs/task_program/test_configured_integration.py`
- Modify: `tests/gym/envs/task_program/test_task_vertical_slices.py`
- Modify: `tests/test_task_program_package_data.py`

**Interfaces:**
- Consumes: the new configured scene binding from Task 4, `ur5_dh_pgi_140_80.yaml`, and `trajectory_open_loop.yaml`.
- Produces: runnable Gym ID `TaskProgramRubiksCubePickPlace-v1`, program ID `rubiks_cube_pick_place`, integration ID `rubiks_cube_pick_place_v1`, and registry ID `task_program_rubiks_cube_pick_place`.

- [ ] **Step 1: Add failing package-data and deployment-inspector expectations**

Extend `_PROGRAMS`, `_DEPLOYMENTS`, and `_RESOURCE_PATHS` in `tests/test_task_program_package_data.py`:

```python
Path("tasks/manipulation/rubiks_cube_pick_place/task_program/program.yaml"): (
    "rubiks_cube_pick_place"
)
```

```python
Path("tasks/manipulation/rubiks_cube_pick_place/task.ur5.yaml"): (
    "rubiks_cube_pick_place",
    "task_program_rubiks_cube_pick_place",
    "ur5_dh_pgi_140_80",
)
```

Add both `env.yaml` and `task_program/integration.yaml` to `_RESOURCE_PATHS`. Add the deployment to the configured-integration and packaged vertical-slice parameterizations, then assert:

```python
assert spec.id == "TaskProgramRubiksCubePickPlace-v1"
assert registration.scene_binding.rigidized_articulations[0].simulation_uid == "rubiks_cube"
assert registration.scene_binding.rigidized_articulation_grasps[0].grasp_link == "lower_two_layers"
```

- [ ] **Step 2: Run resource tests and confirm the files are absent**

Run: `pytest -q tests/test_task_program_package_data.py tests/gym/envs/task_program/test_configured_integration.py tests/gym/envs/task_program/test_task_vertical_slices.py -k rubiks`

Expected: FAIL because the four YAML resources do not exist.

- [ ] **Step 3: Create the physical environment component**

Create `env.yaml` with the repeated-pick-place environment settings and the Rubik's-cube articulation as the only task object:

```yaml
environment_id: rubiks_cube_pick_place
physics: default
max_episodes: 1
max_episode_steps: 1200
num_envs: 1
arena_space: 2.5

simulation:
  light:
    direct:
      - uid: main_light
        light_type: sun
        color: [0.6, 0.6, 0.6]
        intensity: 5.0
        direction: [0.0, 0.0, -1.0]
  background: []
  rigid_object: []
  rigid_object_group: []
  articulation:
    - uid: rubiks_cube
      fpath: RubiksCube/rubiks_cube_001.usdc
      init_pos: [-0.42, -0.10878000028431416, 0.0288]
      init_qpos: [0.0]
      qpos_limits:
        top_turn: [0.0, 0.0]
      root_props:
        fixed_base: false

env:
  sim_steps_per_control: 4
  events: {}
  dataset: {}
```

Preserve the repeated-pick-place environment's explicit `physics: default`, light, episode limits, arena spacing, control frequency, events, and dataset settings. Keep `background` and `rigid_object` empty, and do not place Task Program metadata in this file.

- [ ] **Step 4: Create the provider-independent program and trusted integration**

Create `program.yaml` with the provider-independent three-cycle flow:

```yaml
program_id: rubiks_cube_pick_place
targets:
  drop_pose:
    kind: cyclic_pose
    values:
      - position: [-0.40, 0.48, 0.10]
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
      - position: [-0.42, -0.08, 0.10]
        quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
program:
  kind: repeat
  count: 3
  body:
    kind: segment
    name: move_rubiks_cube
    steps:
      kind: sequence
      items:
        - kind: invoke
          call:
            kind: pick
            object: rubiks_cube
        - kind: invoke
          call:
            kind: place
            object: rubiks_cube
            at:
              kind: target_ref
              target: drop_pose
```

Create `integration.yaml` with:

```yaml
integration_id: rubiks_cube_pick_place_v1
program_id: rubiks_cube_pick_place
requires:
  scene_contract: repeated_pick_place_scene_v1
  embodiment_contract: single_arm_parallel_gripper
scene_binding:
  contract_id: repeated_pick_place_scene_v1
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
profile:
  defaults:
    pick_up:
      primary: primary_manipulator
    place:
      primary: primary_manipulator
  action_options:
    pick:
      kind: pick_up
    place:
      kind: place
  effect_monitors: {}
```

Use the exact `requires`, profile, options, and monitor schema already accepted by the repeated-pick-place integration; change only task-specific IDs and the root binding.

- [ ] **Step 5: Create the UR5 deployment**

Create `task.ur5.yaml` with the standard component composition:

```yaml
id: TaskProgramRubiksCubePickPlace-v1
environment:
  component: env.yaml
task_program:
  program: task_program/program.yaml
  integration: task_program/integration.yaml
  execution_policy: ../../../components/execution_policies/trajectory_open_loop.yaml
embodiment:
  component: ../../../components/embodiments/ur5_dh_pgi_140_80.yaml
  overrides:
    init_qpos: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0]
```

Keep the top-level registration key exactly `id`, matching the current repeated-pick-place deployment schema.

- [ ] **Step 6: Run static example validation**

Run: `pytest -q tests/test_task_program_package_data.py tests/gym/envs/task_program/test_configured_integration.py -k 'rubiks or package_data'`

Run:

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place/task.ur5.yaml
```

Run: `python -m embodichain.cli.list_task | rg 'TaskProgramRubiksCubePickPlace-v1'`

Expected: tests and inspector PASS, and the runnable ID appears exactly once. This completes schema, component resolution, contract composition, catalog preflight, and static Semantic Call lowering without constructing DexSim or requiring the external asset.

- [ ] **Step 7: Record the asset-bound physical qualification command**

After registering the published Rubik's-cube bundle, run with DexSim:

```bash
python -m embodichain.lab.scripts.run_env --task TaskProgramRubiksCubePickPlace-v1 --num-envs 1
```

Qualify automatic asset resolution, positive cube lift, `abs(top_turn) <= 1e-3`, successful Place, and final Task Program acceptance.

- [ ] **Step 8: Format and commit the example unit**

Run: `black .`

Commit:

```bash
git add embodichain_tasks/configs/tasks/manipulation/rubiks_cube_pick_place tests/gym/envs/task_program/test_configured_integration.py tests/gym/envs/task_program/test_task_vertical_slices.py tests/test_task_program_package_data.py
git commit -m "feat(tasks): add Rubik's cube Task Program example"
```

### Task 6: Synchronize Public Documentation and Validate the Branch

**Files:**
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.task_program.integrations.simulation.rst`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.task_program.integrations.rst`
- Modify: `agent_context/topics/task-programs/configuration.md`
- Modify: `agent_context/topics/task-programs/task-programs.md`

**Interfaces:**
- Consumes: all public exports and configuration keys added in Tasks 1-5.
- Produces: API coverage for the geometry function and both binding values, plus accurate project-context routing for `rigidized_articulations`.

- [ ] **Step 1: Run the public API checker before documentation edits**

Run: `python docs/scripts/check_api_docs.py`

Expected: FAIL listing the new public geometry function and/or simulation binding classes as undocumented.

- [ ] **Step 2: Add the exact public API entries**

Add `create_rigidized_articulation_antipodal_affordance` beside the existing articulation geometry exports. Add `SimulationRigidizedArticulationObjectBinding` and `RigidizedArticulationAntipodalGraspBinding` beside the existing simulation binding classes in both integration API pages. Preserve alphabetical grouping and existing `automodule`/`autosummary` conventions.

- [ ] **Step 3: Revise Task Program configuration context**

Replace the sentence that limits nested affordances to `rigid_objects`, `articulations`, or `links` with a four-category contract. Document that:

```text
rigidized_articulations owns required locked_qpos and binds a physical
simulation.articulation UID as SceneObjectRef. Its antipodal_grasp child owns
required grasp_link; ordinary rigid-object antipodal fields remain unchanged.
```

Add `TaskProgramRubiksCubePickPlace-v1` to the reference integration table in `agent_context/topics/task-programs/task-programs.md`; that table is the current owning inventory of configured reference integrations.

- [ ] **Step 4: Run context and API documentation validation**

Run: `python .agents/skills/project-dev-context/scripts/context.py affected --base origin/main --explain`

Run: `python .agents/skills/project-dev-context/scripts/context.py check`

Run: `python -m pytest -q -c /dev/null --noconftest tests/test_agent_context_map.py tests/test_agent_context_tools.py`

Run: `python docs/scripts/check_api_docs.py`

Expected: all commands PASS. The affected-context report identifies `task-programs`; no new topic is needed because the existing configuration detail remains the owner.

- [ ] **Step 5: Run proportional regression suites**

Run:

```bash
pytest -q \
  tests/sim/atomic_actions/test_articulation_geometry.py \
  tests/sim/atomic_actions/test_rigidized_articulation_semantics.py \
  tests/lab/task_program/semantics/test_scene.py \
  tests/gym/envs/task_program/test_simulation.py \
  tests/gym/envs/task_program/test_configured_integration.py \
  tests/gym/envs/task_program/test_task_vertical_slices.py \
  tests/lab/task_program/test_semantic_compiler.py \
  tests/gym/utils/test_gym_utils.py \
  tests/test_task_program_package_data.py
```

Expected: PASS with no skipped tests introduced by this change.

- [ ] **Step 6: Run final style and repository checks**

Run: `black .`

Run: `git diff --check origin/xinyi/atomic03...HEAD`

Run: `git status --short`

Expected: Black reports no further changes, diff check exits zero, and status contains only intentional documentation edits awaiting this task's commit.

- [ ] **Step 7: Commit documentation and context**

Commit:

```bash
git add docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst docs/source/api_reference/embodichain/embodichain.lab.task_program.integrations.simulation.rst docs/source/api_reference/embodichain/embodichain.lab.task_program.integrations.rst agent_context/topics/task-programs/configuration.md agent_context/topics/task-programs/task-programs.md
git commit -m "docs: describe rigidized articulation task programs"
```

- [ ] **Step 8: Review the completed branch against the design**

Verify that the branch diff contains no new Task Program call kind, no program-level robot or simulation identifiers, no articulation getter in the ordinary rigid-object path, and no weakening of the fixed-root or lock checks. Summarize static validation and explicitly state whether the asset-bound DexSim qualification ran.
