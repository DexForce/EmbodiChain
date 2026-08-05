# Newton Runtime Contracts Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current private, stale-ID Newton integration with a
public DexSim contract that supports exact prepare/step semantics,
generation-aware rigid/articulation bindings, transactional runtime rebuild,
full multi-arena frames, and independent simultaneous simulation managers.

**Architecture:** DexSim remains the sole owner of Newton model/state/control
truth and publishes stable entity references, immutable generation-tagged
bindings, prepare results, and rebuild events. EmbodiChain owns environment
semantics through an explicit `BackendSceneContext`, rebinds views after a
generation change, initializes only newly added objects, and keeps its existing
public object and manager calls as compatibility wrappers.

**Tech Stack:** Python 3.11+, DexSim, NVIDIA Newton, NVIDIA Warp, PyTorch,
Gymnasium, pytest, Sphinx Markdown, Black 26.3.1.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-07-13-newton-runtime-contracts-design.md`.
- EmbodiChain branch: `feature/newton-physics-backend` at or after `1d4b9eb`.
- DexSim branch: `feature/embodichain-newton-contracts` from `dev@5281bce`.
- DexSim package version is exactly `0.4.4`; `NEWTON_INTEGRATION_API_VERSION` is exactly `2`.
- `prepare()` creates a ready runtime and never advances simulation time; a requested update performs exactly the requested number of physics steps after prepare.
- Model generation starts at `0`, first successful finalize commits `1`, and only successful model replacement increments it.
- Runtime topology mutation is supported for rigid bodies and articulations; soft-body and cloth mutation fails explicitly.
- Runtime IDs are generation-scoped. Stable public identity is `(world_token, entity_handle, entity_kind)`.
- Public poses use `float32`, `xyzw`, and explicit world/arena-local frames; arena conversion uses the full SE(3) transform.
- Existing `SimulationManager`, `RigidObject`, and `Articulation` public call surfaces remain source compatible.
- Core Newton paths must not use `dexsim.default_world()`, global `get_physics_scene()`, or the default `SimulationManager` instance.
- Default-backend behavior remains unchanged.
- Stage 2 differentiable execution is not implemented by this plan; its plan is written only after this stage's merge gate passes.

---

## Repository Baselines and Reference Files

Run before Task 1:

```bash
git -C /root/sources/dexsim branch --show-current
git -C /root/sources/dexsim rev-parse HEAD
git -C /root/sources/EmbodiChain branch --show-current
git -C /root/sources/EmbodiChain rev-parse HEAD
```

Expected branch names are `feature/embodichain-newton-contracts` and
`feature/newton-physics-backend`. Record the actual starting SHAs in the
execution log; do not reset either repository to the SHAs above.

Use these implementations as focused references, without copying their
singleton/Omniverse assumptions:

- DexSim runtime: `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- DexSim rebuild: `/root/sources/dexsim/python/dexsim/engine/newton_physics/rebuild.py`
- DexSim articulation spans: `/root/sources/dexsim/python/dexsim/engine/newton_physics/articulation/articulation.py`
- IsaacLab clone mapping: `/root/sources/IsaacLab/source/isaaclab_newton/isaaclab_newton/cloner/newton_replicate.py`
- IsaacLab rebinding: `/root/sources/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py`
- IsaacLab FK invalidation: `/root/sources/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`

## File Map

### DexSim files created

- `python/dexsim/engine/newton_physics/contracts.py` — API version, stable references, prepare/rebuild results, subscriptions, model leases, and public errors.
- `python/dexsim/engine/newton_physics/bindings.py` — immutable rigid/articulation bindings and explicit q/qd spans.
- `python/dexsim/engine/newton_physics/runtime_snapshot.py` — generation-independent rigid/articulation snapshot and restore data.
- `python/test/engine/newton_physics/newton_contract_test_utils.py` — shared world, rigid, articulation, and state-array test helpers.
- `python/test/engine/newton_physics/test_newton_public_contract.py` — public API/version/prepare/attachment tests.
- `python/test/engine/newton_physics/test_newton_bindings.py` — generation, cross-world, static-body, and joint-span tests.
- `python/test/engine/newton_physics/test_newton_transactional_rebuild.py` — rigid/articulation preservation and rollback tests.
- `python/test/engine/newton_physics/test_newton_multi_world_runtime.py` — same-device isolation and cleanup tests.

### DexSim files modified

- `version.txt` — package patch version `0.4.4`.
- `python/dexsim/engine/newton_physics/__init__.py` — public exports.
- `python/dexsim/engine/newton_physics/newton_manager.py` — generation, prepare, public attachment/binding delegation, candidate commit, close, events.
- `python/dexsim/engine/newton_physics/rebuild.py` — candidate build/restore/validate/atomic commit.
- `python/dexsim/engine/newton_physics/registry.py` — world-token ownership and public owner lookup.
- `python/dexsim/engine/newton_physics/rigid_body/add_body.py` — legacy patch delegates to public attachment.
- `python/dexsim/engine/newton_physics/rigid_body/registration.py` — canonical descriptor replay into a chosen build target.
- `python/dexsim/engine/newton_physics/articulation/articulation.py` — stable ref and explicit q/qd span export.
- `python/dexsim/engine/newton_physics/articulation/skeleton_bridge.py` — canonical articulation replay and removal delta.
- `python/dexsim/engine/newton_physics/world.py` — prepare-then-step behavior and closed-world checks.
- `python/dexsim/engine/newton_physics/integration.py` — deterministic per-world teardown.
- `python/dexsim/engine/newton_physics/capture_coordinator.py` — weak, device-scoped capture coordination.

### EmbodiChain files created

- `embodichain/lab/sim/physics/context.py` — explicit backend/world/scene/arena ownership and full transform table.
- `tests/sim/newton_contract_test_utils.py` — deterministic Newton manager and asset-config fixtures used by the Stage 1 tests.
- `tests/sim/test_newton_scene_context.py` — ownership and full-frame conversion tests.
- `tests/sim/test_newton_rebuild_bindings.py` — pending initialization and generation refresh tests.
- `tests/sim/test_newton_multi_manager.py` — two-manager isolation and cleanup tests.

### EmbodiChain files modified

- `pyproject.toml` — exact `dexsim_engine==0.4.4` dependency.
- `embodichain/lab/sim/cfg.py` — strict Newton configuration validation.
- `embodichain/lab/sim/common.py` — remove base-constructor virtual reset.
- `embodichain/lab/sim/physics/__init__.py` — export the context/capability types.
- `embodichain/lab/sim/physics/base.py` — structured capability and prepare-result contracts.
- `embodichain/lab/sim/physics/default.py` — default-backend compatibility implementation.
- `embodichain/lab/sim/physics/newton.py` — API handshake, event subscription, generation, pending initialization, close.
- `embodichain/lab/sim/sim_manager.py` — context construction, exact update lifecycle, remove invalidation, idempotent close/reset.
- `embodichain/lab/sim/utility/sim_utils.py` — public `attach_rigid_body` use; remove private registry/meta writes.
- `embodichain/lab/sim/objects/rigid_object.py` — explicit context and deferred initialization.
- `embodichain/lab/sim/objects/articulation.py` — explicit context, separate position/velocity widths, deferred initialization.
- `embodichain/lab/sim/objects/robot.py` — pass owner context and defer reset until fully constructed.
- `embodichain/lab/sim/objects/backends/newton.py` — generation-aware bindings, full frames, FK, explicit unsupported errors.
- `embodichain/lab/sim/objects/backends/default.py` — accept explicit context without behavior changes.
- `tests/sim/test_backend_parity.py` — structured capability matrix.
- `tests/sim/test_newton_finalize_lifecycle.py` — exact prepare and initialization semantics.
- `tests/sim/objects/test_rigid_object.py` — multi-arena/rebuild compatibility cases.
- `tests/sim/objects/test_articulation.py` — q/qd span, FK, link/root frame, rebuild cases.
- `docs/source/overview/sim/sim_manager.md` — public lifecycle, capabilities, topology mutation, and cleanup contract.
- `design/newton-backend-design.md` — replace obsolete Target 4 claims with verified Stage 1 status.

---

### Task 1: Publish the DexSim integration contract and generation lifecycle

**Files:**

- Create: `/root/sources/dexsim/python/dexsim/engine/newton_physics/contracts.py`
- Create: `/root/sources/dexsim/python/test/engine/newton_physics/newton_contract_test_utils.py`
- Create: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_public_contract.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/__init__.py`
- Modify: `/root/sources/dexsim/version.txt`

**Interfaces:**

- Produces: `NEWTON_INTEGRATION_API_VERSION = 2`.
- Produces: `NewtonEntityRef`, `NewtonPrepareResult`, `NewtonPrepareFailure`, `NewtonModelRebuiltEvent`, `NewtonRuntimeStatus`, `NewtonSubscription`, `NewtonModelLease`.
- Produces: `NewtonIntegrationError` subclasses for stale generation, cross-world use, unsupported operation, closed runtime, rebuild failure, and active model lease.
- Produces: `NewtonManager.world_token`, `NewtonManager.model_generation`, `NewtonManager.runtime_status`, and `NewtonManager.prepare()`.
- Produces: `NewtonManager.acquire_model_lease() -> NewtonModelLease`; Stage 2 sessions consume this primitive without changing its rebuild-safety semantics.

- [ ] **Step 1: Add shared deterministic test helpers**

Create `newton_contract_test_utils.py` and import its symbols explicitly from
each new DexSim test file:

```python
DT = 0.01


def make_world(device: str = "cpu"):
    config = dexsim.WorldConfig()
    config.open_windows = False
    config.use_default_physics = False
    config.backend = dexsim.types.Backend.OPENGL
    config.renderer = dexsim.types.Renderer.HYBRID
    world = dexsim.World(config)
    world.set_physics_backend("Newton", cfg=NewtonCfg(device=device))
    manager = get_newton_manager(world)
    manager._dexsim_renderer = config.renderer
    manager._visualizer_disabled = True
    return world, world.get_env(), manager


@pytest.fixture
def newton_world():
    world, env, manager = make_world()
    try:
        yield world, env
    finally:
        world.quit()


@pytest.fixture
def two_newton_worlds():
    first = make_world()
    second = make_world()
    try:
        yield first, second
    finally:
        first[0].quit()
        second[0].quit()


@pytest.fixture
def two_cuda_worlds():
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for same-device capture coordination.")
    first = make_world("cuda:0")
    second = make_world("cuda:0")
    try:
        yield first, second
    finally:
        first[0].quit()
        second[0].quit()


def dynamic_box(arena, name: str, z: float = 1.0):
    obj = arena.create_cube(0.1, 0.1, 0.1)
    obj.set_name(name)
    obj.set_location(0.0, 0.0, z)
    attr = PhysicalAttr()
    attr.mass = 1.0
    obj.add_rigidbody(ActorType.DYNAMIC, RigidBodyShape.BOX, attr)
    return obj


def static_plane(arena, name: str):
    obj = arena.create_plane(0.0, 10.0)
    obj.set_name(name)
    obj.add_rigidbody(ActorType.STATIC, RigidBodyShape.PLANE, PhysicalAttr())
    return obj


def test_articulation(arena, name: str):
    path = get_resources_data_path("Robot", "UR5GPI", "UR5_pgi.urdf")
    articulation = arena.load_urdf(path)
    articulation.set_name(name)
    return articulation


def assign_body_state(state, body_id: int, pose, velocity, acceleration) -> None:
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    body_qdd = state.body_qdd.numpy()
    body_q[body_id] = np.asarray(pose, dtype=np.float32)
    body_qd[body_id] = np.asarray(velocity, dtype=np.float32)
    body_qdd[body_id] = np.asarray(acceleration, dtype=np.float32)
    state.body_q.assign(body_q)
    state.body_qd.assign(body_qd)
    state.body_qdd.assign(body_qdd)
```

Each test file imports `dynamic_box as _dynamic_box`,
`static_plane as _static_plane`, `test_articulation as
_test_urdf_articulation`, and `assign_body_state as _assign_body_state`, so all
helper names in the following snippets are defined.

- [ ] **Step 2: Write contract and prepare tests that fail on the current API**

Add tests with these assertions:

```python
def test_public_contract_version_and_initial_generation(newton_world):
    world, _ = newton_world
    mgr = get_newton_manager(world)
    assert Version(dexsim.__version__).base_version == "0.4.4"
    assert NEWTON_INTEGRATION_API_VERSION == 2
    assert mgr.model_generation == 0
    assert mgr.runtime_status.model_finalized is False


def test_prepare_finalizes_without_advancing_time(newton_world):
    world, env = newton_world
    _dynamic_box(env, "box")
    mgr = get_newton_manager(world)
    before = mgr._sim_time
    result = mgr.prepare()
    assert result.generation == 1
    assert result.did_build is True
    assert result.did_rebuild is False
    assert len(result.added_entities) == 1
    assert result.removed_entities == ()
    assert mgr._sim_time == before
    assert mgr.model_generation == 1
    assert mgr.runtime_status.solver_ready is True


def test_second_prepare_is_idempotent(newton_world):
    world, env = newton_world
    _dynamic_box(env, "box")
    mgr = get_newton_manager(world)
    first = mgr.prepare()
    second = mgr.prepare()
    assert first.generation == second.generation == 1
    assert second.did_build is False
    assert second.did_rebuild is False
```

- [ ] **Step 3: Run the focused test and confirm the missing-contract failure**

Run:

```bash
cd /root/sources/dexsim
pytest -q python/test/engine/newton_physics/test_newton_public_contract.py
```

Expected: collection fails because the new public symbols and `prepare()` do
not exist.

- [ ] **Step 4: Add the public value types and errors**

Implement the contract with immutable, typed values:

```python
from __future__ import annotations


NEWTON_INTEGRATION_API_VERSION = 2


class NewtonIntegrationError(RuntimeError):
    """Base error for the public Newton integration contract."""


class NewtonStaleBindingError(NewtonIntegrationError):
    pass


class NewtonCrossWorldError(NewtonIntegrationError):
    pass


class NewtonUnsupportedOperationError(NewtonIntegrationError):
    pass


class NewtonClosedError(NewtonIntegrationError):
    pass


class NewtonRebuildError(NewtonIntegrationError):
    def __init__(self, failure: NewtonPrepareFailure) -> None:
        self.failure = failure
        super().__init__(failure.message)


class NewtonActiveLeaseError(NewtonIntegrationError):
    pass


@dataclass(frozen=True, slots=True)
class NewtonEntityRef:
    world_token: int
    entity_handle: int
    entity_kind: Literal["rigid", "articulation"]


@dataclass(frozen=True, slots=True)
class NewtonPrepareResult:
    generation: int
    did_build: bool
    did_rebuild: bool
    added_entities: tuple[NewtonEntityRef, ...] = ()
    removed_entities: tuple[NewtonEntityRef, ...] = ()


@dataclass(frozen=True, slots=True)
class NewtonModelRebuiltEvent:
    old_generation: int
    new_generation: int
    added_entities: tuple[NewtonEntityRef, ...]
    removed_entities: tuple[NewtonEntityRef, ...]


@dataclass(frozen=True, slots=True)
class NewtonRuntimeStatus:
    model_finalized: bool
    solver_ready: bool
    running: bool
    stale: bool
    closed: bool


@dataclass(frozen=True, slots=True)
class NewtonPrepareFailure:
    generation: int
    operation: Literal["build", "rebuild"]
    message: str


class NewtonSubscription:
    def __init__(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe = unsubscribe
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()


class NewtonModelLease:
    def __init__(
        self,
        *,
        world_token: int,
        generation: int,
        model: object,
        release: Callable[[], None],
    ) -> None:
        self.world_token = world_token
        self.generation = generation
        self.model = model
        self._release = release
        self._closed = False

    def __enter__(self) -> NewtonModelLease:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._release()
```

Keep `NewtonModelState.BUILDER` and `NewtonModelState.READY` working for
existing callers. Add `_model_generation`, `_has_stepped`, `_closed`, and a
monotonic per-manager `world_token`. `prepare()` delegates build/rebuild to the
later transactional helper, returns an idempotent result, and never calls
`simulate()` or increments `_sim_time`.
Add
`subscribe_model_rebuilt(callback: Callable[[NewtonModelRebuiltEvent], None]) -> NewtonSubscription`;
callbacks are stored per manager and invoked only after an atomic successful
commit. A failed build raises `NewtonRebuildError` carrying a
`NewtonPrepareFailure` and emits no rebuilt event.
`acquire_model_lease()` requires a finalized, non-stale model, increments a
per-manager active-lease counter, and returns a lease that strongly owns that
exact model and generation. Lease release is idempotent and decrements the
counter exactly once. This is a lifecycle primitive only; no differentiable
session or tape behavior is added in Stage 1.

- [ ] **Step 5: Export the contract and bump the package patch version**

Export the new symbols from `newton_physics/__init__.py` and change only:

```text
DEXSIM_VERSION_MAJOR 0
DEXSIM_VERSION_MINOR 4
DEXSIM_VERSION_PATCH 4
```

- [ ] **Step 6: Refresh the editable DexSim package metadata**

The development install points at `build_Release/lib/python_package`; its
Python modules are source symlinks, but its `dexsim/version.txt` is a generated
copy. Refresh it through the repository-supported setup script after changing
the root version:

```bash
cd /root/sources/dexsim
PYTHON_BIN="$(command -v python)" ./setup_dev_python.sh -j12
python - <<'PY'
from packaging.version import Version
import dexsim
assert Version(dexsim.__version__).base_version == "0.4.4"
print(dexsim.__version__, dexsim.__file__)
PY
```

Expected: the editable install reports base version `0.4.4` and imports from
the local development package. Generated `build_Release` contents are never
staged or committed.

- [ ] **Step 7: Run the focused tests**

Run:

```bash
pytest -q python/test/engine/newton_physics/test_newton_public_contract.py
```

Expected: all tests in the file pass; no simulation-time change occurs during
prepare.

- [ ] **Step 8: Commit the public contract**

```bash
git -C /root/sources/dexsim add version.txt python/dexsim/engine/newton_physics/contracts.py python/dexsim/engine/newton_physics/newton_manager.py python/dexsim/engine/newton_physics/__init__.py python/test/engine/newton_physics/newton_contract_test_utils.py python/test/engine/newton_physics/test_newton_public_contract.py
git -C /root/sources/dexsim commit -m "feat(newton): publish runtime integration contract"
```

---

### Task 2: Replace private rigid registration with stable public attachment

**Files:**

- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/contracts.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/rigid_body/add_body.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/rigid_body/registration.py`
- Modify: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_public_contract.py`

**Interfaces:**

- Produces: `NewtonManager.attach_rigid_body(...) -> NewtonEntityRef`.
- Produces: immutable `NewtonRigidDescriptor` replay records owned by DexSim.
- Produces: `NewtonManager.entity_ref(entity)`, `descriptor_for(ref)`, and read-only `descriptors`.
- Consumes: `NewtonEntityRef` and generation lifecycle from Task 1.

- [ ] **Step 1: Add failing public attachment, clone-world, and descriptor-isolation tests**

```python
def test_attach_rigid_body_returns_stable_ref(newton_world):
    world, env = newton_world
    cube = env.create_cube(0.1, 0.2, 0.3)
    mgr = get_newton_manager(world)
    ref = mgr.attach_rigid_body(
        cube,
        actor_type=ActorType.DYNAMIC,
        shape_type=RigidBodyShape.BOX,
        physical_attr=PhysicalAttr(),
    )
    assert ref.world_token == mgr.world_token
    assert ref.entity_handle == cube.get_native_handle()
    assert ref.entity_kind == "rigid"
    assert mgr.entity_ref(cube) == ref


def test_child_arena_attachment_uses_child_world(newton_world):
    world, env = newton_world
    arena = env.add_arena("arena_a")
    cube = arena.create_cube(0.1, 0.1, 0.1)
    mgr = get_newton_manager(world)
    ref = mgr.attach_rigid_body(
        cube,
        actor_type=ActorType.DYNAMIC,
        shape_type=RigidBodyShape.BOX,
        physical_attr=PhysicalAttr(),
    )
    mgr.prepare()
    binding = mgr.bind_rigid_entities((ref,))
    assert binding.world_ids_host.tolist() == [0]
    assert mgr._model.body_world.numpy()[binding.body_ids_host[0]] == 0


def test_clone_descriptors_do_not_alias(newton_world):
    world, env = newton_world
    arena_a = env.add_arena("arena_a")
    arena_b = env.add_arena("arena_b")
    prototype = arena_a.create_sphere(0.1)
    prototype.set_name("prototype")
    prototype.add_rigidbody(
        ActorType.DYNAMIC, RigidBodyShape.SPHERE, PhysicalAttr()
    )
    clone = arena_a.clone_actor_to(
        "prototype", arena_b, "clone", ObjectCloneOptions()
    )
    mgr = get_newton_manager(world)
    source = mgr.descriptor_for(mgr.entity_ref(prototype))
    target = mgr.descriptor_for(mgr.entity_ref(clone))
    assert source is not target
    assert source.world_id == 0
    assert target.world_id == 1


def test_mesh_geometry_descriptor_is_owned_by_dexsim(newton_world):
    world, env = newton_world
    entity = env.create_cube(0.1, 0.1, 0.1)
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    triangles = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32
    )
    geometry = GeometryDesc.mesh(vertices=vertices, triangles=triangles)
    mgr = get_newton_manager(world)
    ref = mgr.attach_rigid_body(
        entity,
        actor_type=ActorType.DYNAMIC,
        shape_type=RigidBodyShape.MESH,
        physical_attr=PhysicalAttr(),
        geometry_desc=geometry,
    )
    vertices[0] = 99.0
    owned = mgr.descriptor_for(ref).geometry_desc
    assert np.allclose(owned.vertices[0], [0.0, 0.0, 0.0])


def test_desc_native_box_prepares_without_shape_parameter_fallback(newton_world):
    world, env = newton_world
    entity = env.create_cube(0.1, 0.2, 0.3)
    mgr = get_newton_manager(world)
    ref = mgr.attach_rigid_body(
        entity,
        actor_type=ActorType.DYNAMIC,
        shape_type=RigidBodyShape.BOX,
        body_desc=RigidBodyPhysicsDesc.dynamic(mass=1.0),
        shape_desc=NewtonCollisionDesc(ke=1000.0, kd=50.0, margin=0.01),
        geometry_desc=GeometryDesc.cube((0.1, 0.2, 0.3)),
    )
    result = mgr.prepare()
    assert result.generation == 1
    assert mgr.bind_rigid_entities((ref,)).body_ids_host[0] >= 0


def test_desc_native_sphere_prepares_from_owned_geometry(newton_world):
    world, env = newton_world
    entity = env.create_sphere(0.2)
    mgr = get_newton_manager(world)
    ref = mgr.attach_rigid_body(
        entity,
        actor_type=ActorType.DYNAMIC,
        shape_type=RigidBodyShape.SPHERE,
        body_desc=RigidBodyPhysicsDesc.dynamic(mass=1.0),
        shape_desc=NewtonCollisionDesc(ke=1000.0, kd=50.0),
        geometry_desc=GeometryDesc.sphere(0.2),
    )
    mgr.prepare()
    descriptor = mgr.descriptor_for(ref)
    assert descriptor.geometry_desc.radius == pytest.approx(0.2)
    assert mgr.bind_rigid_entities((ref,)).body_ids_host[0] >= 0
```

- [ ] **Step 2: Run the tests and confirm public attachment is absent**

Run the attachment tests above with `pytest -q`. Expected: failures report
missing `attach_rigid_body`, `entity_ref`, or `descriptor_for`.

- [ ] **Step 3: Add canonical descriptor ownership and the public method**

Add an immutable descriptor that deep-copies mutable desc-native inputs:

```python
@dataclass(frozen=True, slots=True)
class NewtonRigidDescriptor:
    entity_ref: NewtonEntityRef
    arena_handle: int
    world_id: int
    actor_type: ActorType
    shape_type: RigidBodyShape
    node_scale: tuple[float, float, float]
    body_scale: tuple[float, float, float]
    physical_attr: PhysicalAttr | None
    body_desc: object | None
    shape_desc: object | None
    geometry_desc: object | None
```

The `object` annotations above stand for the existing concrete DexSim spawn
descriptor types, not borrowed arbitrary objects. At attachment time, normalize
legacy `PhysicalAttr` into owned body/collision values and recursively copy all
desc-native data. Copy NumPy arrays with canonical `float32`/`int32` dtypes and
mark them read-only. Resolve file-backed mesh data, convex/ACD hulls, and SDF
mesh/config inputs into descriptor-owned replay data so a rebuild neither reads
mutable entity metadata nor depends on a later file change. `descriptor_for()`
returns this immutable snapshot; it never returns `dexsim_meta`.

Implement the public method with this exact surface:

```python
def attach_rigid_body(
    self,
    entity,
    *,
    actor_type,
    shape_type,
    physical_attr=None,
    body_desc=None,
    shape_desc=None,
    geometry_desc=None,
) -> NewtonEntityRef:
    self._assert_open()
    ref = self._ref_for_entity(entity, "rigid")
    descriptor = make_rigid_descriptor(
        manager=self,
        entity=entity,
        entity_ref=ref,
        actor_type=actor_type,
        shape_type=shape_type,
        physical_attr=physical_attr,
        body_desc=body_desc,
        shape_desc=shape_desc,
        geometry_desc=geometry_desc,
    )
    self._rigid_descriptors[ref] = descriptor
    replay_rigid_descriptor(self, descriptor, entity)
    self._record_added(ref)
    self.mark_runtime_model_stale()
    return ref
```

The legacy `MeshObject.add_rigidbody` patches call this method and return its
reference. `register_mesh_object_to_newton_patch` remains temporarily importable
for DexSim compatibility but delegates through descriptor replay and is no
longer called by EmbodiChain.
Legacy `add_sdf_rigidbody` and `add_acd_rigidbody` project their SDF config or
resolved convex hulls into the same `geometry_desc` contract before delegation;
the existing SDF/ACD regression tests must keep passing.

- [ ] **Step 4: Run attachment and existing scene mutation tests**

```bash
pytest -q python/test/engine/newton_physics/test_newton_public_contract.py python/test/engine/newton_physics/test_newton_scene_mutations.py
```

Expected: both files pass, including global world `-1` and child world IDs.

- [ ] **Step 5: Commit public rigid attachment**

```bash
git -C /root/sources/dexsim add python/dexsim/engine/newton_physics/contracts.py python/dexsim/engine/newton_physics/newton_manager.py python/dexsim/engine/newton_physics/rigid_body/add_body.py python/dexsim/engine/newton_physics/rigid_body/registration.py python/test/engine/newton_physics/test_newton_public_contract.py
git -C /root/sources/dexsim commit -m "feat(newton): add stable rigid attachment API"
```

---

### Task 3: Add immutable generation-aware rigid and articulation bindings

**Files:**

- Create: `/root/sources/dexsim/python/dexsim/engine/newton_physics/bindings.py`
- Create: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_bindings.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/articulation/articulation.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/__init__.py`

**Interfaces:**

- Produces: `NewtonJointSpan`, `NewtonArticulationSpan`, `RigidEntityBinding`, `ArticulationBinding`.
- Produces: `joint_span_from_builder(builder, joint_id) -> NewtonJointSpan`.
- Produces: `NewtonManager.bind_rigid_entities(refs, device=None)`.
- Produces: `NewtonManager.bind_articulations(refs, device=None)`.
- Consumes: stable references from Task 2.

- [ ] **Step 1: Add failing binding contract tests**

```python
def test_rigid_binding_is_int32_and_generation_scoped(newton_world):
    world, env = newton_world
    dynamic = _dynamic_box(env, "dynamic")
    static = _static_plane(env, "static")
    mgr = get_newton_manager(world)
    result = mgr.prepare()
    binding = mgr.bind_rigid_entities(
        (mgr.entity_ref(dynamic), mgr.entity_ref(static))
    )
    assert binding.generation == result.generation
    assert binding.body_ids_host.dtype == np.int32
    assert binding.shape_ids_host.dtype == np.int32
    assert binding.body_ids_host[1] == -1
    binding.assert_current(mgr)


def test_binding_rejects_other_world(two_newton_worlds):
    (world_a, env_a, mgr_a), (_, _, mgr_b) = two_newton_worlds
    box = _dynamic_box(env_a, "box")
    mgr_a.prepare()
    with pytest.raises(NewtonCrossWorldError):
        mgr_b.bind_rigid_entities((mgr_a.entity_ref(box),))


def test_articulation_binding_exposes_distinct_q_and_qd_spans(newton_world):
    world, env = newton_world
    art = _test_urdf_articulation(env, "arm")
    mgr = get_newton_manager(world)
    mgr.prepare()
    binding = mgr.bind_articulations((mgr.entity_ref(art),))
    assert binding.qpos_width > 0
    assert binding.qvel_width > 0
    assert all(
        span.q_width >= 0 for spans in binding.joint_spans for span in spans
    )
    assert all(
        span.qd_width >= 0 for spans in binding.joint_spans for span in spans
    )
    assert binding.joint_spans_wp.dtype == wp.int32
```

Add builder-only unit cases for revolute, spherical, and free-joint widths:

```python
@pytest.mark.parametrize(
    "q_width, qd_width",
    [(1, 1), (4, 3), (7, 6)],
)
def test_joint_span_uses_distinct_position_and_velocity_widths(
    q_width, qd_width
):
    builder = SimpleNamespace(
        joint_q_start=[0, q_width],
        joint_qd_start=[0, qd_width],
        joint_q=[0.0] * q_width,
        joint_qd=[0.0] * qd_width,
        joint_target_pos=[0.0] * qd_width,
        joint_target_vel=[0.0] * qd_width,
    )
    span = joint_span_from_builder(builder, joint_id=0)
    assert span.q_start == span.qd_start == 0
    assert span.q_width == q_width
    assert span.qd_width == qd_width
    assert span.target_q_start == span.qd_start
    assert span.target_q_width == qd_width
    assert span.target_qd_start == span.qd_start
    assert span.target_qd_width == qd_width
```

- [ ] **Step 2: Run the binding file and confirm collection/API failures**

```bash
pytest -q python/test/engine/newton_physics/test_newton_bindings.py
```

Expected: missing binding classes/methods.

- [ ] **Step 3: Implement immutable bindings with O(1) validation**

```python
@dataclass(frozen=True, slots=True)
class NewtonJointSpan:
    joint_id: int
    q_start: int
    q_width: int
    qd_start: int
    qd_width: int
    target_q_start: int
    target_q_width: int
    target_qd_start: int
    target_qd_width: int


@dataclass(frozen=True, slots=True)
class NewtonArticulationSpan:
    ref: NewtonEntityRef
    q_start: int
    q_width: int
    qd_start: int
    qd_width: int
    target_q_start: int
    target_q_width: int
    target_qd_start: int
    target_qd_width: int
    control_start: int
    control_width: int


@dataclass(frozen=True, slots=True)
class RigidEntityBinding:
    world_token: int
    generation: int
    refs: tuple[NewtonEntityRef, ...]
    body_ids_host: np.ndarray
    shape_ids_host: np.ndarray
    world_ids_host: np.ndarray
    body_ids_wp: wp.array
    shape_ids_wp: wp.array
    world_ids_wp: wp.array

    def assert_current(self, manager: NewtonManager) -> None:
        assert_binding_owner_and_generation(self, manager)


@dataclass(frozen=True, slots=True)
class ArticulationBinding:
    world_token: int
    generation: int
    refs: tuple[NewtonEntityRef, ...]
    articulation_ids_host: np.ndarray
    root_body_ids_host: np.ndarray
    link_body_ids_host: np.ndarray
    world_ids_host: np.ndarray
    articulation_spans: tuple[NewtonArticulationSpan, ...]
    joint_spans: tuple[tuple[NewtonJointSpan, ...], ...]
    joint_spans_wp: wp.array
    qpos_width: int
    qvel_width: int
    target_qpos_width: int
    target_qvel_width: int
```

Resolve all host arrays once, upload `int32` device arrays once, make NumPy
arrays read-only, and validate a binding by comparing only `world_token` and
`generation`. Do not resolve entity IDs in steady-state fetch/apply calls.
`joint_span_from_builder` computes each end from the next start entry, or from
the corresponding flat array length for the final joint. Current position uses
`joint_q_start/q_width`; current velocity uses `joint_qd_start/qd_width`.
Newton `joint_target_pos`, `joint_target_vel`, and `joint_f` are per-DOF, so
their starts and widths use the qd span even for spherical/free joints. Do not
reuse the coordinate-width q span for position targets.

- [ ] **Step 4: Run binding and simulation-index regressions**

```bash
pytest -q python/test/engine/newton_physics/test_newton_bindings.py python/test/engine/newton_physics/test_newton_sim_index.py
```

Expected: all tests pass; old `get_sim_index()` remains a generation-local
compatibility view, not a stable handle.

- [ ] **Step 5: Commit bindings**

```bash
git -C /root/sources/dexsim add python/dexsim/engine/newton_physics/bindings.py python/dexsim/engine/newton_physics/newton_manager.py python/dexsim/engine/newton_physics/articulation/articulation.py python/dexsim/engine/newton_physics/__init__.py python/test/engine/newton_physics/test_newton_bindings.py
git -C /root/sources/dexsim commit -m "feat(newton): add generation-aware entity bindings"
```

---

### Task 4: Make rigid rebuild transactional and preserve both state buffers

**Files:**

- Create: `/root/sources/dexsim/python/dexsim/engine/newton_physics/runtime_snapshot.py`
- Create: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_transactional_rebuild.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/rebuild.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/world.py`

**Interfaces:**

- Produces: `NewtonRuntimeSnapshot`, `RigidRuntimeSnapshot`, and candidate-runtime commit.
- Produces: successful rebuild event after commit only.
- Produces: rebuild rejection while a model-generation lease is active.
- Consumes: descriptors and bindings from Tasks 2–3.

- [ ] **Step 1: Add failing state-preservation, rollback, and exact-step tests**

```python
def test_rigid_rebuild_preserves_both_states_and_wrench(newton_world):
    world, env = newton_world
    box = _dynamic_box(env, "box")
    mgr = get_newton_manager(world)
    mgr.prepare()
    ref = mgr.entity_ref(box)
    body_id = mgr.bind_rigid_entities((ref,)).body_ids_host[0]
    pose0 = np.array([0.1, 0.2, 1.3, 0.0, 0.0, 0.0, 1.0], np.float32)
    pose1 = np.array([0.2, 0.3, 1.4, 0.0, 0.0, 0.0, 1.0], np.float32)
    _assign_body_state(
        mgr._state_0,
        body_id,
        pose0,
        [1, 2, 3, 4, 5, 6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    _assign_body_state(
        mgr._state_1,
        body_id,
        pose1,
        [6, 5, 4, 3, 2, 1],
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    external_forces = mgr._external_forces.numpy()
    external_forces[body_id] = [1, 2, 3, 4, 5, 6]
    mgr._external_forces.assign(external_forces)
    _dynamic_box(env, "new_box")
    result = mgr.prepare()
    new_id = mgr.bind_rigid_entities((ref,)).body_ids_host[0]
    assert result.did_rebuild is True
    assert np.allclose(mgr._state_0.body_q.numpy()[new_id], pose0)
    assert np.allclose(mgr._state_1.body_q.numpy()[new_id], pose1)
    assert np.allclose(
        mgr._state_0.body_qdd.numpy()[new_id],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    assert np.allclose(
        mgr._state_1.body_qdd.numpy()[new_id],
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    )
    assert np.allclose(mgr._external_forces.numpy()[new_id], [1, 2, 3, 4, 5, 6])


def test_failed_candidate_keeps_old_runtime_and_generation(newton_world, monkeypatch):
    world, env = newton_world
    box = _dynamic_box(env, "box")
    mgr = get_newton_manager(world)
    mgr.prepare()
    old_model = mgr._model
    old_generation = mgr.model_generation
    _dynamic_box(env, "new_box")
    monkeypatch.setattr(rebuild, "_validate_candidate", lambda candidate: (_ for _ in ()).throw(ValueError("injected")))
    with pytest.raises(NewtonRebuildError, match="injected"):
        mgr.prepare()
    assert mgr._model is old_model
    assert mgr.model_generation == old_generation
    assert mgr.lifecycle_state is NewtonModelState.STALE


def test_world_update_prepares_then_executes_one_step(newton_world):
    world, env = newton_world
    _dynamic_box(env, "box", z=1.0)
    mgr = get_newton_manager(world)
    before = mgr._sim_time
    world.update(0.01)
    assert mgr.model_generation == 1
    assert mgr._sim_time == pytest.approx(before + 0.01)


def test_active_model_lease_blocks_rebuild_until_released(newton_world):
    world, env = newton_world
    _dynamic_box(env, "box")
    mgr = get_newton_manager(world)
    mgr.prepare()
    lease = mgr.acquire_model_lease()
    assert lease.world_token == mgr.world_token
    assert lease.generation == mgr.model_generation
    assert lease.model is mgr._model
    _dynamic_box(env, "new_box")
    with pytest.raises(NewtonActiveLeaseError, match="generation 1"):
        mgr.prepare()
    assert mgr.model_generation == 1
    lease.close()
    lease.close()
    assert mgr.prepare().generation == 2
```

- [ ] **Step 2: Run the new file and confirm current rebuild destroys/aliases state**

```bash
pytest -q python/test/engine/newton_physics/test_newton_transactional_rebuild.py
```

Expected: failures show missing snapshot types, non-transactional clearing, or
the current first-update skip; the active-lease case currently rebuilds or has
no lease surface.

- [ ] **Step 3: Add complete rigid snapshots keyed by stable reference**

```python
@dataclass(frozen=True, slots=True)
class RigidRuntimeSnapshot:
    ref: NewtonEntityRef
    state_0_pose: np.ndarray
    state_0_velocity: np.ndarray
    state_0_acceleration: np.ndarray
    state_1_pose: np.ndarray
    state_1_velocity: np.ndarray
    state_1_acceleration: np.ndarray
    external_wrench: np.ndarray


@dataclass(frozen=True, slots=True)
class NewtonRuntimeSnapshot:
    generation: int
    rigid: dict[NewtonEntityRef, RigidRuntimeSnapshot]
    articulations: dict[NewtonEntityRef, ArticulationRuntimeSnapshot]
```

Capture only references present in the old finalized model. Restore surviving
references after candidate finalization using the candidate binding. Copy
arrays into both candidate states; do not make both buffers identical when the
old buffers differed.

- [ ] **Step 4: Build and validate a candidate before mutating the live manager**

Use an unregistered candidate manager that shares only the stable world token:

```python
candidate = NewtonManager(
    cfg=copy.deepcopy(manager.cfg),
    world_token=manager.world_token,
    register_live=False,
)
candidate.set_dexsim_world(world)
candidate.replay_descriptors(manager.descriptors)
candidate.start_simulation()
restore_runtime_snapshot(candidate, snapshot)
_validate_candidate(candidate)
manager._commit_candidate(candidate, delta)
```

Before constructing a candidate, `prepare()` checks the active-lease count. A
topology-changing prepare raises `NewtonActiveLeaseError` with the world token
and leased generation while the count is non-zero; idempotent prepare of an
unchanged READY model remains allowed. The live model, generation, topology
delta, and STALE status remain unchanged after rejection. Stage 2 must acquire
this lease before recording a tape and close it only after backward or explicit
session detach.

`_commit_candidate` swaps builder/model/states/control/contacts/pipeline/solver,
entity mappings, descriptors, articulation runtime bindings, caches, and graph
as one non-raising assignment block. It increments generation once, sets READY,
detaches the transferred resources from the candidate, then emits one rebuilt
event. Retire the old runtime only after the swap; cleanup failure is logged
without rolling back a runtime already published to subscribers. Candidate
failure before the swap closes candidate resources,
leaves the live runtime fields unchanged, keeps STALE, and raises
`NewtonRebuildError` chained from the original exception.

- [ ] **Step 5: Make World.update call prepare and still step**

Replace skip-on-build behavior with:

```python
prepare_result = mgr.prepare()
mgr.step(dt)
if mgr.should_sync_to_dexsim(step_override=sync_to_dexsim):
    _push_newton_state_to_dexsim(mgr)
```

The result is available to integrations but never consumes the requested
physics step.

- [ ] **Step 6: Run lifecycle and mutation regressions**

```bash
pytest -q python/test/engine/newton_physics/test_newton_transactional_rebuild.py python/test/engine/newton_physics/test_newton_scene_lifecycle.py python/test/engine/newton_physics/test_newton_scene_mutations.py python/test/engine/newton_physics/test_newton_body_dynamics.py
```

Expected: all pass; update time increments on the first call and after rebuild.

- [ ] **Step 7: Commit transactional rigid rebuild**

```bash
git -C /root/sources/dexsim add python/dexsim/engine/newton_physics/runtime_snapshot.py python/dexsim/engine/newton_physics/rebuild.py python/dexsim/engine/newton_physics/newton_manager.py python/dexsim/engine/newton_physics/world.py python/test/engine/newton_physics/test_newton_transactional_rebuild.py
git -C /root/sources/dexsim commit -m "refactor(newton): rebuild rigid runtime transactionally"
```

---

### Task 5: Preserve articulation runtime state and rebind explicit spans

**Files:**

- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/runtime_snapshot.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/rebuild.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/articulation/articulation.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/articulation/skeleton_bridge.py`
- Modify: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_transactional_rebuild.py`
- Modify: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_bindings.py`

**Interfaces:**

- Produces: complete `ArticulationRuntimeSnapshot` and canonical articulation replay.
- Produces: `NewtonManager.forward_kinematics(articulation_mask=None)` as the public FK synchronization entry point; it updates both ping-pong states.
- Consumes: `ArticulationBinding` from Task 3 and candidate transaction from Task 4.

- [ ] **Step 1: Add failing articulation preservation and removal tests**

Define these local test helpers above the tests:

```python
def _assign_slice(owner, name: str, start: int, values: np.ndarray) -> None:
    source = getattr(owner, name)
    data = source.numpy()
    data[start : start + len(values)] = values
    source.assign(data)


def _read_slice(owner, name: str, start: int, width: int) -> np.ndarray:
    return getattr(owner, name).numpy()[start : start + width].copy()


def _assign_existing_owners(
    manager, owner_names: tuple[str, ...], name: str, start: int, values
) -> None:
    assigned = False
    for owner_name in owner_names:
        owner = getattr(manager, owner_name, None)
        if owner is None or getattr(owner, name, None) is None:
            continue
        _assign_slice(owner, name, start, values)
        assigned = True
    assert assigned, f"{name} is unavailable on {owner_names}"


def _read_first_owner(
    manager, owner_names: tuple[str, ...], name: str, start: int, width: int
) -> np.ndarray:
    for owner_name in owner_names:
        owner = getattr(manager, owner_name, None)
        if owner is not None and getattr(owner, name, None) is not None:
            return _read_slice(owner, name, start, width)
    raise AssertionError(f"{name} is unavailable on {owner_names}")


def _write_articulation_arrays(
    manager,
    binding,
    state_0_q,
    state_0_qd,
    state_1_q,
    state_1_qd,
    model_target_q,
    model_target_qd,
    control_target_q,
    control_target_qd,
    generalized_force,
    active_control,
) -> None:
    span = binding.articulation_spans[0]
    _assign_slice(manager._state_0, "joint_q", span.q_start, state_0_q)
    _assign_slice(manager._state_0, "joint_qd", span.qd_start, state_0_qd)
    _assign_slice(manager._state_1, "joint_q", span.q_start, state_1_q)
    _assign_slice(manager._state_1, "joint_qd", span.qd_start, state_1_qd)
    _assign_existing_owners(
        manager,
        ("_model",),
        "joint_target_pos",
        span.target_q_start,
        model_target_q,
    )
    _assign_existing_owners(
        manager,
        ("_model",),
        "joint_target_vel",
        span.target_qd_start,
        model_target_qd,
    )
    _assign_existing_owners(
        manager,
        ("_control",),
        "joint_target_pos",
        span.target_q_start,
        control_target_q,
    )
    _assign_existing_owners(
        manager,
        ("_control",),
        "joint_target_vel",
        span.target_qd_start,
        control_target_qd,
    )
    _assign_existing_owners(
        manager,
        ("_model",),
        "joint_f",
        span.control_start,
        generalized_force,
    )
    _assign_existing_owners(
        manager,
        ("_control",),
        "joint_f",
        span.control_start,
        active_control,
    )


def _read_q(manager, binding, state_name="_state_0"):
    span = binding.articulation_spans[0]
    return _read_slice(
        getattr(manager, state_name), "joint_q", span.q_start, span.q_width
    )


def _read_qd(manager, binding, state_name="_state_0"):
    span = binding.articulation_spans[0]
    return _read_slice(
        getattr(manager, state_name), "joint_qd", span.qd_start, span.qd_width
    )


def _read_target_q(manager, binding, owner_name):
    span = binding.articulation_spans[0]
    return _read_first_owner(
        manager,
        (owner_name,),
        "joint_target_pos",
        span.target_q_start,
        span.target_q_width,
    )


def _read_target_qd(manager, binding, owner_name):
    span = binding.articulation_spans[0]
    return _read_first_owner(
        manager,
        (owner_name,),
        "joint_target_vel",
        span.target_qd_start,
        span.target_qd_width,
    )


def _read_generalized_force(manager, binding):
    span = binding.articulation_spans[0]
    return _read_first_owner(
        manager,
        ("_model",),
        "joint_f",
        span.control_start,
        span.control_width,
    )


def _read_active_control(manager, binding):
    span = binding.articulation_spans[0]
    return _read_first_owner(
        manager,
        ("_control",),
        "joint_f",
        span.control_start,
        span.control_width,
    )
```

```python
def test_articulation_rebuild_preserves_current_target_and_control(newton_world):
    world, env = newton_world
    art = _test_urdf_articulation(env, "arm")
    mgr = get_newton_manager(world)
    mgr.prepare()
    ref = mgr.entity_ref(art)
    binding = mgr.bind_articulations((ref,))
    q0 = np.linspace(0.01, 0.01 * binding.qpos_width, binding.qpos_width, dtype=np.float32)
    qd0 = np.linspace(0.02, 0.02 * binding.qvel_width, binding.qvel_width, dtype=np.float32)
    q1 = q0 + 0.4
    qd1 = qd0 + 0.5
    model_target_q = np.linspace(
        -0.03,
        -0.03 * binding.target_qpos_width,
        binding.target_qpos_width,
        dtype=np.float32,
    )
    model_target_qd = np.linspace(
        -0.04,
        -0.04 * binding.target_qvel_width,
        binding.target_qvel_width,
        dtype=np.float32,
    )
    control_target_q = model_target_q - 0.7
    control_target_qd = model_target_qd - 0.8
    generalized_force = np.full(binding.qvel_width, 0.3, dtype=np.float32)
    active_control = np.full(binding.qvel_width, -0.6, dtype=np.float32)
    _write_articulation_arrays(
        mgr,
        binding,
        q0,
        qd0,
        q1,
        qd1,
        model_target_q,
        model_target_qd,
        control_target_q,
        control_target_qd,
        generalized_force,
        active_control,
    )
    _dynamic_box(env, "topology_change")
    mgr.prepare()
    rebound = mgr.bind_articulations((ref,))
    assert np.allclose(_read_q(mgr, rebound), q0)
    assert np.allclose(_read_qd(mgr, rebound), qd0)
    assert np.allclose(_read_q(mgr, rebound, "_state_1"), q1)
    assert np.allclose(_read_qd(mgr, rebound, "_state_1"), qd1)
    assert np.allclose(
        _read_target_q(mgr, rebound, "_model"), model_target_q
    )
    assert np.allclose(
        _read_target_qd(mgr, rebound, "_model"), model_target_qd
    )
    assert np.allclose(
        _read_target_q(mgr, rebound, "_control"), control_target_q
    )
    assert np.allclose(
        _read_target_qd(mgr, rebound, "_control"), control_target_qd
    )
    assert np.allclose(
        _read_generalized_force(mgr, rebound), generalized_force
    )
    assert np.allclose(_read_active_control(mgr, rebound), active_control)


def test_removed_articulation_ref_cannot_rebind(newton_world):
    world, env = newton_world
    art = _test_urdf_articulation(env, "arm")
    mgr = get_newton_manager(world)
    mgr.prepare()
    ref = mgr.entity_ref(art)
    env.remove_skeleton("arm")
    mgr.prepare()
    with pytest.raises(NewtonStaleBindingError, match="removed"):
        mgr.bind_articulations((ref,))


def test_articulation_rebuild_preserves_drive_limits_and_feedforward(newton_world):
    world, env = newton_world
    art = _test_urdf_articulation(env, "arm")
    mgr = get_newton_manager(world)
    mgr.prepare()
    ref = mgr.entity_ref(art)
    binding = mgr.bind_articulations((ref,))
    span = binding.articulation_spans[0]
    width = span.control_width
    expected = {
        "joint_target_ke": np.full(width, 11.0, np.float32),
        "joint_target_kd": np.full(width, 1.2, np.float32),
        "joint_friction": np.full(width, 0.13, np.float32),
        "joint_armature": np.full(width, 0.07, np.float32),
        "joint_target_mode": np.full(width, 1, np.int32),
        "joint_effort_limit": np.full(width, 9.0, np.float32),
        "joint_velocity_limit": np.full(width, 4.0, np.float32),
        "joint_limit_lower": np.full(width, -0.9, np.float32),
        "joint_limit_upper": np.full(width, 0.9, np.float32),
    }
    for name, values in expected.items():
        _assign_slice(mgr._model, name, span.control_start, values)
    feedforward = np.full(width, 0.23, np.float32)
    _assign_slice(mgr._control, "joint_act", span.control_start, feedforward)
    _dynamic_box(env, "topology_change")
    mgr.prepare()
    rebound = mgr.bind_articulations((ref,)).articulation_spans[0]
    for name, values in expected.items():
        actual = _read_slice(
            mgr._model, name, rebound.control_start, rebound.control_width
        )
        assert np.allclose(actual, values)
    assert np.allclose(
        _read_slice(
            mgr._control,
            "joint_act",
            rebound.control_start,
            rebound.control_width,
        ),
        feedforward,
    )
```

- [ ] **Step 2: Run the articulation tests and confirm runtime data is lost**

Expected: current rebuild either omits the articulation or loses current/target
state and control.

- [ ] **Step 3: Implement complete articulation snapshots**

```python
@dataclass(frozen=True, slots=True)
class ArticulationRuntimeSnapshot:
    ref: NewtonEntityRef
    state_0_joint_q: np.ndarray
    state_0_joint_qd: np.ndarray
    state_1_joint_q: np.ndarray
    state_1_joint_qd: np.ndarray
    model_target_joint_q: np.ndarray | None
    model_target_joint_qd: np.ndarray | None
    control_target_joint_q: np.ndarray | None
    control_target_joint_qd: np.ndarray | None
    model_joint_f: np.ndarray | None
    control_joint_f: np.ndarray | None
    control_joint_act: np.ndarray | None
    drive_stiffness: np.ndarray
    drive_damping: np.ndarray
    drive_friction: np.ndarray
    drive_armature: np.ndarray
    drive_target_mode: np.ndarray
    drive_effort_limit: np.ndarray
    drive_velocity_limit: np.ndarray
    joint_limit_lower: np.ndarray
    joint_limit_upper: np.ndarray
    root_state_0: np.ndarray
    root_state_1: np.ndarray
```

Read/write each field through explicit spans from `ArticulationBinding`.
Model defaults and active `Control` targets/forces are captured separately;
never collapse them just because the normal setter currently writes both.
An owner field is `None` only when that Newton model/control array is genuinely
absent, and restore preserves that absence.
Capture active feed-forward control from `Control.joint_act`. Capture drive and
limit arrays from `joint_target_ke`, `joint_target_kd`, `joint_friction`,
`joint_armature`, `joint_target_mode`, `joint_effort_limit`,
`joint_velocity_limit`, `joint_limit_lower`, and `joint_limit_upper` on their
live owner (`Control` when exposed, otherwise `Model`) and restore them before
candidate validation. Each `root_state_*` is exactly 13 `float32` values: world pose in
`xyz+xyzw` followed by linear and angular velocity. Contacts are not captured;
the candidate collision pipeline regenerates them.
Canonical articulation replay reconstructs links/joints/drives into the
candidate and then refreshes existing `NewtonArticulation` wrapper metadata at
commit. Never interpret an active-joint ordinal as a flattened q/qd index.

- [ ] **Step 4: Invalidate FK after current q writes**

When current q is restored or written, build a boolean articulation mask with
shape `(model.articulation_count,)`. Evaluate FK independently against both
ping-pong states so their distinct q/qd histories remain distinct:

```python
for state in (manager._state_0, manager._state_1):
    eval_fk(
        manager._model,
        state.joint_q,
        state.joint_qd,
        state,
        articulation_mask,
    )
```

Run visual synchronization only after FK is current.

- [ ] **Step 5: Run articulation, binding, and rebuild tests**

```bash
pytest -q python/test/engine/newton_physics/test_newton_transactional_rebuild.py python/test/engine/newton_physics/test_newton_bindings.py python/test/engine/newton_physics/test_newton_physics_scene.py -k articulation
```

Expected: all selected tests pass, including spherical/free-joint span cases.

- [ ] **Step 6: Commit articulation preservation**

```bash
git -C /root/sources/dexsim add python/dexsim/engine/newton_physics/runtime_snapshot.py python/dexsim/engine/newton_physics/rebuild.py python/dexsim/engine/newton_physics/articulation/articulation.py python/dexsim/engine/newton_physics/articulation/skeleton_bridge.py python/test/engine/newton_physics/test_newton_transactional_rebuild.py python/test/engine/newton_physics/test_newton_bindings.py
git -C /root/sources/dexsim commit -m "feat(newton): preserve articulation state across rebuild"
```

---

### Task 6: Isolate same-device worlds and make DexSim cleanup deterministic

**Files:**

- Create: `/root/sources/dexsim/python/test/engine/newton_physics/test_newton_multi_world_runtime.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/registry.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/integration.py`
- Modify: `/root/sources/dexsim/python/dexsim/engine/newton_physics/capture_coordinator.py`

**Interfaces:**

- Produces: idempotent `NewtonManager.close()` and public `manager_for_entity(entity)`.
- Produces: weak device-level CUDA capture coordination without shared physics state.
- Consumes: per-world tokens and generations from Tasks 1–5.

- [ ] **Step 1: Add failing two-world isolation and close tests**

```python
def test_two_worlds_build_step_rebuild_and_close_independently(two_newton_worlds):
    (world_a, env_a, mgr_a), (world_b, env_b, mgr_b) = two_newton_worlds
    box_a = _dynamic_box(env_a, "a")
    box_b = _dynamic_box(env_b, "b")
    world_a.update(0.01)
    world_b.update(0.02)
    assert mgr_a.world_token != mgr_b.world_token
    assert mgr_a.model_generation == mgr_b.model_generation == 1
    assert mgr_a._model is not mgr_b._model
    _dynamic_box(env_a, "a2")
    world_a.update(0.01)
    assert mgr_a.model_generation == 2
    assert mgr_b.model_generation == 1
    mgr_a.close()
    mgr_a.close()
    with pytest.raises(NewtonClosedError):
        mgr_a.bind_rigid_entities((mgr_a.entity_ref(box_a),))
    assert manager_for_entity(box_b) is mgr_b
    world_b.update(0.02)


def test_capture_coordinator_holds_only_weak_manager_refs(two_cuda_worlds):
    (_, _, mgr_a), (_, _, mgr_b) = two_cuda_worlds
    coordinator = capture_coordinator_for_device(mgr_a.device)
    assert coordinator.manager_count == 2
    mgr_a.close()
    assert mgr_a not in tuple(coordinator.managers)
    assert mgr_b in tuple(coordinator.managers)
    assert coordinator.manager_count == 1
```

- [ ] **Step 2: Run the file; confirm leakage/cross-world assumptions fail**

```bash
pytest -q python/test/engine/newton_physics/test_newton_multi_world_runtime.py
```

Expected: failures expose absent close/owner lookup or strong global state.

- [ ] **Step 3: Implement deterministic per-world teardown**

`NewtonManager.close()` sets closed once, invalidates graphs/caches, clears
callbacks and subscriptions, releases model/state/control/contact/solver and
renderer resources, unregisters arena/entity ownership, and removes only this
manager from weak coordination. Every public method calls `_assert_open()`.

Expose owner lookup without leaking private registries:

```python
def manager_for_entity(entity) -> NewtonManager | None:
    arena = entity.get_arena()
    return manager_for_arena(arena)
```

The CUDA coordinator stores `weakref.WeakSet[NewtonManager]`, serializes only
capture operations for a device, and has a finite diagnostic timeout. It owns
no builder/model/state/control/solver arrays. Expose a read-only `managers`
tuple and derived `manager_count` for diagnostics/tests.

- [ ] **Step 4: Run the DexSim Stage 1 suite**

```bash
pytest -q python/test/engine/newton_physics/test_newton_public_contract.py python/test/engine/newton_physics/test_newton_bindings.py python/test/engine/newton_physics/test_newton_transactional_rebuild.py python/test/engine/newton_physics/test_newton_multi_world_runtime.py python/test/engine/newton_physics/test_newton_scene_lifecycle.py python/test/engine/newton_physics/test_newton_scene_mutations.py python/test/engine/newton_physics/test_newton_sim_index.py python/test/engine/newton_physics/test_newton_physics_scene.py
```

Expected: zero failures and no surviving per-world registrations after fixture
teardown.

- [ ] **Step 5: Commit world isolation and cleanup**

```bash
git -C /root/sources/dexsim add python/dexsim/engine/newton_physics/newton_manager.py python/dexsim/engine/newton_physics/registry.py python/dexsim/engine/newton_physics/integration.py python/dexsim/engine/newton_physics/capture_coordinator.py python/test/engine/newton_physics/test_newton_multi_world_runtime.py
git -C /root/sources/dexsim commit -m "fix(newton): isolate and close per-world runtimes"
```

---

### Task 7: Add EmbodiChain API handshake, structured capabilities, and scene context

**Files:**

- Create: `embodichain/lab/sim/physics/context.py`
- Create: `tests/sim/newton_contract_test_utils.py`
- Create: `tests/sim/test_newton_scene_context.py`
- Modify: `pyproject.toml`
- Modify: `embodichain/lab/sim/cfg.py`
- Modify: `embodichain/lab/sim/physics/base.py`
- Modify: `embodichain/lab/sim/physics/default.py`
- Modify: `embodichain/lab/sim/physics/newton.py`
- Modify: `embodichain/lab/sim/physics/__init__.py`
- Modify: `tests/sim/test_backend_parity.py`

**Interfaces:**

- Produces: `PhysicsCapabilities`, `PhysicsPrepareResult`, `BackendSceneContext`.
- Produces: exact DexSim package/API validation during Newton activation.
- Consumes: DexSim public contract from Tasks 1–6.

- [ ] **Step 1: Add shared EmbodiChain contract test fixtures**

Create `tests/sim/newton_contract_test_utils.py`; each new Stage 1 test file
imports the fixtures and factories it uses:

```python
ART_PATH = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"


def box_cfg(uid: str, z: float = 1.0) -> RigidObjectCfg:
    return RigidObjectCfg.from_dict(
        {
            "uid": uid,
            "shape": {"shape_type": "Cube", "size": [0.1, 0.1, 0.1]},
            "attrs": {"mass": 1.0},
            "body_type": "dynamic",
            "init_pos": (0.0, 0.0, z),
        }
    )


def arm_cfg(uid: str) -> ArticulationCfg:
    return ArticulationCfg.from_dict(
        {
            "uid": uid,
            "fpath": get_data_path(ART_PATH),
            "drive_pros": {"drive_type": "force"},
        }
    )


@pytest.fixture
def newton_sim():
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=2,
            physics_cfg=NewtonPhysicsCfg(
                device="cpu", num_substeps=2, use_cuda_graph=False
            ),
        )
    )
    try:
        yield sim
    finally:
        close = getattr(sim, "close", None)
        if close is None:
            sim.destroy(exit_process=False)
        else:
            close()


@pytest.fixture
def fake_manager():
    identity = np.eye(4, dtype=np.float32)
    rotated = np.eye(4, dtype=np.float32)
    rotated[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    rotated[:3, 3] = [2.0, 3.0, 0.0]

    class Root:
        def __init__(self, pose):
            self.pose = pose
        def get_world_pose(self):
            return self.pose.copy()

    class Arena:
        def __init__(self, pose):
            self.root = Root(pose)
        def get_root_node(self):
            return self.root

    class Manager:
        def __init__(self):
            self._arenas = [Arena(identity), Arena(rotated)]
            self.device = torch.device("cpu")
            self.is_closed = False

    return Manager()
```

- [ ] **Step 2: Add failing version, capability, and context tests**

```python
def test_newton_backend_requires_exact_contract(monkeypatch):
    monkeypatch.setattr(dexsim, "__version__", "0.4.3")
    backend = NewtonPhysicsBackend(SimpleNamespace())
    with pytest.raises(RuntimeError, match="0.4.4"):
        backend._validate_integration_contract()


def test_capabilities_are_structured():
    caps = NewtonPhysicsBackend(SimpleNamespace()).capabilities
    assert caps.runtime_topology_mutation == frozenset({"rigid", "articulation"})
    assert caps.multi_world is True
    assert caps.articulation_acceleration is False
    assert "soft_body" not in caps.asset_kinds
    assert "cloth" not in caps.asset_kinds


def test_scene_context_uses_full_arena_transform(fake_manager):
    context = BackendSceneContext(fake_manager)
    transforms = context.arena_transforms
    assert transforms.shape == (2, 4, 4)
    local = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    world = context.local_to_world(local, torch.tensor([1]))
    roundtrip = context.world_to_local(world, torch.tensor([1]))
    assert torch.allclose(roundtrip, local, atol=1e-6)
    assert not torch.allclose(world[:, :3], local[:, :3] + transforms[1, :3, 3])
```

The fake second arena must contain both non-zero translation and a 90-degree Z
rotation so the last assertion detects translation-only conversion.

- [ ] **Step 3: Run the pure-Python tests and confirm missing types/validation**

```bash
pytest -q tests/sim/test_backend_parity.py tests/sim/test_newton_scene_context.py -m "not requires_sim"
```

Expected: missing capability/context contracts.

- [ ] **Step 4: Add structured backend contracts**

```python
@dataclass(frozen=True, slots=True)
class PhysicsCapabilities:
    asset_kinds: frozenset[str]
    runtime_topology_mutation: frozenset[str]
    solver_gradients: frozenset[str]
    cuda_graph: bool
    partial_reset: bool
    forward_kinematics: bool
    heterogeneous_joint_spans: bool
    runtime_collision_filter: bool
    contact_sensor: bool
    articulation_acceleration: bool
    multi_world: bool


@dataclass(frozen=True, slots=True)
class PhysicsPrepareResult:
    generation: int | None
    did_build: bool
    did_rebuild: bool
    added_entities: tuple[object, ...] = ()
    removed_entities: tuple[object, ...] = ()
```

Keep every existing `supports_*` property as a wrapper over `capabilities`.
The default backend returns `generation=None` and preserves existing behavior.
Add abstract/default implementations for `model_generation`,
`queue_initialization(obj)`, `prepare() -> PhysicsPrepareResult`, and
idempotent `close()` so later tasks do not branch on backend names.

- [ ] **Step 5: Add strict config and dependency validation**

Change the dependency to:

```toml
"dexsim_engine==0.4.4",
```

Validate positive `physics_dt`, positive `num_substeps`, normalized device,
recognized solver parameters, gradient/solver compatibility, broad phase, and
CUDA graph combinations before world construction. On backend activation,
require both `dexsim.__version__ == "0.4.4"` and
`NEWTON_INTEGRATION_API_VERSION == 2`. Accept local source builds whose public
version is `0.4.4+<development metadata>` by comparing
`packaging.version.Version(dexsim.__version__).base_version` to `"0.4.4"`;
reject every other base version.

- [ ] **Step 6: Implement explicit owner context**

```python
class BackendSceneContext:
    def __init__(self, manager: SimulationManager) -> None:
        self._manager_ref = weakref.ref(manager)

    @property
    def manager(self) -> SimulationManager:
        manager = self._manager_ref()
        if manager is None or manager.is_closed:
            raise RuntimeError("BackendSceneContext owner is closed.")
        return manager

    @property
    def world(self):
        return self.manager.get_world()

    @property
    def scene(self):
        return self.manager.physics.get_scene()

    @property
    def physics(self):
        return self.manager.physics

    @property
    def generation(self) -> int | None:
        return self.manager.physics.model_generation
```

Build device-resident `(N, 4, 4)` world transforms and inverses from each
arena root node's full world pose. Provide batched `local_to_world()` and
`world_to_local()` for pose tensors in `xyzw`. Maintain weak maps from arena
and entity native handles to contexts. `register_entities()` and
`for_entities()` provide the source-compatible constructor fallback without
consulting a default world or default SimulationManager; unknown external
entities raise an ownership error that tells the caller to pass `context=`.
Expose `register_entity(entity, ref=None)`, `register_entities(entities)`, and
`entity_ref(entity)`; the last method returns the DexSim stable reference
recorded during Newton attachment.

Use homogeneous composition for both conversion directions:

```python
def _xyzw_pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    matrix = torch.eye(4, dtype=pose.dtype, device=pose.device).repeat(
        pose.shape[0], 1, 1
    )
    matrix[:, :3, 3] = pose[:, :3]
    matrix[:, :3, :3] = matrix_from_quat(
        convert_quat(pose[:, 3:7], to="wxyz")
    )
    return matrix


def _matrix_to_xyzw_pose(matrix: torch.Tensor) -> torch.Tensor:
    quat = convert_quat(quat_from_matrix(matrix[:, :3, :3]), to="xyzw")
    return torch.cat((matrix[:, :3, 3], quat), dim=-1)


def local_to_world(self, pose: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    local = _xyzw_pose_to_matrix(pose)
    world = torch.bmm(self.arena_transforms[env_ids.long()], local)
    return _matrix_to_xyzw_pose(world)


def world_to_local(self, pose: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    world = _xyzw_pose_to_matrix(pose)
    local = torch.bmm(self.inverse_arena_transforms[env_ids.long()], world)
    return _matrix_to_xyzw_pose(local)
```

- [ ] **Step 7: Run config/context/capability tests**

```bash
pytest -q tests/sim/test_backend_parity.py tests/sim/test_newton_scene_context.py tests/sim/test_physics_attrs.py
```

Expected: all pass without creating a real simulation for pure contract cases.

- [ ] **Step 8: Commit the EmbodiChain contract foundation**

```bash
git add pyproject.toml embodichain/lab/sim/cfg.py embodichain/lab/sim/physics/context.py embodichain/lab/sim/physics/base.py embodichain/lab/sim/physics/default.py embodichain/lab/sim/physics/newton.py embodichain/lab/sim/physics/__init__.py tests/sim/newton_contract_test_utils.py tests/sim/test_backend_parity.py tests/sim/test_newton_scene_context.py
git commit -m "refactor(sim): add explicit physics scene contracts"
```

---

### Task 8: Route EmbodiChain spawning and objects through explicit ownership

**Files:**

- Modify: `embodichain/lab/sim/common.py`
- Modify: `embodichain/lab/sim/sim_manager.py`
- Modify: `embodichain/lab/sim/utility/sim_utils.py`
- Modify: `embodichain/lab/sim/objects/rigid_object.py`
- Modify: `embodichain/lab/sim/objects/articulation.py`
- Modify: `embodichain/lab/sim/objects/robot.py`
- Modify: `embodichain/lab/sim/objects/backends/default.py`
- Modify: `tests/sim/test_newton_scene_context.py`
- Modify: `tests/sim/test_newton_finalize_lifecycle.py`

**Interfaces:**

- Produces: manager-owned context passed into all physical objects/views.
- Produces: explicit pending-initialization queue; constructors never invoke overridable `reset()`.
- Consumes: `BackendSceneContext` and DexSim `attach_rigid_body`.

- [ ] **Step 1: Add failing no-global and no-constructor-reset tests**

```python
def test_rigid_and_articulation_construction_do_not_use_default_world(
    monkeypatch, newton_sim
):
    monkeypatch.setattr(dexsim, "default_world", lambda: (_ for _ in ()).throw(AssertionError("global")))
    rigid = newton_sim.add_rigid_object(box_cfg("box"))
    art = newton_sim.add_articulation(arm_cfg("arm"))
    assert rigid.context is newton_sim.scene_context
    assert art.context is newton_sim.scene_context


def test_physical_object_context_keyword_is_source_compatible():
    assert inspect.signature(RigidObject).parameters["context"].default is None
    assert inspect.signature(Articulation).parameters["context"].default is None


def test_batch_entity_constructor_never_calls_virtual_reset():
    class Probe(BatchEntity):
        def reset(self, env_ids=None):
            raise AssertionError("virtual reset from base constructor")
        def set_local_pose(self, pose, env_ids=None):
            return None
        def get_local_pose(self, to_matrix=False):
            return torch.zeros(1, 7)
    Probe(
        cfg=ObjectBaseCfg(uid="probe"),
        entities=[object()],
        device=torch.device("cpu"),
    )
```

- [ ] **Step 2: Run focused tests and confirm global lookup/base reset failures**

```bash
pytest -q tests/sim/test_newton_scene_context.py tests/sim/test_newton_finalize_lifecycle.py
```

- [ ] **Step 3: Remove base virtual reset and pass context from the manager**

`BatchEntity.__init__` stores fields only. Keep the `auto_reset` keyword for
source compatibility, but ignore it and emit a one-time deprecation warning
when `True`; no virtual method is called.

Create `self.scene_context = BackendSceneContext(self)` immediately after
backend activation and arena construction. Pass it explicitly:

```python
rigid_obj = RigidObject(
    cfg=cfg,
    entities=obj_list,
    device=self.device,
    context=self.scene_context,
)
self.physics.queue_initialization(rigid_obj)
```

Use the same pattern for articulations and robots. Default-only light and rigid
group constructors call their own `reset()` explicitly after all subclass
fields are initialized, preserving current behavior.

Add `context: BackendSceneContext | None = None` as the final keyword to rigid,
articulation, and robot constructors. Resolve `None` with
`BackendSceneContext.for_entities(entities)`. The manager registers spawned
entities against its context before constructing their wrapper, so existing
positional constructor calls retain their signature and no global owner lookup
is needed.

- [ ] **Step 4: Replace EmbodiChain private Newton attachment**

In `_attach_newton_rigidbody_desc`, retain EmbodiChain descriptor resolution
and warnings, then call only:

```python
manager = context.physics.newton_manager
entity_ref = manager.attach_rigid_body(
    obj,
    actor_type=body_type,
    shape_type=shape_type,
    body_desc=body,
    shape_desc=shape,
)
context.register_entity(obj, entity_ref)
```

Delete imports of `register_mesh_object_to_newton_patch`,
`_get_entity_native_handle`, writes to `mgr.dexsim_meta`, and hard-coded world
`-1`. The standard legacy `add_rigidbody` route stores the returned reference
through the same context registration method.

Thread `context` through `load_mesh_objects_from_cfg`,
`spawn_rigid_object_entities`, `spawn_articulation_entities`, and
`spawn_usd_articulation_entities`. Compatibility defaults resolve from the
provided entities; the SimulationManager core path always passes its context.
Replace `_is_newton_backend_active()`, `_newton_solver_type()`, and
`get_dexsim_arenas()` use in physical spawn/object paths with context
properties. Object `destroy()` methods remove entities from their owning
context arenas, never from a default world.

- [ ] **Step 5: Run spawn and default-backend regressions**

```bash
pytest -q tests/sim/test_newton_scene_context.py tests/sim/test_newton_finalize_lifecycle.py
pytest -q tests/sim/objects/test_rigid_object.py tests/sim/objects/test_articulation.py -k "constructor or spawn or desc_native"
```

Expected: selected tests pass; no core object construction resolves a default
world.

- [ ] **Step 6: Commit explicit ownership and spawning**

```bash
git add embodichain/lab/sim/common.py embodichain/lab/sim/sim_manager.py embodichain/lab/sim/utility/sim_utils.py embodichain/lab/sim/objects/rigid_object.py embodichain/lab/sim/objects/articulation.py embodichain/lab/sim/objects/robot.py embodichain/lab/sim/objects/backends/default.py tests/sim/test_newton_scene_context.py tests/sim/test_newton_finalize_lifecycle.py
git commit -m "refactor(sim): make physical object ownership explicit"
```

---

### Task 9: Rebind rigid views by generation and initialize only new objects

**Files:**

- Create: `tests/sim/test_newton_rebuild_bindings.py`
- Modify: `embodichain/lab/sim/physics/newton.py`
- Modify: `embodichain/lab/sim/objects/backends/newton.py`
- Modify: `embodichain/lab/sim/objects/rigid_object.py`
- Modify: `tests/sim/objects/test_rigid_object.py`
- Modify: `tests/sim/test_newton_finalize_lifecycle.py`

**Interfaces:**

- Produces: `NewtonRigidBodyView._ensure_binding()` with O(1) generation check.
- Produces: event-driven cache invalidation and exact pending initialization.
- Consumes: DexSim `RigidEntityBinding`, prepare result, and rebuild event.

- [ ] **Step 1: Add failing rebind/state-preservation/initialization tests**

```python
def test_rigid_view_refreshes_once_after_generation_change(newton_sim):
    old = newton_sim.add_rigid_object(box_cfg("old", z=1.0))
    first = newton_sim.physics.prepare()
    old_generation = old._data.body_view.binding.generation
    old_pose = old.get_local_pose().clone()
    new = newton_sim.add_rigid_object(box_cfg("new", z=2.0))
    second = newton_sim.physics.prepare()
    assert second.generation == first.generation + 1
    assert old_generation == first.generation
    assert old._data.body_view.binding.generation == second.generation
    assert new._data.body_view.binding.generation == second.generation
    assert torch.allclose(old.get_local_pose(), old_pose, atol=1e-5)
    assert torch.allclose(new.get_local_pose()[:, 2], torch.tensor([2.0] * new.num_instances))


def test_existing_object_is_not_reset_on_rebuild(newton_sim, mocker):
    old = newton_sim.add_rigid_object(box_cfg("old"))
    newton_sim.physics.prepare()
    reset = mocker.spy(old, "reset")
    newton_sim.add_rigid_object(box_cfg("new"))
    newton_sim.physics.prepare()
    reset.assert_not_called()
```

- [ ] **Step 2: Run the new file and observe stale IDs/global reset**

```bash
pytest -q tests/sim/test_newton_rebuild_bindings.py
```

Expected: stale binding or `_reset_entities_after_finalize` resets existing
objects.

- [ ] **Step 3: Replace permanent IDs with one binding per view**

```python
def _ensure_binding(self) -> RigidEntityBinding:
    generation = self._context.generation
    if self._binding is None or self._binding.generation != generation:
        refs = tuple(self._context.entity_ref(entity) for entity in self._entities)
        self._binding = self._manager.bind_rigid_entities(refs, device=self._device)
        self._invalidate_derived_caches()
    self._binding.assert_current(self._manager)
    return self._binding

@property
def binding(self) -> RigidEntityBinding:
    return self._ensure_binding()
```

Every fetch/apply operation calls this once and passes its batched device IDs
to `NewtonPhysicsScene`. Remove permanent `_body_ids`, sorted-ID, and XY-offset
caches; rebuild derived caches only inside `_invalidate_derived_caches()`.

- [ ] **Step 4: Subscribe backend lifecycle and initialize only prepare additions**

`NewtonPhysicsBackend.activate()` subscribes to model rebuilt events and marks
registered views dirty. `queue_initialization(obj)` stores object identity and
its stable refs. After `prepare()` and view rebind, initialize only queued
objects whose refs appear in `result.added_entities`, then remove them from the
queue. First build follows the same path. Existing objects are never reset by a
rebuild.

- [ ] **Step 5: Convert rigid poses with full context transforms**

`fetch_pose()` converts DexSim world poses to arena-local with
`context.world_to_local`; `apply_pose()` converts local to world with
`context.local_to_world`. Remove all `[:2, 3]`, XY-only, and
`get_all_arenas()` conversion logic from the Newton view.

- [ ] **Step 6: Run rigid and lifecycle tests**

```bash
pytest -q tests/sim/test_newton_rebuild_bindings.py tests/sim/test_newton_finalize_lifecycle.py tests/sim/objects/test_rigid_object.py -k "Newton or newton or local_pose or reset"
```

Expected: selected tests pass for 1, 2, and 8 arenas, including a rotated
arena fixture.

- [ ] **Step 7: Commit rigid rebinding**

```bash
git add embodichain/lab/sim/physics/newton.py embodichain/lab/sim/objects/backends/newton.py embodichain/lab/sim/objects/rigid_object.py tests/sim/test_newton_rebuild_bindings.py tests/sim/objects/test_rigid_object.py tests/sim/test_newton_finalize_lifecycle.py
git commit -m "fix(newton): rebind rigid views after runtime rebuild"
```

---

### Task 10: Rebind articulation views, separate q/qd widths, and enforce FK/capabilities

**Files:**

- Modify: `embodichain/lab/sim/objects/backends/newton.py`
- Modify: `embodichain/lab/sim/objects/articulation.py`
- Modify: `embodichain/lab/sim/objects/robot.py`
- Modify: `tests/sim/test_newton_rebuild_bindings.py`
- Modify: `tests/sim/objects/test_articulation.py`
- Modify: `tests/sim/objects/test_rigid_object.py`
- Modify: `tests/sim/objects/test_robot.py`

**Interfaces:**

- Produces: `NewtonArticulationView._ensure_binding()` and real `compute_kinematics()`.
- Produces: `_articulation_buffer_shapes(num_instances, qpos_width, qvel_width, target_qpos_width, target_qvel_width)` used by `ArticulationData`.
- Produces: explicit unsupported-operation errors for q acceleration and other absent capabilities.
- Consumes: DexSim `ArticulationBinding` and full context transforms.

- [ ] **Step 1: Add failing articulation rebind, frame, width, and FK tests**

```python
def test_articulation_view_rebinds_and_preserves_targets(newton_sim):
    arm = newton_sim.add_articulation(arm_cfg("arm"))
    newton_sim.physics.prepare()
    q = torch.full((arm.num_instances, arm.qpos_width), 0.1, device=arm.device)
    qd = torch.full((arm.num_instances, arm.qvel_width), 0.2, device=arm.device)
    target_q = torch.full(
        (arm.num_instances, arm.target_qpos_width),
        -0.1,
        device=arm.device,
    )
    target_qd = torch.full(
        (arm.num_instances, arm.target_qvel_width),
        -0.2,
        device=arm.device,
    )
    arm.set_qpos(q, target=False)
    arm.set_qvel(qd, target=False)
    arm.set_qpos(target_q, target=True)
    arm.set_qvel(target_qd, target=True)
    before_link = arm.get_link_pose(arm.link_names[-1]).clone()
    newton_sim.add_rigid_object(box_cfg("topology_change"))
    newton_sim.physics.prepare()
    assert torch.allclose(arm.get_qpos(), q)
    assert torch.allclose(arm.get_qvel(), qd)
    assert torch.allclose(arm.get_qpos(target=True), target_q)
    assert torch.allclose(arm.get_qvel(target=True), target_qd)
    assert torch.allclose(arm.get_link_pose(arm.link_names[-1]), before_link, atol=1e-5)


def test_qpos_write_updates_link_pose_without_physics_step(newton_sim):
    arm = newton_sim.add_articulation(arm_cfg("arm"))
    newton_sim.physics.prepare()
    before = arm.get_link_pose(arm.link_names[-1]).clone()
    q = arm.get_qpos().clone()
    q[:, 0] += 0.2
    arm.set_qpos(q, target=False)
    after = arm.get_link_pose(arm.link_names[-1])
    assert not torch.allclose(after, before)


def test_newton_qacc_is_explicitly_unsupported(newton_sim):
    arm = newton_sim.add_articulation(arm_cfg("arm"))
    newton_sim.physics.prepare()
    with pytest.raises(NotImplementedError, match="acceleration"):
        _ = arm.body_data.qacc
```

Add pure allocation coverage for distinct widths:

```python
def test_articulation_buffer_shapes_keep_q_and_qd_widths_distinct():
    shapes = _articulation_buffer_shapes(
        num_instances=3,
        qpos_width=7,
        qvel_width=6,
        target_qpos_width=6,
        target_qvel_width=6,
    )
    assert shapes["qpos"] == (3, 7)
    assert shapes["target_qpos"] == (3, 6)
    assert shapes["qvel"] == (3, 6)
    assert shapes["target_qvel"] == (3, 6)
    assert shapes["qf"] == (3, 6)
```

- [ ] **Step 2: Run the selected articulation tests and confirm stale/FK/zero-qacc failures**

```bash
pytest -q tests/sim/test_newton_rebuild_bindings.py tests/sim/objects/test_articulation.py -k "Newton or newton or qpos or qacc or link_pose"
```

- [ ] **Step 3: Bind articulation IDs, link IDs, and spans as one unit**

Implement the same O(1) generation guard as the rigid view. Resolve root/link
IDs and joint spans only through `manager.bind_articulations`; delete direct
reads of `dexsim_meta_links`, `get_gpu_index()`, and permanent articulation ID
lists from the Newton view.

Expose and use separate widths:

```python
@property
def qpos_width(self) -> int:
    return self._ensure_binding().qpos_width

@property
def qvel_width(self) -> int:
    return self._ensure_binding().qvel_width

@property
def target_qpos_width(self) -> int:
    return self._ensure_binding().target_qpos_width

@property
def target_qvel_width(self) -> int:
    return self._ensure_binding().target_qvel_width
```

Allocate current `qpos` with `qpos_width`, current `qvel/qf` with
`qvel_width`, and targets with their explicit binding widths. For DexSim 0.4.4
both target position and target velocity are per-DOF and therefore their
widths equal `qvel_width`, but callers consume the explicit properties rather
than inferring that relationship. Existing `dof` remains a compatibility alias
for all-1-DOF assets and raises a clear error when its old ambiguous meaning
would truncate a non-scalar joint.
Expose matching read-only `Articulation.qpos_width` and
`Articulation.qvel_width` properties plus `target_qpos_width` and
`target_qvel_width` that delegate to the view.

```python
def _articulation_buffer_shapes(
    num_instances: int,
    qpos_width: int,
    qvel_width: int,
    target_qpos_width: int,
    target_qvel_width: int,
) -> dict[str, tuple[int, int]]:
    return {
        "qpos": (num_instances, qpos_width),
        "target_qpos": (num_instances, target_qpos_width),
        "qvel": (num_instances, qvel_width),
        "target_qvel": (num_instances, target_qvel_width),
        "qf": (num_instances, qvel_width),
    }
```

- [ ] **Step 4: Implement frame-correct root/link reads and FK**

Convert root and link world poses with the complete arena transform. On current
q writes, call the DexSim public FK invalidation/evaluation path for the
affected articulation IDs. Implement `compute_kinematics(env_ids)` by mapping
the selected environment rows through the binding's `articulation_ids_host`,
constructing a boolean mask of length `model.articulation_count`, and calling
the public manager FK method; it must not be a no-op.

Replace fabricated q acceleration with:

```python
raise NotImplementedError(
    "Newton articulation joint acceleration is not exposed by DexSim 0.4.4."
)
```

Use the same explicit pattern for unsupported runtime collision-filter or
sensor operations; do not return plausible zeros or success.

Add `@pytest.mark.gpu` to the Newton-backed rigid, articulation, and robot test
classes because they configure `device="cuda"` even though their node IDs do
not contain `cuda`. This keeps them out of the CPU job and includes them in the
serial `--run-gpu -m gpu` merge gate.

- [ ] **Step 5: Run articulation and robot regressions**

```bash
pytest -q tests/sim/test_newton_rebuild_bindings.py tests/sim/objects/test_articulation.py tests/sim/objects/test_robot.py -k "Newton or newton"
```

Expected: Newton articulation/robot selections pass; skips only correspond to
capabilities explicitly outside Stage 1.

- [ ] **Step 6: Commit articulation rebinding**

```bash
git add embodichain/lab/sim/objects/backends/newton.py embodichain/lab/sim/objects/articulation.py embodichain/lab/sim/objects/robot.py tests/sim/test_newton_rebuild_bindings.py tests/sim/objects/test_articulation.py tests/sim/objects/test_rigid_object.py tests/sim/objects/test_robot.py
git commit -m "fix(newton): bind articulation state by model generation"
```

---

### Task 11: Complete SimulationManager mutation, exact update, multi-instance, and close lifecycle

**Files:**

- Create: `tests/sim/test_newton_multi_manager.py`
- Modify: `embodichain/lab/sim/sim_manager.py`
- Modify: `embodichain/lab/sim/physics/base.py`
- Modify: `embodichain/lab/sim/physics/default.py`
- Modify: `embodichain/lab/sim/physics/newton.py`
- Modify: `tests/sim/test_newton_finalize_lifecycle.py`
- Modify: `tests/sim/test_newton_rebuild_bindings.py`

**Interfaces:**

- Produces: idempotent `SimulationManager.close()` and close-before-remove `reset()`.
- Produces: exact requested-step update after prepare.
- Consumes: pending initialization and per-world DexSim close.

- [ ] **Step 1: Add failing update/remove/close/two-manager tests**

```python
def test_update_runs_exact_requested_steps_after_rebuild(newton_sim):
    box = newton_sim.add_rigid_object(box_cfg("box", z=1.0))
    manager = newton_sim.newton_manager
    before = manager._sim_time
    newton_sim.update(physics_dt=0.01, step=3)
    assert manager._sim_time == pytest.approx(before + 0.03)
    newton_sim.add_rigid_object(box_cfg("new", z=2.0))
    before = manager._sim_time
    newton_sim.update(physics_dt=0.01, step=2)
    assert manager._sim_time == pytest.approx(before + 0.02)


def test_remove_invalidates_and_survivor_rebinds(newton_sim):
    keep = newton_sim.add_rigid_object(box_cfg("keep"))
    remove = newton_sim.add_rigid_object(box_cfg("remove"))
    newton_sim.physics.prepare()
    keep_pose = keep.get_local_pose().clone()
    assert newton_sim.remove_asset("remove") is True
    result = newton_sim.physics.prepare()
    assert result.did_rebuild is True
    assert torch.allclose(keep.get_local_pose(), keep_pose, atol=1e-5)
    with pytest.raises(Exception, match="removed|closed|stale"):
        remove.get_local_pose()


def test_close_and_reset_are_idempotent(newton_sim):
    instance_id = newton_sim.instance_id
    world = newton_sim.get_world()
    newton_sim.close()
    newton_sim.close()
    assert newton_sim.is_closed is True
    assert SimulationManager.is_instantiated(instance_id) is False
    SimulationManager.reset(instance_id)
    assert dexsim.engine.newton_physics.get_newton_manager(world) is None


@pytest.mark.gpu
def test_two_same_device_managers_are_isolated():
    def make_sim():
        return SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cuda:0",
                num_envs=2,
                physics_cfg=NewtonPhysicsCfg(
                    device="cuda:0", use_cuda_graph=False
                ),
            )
        )

    first = make_sim()
    second = make_sim()
    try:
        first_box = first.add_rigid_object(box_cfg("first"))
        second_box = second.add_rigid_object(box_cfg("second"))
        first.update(physics_dt=0.01, step=1)
        second.update(physics_dt=0.01, step=1)
        assert first.get_world() is not second.get_world()
        assert first.get_physics_scene() is not second.get_physics_scene()
        assert first.scene_context is not second.scene_context
        assert first.newton_manager.world_token != second.newton_manager.world_token
        first.add_rigid_object(box_cfg("first_new"))
        first.update(physics_dt=0.01, step=1)
        assert first.newton_manager.model_generation == 2
        assert second.newton_manager.model_generation == 1
        first.close()
        second.update(physics_dt=0.01, step=1)
        second_pose = second_box.get_local_pose().clone()
        assert torch.isfinite(second_pose).all()
        assert first_box.context is not second_box.context
    finally:
        first.close()
        second.close()
```

- [ ] **Step 2: Run the lifecycle files and confirm current reset/leak behavior**

```bash
pytest -q tests/sim/test_newton_finalize_lifecycle.py tests/sim/test_newton_rebuild_bindings.py tests/sim/test_newton_multi_manager.py --run-gpu
```

- [ ] **Step 3: Return prepare results and preserve exact update count**

Change `PhysicsBackend.prepare()` and `ensure_initialized()` to return
`PhysicsPrepareResult`. `SimulationManager.update()` calls prepare once, then
executes the existing world update loop exactly `step` times. It never adds a
warmup step and never drops the first requested step.

- [ ] **Step 4: Make every topology mutation invalidate and every removal close its view**

After successful rigid/articulation/robot add or remove, call
`physics.invalidate()`. Removal destroys the wrapper, unregisters its context
refs, and leaves surviving refs queued for rebind but not reset. Soft/cloth
add/remove on Newton raises `NotImplementedError` before mutating registries.

- [ ] **Step 5: Add idempotent close and safe registry allocation**

```python
def close(self) -> None:
    if self._is_closed:
        return
    self._is_closed = True
    first_error = None
    try:
        self.wait_window_record_saves()
        self.physics.close()
    except Exception as exc:
        first_error = exc
    try:
        if self._world is not None:
            self._world.quit()
    except Exception as exc:
        if first_error is None:
            first_error = exc
    finally:
        self._instances.pop(self.instance_id, None)
    if first_error is not None:
        raise first_error
```

`reset(instance_id)` calls `instance.close()` before removing it. Preserve
`destroy(exit_process=...)` as a wrapper around close plus its documented
process-exit policy. Allocate instance IDs monotonically rather than from
`len(_instances)`, so closing a non-last manager cannot overwrite a live entry.
DexSim manager close is idempotent, so the backend-close/world-quit sequence is
safe even when `World.quit()` invokes the same integration teardown again.

- [ ] **Step 6: Run lifecycle, multi-manager, and default-backend tests**

```bash
pytest -q tests/sim/test_newton_finalize_lifecycle.py tests/sim/test_newton_rebuild_bindings.py tests/sim/test_newton_multi_manager.py tests/sim/test_backend_parity.py tests/sim/objects/test_rigid_object.py tests/sim/objects/test_articulation.py --run-gpu
```

Expected: zero failures; fixture teardown finds no stale world or manager
registration.

- [ ] **Step 7: Commit manager lifecycle completion**

```bash
git add embodichain/lab/sim/sim_manager.py embodichain/lab/sim/physics/base.py embodichain/lab/sim/physics/default.py embodichain/lab/sim/physics/newton.py tests/sim/test_newton_finalize_lifecycle.py tests/sim/test_newton_rebuild_bindings.py tests/sim/test_newton_multi_manager.py
git commit -m "fix(sim): close and rebuild Newton managers safely"
```

---

### Task 12: Document the contract and run the Stage 1 merge gate

**Files:**

- Modify: `docs/source/overview/sim/sim_manager.md`
- Modify: `design/newton-backend-design.md`
- Verify: both repositories' Stage 1 diffs.

**Interfaces:**

- Consumes: all Stage 1 interfaces and tests.
- Produces: a verified foundation for the later Stage 2 differentiable plan.

- [ ] **Step 1: Update public documentation with executable examples**

Document:

```python
sim = SimulationManager(
    SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(device="cuda:0"),
        num_envs=4,
        headless=True,
    )
)
try:
    cube = sim.add_rigid_object(cube_cfg)
    sim.finalize_newton_physics()
    sim.update(step=1)
finally:
    sim.close()
```

State that prepare/finalize does not advance time, add/remove of rigid bodies
and articulations rebuilds transactionally, old runtime IDs must not be cached,
soft/cloth topology mutation is unsupported, q acceleration is unavailable,
and two managers are isolated. Mark the old Target 4 implementation claims as
superseded by the 2026-07-13 design and record only tests actually passing.

- [ ] **Step 2: Run DexSim formatting and the full Newton test directory**

```bash
cd /root/sources/dexsim
black --check --diff python/dexsim/engine/newton_physics python/test/engine/newton_physics
pytest -q python/test/engine/newton_physics
```

Expected: Black exits zero and pytest reports zero failures.

- [ ] **Step 3: Run EmbodiChain focused CPU/headless tests**

```bash
cd /root/sources/EmbodiChain
pytest -q tests/sim/test_backend_parity.py tests/sim/test_physics_attrs.py tests/sim/test_newton_scene_context.py tests/sim/test_newton_finalize_lifecycle.py
```

Expected: zero failures.

- [ ] **Step 4: Run EmbodiChain serial GPU Newton integration tests**

```bash
pytest -q tests/sim/test_newton_rebuild_bindings.py tests/sim/test_newton_multi_manager.py tests/sim/objects/test_rigid_object.py tests/sim/objects/test_articulation.py tests/sim/objects/test_robot.py --run-gpu -m gpu
```

Expected: zero failures; skips are listed and checked against structured
capabilities.

- [ ] **Step 5: Run the complete EmbodiChain regression and docs build**

```bash
pytest -q tests
black --check --diff --color ./
LC_ALL=C.UTF-8 LANG=C.UTF-8 make -C docs html
```

Expected: zero pytest failures, Black leaves all files unchanged, and Sphinx
builds without new warnings/errors.

- [ ] **Step 6: Inspect both diffs and dependency/API versions**

```bash
if rg -n "dexsim\.default_world\(\)|get_physics_scene\(\)|SimulationManager\.get_instance\(" embodichain/lab/sim/objects/rigid_object.py embodichain/lab/sim/objects/articulation.py embodichain/lab/sim/objects/backends/newton.py; then
  echo "core Newton object/view path still contains global owner lookup" >&2
  exit 1
fi
if rg -n "register_mesh_object_to_newton_patch|_get_entity_native_handle|dexsim_meta" embodichain/lab/sim/utility/sim_utils.py embodichain/lab/sim/objects/backends/newton.py; then
  echo "EmbodiChain still consumes private DexSim Newton integration state" >&2
  exit 1
fi
git -C /root/sources/dexsim diff --check dev...HEAD
git -C /root/sources/dexsim log --oneline --decorate dev..HEAD
git -C /root/sources/EmbodiChain diff --check main...HEAD
git -C /root/sources/EmbodiChain log --oneline --decorate main..HEAD
python - <<'PY'
import dexsim
from dexsim.engine.newton_physics import NEWTON_INTEGRATION_API_VERSION
assert dexsim.__version__.split("+")[0] == "0.4.4"
assert NEWTON_INTEGRATION_API_VERSION == 2
print(dexsim.__version__, NEWTON_INTEGRATION_API_VERSION)
PY
```

Expected: no whitespace errors, reviewable commit series in each repository,
package base version `0.4.4`, API version `2`.

- [ ] **Step 7: Commit Stage 1 documentation**

```bash
git add docs/source/overview/sim/sim_manager.md design/newton-backend-design.md
git commit -m "docs: describe Newton runtime lifecycle contracts"
```

- [ ] **Step 8: Stop at the Stage 1 review gate**

Report exact command outputs, failures/skips, both branch heads, and remaining
known upstream limitations. Do not start Stage 2. After Stage 1 is accepted,
use `superpowers:brainstorming` only if Stage 1 changed the approved design;
otherwise use `superpowers:writing-plans` to create the dependent
differentiable dynamics/kinematics implementation plan.

---

## Spec Coverage Checklist

| Specification requirement | Implemented by |
|---|---|
| DexSim public API/version/generation/prepare | Tasks 1–3 |
| Public rigid attachment and canonical descriptors | Task 2 |
| Generation-aware rigid/articulation bindings | Tasks 3, 9, 10 |
| Transactional rebuild and rigid state preservation | Task 4 |
| Active model-generation lease blocks rebuild | Tasks 1, 4 |
| Articulation state/control preservation and q/qd spans | Task 5 |
| Full arena/world frame conversion | Tasks 7, 9, 10 |
| No global/default-world core ownership | Tasks 7–8 |
| Pending initialization without constructor virtual reset | Tasks 8–9 |
| Two same-device worlds/managers and deterministic cleanup | Tasks 6, 11 |
| Structured capabilities and explicit unsupported errors | Tasks 7, 10 |
| Exact prepare/step count | Tasks 1, 4, 11 |
| Existing public API and default backend compatibility | Tasks 7–12 |
| Documentation and complete merge gate | Task 12 |
