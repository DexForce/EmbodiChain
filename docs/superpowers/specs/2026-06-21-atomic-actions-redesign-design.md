# Atomic Actions Redesign — Design Spec

**Date:** 2026-06-21
**Scope:** `embodichain/lab/sim/atomic_actions/` (`core.py`, `actions.py`, `engine.py`) and the three tutorial scripts that drive it.
**Status:** Design — pending review.

---

## 1. Motivation

The atomic actions package on branch `ljd/move_object_atomic_actions` introduces a unified abstraction for the reach / pick / move-object / place primitives that the upper layers (RL tasks, expert demos, agent skills) will eventually drive. The current implementation captures the right ideas — batched tensor signatures, a `HeldObjectState` to bridge pickup → move → place, `@configclass` configs, a registry — but the abstraction has accreted enough ceremony that adding a new action is harder than it should be and the existing four are coupled by inheritance in ways that no longer serve them.

Concretely:

- `HeldObjectState` flows through four channels at once (`kwargs["action_context"]`, `kwargs["held_object_state"]`, `self._held_object_state`, `get_held_object_state()` + `ClassVar updates_held_object_state`).
- The `target` parameter is a five-way `Union` resolved by a 70-line mini-DSL inside the engine (`_resolve_target` accepts `Tensor | str | ObjectSemantics | MoveObjectTarget | Dict[str, Any]`).
- `PickUpAction`, `PlaceAction`, and `MoveObjectAction` inherit from `MoveAction` (directly or via `_HandCloseAction`) only to share helper methods, not because they are kinds-of `MoveAction`.
- The engine's `execute_static` couples step order to construction order — there is no name-keyed sequencing API.
- `SemanticAnalyzer` is a stub that returns hardcoded defaults keyed by string label; it adds no real value in the runtime path.
- `ObjectSemantics.__post_init__` aliases its `geometry` dict into the `Affordance`, creating a hidden mutation hazard.
- `validate(...)` is declared on the ABC but every concrete subclass returns `True  # TODO`.
- The return tuple `(success, trajectory, joint_ids)` requires the engine to re-index `traj_full[:, :, joint_ids] = traj` and forces callers to inspect `joint_ids` to know which columns are arm vs hand.

Total package size today is ~1,900 lines, and the two driving tutorial scripts are ~450 lines each, which is a strong signal that the abstraction is leaking ceremony into call sites.

The only downstream consumers today are:

- `scripts/tutorials/sim/atomic_actions.py`
- `scripts/tutorials/atomic_action/pickup_atomic_actions.py`
- `scripts/tutorials/atomic_action/move_object_atomic_actions.py`
- `tests/sim/atomic_actions/` (test_core, test_actions, test_engine)

No production task envs, RL code, agent code, or solvers import this package. The redesign can therefore be a hard cut on the same branch with all four call-site groups migrated together — no deprecation window required.

## 2. Goals

1. Replace the `Union` + dict-DSL target with **typed, disjoint target dataclasses**, one per action.
2. Replace the four-channel held-state passing with a **single explicit `WorldState`** threaded through every `execute` call.
3. Replace the `MoveAction`-as-parent inheritance tree with **composition** — actions hold a stateless `TrajectoryBuilder` helper rather than inheriting from each other.
4. Give the engine a **name-keyed, sequence-as-data** API (`engine.run([(name, target), ...])`) so step order lives at the call site, not in registration order.
5. Drop the `SemanticAnalyzer` and the dict/string target-resolution path from the engine.
6. Fix the `Affordance` / `ObjectSemantics` geometry aliasing.
7. Either implement `validate` or remove it from the contract.
8. Make every action return a single full-robot trajectory `(n_envs, n_waypoints, robot.dof)`, removing the per-call `joint_ids` re-indexing.

## 3. Non-Goals

- Adding new action types (push, rotate, slide). The redesign keeps the same four primitives (`move`, `pick_up`, `move_object`, `place`).
- Changing the grasp generator (`AntipodalAffordance` / `GraspGenerator`) internals.
- Changing the motion-planner / TOPP-RA / IK-solver layers.
- Changing `BatchEntity` or the `Robot` object's contract for `compute_ik` / `compute_fk` / `get_joint_ids`.
- Removing the `@configclass` pattern — keep all `*ActionCfg` classes, only reshape the inheritance.
- Replacing the global function-style `register_action` / `unregister_action` / `get_registered_actions` registry; it stays for third-party extension.
- Backward-compatible shims — the four call sites are updated in the same PR.

## 4. Proposed Architecture

### 4.1 Module Layout

```
embodichain/lab/sim/atomic_actions/
├── __init__.py              # public exports
├── core.py                  # ~150 lines — typed primitives + AtomicAction ABC + WorldState + ActionResult
├── trajectory.py            # ~250 lines — NEW: stateless TrajectoryBuilder (helpers extracted from MoveAction)
├── affordance.py            # ~200 lines — NEW: Affordance, AntipodalAffordance, InteractionPoints (extracted from core.py)
├── actions.py               # ~550 lines — four concrete actions, each inherits AtomicAction directly
└── engine.py                # ~80 lines — registry + name-keyed runner
```

Splitting `core.py` is mostly bookkeeping; the meaningful change is **`trajectory.py`** (composition target) and the disappearance of `_HandCloseAction` + the `MoveAction`-as-parent role.

### 4.2 Typed Targets

In `core.py`, replace the polymorphic `target` with one dataclass per action:

```python
@dataclass(frozen=True)
class PoseTarget:
    """End-effector target pose. Used by MoveAction and PlaceAction."""
    xpos: torch.Tensor   # (4, 4) or (n_envs, 4, 4)

@dataclass(frozen=True)
class GraspTarget:
    """Pickup target. The grasp pose is solved from the affordance + entity at execute time."""
    semantics: ObjectSemantics

@dataclass(frozen=True)
class HeldObjectTarget:
    """Move the currently-held object to a desired object pose. Keeps the gripper closed."""
    object_target_pose: torch.Tensor   # (4, 4) or (n_envs, 4, 4)

Target = PoseTarget | GraspTarget | HeldObjectTarget
```

Every `AtomicAction` subclass declares the target type it accepts:

```python
class PickUpAction(AtomicAction):
    TargetType: ClassVar[type[Target]] = GraspTarget
    ...
```

The engine validates `isinstance(target, action.TargetType)` once before dispatch; the action body never re-validates.

The dict-DSL target format (`{"label": "apple", "geometry": ..., "custom_config": ...}`) and the bare string label format are **removed** from the engine. Building an `ObjectSemantics` from a label is a separate concern; if a tutorial or test needs it, it constructs `ObjectSemantics` itself (or via a future task-side helper that lives outside `atomic_actions/`).

### 4.3 `WorldState` and `ActionResult`

Replace the four-channel held-state passing with a single threaded value:

```python
@dataclass
class WorldState:
    """State the engine threads through a sequence of actions."""
    last_qpos: torch.Tensor                       # (n_envs, robot.dof)
    held_object: HeldObjectState | None = None

@dataclass
class ActionResult:
    success: bool
    trajectory: torch.Tensor                      # ALWAYS (n_envs, n_waypoints, robot.dof)
    next_state: WorldState                        # explicit successor state
```

The `AtomicAction.execute` signature becomes:

```python
def execute(self, target: Target, state: WorldState) -> ActionResult:
    ...
```

Removed:

- The `**kwargs` bag (`action_context`, `held_object_state`).
- The `updates_held_object_state: ClassVar[bool]` flag.
- The `get_held_object_state()` method on `AtomicAction`.
- The `self._held_object_state` instance attribute.
- The `(success, trajectory, joint_ids)` tuple return.

Engine behaviour after a successful action:

```python
state = result.next_state   # PickUp sets state.held_object; Place clears it; Move leaves it alone.
```

Actions construct the successor `WorldState` explicitly. There is exactly one place where held-object state lives in flight: in the `WorldState` value passed between actions.

### 4.4 `TrajectoryBuilder` (Composition Replaces Inheritance)

Extract every helper method currently sitting on `MoveAction` into a stateless collaborator (`trajectory.py`):

```python
class TrajectoryBuilder:
    """Stateless trajectory utilities shared by all atomic actions."""

    def __init__(self, motion_generator: MotionGenerator):
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = self.robot.device

    def plan_arm_traj(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
    ) -> tuple[bool, torch.Tensor]: ...

    def interpolate_hand_qpos(
        self,
        start_hand_qpos: torch.Tensor,
        end_hand_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor: ...

    def split_three_phase(
        self,
        sample_interval: int,
        hand_interp_steps: int,
        *,
        first_phase_ratio: float = 0.6,
    ) -> tuple[int, int, int]: ...

    def resolve_pose_target(
        self,
        target: torch.Tensor,
        n_envs: int,
    ) -> torch.Tensor: ...

    def resolve_start_qpos(
        self,
        start_qpos: torch.Tensor | None,
        n_envs: int,
        arm_dof: int,
        control_part: str,
    ) -> torch.Tensor: ...

    def apply_local_offset(
        self,
        pose: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor: ...

    def all_envs_success(self, is_success: bool | torch.Tensor) -> bool: ...
```

Every concrete action holds a `TrajectoryBuilder` and inherits directly from `AtomicAction`:

```python
class PickUpAction(AtomicAction):
    TargetType: ClassVar = GraspTarget

    def __init__(self, motion_generator, cfg: PickUpActionCfg | None = None):
        super().__init__(motion_generator, cfg or PickUpActionCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        ...
```

Killed:

- `MoveAction` is no longer a parent. It is a peer of `PickUp`, `Place`, `MoveObject` — same depth in the hierarchy.
- `_HandCloseAction` (the internal private parent for `MoveObjectAction`) is deleted; its helpers (`_expand_hand_qpos`, `_repeat_hand_qpos`) move to `TrajectoryBuilder`.
- `IK` and `FK` convenience methods on `AtomicAction` (`_ik_solve`, `_fk_compute`, `_apply_offset`) move to `TrajectoryBuilder`. The base class shrinks to ~20 lines.

This also unblocks a future `PushAction` or `RotateAction` that has no business inheriting from `MoveAction`.

### 4.5 Engine API

```python
class AtomicActionEngine:
    def __init__(self, motion_generator: MotionGenerator):
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = self.robot.device
        self._actions: dict[str, AtomicAction] = {}

    def register(self, action: AtomicAction, *, name: str | None = None) -> None:
        """Register a concrete action instance under its cfg.name (or override)."""
        ...

    def run(
        self,
        steps: list[tuple[str, Target]],
        state: WorldState | None = None,
    ) -> tuple[bool, torch.Tensor, WorldState]:
        """Run a named sequence of actions.

        Each step is (action_name, typed_target). State is threaded through;
        if not supplied, the engine seeds it from `robot.get_qpos()`.
        Returns (success, concatenated_trajectory, final_state).
        """
        ...
```

Removed from the engine:

- `_resolve_target` (the 70-line dict/string/Union DSL).
- `SemanticAnalyzer` (and its `_object_cache`, `analyze`, `clear_cache`).
- `_action_context` instance dict.
- `execute_static(target_list)` — replaced by `run(steps)` where step order is data.
- `validate(action_name, target, ...)` — see §4.7.

The global module-level `register_action` / `unregister_action` / `get_registered_actions` registry stays; it is the extension point for third-party action classes. The engine consults it during a `from_config` factory if/when we add one — for the first cut, callers instantiate actions and call `engine.register(...)` directly.

### 4.6 Affordance & ObjectSemantics

Fix the aliasing hazard:

- `Affordance` no longer stores a `geometry` dict that is mutated by `ObjectSemantics.__post_init__`.
- `Affordance` is constructed with the **specific tensors it needs** (e.g., `mesh_vertices`, `mesh_triangles` for `AntipodalAffordance`). The construction takes references, not aliases of a dict that another object also holds.
- `Affordance.mesh_vertices` / `mesh_triangles` properties either return the stored tensor or are removed; they no longer hide a dict lookup.
- `ObjectSemantics.__post_init__` no longer mutates the affordance. It still sets `affordance.object_label = self.label` if that field exists, but does not transplant `geometry`.

Concrete `AntipodalAffordance` constructor becomes:

```python
@dataclass
class AntipodalAffordance(Affordance):
    mesh_vertices: torch.Tensor | None = None
    mesh_triangles: torch.Tensor | None = None
    generator_cfg: GraspGeneratorCfg | None = None
    gripper_collision_cfg: GripperCollisionCfg | None = None
    force_reannotate: bool = False
    is_draw_grasp_xpos: bool = False
    # generator is lazily initialised on first use
    _generator: GraspGenerator | None = field(default=None, init=False, repr=False)
```

`ObjectSemantics` keeps `geometry` and `properties` as plain dicts — they are user metadata, not shared state.

### 4.7 `validate` — drop it

Every current implementation is `return True  # TODO`. We will remove `validate` from the `AtomicAction` ABC entirely. If a future use case (cheap pre-flight IK check at task-generation time) materialises, we will add it back with a concrete contract and at least one real implementation. Keeping a stub on the ABC for three months without ever using it has cost (it shows up in docs, in agent context, in skill scaffolds) and zero value.

### 4.8 Trajectory Shape

Every action returns `trajectory` shaped `(n_envs, n_waypoints, robot.dof)` — the full robot DoF, with arm columns set by the planner and hand columns padded with the appropriate qpos (`hand_open_qpos` during reach, `hand_close_qpos` during lift, etc., as today).

Inside `actions.py`, the helper assembles a full-DoF tensor by writing into `robot.get_joint_ids("arm")` and `robot.get_joint_ids("hand")` columns. This is the same code that lives in the engine today, just moved one level down.

The engine then becomes a `torch.cat` over the per-step full-DoF trajectories — no re-indexing.

## 5. Component Inventory After Refactor

### 5.1 `core.py` (~150 lines)

```python
# Held-state primitive (kept, signature unchanged)
@dataclass
class HeldObjectState:
    semantics: ObjectSemantics
    object_to_eef: torch.Tensor
    grasp_xpos: torch.Tensor

# Typed targets (NEW — replaces Union[...] + dict DSL)
@dataclass(frozen=True)
class PoseTarget: ...
@dataclass(frozen=True)
class GraspTarget: ...
@dataclass(frozen=True)
class HeldObjectTarget: ...
Target = PoseTarget | GraspTarget | HeldObjectTarget

# Engine ↔ action contract (NEW)
@dataclass
class WorldState:
    last_qpos: torch.Tensor
    held_object: HeldObjectState | None = None

@dataclass
class ActionResult:
    success: bool
    trajectory: torch.Tensor
    next_state: WorldState

# ObjectSemantics (kept, no more __post_init__ aliasing)
@dataclass
class ObjectSemantics:
    affordance: Affordance
    geometry: dict[str, Any]
    properties: dict[str, Any] = field(default_factory=dict)
    label: str = "none"
    entity: BatchEntity | None = None

# Config base (kept)
@configclass
class ActionCfg: ...

# Action ABC (slim)
class AtomicAction(ABC):
    TargetType: ClassVar[type[Target]]
    def __init__(self, motion_generator, cfg): ...
    @abstractmethod
    def execute(self, target: Target, state: WorldState) -> ActionResult: ...
```

### 5.2 `affordance.py` (~200 lines)

`Affordance`, `AntipodalAffordance`, `InteractionPoints` — extracted from current `core.py`, with the geometry-aliasing removed. `AntipodalAffordance.get_valid_grasp_poses` / `get_best_grasp_poses` / `_draw_grasp_xpos` are kept as-is in behaviour, just relocated.

### 5.3 `trajectory.py` (~250 lines)

`TrajectoryBuilder` — all the helpers currently on `MoveAction` / `AtomicAction`. Stateless given `motion_generator`. Used by composition.

### 5.4 `actions.py` (~550 lines)

```python
class MoveAction(AtomicAction):
    TargetType = PoseTarget
    def execute(self, target: PoseTarget, state: WorldState) -> ActionResult: ...

class PickUpAction(AtomicAction):
    TargetType = GraspTarget
    def execute(self, target: GraspTarget, state: WorldState) -> ActionResult: ...
    # sets next_state.held_object

class MoveObjectAction(AtomicAction):
    TargetType = HeldObjectTarget
    def execute(self, target: HeldObjectTarget, state: WorldState) -> ActionResult: ...
    # requires state.held_object; preserves it

class PlaceAction(AtomicAction):
    TargetType = PoseTarget
    def execute(self, target: PoseTarget, state: WorldState) -> ActionResult: ...
    # clears next_state.held_object
```

All four are **siblings** inheriting from `AtomicAction` directly. Each holds a `TrajectoryBuilder` for shared trajectory math. The `*ActionCfg` classes keep their existing `@configclass` shapes, but the inheritance chain among them is also flattened:

```
ActionCfg
├── MoveActionCfg
├── PickUpActionCfg   (used to extend GraspActionCfg → MoveActionCfg)
├── PlaceActionCfg    (used to extend GraspActionCfg → MoveActionCfg)
└── MoveObjectActionCfg (used to extend HandCloseActionCfg → MoveActionCfg)
```

`GraspActionCfg` and `HandCloseActionCfg` become **mixins or plain field bundles**, not parents. The exact representation (mixin vs explicit field repetition) is an implementation detail to be decided during the implementation plan — both work; the test is that no `*ActionCfg` ends up depending on `MoveActionCfg`'s fields it does not use.

### 5.5 `engine.py` (~80 lines)

```python
class AtomicActionEngine:
    def __init__(self, motion_generator): ...
    def register(self, action: AtomicAction, *, name: str | None = None) -> None: ...
    def run(
        self,
        steps: list[tuple[str, Target]],
        state: WorldState | None = None,
    ) -> tuple[bool, torch.Tensor, WorldState]: ...
```

Plus the existing module-level `register_action` / `unregister_action` / `get_registered_actions` for third-party action class registration (these are about **classes**, not engine instances — they survive).

## 6. Data Flow Example

Pickup → move → place from a tutorial:

```python
engine = AtomicActionEngine(motion_generator)
engine.register(PickUpAction(motion_generator, cfg=pickup_cfg))
engine.register(MoveObjectAction(motion_generator, cfg=move_cfg))
engine.register(PlaceAction(motion_generator, cfg=place_cfg))

apple_semantics = ObjectSemantics(
    affordance=AntipodalAffordance(mesh_vertices=v, mesh_triangles=t),
    geometry={"bounding_box": [0.07, 0.07, 0.07]},
    label="apple",
    entity=apple_entity,
)
target_pose = torch.eye(4); target_pose[:3, 3] = torch.tensor([0.4, 0.0, 0.1])
place_pose  = torch.eye(4); place_pose[:3, 3]  = torch.tensor([0.6, 0.0, 0.05])

success, traj, final_state = engine.run([
    ("pick_up",     GraspTarget(apple_semantics)),
    ("move_object", HeldObjectTarget(target_pose)),
    ("place",       PoseTarget(place_pose)),
])
```

- After `pick_up`, the engine's internal `state.held_object` is populated by `PickUpAction.execute(...).next_state`.
- `MoveObjectAction.execute` reads `state.held_object` directly. Raising if `None` is part of its contract.
- After `place`, `next_state.held_object` is `None`.

There is no `_action_context` dict, no `**kwargs` bag, no `ClassVar` flag, no `get_held_object_state()` method.

## 7. Migration

Because the only consumers are the three tutorial scripts and the test suite, all changes ship in the same PR. There is no deprecation window.

| File | Change |
|---|---|
| `embodichain/lab/sim/atomic_actions/__init__.py` | Update exports to the new public API. |
| `embodichain/lab/sim/atomic_actions/core.py` | Rewrite per §5.1. |
| `embodichain/lab/sim/atomic_actions/affordance.py` | New file, content extracted from old `core.py`. |
| `embodichain/lab/sim/atomic_actions/trajectory.py` | New file, content extracted from old `actions.py` helpers. |
| `embodichain/lab/sim/atomic_actions/actions.py` | Rewrite per §5.4. |
| `embodichain/lab/sim/atomic_actions/engine.py` | Rewrite per §5.5. |
| `scripts/tutorials/sim/atomic_actions.py` | Update to new engine API. Expected ~50% shorter. |
| `scripts/tutorials/atomic_action/pickup_atomic_actions.py` | Same. |
| `scripts/tutorials/atomic_action/move_object_atomic_actions.py` | Same. |
| `tests/sim/atomic_actions/test_core.py` | Rewrite for new typed targets + `WorldState`. |
| `tests/sim/atomic_actions/test_actions.py` | Rewrite per-action tests; drop mocks of removed channels. |
| `tests/sim/atomic_actions/test_engine.py` | Rewrite for `engine.run(steps, state)`; drop `_resolve_target` tests. |
| `docs/source/overview/sim/atomic_actions.md` | Update narrative to match new API. |
| `docs/source/tutorial/atomic_actions.rst` | Update tutorial walkthrough. |
| `agent_context/...` topic for `atomic_actions` | Update if present. |
| `.claude/skills/add-atomic-action/SKILL.md` | Update scaffold to emit new shape. |

## 8. Behavioural Equivalence

The redesign is a **pure refactor of the abstraction** — the trajectories generated end-to-end for the same inputs must be the same. Tests that today assert "given mocked motion_generator returning X, pickup-then-move-then-place produces trajectory Y" should pass on the new code without numerical drift, modulo:

- The `joint_ids` field disappears from the action return — equivalent tests now assert directly on the full-DoF trajectory tensor.
- `validate` tests are deleted along with the method.
- `_resolve_target` tests for dict/string formats are deleted along with the resolver.

## 9. Open Questions

These are deliberately deferred to the implementation plan, not to be answered in the spec:

1. **`*ActionCfg` flattening:** mixin classes vs explicit field repetition vs a `@configclass` "include" mechanism. Either works; pick during implementation.
2. **`engine.run` failure semantics:** today `execute_static` returns the partial trajectory accumulated up to the failing action. New `run` should preserve that. Confirm by reading the existing test, then carry the same behaviour.
3. **`TrajectoryBuilder` instance per action vs singleton on the engine:** today each action creates its own. Sharing one on the engine and injecting via `__init__` is slightly cleaner; deferred.
4. **`InteractionPoints` use sites:** the class exists in core but nothing currently consumes it. The redesign keeps it (extracted into `affordance.py`) on the bet that future actions will use it; if implementation reveals it is truly dead, drop it.

## 10. Acceptance Criteria

The redesign is done when:

1. `embodichain/lab/sim/atomic_actions/` matches §5: `core.py`, `affordance.py`, `trajectory.py`, `actions.py`, `engine.py`, `__init__.py`, with no remaining `_HandCloseAction`, no `_resolve_target`, no `SemanticAnalyzer`, no `action_context` kwarg, no `updates_held_object_state` ClassVar.
2. All four concrete action classes inherit directly from `AtomicAction`.
3. `AtomicAction.execute` signature is `(self, target: Target, state: WorldState) -> ActionResult`. No `**kwargs`.
4. Engine has `run(steps, state=None)`; no `execute_static`, no `validate`.
5. `ObjectSemantics.__post_init__` does not mutate `affordance.geometry`.
6. The three tutorial scripts each shrink by at least 30% line-count and run end-to-end producing visually-equivalent trajectories on the bundled demo scenes.
7. The test suite under `tests/sim/atomic_actions/` is rewritten and passes; total test count is the same order of magnitude (within ±20%).
8. `black .` + `/pre-commit-check` clean.
9. Sphinx docs build clean and the tutorial page reflects the new API.

---

**Next step:** invoke `superpowers:writing-plans` to turn this spec into an executable implementation plan.
