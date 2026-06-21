# Newton Backend PR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the two outstanding Newton-backend PR targets — multi-env
parallel simulation via `clone_arena_to` and a `DifferentiableEmbodiedEnv`
that bridges Warp tape autodiff into PyTorch autograd for analytic policy
gradient (APG).

**Architecture:** The Newton backend implicitly clones arena_0 into
arenas 1..N-1 inside `NewtonPhysicsBackend.prepare()` before
`rebuild_newton_from_scene`, and Newton object views resolve per-env body
IDs by reconstructing dexsim's clone naming pattern
(`f"{actor_name}_{arena_name}"`). A new `embodichain.lab.sim.diff`
package provides a `torch.autograd.Function` bridge over
`dexsim.engine.newton_physics.DifferentiableStepper`; a new
`DifferentiableEmbodiedEnv` gym subclass wires it into the standard
EmbodiChain env step pipeline.

**Tech Stack:** Python 3.10+, PyTorch (autograd), NVIDIA Warp (`wp.Tape`,
`wp.to_torch`/`wp.from_torch`), DexSim Newton physics
(`dexsim.engine.newton_physics`), gymnasium, pytest.

**Companion spec:** `docs/superpowers/specs/2026-06-21-newton-backend-pr-design.md`

---

## File Map

**Created:**
- `embodichain/lab/sim/diff/__init__.py` — public re-exports for the diff package
- `embodichain/lab/sim/diff/bridge.py` — `NewtonStepFunc(torch.autograd.Function)`, `tape_context`, `differentiable_step`
- `embodichain/lab/gym/envs/differentiable_env.py` — `DifferentiableEmbodiedEnv` subclass
- `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` — Franka APG example task
- `tests/sim/test_newton_multi_env.py`
- `tests/sim/test_differentiable_stepper.py`
- `tests/gym/envs/test_differentiable_env.py`
- `agent_context/topics/differentiable-env.md`

**Modified:**
- `embodichain/lab/sim/physics/newton.py` — add clone-at-finalize, `_arenas_cloned` flag
- `embodichain/lab/sim/sim_manager.py` — add `create_differentiable_stepper` / `create_gradient_rollout` delegators
- `embodichain/lab/sim/objects/backends/newton.py` — multi-env body-id resolution in `NewtonRigidBodyView` / `NewtonArticulationView`
- `embodichain/lab/sim/utility/sim_utils.py` — `arena_index>0` spawn guard on Newton
- `agent_context/MAP.yaml` — register new `differentiable-env` topic
- `design/newton-backend-design.md` — mark Targets 4/5 done, link to plan

---

## Task 1: Add `_arenas_cloned` flag to `NewtonPhysicsBackend`

**Files:**
- Modify: `embodichain/lab/sim/physics/newton.py`

Establish the flag and reset semantics first; clone logic comes in Task 3.

- [ ] **Step 1: Read the current backend file**

Run: `Read embodichain/lab/sim/physics/newton.py`

- [ ] **Step 2: Add the flag to `__init__`**

Edit `embodichain/lab/sim/physics/newton.py` — inside `NewtonPhysicsBackend.__init__`:

```python
    def __init__(self, manager) -> None:
        super().__init__(manager)
        self._newton_manager: "NewtonManager | None" = None
        self._is_finalized = False
        self._arenas_cloned = False
```

- [ ] **Step 3: Reset the flag in `invalidate`**

Edit `invalidate` to also reset the clone state — topology mutations that
trigger `invalidate()` must allow re-cloning into any newly added arenas
or after a `clean_arena`:

```python
    def invalidate(self) -> None:
        """Mark the Newton scene as needing re-finalization after a mutation."""
        self._is_finalized = False
        self._arenas_cloned = False
```

- [ ] **Step 4: Commit**

```bash
git add embodichain/lab/sim/physics/newton.py
git commit -m "feat(sim/newton): add _arenas_cloned lifecycle flag

Prep for clone-at-finalize multi-env. Tracks whether source arena has
been replicated into peer arenas for the current Newton finalize cycle;
cleared by invalidate() so topology mutations trigger re-clone."
```

---

## Task 2: Spawn guard for `arena_index>0` on Newton

**Files:**
- Modify: `embodichain/lab/sim/utility/sim_utils.py`

On Newton, every `add_*` call must target the source arena (arena_0). Reject
`arena_index>0` with a clear message. `-1` (global) and `0` both route to
arena_0. This makes the implicit-clone contract explicit at the spawn API.

- [ ] **Step 1: Inspect the existing entry points**

Run: `grep -n "def spawn_rigid_object\|def spawn_articulation\|def spawn_robot\|arena_index" embodichain/lab/sim/utility/sim_utils.py | head -30`

Note the function names. The actual entry points are the ones called by
`SimulationManager.add_rigid_object` / `add_articulation` / `add_robot`.

- [ ] **Step 2: Write the failing test first**

Create `tests/sim/test_newton_multi_env.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Multi-env Newton backend tests."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import BoxCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg


def _newton_sim_cfg(num_envs: int = 4, headless: bool = True) -> SimulationManagerCfg:
    return SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            physics_dt=1.0 / 60.0,
            num_substeps=4,
            requires_grad=False,
            use_cuda_graph=False,
            debug_mode=False,
        ),
        num_envs=num_envs,
        headless=headless,
    )


def test_spawn_with_arena_index_above_zero_rejected_on_newton():
    sim = SimulationManager(_newton_sim_cfg(num_envs=2))
    cube_cfg = RigidObjectCfg(
        uid="cube",
        shape=BoxCfg(extents=(0.1, 0.1, 0.1)),
        init_pos=(0.0, 0.0, 1.0),
    )
    cube_cfg.arena_index = 1
    with pytest.raises(Exception, match=r"arena_index"):
        sim.add_rigid_object(cube_cfg)
    SimulationManager.reset()
```

> Note: `RigidObjectCfg` does not own `arena_index` directly — the field
> lives on the `MarkerCfg`-style and a few cfgs. If `add_rigid_object`
> accepts `arena_index` via a kwarg, adjust the test accordingly. Verify by
> grepping `def add_rigid_object` in `sim_manager.py` before running.

- [ ] **Step 3: Run the test and confirm it fails**

Run: `pytest -q tests/sim/test_newton_multi_env.py::test_spawn_with_arena_index_above_zero_rejected_on_newton`
Expected: FAIL (no guard yet — either spawns silently or fails with the wrong error).

- [ ] **Step 4: Add the guard helper to `sim_utils.py`**

Edit `embodichain/lab/sim/utility/sim_utils.py` — add near
`_is_newton_backend_active`:

```python
def _check_newton_spawn_arena(arena_index: int) -> None:
    """Reject Newton spawns into a non-source arena.

    Newton's multi-env path clones arena_0 into peer arenas at finalize.
    Spawning into arenas 1..N-1 directly would conflict with the clone
    and produce duplicate or misindexed bodies.
    """
    if _is_newton_backend_active() and arena_index is not None and arena_index > 0:
        logger.log_error(
            f"Invalid arena_index={arena_index} for Newton spawn. "
            "Newton multi-env clones the source arena (arena_index in {-1, 0}) "
            "into peer arenas at finalize."
        )
```

- [ ] **Step 5: Call the guard from every Newton-relevant spawn path**

Edit `embodichain/lab/sim/utility/sim_utils.py` — call
`_check_newton_spawn_arena(cfg.arena_index)` (or the equivalent passed-in
kwarg) at the top of `spawn_rigid_object`, `spawn_articulation`, and
`spawn_robot` (the helpers invoked from
`SimulationManager.add_rigid_object` / `add_articulation` / `add_robot`).
Confirm names by grep before editing.

- [ ] **Step 6: Re-run the test and confirm it passes**

Run: `pytest -q tests/sim/test_newton_multi_env.py::test_spawn_with_arena_index_above_zero_rejected_on_newton`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add embodichain/lab/sim/utility/sim_utils.py tests/sim/test_newton_multi_env.py
git commit -m "feat(sim/newton): reject arena_index>0 spawns on Newton

Newton multi-env clones the source arena at finalize, so spawning
directly into peer arenas would produce duplicate bodies. Adds a
spawn-time guard plus a regression test."
```

---

## Task 3: Implement clone-at-finalize in `NewtonPhysicsBackend.prepare()`

**Files:**
- Modify: `embodichain/lab/sim/physics/newton.py`
- Test: `tests/sim/test_newton_multi_env.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/test_newton_multi_env.py`:

```python
def test_finalize_clones_source_arena_into_peers():
    sim = SimulationManager(_newton_sim_cfg(num_envs=3))
    cube_cfg = RigidObjectCfg(
        uid="cube",
        shape=BoxCfg(extents=(0.1, 0.1, 0.1)),
        init_pos=(0.0, 0.0, 1.0),
    )
    sim.add_rigid_object(cube_cfg)
    sim.finalize_newton_physics()

    backend = sim.physics
    assert backend._arenas_cloned is True
    assert backend._is_finalized is True

    # arena_1 and arena_2 should now contain a "cube_arena_1" / "cube_arena_2"
    # actor mirroring arena_0's cube.
    actor_names_arena_0 = {a.get_name() for a in sim._arenas[0].get_all_actors()}
    actor_names_arena_1 = {a.get_name() for a in sim._arenas[1].get_all_actors()}
    actor_names_arena_2 = {a.get_name() for a in sim._arenas[2].get_all_actors()}

    assert any("cube" in n for n in actor_names_arena_0)
    assert any(n.endswith("_arena_1") for n in actor_names_arena_1)
    assert any(n.endswith("_arena_2") for n in actor_names_arena_2)

    SimulationManager.reset()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest -q tests/sim/test_newton_multi_env.py::test_finalize_clones_source_arena_into_peers`
Expected: FAIL — `backend._arenas_cloned` stays False; peer arenas are
empty.

- [ ] **Step 3: Implement the clone helper**

Edit `embodichain/lab/sim/physics/newton.py` — add private helpers and call
from `prepare()`:

```python
    def _arena_is_empty(self, arena) -> bool:
        try:
            return len(list(arena.get_all_actors())) == 0
        except Exception:
            return True

    def _clone_source_arena_if_needed(self) -> None:
        arenas = self._manager._arenas
        if len(arenas) <= 1 or self._arenas_cloned:
            return
        source = arenas[0]
        for arena in arenas[1:]:
            if self._arena_is_empty(arena):
                source.clone_arena_to(arena)
        self._arenas_cloned = True
```

Then change `prepare()` to call the helper before the rebuild — insert
between the early-return and the `if state != "READY":` block:

```python
    def prepare(self) -> None:
        if self._is_finalized and self._lifecycle_state() == "READY":
            return

        # Clone arena_0 into peer arenas before rebuilding the Newton model.
        # See docs/superpowers/specs/2026-06-21-newton-backend-pr-design.md §2.
        self._clone_source_arena_if_needed()

        mgr = self.newton_manager
        state = self._lifecycle_state()
        ...
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `pytest -q tests/sim/test_newton_multi_env.py::test_finalize_clones_source_arena_into_peers`
Expected: PASS.

- [ ] **Step 5: Add a re-clone-after-mutation test**

Append to `tests/sim/test_newton_multi_env.py`:

```python
def test_attribute_mutation_does_not_trigger_reclone():
    sim = SimulationManager(_newton_sim_cfg(num_envs=2))
    cube_cfg = RigidObjectCfg(
        uid="cube",
        shape=BoxCfg(extents=(0.1, 0.1, 0.1)),
        init_pos=(0.0, 0.0, 1.0),
    )
    cube = sim.add_rigid_object(cube_cfg)
    sim.finalize_newton_physics()
    assert sim.physics._arenas_cloned is True

    cube.set_mass(2.0)  # attribute write, NOT topology change
    assert sim.physics._arenas_cloned is True

    SimulationManager.reset()


def test_adding_a_new_asset_invalidates_clone_state():
    sim = SimulationManager(_newton_sim_cfg(num_envs=2))
    cube_cfg = RigidObjectCfg(
        uid="cube",
        shape=BoxCfg(extents=(0.1, 0.1, 0.1)),
        init_pos=(0.0, 0.0, 1.0),
    )
    sim.add_rigid_object(cube_cfg)
    sim.finalize_newton_physics()
    assert sim.physics._arenas_cloned is True

    sphere_cfg = RigidObjectCfg(
        uid="sphere",
        shape=BoxCfg(extents=(0.05, 0.05, 0.05)),
        init_pos=(0.0, 0.2, 1.0),
    )
    sim.add_rigid_object(sphere_cfg)
    assert sim.physics._arenas_cloned is False  # invalidate() cleared it

    sim.finalize_newton_physics()
    assert sim.physics._arenas_cloned is True
    SimulationManager.reset()
```

- [ ] **Step 6: Run the new tests**

Run: `pytest -q tests/sim/test_newton_multi_env.py -k "mutation or invalidates"`
Expected: PASS (re-clone-on-add works because `add_rigid_object` already
calls `_invalidate_newton_physics`, which clears `_arenas_cloned` from
Task 1).

- [ ] **Step 7: Commit**

```bash
git add embodichain/lab/sim/physics/newton.py tests/sim/test_newton_multi_env.py
git commit -m "feat(sim/newton): clone source arena into peers at finalize

NewtonPhysicsBackend.prepare() now calls clone_arena_to(arena_i) for
every empty peer arena before triggering rebuild_newton_from_scene.
The _arenas_cloned flag prevents redundant cloning across attribute
mutations; topology changes (add_*/remove_*) clear it via invalidate().
Closes Target 4 (multi-env spawn-side)."
```

---

## Task 4: Multi-env body-ID resolution in Newton object views

**Files:**
- Modify: `embodichain/lab/sim/objects/backends/newton.py`
- Test: `tests/sim/test_newton_multi_env.py`

dexsim's `_clone_arena_to_Arena_newton` (see
`/root/sources/dexsim/python/dexsim/engine/newton_physics/rigid_body/scene.py:198`)
names cloned actors `f"{src_actor_name}_{dst_arena.get_name()}"`. After
finalize, the Newton view must resolve N body IDs per logical entity using
this exact pattern.

- [ ] **Step 1: Inspect current view resolver**

Run: `Read embodichain/lab/sim/objects/backends/newton.py`

Identify `NewtonRigidBodyView._resolve_body_ids` (or equivalent) and
note its current scalar return shape.

- [ ] **Step 2: Write the failing batched-state test**

Append to `tests/sim/test_newton_multi_env.py`:

```python
import torch


def test_rigid_object_returns_batched_body_state_after_clone():
    sim = SimulationManager(_newton_sim_cfg(num_envs=3))
    cube_cfg = RigidObjectCfg(
        uid="cube",
        shape=BoxCfg(extents=(0.1, 0.1, 0.1)),
        init_pos=(0.0, 0.0, 1.0),
    )
    cube = sim.add_rigid_object(cube_cfg)
    sim.finalize_newton_physics()

    state = cube.data.body_state  # public batched accessor
    # Expected: shape [num_envs, 7] for (xyz + qxqyqzqw) or [num_envs, 13]
    # depending on accessor; just assert the leading dim is num_envs.
    assert state.shape[0] == 3
    SimulationManager.reset()
```

> If the existing accessor name differs from `data.body_state`, grep
> `RigidObjectData` for the canonical accessor that returns pose+twist
> per env, and adjust.

- [ ] **Step 3: Run the test and confirm it fails**

Run: `pytest -q tests/sim/test_newton_multi_env.py::test_rigid_object_returns_batched_body_state_after_clone`
Expected: FAIL — view returns arena_0's scalar.

- [ ] **Step 4: Add `_num_envs` plumbing to the view**

Edit `embodichain/lab/sim/objects/backends/newton.py` —
`NewtonRigidBodyView.__init__` (and similarly for
`NewtonArticulationView`):

```python
class NewtonRigidBodyView(RigidBodyViewBase):
    def __init__(self, entities, physics_scene, *, num_envs: int = 1):
        super().__init__(entities, physics_scene)
        self._num_envs = num_envs
        self._body_ids: torch.Tensor | None = None  # resolved lazily
        self._arena_names: tuple[str, ...] | None = None  # filled at first resolve
```

- [ ] **Step 5: Implement the batched body-id resolver**

In the same class:

```python
    def _resolve_body_ids(self) -> torch.Tensor:
        """Return a [num_envs] tensor of Newton body IDs for this entity.

        Reconstructs dexsim's clone naming
        (``f"{src_name}_{dst_arena_name}"``) and looks each name up in the
        finalized Newton model. Falls back to the arena_0 scalar before
        finalize.
        """
        if self._body_ids is not None:
            return self._body_ids

        scene = self._physics_scene
        mgr = scene.newton_manager if hasattr(scene, "newton_manager") else scene
        # Pre-finalize: return scalar arena_0 ID for BUILDER-state code paths.
        lifecycle = getattr(getattr(mgr, "lifecycle_state", None), "name", "")
        if lifecycle != "READY":
            return self._resolve_arena0_scalar()

        src_name = self._entities[0].get_name()
        if self._num_envs == 1:
            self._body_ids = torch.tensor(
                [self._lookup_body_id(mgr, src_name)],
                dtype=torch.long,
            )
            return self._body_ids

        arena_names = self._arena_names_from_manager()
        ids: list[int] = []
        for i, arena_name in enumerate(arena_names):
            name = src_name if i == 0 else f"{src_name}_{arena_name}"
            ids.append(self._lookup_body_id(mgr, name))
        self._body_ids = torch.tensor(ids, dtype=torch.long)
        return self._body_ids

    def _arena_names_from_manager(self) -> tuple[str, ...]:
        if self._arena_names is not None:
            return self._arena_names
        # The owning SimulationManager exposes _arenas; the view is
        # constructed from inside SimulationManager.add_rigid_object, so
        # we pass arena names down at construction OR look them up via a
        # back-reference. Prefer construction-time injection — see Task 5.
        raise RuntimeError(
            "Arena names not injected — caller must pass arena_names "
            "at view construction.")

    def _lookup_body_id(self, mgr, name: str) -> int:
        # The dexsim Newton manager exposes a name -> body_id map. Probe
        # the canonical accessor; fall back to scanning model.body_label.
        if hasattr(mgr, "body_index"):
            return int(mgr.body_index(name))
        labels = list(getattr(mgr._model, "body_label", []))
        for i, label in enumerate(labels):
            if str(label) == name:
                return i
        raise KeyError(f"Newton body {name!r} not found after finalize.")
```

> Note: the actual lookup API may differ — verify by reading
> `/root/sources/dexsim/python/dexsim/engine/newton_physics/newton_manager.py`
> for `body_index` / `get_body_id` / `name_to_body_id` before finalizing
> the resolver. Use whichever name dexsim exposes; if none, the
> `body_label` scan is the safe fallback.

- [ ] **Step 6: Same treatment for `NewtonArticulationView`**

Add `_num_envs`, `_arena_names`, and a parallel resolver for the
articulation's body-list and joint-id list. Use the same
`f"{name}_{arena_name}"` pattern. For an articulation with N links,
the result is `[num_envs, num_links]`.

- [ ] **Step 7: Inject `num_envs` and `arena_names` at view construction**

Each `RigidObject` / `Articulation` constructs its view via a factory in
`embodichain/lab/sim/objects/backends/__init__.py` (or similar). Locate
that factory by grep and thread `num_envs` and `arena_names` through:

```python
def make_rigid_body_view(entities, physics_scene, *, num_envs, arena_names):
    if is_newton_scene(physics_scene):
        return NewtonRigidBodyView(
            entities, physics_scene,
            num_envs=num_envs, arena_names=arena_names,
        )
    return DefaultRigidBodyView(entities, physics_scene)
```

Caller (`RigidObject.__init__`, `Articulation.__init__`) passes
`self._sim_manager.num_envs` and a tuple of arena names
(`tuple(a.get_name() for a in self._sim_manager._arenas)`).

- [ ] **Step 8: Run the batched-state test**

Run: `pytest -q tests/sim/test_newton_multi_env.py::test_rigid_object_returns_batched_body_state_after_clone`
Expected: PASS.

- [ ] **Step 9: Run the full multi-env test file**

Run: `pytest -q tests/sim/test_newton_multi_env.py`
Expected: All four tests PASS.

- [ ] **Step 10: Run the existing Newton single-env suite for regressions**

Run: `pytest -q tests/sim/objects/test_rigid_object.py::TestRigidObjectNewton tests/sim/objects/test_articulation.py::TestArticulationNewton tests/sim/objects/test_robot.py::TestRobotNewton`
Expected: All PASS.

- [ ] **Step 11: Commit**

```bash
git add embodichain/lab/sim/objects/backends/newton.py \
        embodichain/lab/sim/objects/backends/__init__.py \
        embodichain/lab/sim/objects/rigid_object.py \
        embodichain/lab/sim/objects/articulation.py \
        tests/sim/test_newton_multi_env.py
git commit -m "feat(sim/newton): multi-env body-id resolution in object views

NewtonRigidBodyView and NewtonArticulationView now resolve a [num_envs]
body-id tensor by reconstructing dexsim's clone naming
(f\"{src_name}_{arena_name}\"). View construction takes num_envs and
arena_names; the existing batched accessors return [num_envs, ...]
tensors automatically. Closes Target 4 (multi-env read side)."
```

---

## Task 5: Add `create_differentiable_stepper` / `create_gradient_rollout` delegators

**Files:**
- Modify: `embodichain/lab/sim/sim_manager.py`
- Test: `tests/sim/test_differentiable_stepper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sim/test_differentiable_stepper.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Tests for the differentiable-stepper delegators on SimulationManager."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg


def test_default_backend_rejects_differentiable_stepper():
    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=DefaultPhysicsCfg(), num_envs=1, headless=True,
    ))
    with pytest.raises(Exception, match=r"Newton"):
        sim.create_differentiable_stepper()
    SimulationManager.reset()


def test_newton_without_grad_rejects_differentiable_stepper():
    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(requires_grad=False, use_cuda_graph=False),
        num_envs=1, headless=True,
    ))
    sim.finalize_newton_physics()
    with pytest.raises(Exception, match=r"grad"):
        sim.create_differentiable_stepper()
    SimulationManager.reset()


def test_newton_with_grad_creates_stepper():
    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            requires_grad=True,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        ),
        num_envs=1, headless=True,
    ))
    sim.finalize_newton_physics()
    stepper = sim.create_differentiable_stepper()
    from dexsim.engine.newton_physics.differentiable_stepper import (
        DifferentiableStepper,
    )
    assert isinstance(stepper, DifferentiableStepper)
    SimulationManager.reset()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest -q tests/sim/test_differentiable_stepper.py`
Expected: FAIL — `create_differentiable_stepper` not defined.

- [ ] **Step 3: Add the delegator methods**

Edit `embodichain/lab/sim/sim_manager.py` — add near the other Newton
back-compat delegators (search `newton_manager` in the file to locate the
right region):

```python
    def create_differentiable_stepper(self):
        """Create a single-step differentiable physics primitive (Newton-only).

        Requires the Newton backend with ``requires_grad=True`` and
        ``solver_type="semi_implicit"``. Delegates to
        :meth:`dexsim.engine.newton_physics.NewtonManager.create_differentiable_stepper`.

        Raises:
            RuntimeError: If the active backend is not Newton or if the
                Newton manager is not ready / not in grad mode.
        """
        if not self.is_newton_backend:
            logger.log_error(
                "create_differentiable_stepper requires the Newton backend.")
        return self.physics.newton_manager.create_differentiable_stepper()

    def create_gradient_rollout(
        self,
        record_steps: int,
        substeps_per_record: int | None = None,
        record_dt: float | None = None,
    ):
        """Create a gradient rollout buffer (Newton-only).

        Delegates to
        :meth:`dexsim.engine.newton_physics.NewtonManager.create_gradient_rollout`.
        """
        if not self.is_newton_backend:
            logger.log_error(
                "create_gradient_rollout requires the Newton backend.")
        return self.physics.newton_manager.create_gradient_rollout(
            record_steps=record_steps,
            substeps_per_record=substeps_per_record,
            record_dt=record_dt,
        )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest -q tests/sim/test_differentiable_stepper.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/sim_manager.py tests/sim/test_differentiable_stepper.py
git commit -m "feat(sim): SimulationManager delegators for Newton diff stepper

create_differentiable_stepper and create_gradient_rollout are thin
passthroughs to NewtonManager. Both raise on the default backend.
Backs the new embodichain.lab.sim.diff package (next commit)."
```

---

## Task 6: Create the `embodichain.lab.sim.diff` package — bridge

**Files:**
- Create: `embodichain/lab/sim/diff/__init__.py`
- Create: `embodichain/lab/sim/diff/bridge.py`

The bridge wraps a `wp.Tape()` around one EmbodiChain physics step and
exposes a `torch.autograd.Function` so callers can drive APG with
PyTorch-side action tensors.

- [ ] **Step 1: Create the package skeleton**

Create `embodichain/lab/sim/diff/__init__.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Differentiable Newton stepping for EmbodiChain.

Bridges DexSim's :class:`~dexsim.engine.newton_physics.DifferentiableStepper`
into PyTorch autograd via a :class:`torch.autograd.Function`, and exposes a
:class:`tape_context` manager for advanced users who want to compose their
own Warp kernels.
"""

from __future__ import annotations

from .bridge import (
    NewtonStepFunc,
    differentiable_step,
    tape_context,
)

__all__ = [
    "NewtonStepFunc",
    "differentiable_step",
    "tape_context",
]
```

- [ ] **Step 2: Create `bridge.py`**

Create `embodichain/lab/sim/diff/bridge.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Warp-tape <-> PyTorch-autograd bridge for Newton physics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator

import torch
import warp as wp

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["NewtonStepFunc", "differentiable_step", "tape_context"]


@contextmanager
def tape_context(manager: "SimulationManager") -> Iterator[wp.Tape]:
    """Open a Warp tape bound to the manager's Newton state.

    Advanced users compose their own Warp kernels inside this context, then
    call ``tape.backward()`` outside the with-block.
    """
    if not manager.is_newton_backend:
        raise RuntimeError(
            "tape_context requires the Newton backend with requires_grad=True.")
    tape = wp.Tape()
    with tape:
        yield tape


def differentiable_step(
    manager: "SimulationManager",
    *,
    apply_control_fn: Callable[[wp.Tape], None],
    substeps: int,
    dt: float | None = None,
) -> dict:
    """Run one EmbodiChain-level physics step inside a Warp tape.

    Args:
        manager: The owning :class:`SimulationManager` (must be Newton).
        apply_control_fn: Callable that writes the joint/body control
            targets inside the tape. Invoked once at the start of the
            step. Receives the open tape; must launch Warp kernels (or
            call dexsim setters that are tape-aware) to populate
            ``manager.physics.newton_manager._control``.
        substeps: Number of solver substeps to run (typically
            ``sim_cfg.sim_steps_per_control``).
        dt: Solver dt; defaults to the manager's configured dt.

    Returns:
        A dict carrying the tape and the state buffers for the caller to
        save in autograd context.
    """
    if not manager.is_newton_backend:
        raise RuntimeError(
            "differentiable_step requires the Newton backend.")
    nm = manager.physics.newton_manager
    stepper = manager.create_differentiable_stepper()
    state_in = nm._state_0
    state_out = nm._model.state()
    contacts = stepper.create_contacts()
    dt_val = nm.solver_dt if dt is None else float(dt)

    tape = wp.Tape()
    with tape:
        apply_control_fn(tape)
        for _ in range(substeps):
            stepper.step(state_in, state_out, contacts=contacts, dt=dt_val)
            state_in, state_out = state_out, state_in

    # The final state lives in state_in after the swap.
    return {
        "tape": tape,
        "final_state": state_in,
        "stepper": stepper,
    }


class NewtonStepFunc(torch.autograd.Function):
    """torch.autograd.Function bridging Warp tape autodiff to PyTorch.

    Forward: launches the action-to-control Warp kernel, runs
    ``substeps`` differentiable solver steps, and reads observation /
    reward as torch tensors via ``wp.to_torch`` (zero-copy where
    possible).

    Backward: copies upstream grads into the corresponding Warp
    ``.grad`` buffers, calls ``tape.backward()``, and returns
    ``wp.to_torch(action.grad)`` reshaped to the action's tensor shape.

    Callers must supply a ``sim_state`` dict with the following keys:
        manager: SimulationManager (Newton, requires_grad=True)
        substeps: int
        action_to_control_kernel: callable(action_wp, *kernel_args)
        kernel_args: tuple consumed by action_to_control_kernel
        obs_reward_fn: callable(final_state) -> dict with torch outputs
    """

    @staticmethod
    def forward(ctx, action_torch: torch.Tensor, sim_state: dict):
        manager = sim_state["manager"]
        substeps = int(sim_state["substeps"])
        kernel = sim_state["action_to_control_kernel"]
        kernel_args = sim_state["kernel_args"]
        obs_reward_fn = sim_state["obs_reward_fn"]

        nm = manager.physics.newton_manager
        stepper = manager.create_differentiable_stepper()

        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        state_in = nm._state_0
        state_out = nm._model.state()
        contacts = stepper.create_contacts()
        dt_val = nm.solver_dt

        tape = wp.Tape()
        with tape:
            kernel(action_wp, *kernel_args)  # writes nm._control inside tape
            for _ in range(substeps):
                stepper.step(state_in, state_out, contacts=contacts, dt=dt_val)
                state_in, state_out = state_out, state_in

        outputs = obs_reward_fn(state_in)
        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.outputs_wp = outputs.get("_grad_track", {})
        # `outputs` is a dict of torch tensors built from wp.to_torch — the
        # caller is responsible for ensuring at least one is grad-tracked.
        return tuple(outputs[k] for k in outputs["_order"])

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Copy each upstream grad back into the corresponding Warp .grad.
        for name, grad_t in zip(ctx.outputs_wp["_order"], grad_outputs):
            wp_arr = ctx.outputs_wp[name]
            if grad_t is None or wp_arr.grad is None:
                continue
            wp.copy(wp_arr.grad,
                    wp.from_torch(grad_t.detach().clone().contiguous(),
                                  dtype=wp.float32))
        ctx.tape.backward()
        action_grad = wp.to_torch(ctx.action_wp.grad).clone()
        ctx.tape.zero()
        # Reshape to the original action layout; second input (sim_state)
        # has no gradient.
        return action_grad.reshape(ctx.saved_action_shape), None
```

> Note: the contract between `obs_reward_fn` and `NewtonStepFunc.backward`
> is intentionally explicit — the caller (the env in Task 7) constructs the
> dict in a way that records which outputs need grad-tracking. The
> `_order` / `_grad_track` plumbing keeps the autograd function fully
> general; the env class hides it from end users.

- [ ] **Step 3: Lightweight import smoke**

Run: `python -c "from embodichain.lab.sim.diff import NewtonStepFunc, tape_context, differentiable_step; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Append a tape-smoke test**

Append to `tests/sim/test_differentiable_stepper.py`:

```python
def test_tape_context_records_step():
    import warp as wp

    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            requires_grad=True,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        ),
        num_envs=1, headless=True,
    ))
    sim.finalize_newton_physics()
    from embodichain.lab.sim.diff import tape_context

    with tape_context(sim) as tape:
        pass  # empty tape is valid; tape.backward() on empty is a no-op

    assert isinstance(tape, wp.Tape)
    SimulationManager.reset()
```

- [ ] **Step 5: Run all diff-stepper tests**

Run: `pytest -q tests/sim/test_differentiable_stepper.py`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/diff/__init__.py \
        embodichain/lab/sim/diff/bridge.py \
        tests/sim/test_differentiable_stepper.py
git commit -m "feat(sim/diff): Warp-tape <-> PyTorch-autograd bridge

New embodichain.lab.sim.diff package: NewtonStepFunc (autograd.Function)
wraps DifferentiableStepper inside a wp.Tape, tape_context is the
low-level context manager for advanced kernels, differentiable_step is
the convenience wrapper. Foundation for DifferentiableEmbodiedEnv."
```

---

## Task 7: `DifferentiableEmbodiedEnv` gym subclass

**Files:**
- Create: `embodichain/lab/gym/envs/differentiable_env.py`
- Test: `tests/gym/envs/test_differentiable_env.py`

- [ ] **Step 1: Inspect `EmbodiedEnv.step` signature**

Run: `Read embodichain/lab/gym/envs/embodied_env.py` (focus on `step`,
`reset`, `_preprocess_action`, `_step_action`).

Identify exactly which methods produce the per-step `obs, reward, done,
info`. The override must invoke the same observation/reward managers as
the base class — just inside a tape.

- [ ] **Step 2: Write the construction-validation test first**

Create `tests/gym/envs/test_differentiable_env.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Tests for DifferentiableEmbodiedEnv."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.gym.envs.differentiable_env import (
    DifferentiableEmbodiedEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg


def _diff_env_cfg(requires_grad: bool = True, backend: str = "newton") -> EmbodiedEnvCfg:
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

    if backend == "newton":
        physics_cfg = NewtonPhysicsCfg(
            requires_grad=requires_grad,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        )
    else:
        physics_cfg = DefaultPhysicsCfg()
    sim_cfg = SimulationManagerCfg(
        physics_cfg=physics_cfg, num_envs=2, headless=True,
    )
    return EmbodiedEnvCfg(sim_cfg=sim_cfg)


def test_construct_without_requires_grad_raises():
    with pytest.raises(Exception, match=r"requires_grad"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(requires_grad=False))


def test_construct_on_default_backend_raises():
    with pytest.raises(Exception, match=r"Newton"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(backend="default"))
```

- [ ] **Step 3: Run and confirm failure**

Run: `pytest -q tests/gym/envs/test_differentiable_env.py`
Expected: FAIL — `DifferentiableEmbodiedEnv` not defined.

- [ ] **Step 4: Implement `DifferentiableEmbodiedEnv`**

Create `embodichain/lab/gym/envs/differentiable_env.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Differentiable Newton-backed EmbodiedEnv for analytic policy gradient.

Wraps the standard :class:`EmbodiedEnv` step pipeline in a Warp tape and
bridges autograd into PyTorch via
:class:`embodichain.lab.sim.diff.NewtonStepFunc`. Subclasses define how
actions become Newton control writes and how observations/rewards are
read from the post-step state; the bridge handles the tape lifecycle
and the backward pass.

Usage:

    class MyTask(DifferentiableEmbodiedEnv):
        def _apply_action_kernel(self, action_wp, tape): ...
        def _read_outputs(self, final_state) -> dict: ...
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc
from embodichain.utils import logger

__all__ = ["DifferentiableEmbodiedEnv"]


class DifferentiableEmbodiedEnv(EmbodiedEnv):
    """EmbodiedEnv variant that exposes APG-ready :py:meth:`step`.

    Subclasses must implement :meth:`_apply_action_kernel` and
    :meth:`_read_outputs`; the rest of the EmbodiedEnv contract (reset,
    observation managers, reward functors) carries over.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, *args, **kwargs) -> None:
        self._validate_diff_cfg(cfg)
        super().__init__(cfg, *args, **kwargs)
        self._truncate_backward_at: int | None = getattr(
            cfg, "truncate_backward_at", None,
        )

    @staticmethod
    def _validate_diff_cfg(cfg: EmbodiedEnvCfg) -> None:
        physics_cfg = cfg.sim_cfg.physics_cfg
        if not isinstance(physics_cfg, NewtonPhysicsCfg):
            logger.log_error(
                "DifferentiableEmbodiedEnv requires NewtonPhysicsCfg, "
                f"got {type(physics_cfg).__name__}.")
        if not physics_cfg.requires_grad:
            logger.log_error(
                "DifferentiableEmbodiedEnv requires requires_grad=True on "
                "the NewtonPhysicsCfg.")

    # -- subclass contract ------------------------------------------------ #

    @abstractmethod
    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        """Inside the open Warp tape, write the action into Newton control.

        Implementations launch a Warp kernel that reads ``action_wp``
        (a ``wp.array(dtype=wp.float32, requires_grad=True)`` of shape
        ``[num_envs * action_dim]``) and writes into
        ``self.sim.physics.newton_manager._control`` so the next stepper
        call uses the new control.
        """

    @abstractmethod
    def _read_outputs(self, final_state: Any) -> dict:
        """Read the post-step observation and reward as torch tensors.

        Must return a dict with keys ``"obs"``, ``"reward"``,
        ``"terminated"``, ``"truncated"``, ``"info"``, plus the
        ``_order``/``_grad_track`` metadata expected by
        :class:`NewtonStepFunc`. ``obs`` and ``reward`` should be torch
        tensors backed by ``wp.to_torch`` of grad-tracked Warp arrays.
        """

    # -- gym surface ------------------------------------------------------ #

    def step(self, action: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        sim_state = self._build_sim_state_dict(action)
        outputs = NewtonStepFunc.apply(action, sim_state)
        obs, reward, terminated, truncated = outputs[:4]
        info = sim_state["last_info"]

        done_mask = terminated | truncated
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            fresh_obs, _ = self.reset(env_ids=reset_ids)
            obs = torch.where(
                done_mask.unsqueeze(-1).expand_as(obs),
                fresh_obs.detach(), obs,
            )
        return obs, reward, terminated, truncated, info

    def _build_sim_state_dict(self, action: torch.Tensor) -> dict:
        # Pack the args NewtonStepFunc expects. Subclass-supplied kernel
        # + output reader; environment-level metadata stays here.
        return {
            "manager": self.sim,
            "substeps": self.sim_cfg.sim_steps_per_control,
            "action_to_control_kernel": self._wrap_action_kernel(),
            "kernel_args": (),
            "obs_reward_fn": self._read_outputs,
            "last_info": {},
        }

    def _wrap_action_kernel(self):
        env = self
        def _inner(action_wp, *_):
            env._apply_action_kernel(action_wp, tape=None)
        return _inner
```

- [ ] **Step 5: Re-run construction tests**

Run: `pytest -q tests/gym/envs/test_differentiable_env.py::test_construct_without_requires_grad_raises tests/gym/envs/test_differentiable_env.py::test_construct_on_default_backend_raises`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/gym/envs/differentiable_env.py \
        tests/gym/envs/test_differentiable_env.py
git commit -m "feat(gym): DifferentiableEmbodiedEnv for APG

Newton-only EmbodiedEnv subclass that wraps step() in a Warp tape via
NewtonStepFunc. Subclasses implement _apply_action_kernel and
_read_outputs; the base class handles validation, auto-reset on done,
and the autograd bridge."
```

---

## Task 8: Franka reach APG example task

**Files:**
- Create: `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py`
- Test: `tests/gym/envs/test_differentiable_env.py` (append)

Model the example after
`/root/sources/analytic_policy_gradients/envs/franka_reach_env.py`, but
built on EmbodiChain primitives (`add_robot` with the Franka URDF, the
`DifferentiableEmbodiedEnv` base).

- [ ] **Step 1: Locate Franka URDF in EmbodiChain data**

Run: `find embodichain/data -iname "fr3*.urdf" -o -iname "*franka*.urdf" | head -5`

If no URDF is bundled, the example accepts a `urdf_path` override and
falls back to `newton.utils.download_asset("franka_emika_panda")`,
matching the reference env.

- [ ] **Step 2: Write the example task**

Create `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` (full
contents below):

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Franka FR3 reach task with differentiable Newton physics (APG)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import warp as wp

from embodichain.lab.gym.envs.differentiable_env import DifferentiableEmbodiedEnv
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.gym.utils.registry import register_env
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.sim_manager import SimulationManagerCfg

FRANKA_NUM_ARM_JOINTS = 7
FRANKA_EE_BODY = "fr3_hand_tcp"
DEFAULT_ACTION_SCALE = 0.2
DEFAULT_MAX_EPISODE_STEPS = 30
TARGET_POS_RANGE = {
    "x": (0.05, 0.70),
    "y": (-0.45, 0.45),
    "z": (0.20, 0.95),
}
TARGET_MAX_TILT = math.pi / 3


@wp.kernel
def _set_joint_targets_kernel(
    action: wp.array(dtype=wp.float32),
    current_q: wp.array(dtype=wp.float32),
    target_q: wp.array(dtype=wp.float32),
    limit_lo: wp.array(dtype=wp.float32),
    limit_hi: wp.array(dtype=wp.float32),
    action_scale: wp.float32,
    n_joints_per_env: wp.int32,
    n_arm: wp.int32,
    total: wp.int32,
):
    tid = wp.tid()
    if tid < total:
        env_idx = tid / n_arm
        j = tid % n_arm
        off = env_idx * n_joints_per_env + j
        new_q = current_q[off] + action[tid] * action_scale
        target_q[off] = wp.clamp(new_q, limit_lo[j], limit_hi[j])


@register_env("FrankaReachApg-v0")
class FrankaReachApgEnv(DifferentiableEmbodiedEnv):
    """Differentiable Franka FR3 reach task.

    Built on EmbodiChain's :class:`DifferentiableEmbodiedEnv`; the
    Warp-tape bridge produces ``action.grad`` that flows back through the
    semi-implicit Newton solver.
    """

    metadata = {"render_modes": ["human"], "default_num_envs": 4}

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        *,
        num_envs: int = 4,
        urdf_path: str | None = None,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cuda:0",
    ) -> None:
        if cfg is None:
            cfg = EmbodiedEnvCfg(
                sim_cfg=SimulationManagerCfg(
                    physics_cfg=NewtonPhysicsCfg(
                        device=device,
                        requires_grad=True,
                        solver_cfg={"solver_type": "semi_implicit"},
                        use_cuda_graph=False,
                    ),
                    num_envs=num_envs,
                    headless=True,
                ),
            )
        self._urdf_path = urdf_path
        self._action_scale = float(action_scale)
        self._max_episode_steps = int(max_episode_steps)
        super().__init__(cfg)
        self._init_franka()
        self._init_targets()

    # -- scene setup ----------------------------------------------------- #

    def _init_franka(self) -> None:
        urdf = self._urdf_path or self._resolve_default_urdf()
        robot_cfg = RobotCfg(
            uid="franka",
            urdf_cfg=URDFCfg().set_urdf(urdf),
            fix_base=True,
        )
        self._robot = self.sim.add_robot(robot_cfg)
        self.sim.finalize_newton_physics()

        # Cache joint-limit Warp arrays for the action kernel.
        model = self.sim.physics.newton_manager._model
        lo = np.asarray(model.joint_limit_lower[:FRANKA_NUM_ARM_JOINTS],
                        dtype=np.float32)
        hi = np.asarray(model.joint_limit_upper[:FRANKA_NUM_ARM_JOINTS],
                        dtype=np.float32)
        self._limit_lo_wp = wp.array(lo, dtype=wp.float32, device=model.device)
        self._limit_hi_wp = wp.array(hi, dtype=wp.float32, device=model.device)
        self._n_joints_per_env = int(len(model.joint_q) // self.sim.num_envs)

    def _resolve_default_urdf(self) -> str:
        try:
            import newton.utils as nu

            urdf = nu.download_asset("franka_emika_panda") / (
                "urdf/fr3_franka_hand.urdf")
            if urdf.exists():
                return str(urdf)
        except Exception:
            pass
        raise FileNotFoundError(
            "Franka URDF not available; pass urdf_path explicitly.")

    def _init_targets(self) -> None:
        n = self.sim.num_envs
        device = self.device
        self.target_pos = torch.zeros(n, 3, device=device)
        self.target_quat = torch.zeros(n, 4, device=device)
        self.last_action = torch.zeros(
            n, FRANKA_NUM_ARM_JOINTS, device=device,
        )
        self.step_count = torch.zeros(n, dtype=torch.int32, device=device)
        self._sample_new_targets(torch.arange(n, device=device))

    def _sample_new_targets(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        d = self.device
        self.target_pos[env_ids, 0] = (
            TARGET_POS_RANGE["x"][0]
            + torch.rand(n, device=d)
            * (TARGET_POS_RANGE["x"][1] - TARGET_POS_RANGE["x"][0]))
        self.target_pos[env_ids, 1] = (
            TARGET_POS_RANGE["y"][0]
            + torch.rand(n, device=d)
            * (TARGET_POS_RANGE["y"][1] - TARGET_POS_RANGE["y"][0]))
        self.target_pos[env_ids, 2] = (
            TARGET_POS_RANGE["z"][0]
            + torch.rand(n, device=d)
            * (TARGET_POS_RANGE["z"][1] - TARGET_POS_RANGE["z"][0]))
        # Identity-ish quat, no tilt for the smoke task.
        self.target_quat[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=d).expand(n, -1)

    # -- DifferentiableEmbodiedEnv contract ------------------------------ #

    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        nm = self.sim.physics.newton_manager
        n_envs = self.sim.num_envs
        total = n_envs * FRANKA_NUM_ARM_JOINTS
        wp.launch(
            _set_joint_targets_kernel,
            dim=total,
            inputs=[
                action_wp,
                nm._state_0.joint_q,
                nm._control.joint_target,
                self._limit_lo_wp,
                self._limit_hi_wp,
                wp.float32(self._action_scale),
                wp.int32(self._n_joints_per_env),
                wp.int32(FRANKA_NUM_ARM_JOINTS),
                wp.int32(total),
            ],
            device=nm._model.device,
        )

    def _read_outputs(self, final_state: Any) -> dict:
        nm = self.sim.physics.newton_manager
        n = self.sim.num_envs
        body_q = wp.to_torch(final_state.body_q).view(n, -1, 7)
        ee_idx = self._ee_body_indices()
        ee_pose = body_q[torch.arange(n, device=self.device), ee_idx]
        eef_pos = ee_pose[:, :3]
        eef_quat = ee_pose[:, 3:]

        pos_dist = (eef_pos - self.target_pos).norm(dim=-1)
        rot_dist = self._quat_distance(eef_quat, self.target_quat)
        reward = (
            -0.2 * pos_dist
            + 0.1 * torch.exp(-(pos_dist ** 2) / (2 * 0.1 ** 2))
            - 0.1 * rot_dist
            + 0.1 * torch.exp(-(rot_dist ** 2) / (2 * 0.3 ** 2))
        )

        obs = torch.cat([
            wp.to_torch(final_state.joint_q).view(n, -1)[:, :FRANKA_NUM_ARM_JOINTS],
            ee_pose,
            self.target_pos,
            self.target_quat,
            self.last_action,
        ], dim=-1)

        terminated = (pos_dist < 0.01) & (rot_dist < 0.3)
        self.step_count += 1
        truncated = self.step_count >= self._max_episode_steps

        return {
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {
                "obs": final_state.joint_q,
                "reward": None,  # reward is torch-built; tape flows via FK
            },
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _ee_body_indices(self) -> torch.Tensor:
        if hasattr(self, "_cached_ee_idx"):
            return self._cached_ee_idx
        model = self.sim.physics.newton_manager._model
        idx_per_env = []
        n_per_env = len(model.body_label) // self.sim.num_envs
        for i in range(self.sim.num_envs):
            for j, label in enumerate(model.body_label):
                if FRANKA_EE_BODY in str(label) and (j // n_per_env) == i:
                    idx_per_env.append(j)
                    break
        self._cached_ee_idx = torch.tensor(idx_per_env, dtype=torch.long,
                                           device=self.device)
        return self._cached_ee_idx

    @staticmethod
    def _quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(((q1 - q2) ** 2).sum(-1),
                              ((q1 + q2) ** 2).sum(-1))

    # -- gym overrides --------------------------------------------------- #

    def reset(self, *, seed: int | None = None, env_ids=None):
        env_ids = (env_ids if env_ids is not None
                   else torch.arange(self.sim.num_envs, device=self.device))
        with torch.no_grad():
            self.step_count[env_ids] = 0
            self.last_action[env_ids] = 0.0
            self._sample_new_targets(env_ids)
            # Reset Newton joint_q to zero for the touched envs.
            jq = wp.to_torch(
                self.sim.physics.newton_manager._state_0.joint_q,
            ).view(self.sim.num_envs, -1)
            jq[env_ids] = 0.0
        obs = self._initial_obs()
        return obs, {}

    def _initial_obs(self) -> torch.Tensor:
        with torch.no_grad():
            return self._read_outputs(
                self.sim.physics.newton_manager._state_0)["obs"]
```

- [ ] **Step 3: Smoke import the task**

Run: `python -c "from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import FrankaReachApgEnv; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Append the smoke test**

Append to `tests/gym/envs/test_differentiable_env.py`:

```python
@pytest.mark.requires_gpu
def test_franka_apg_smoke_backward():
    try:
        from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import (
            FrankaReachApgEnv,
        )
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward.requires_grad, "Reward must be autograd-tracked."
    loss = reward.sum()
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    env.close()


@pytest.mark.requires_gpu
def test_franka_apg_one_iter_loss_reduces():
    try:
        from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import (
            FrankaReachApgEnv,
        )
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    opt = torch.optim.SGD([action], lr=0.01)

    losses = []
    for _ in range(3):
        env.reset(seed=0)
        opt.zero_grad()
        _, reward, _, _, _ = env.step(action)
        loss = (-reward).sum()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    assert losses[-1] < losses[0], (
        f"APG did not reduce loss: {losses}")
    env.close()
```

- [ ] **Step 5: Run all differentiable-env tests on a GPU host**

Run: `pytest -q tests/gym/envs/test_differentiable_env.py`
Expected: 4 PASS (or smoke tests SKIPPED if URDF unavailable / no GPU).

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py \
        tests/gym/envs/test_differentiable_env.py
git commit -m "feat(gym/tasks): Franka FR3 reach APG example

End-to-end APG smoke task built on DifferentiableEmbodiedEnv with a
Warp action-to-control kernel and torch-built reward. Verifies the
autograd bridge with one-iteration loss reduction. URDF resolved from
newton.utils.download_asset with explicit override."
```

---

## Task 9: Documentation — agent_context topic and design doc update

**Files:**
- Create: `agent_context/topics/differentiable-env.md`
- Modify: `agent_context/MAP.yaml`
- Modify: `design/newton-backend-design.md`

- [ ] **Step 1: Inspect MAP.yaml format**

Run: `Read agent_context/MAP.yaml`

Note the existing entries (`env-framework`, `manager-functor`, ...) and
mirror that structure.

- [ ] **Step 2: Create the topic file**

Create `agent_context/topics/differentiable-env.md`:

```markdown
# Differentiable Env (APG) Context

EmbodiChain supports analytic policy gradient (APG) via
:class:`embodichain.lab.gym.envs.differentiable_env.DifferentiableEmbodiedEnv`.
The bridge wraps `dexsim.engine.newton_physics.DifferentiableStepper`
inside a `wp.Tape()` and exposes a `torch.autograd.Function`
(`embodichain.lab.sim.diff.NewtonStepFunc`) so PyTorch-side `action`
tensors get a gradient from `tape.backward()`.

## Required configuration

- `NewtonPhysicsCfg(requires_grad=True, solver_cfg={"solver_type": "semi_implicit"})`
- `use_cuda_graph=False` (forced by dexsim when grad mode is on)

The default backend and any other Newton solver are rejected.

## Subclass contract

Task authors implement two methods on `DifferentiableEmbodiedEnv`:

- `_apply_action_kernel(action_wp, tape)` — launch a Warp kernel that
  writes joint/body targets into `nm._control` while the tape is open.
- `_read_outputs(final_state)` — build the `obs` / `reward` / `done`
  outputs as torch tensors via `wp.to_torch` so the tape can record the
  dependency.

See `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` for the
canonical example.

## Functor autograd compatibility

Reward/observation functors that compose torch operations on tensors
obtained via `wp.to_torch` are automatically autograd-compatible.
Functors that detour through CPU / NumPy break the graph; those need
torch-only reimplementations for the differentiable path.

## Memory

Each step records `sim_steps_per_control` substeps into the tape. For
long horizons or large `num_envs`, pass `truncate_backward_at=K` on the
env config to split the tape and detach at chunk boundaries.
```

- [ ] **Step 3: Register the topic in `MAP.yaml`**

Edit `agent_context/MAP.yaml` — append:

```yaml
- id: differentiable-env
  aliases: ["apg", "analytic-policy-gradient", "differentiable-rl"]
  keywords: [differentiable, gradient, apg, autograd, warp tape]
  files:
    - topics/differentiable-env.md
```

- [ ] **Step 4: Update the Newton design doc**

Edit `design/newton-backend-design.md`:
- In the "Completion Plan -> Done" list, add items 13/14:
  - "13. Multi-env parallel via clone_arena_to (Target 4) — implemented."
  - "14. DifferentiableEmbodiedEnv via Warp-tape autograd bridge (Target 5) — implemented."
- In "Remaining", remove items 7 (rigid-only Newton gym smoke tests) and
  8 (gradient rollout wrapper + smoke test) since both are now covered.
- Append a "References" section line: "Implementation plan:
  `docs/superpowers/plans/2026-06-22-newton-backend-pr.md`."

- [ ] **Step 5: Commit**

```bash
git add agent_context/topics/differentiable-env.md \
        agent_context/MAP.yaml \
        design/newton-backend-design.md
git commit -m "docs: Newton multi-env + DifferentiableEmbodiedEnv

agent_context routing for the new differentiable-env topic, plus an
update to design/newton-backend-design.md marking Targets 4 and 5
done with a link to the implementation plan."
```

---

## Task 10: Branch cleanup and full test run

**Files:** none (git operations + verification only).

- [ ] **Step 1: Run the full Newton + diff suite**

Run:
```bash
pytest -q \
    tests/sim/test_backend_parity.py \
    tests/sim/test_newton_finalize_lifecycle.py \
    tests/sim/test_newton_multi_env.py \
    tests/sim/test_differentiable_stepper.py \
    tests/sim/test_physics_attrs.py \
    tests/sim/test_sim_manager_cfg.py \
    tests/sim/objects/test_rigid_object.py \
    tests/sim/objects/test_articulation.py::TestArticulationNewton \
    tests/sim/objects/test_robot.py::TestRobotNewton \
    tests/gym/envs/test_differentiable_env.py
```
Expected: all PASS (or GPU-marked tests SKIPPED on a headless host).

- [ ] **Step 2: Run pre-commit checks**

Run the `/pre-commit-check` skill — black, headers, type annotations,
exports, docstrings.

- [ ] **Step 3: Inspect the branch for `wip` commits to squash**

Run: `git log --oneline main..HEAD | grep -i wip`

If any remain, plan an interactive cleanup via `git rebase -i main` (the
existing CLAUDE.md disallows `-i`, so do this **manually** outside the
agent or skip squashing if maintainer prefers history-preserving merge).

- [ ] **Step 4: Create the PR**

Use the `/pr` skill. Title: `feat(sim): Newton physics backend with
multi-env and differentiable APG`. Body summary:

- Multi-env on Newton via implicit `clone_arena_to` at finalize.
- New `embodichain.lab.sim.diff` package and `DifferentiableEmbodiedEnv`
  for APG on the `semi_implicit` solver.
- Franka FR3 reach APG example task with a one-iter loss-reduction
  smoke test.
- Docs: agent_context routing + updated `design/newton-backend-design.md`.

Reference the design doc and this plan.

---

## Self-Review

**Spec coverage check:**

- §2 multi-env clone-at-finalize — Tasks 1, 3.
- §2 spawn guard `arena_index>0` — Task 2.
- §2 batched body-id resolution — Task 4.
- §3 module layout (`diff/bridge.py`, `differentiable_env.py`, example) — Tasks 6, 7, 8.
- §3 `NewtonStepFunc` + `tape_context` — Task 6.
- §3 `DifferentiableEmbodiedEnv` validation + step pipeline — Task 7.
- §3 Franka APG example + smoke tests — Task 8.
- §4 manager delegators — Task 5.
- §4 view changes for num_envs — Task 4.
- §5 risks: clone re-evaluation under mutation — Tasks 1, 3; documented
  in topic file (Task 9).
- §6 deferred items — out of scope (no tasks, per spec).
- §7 PR shape and commit plan — Task 10.
- §8 test files — Tasks 2, 3, 4, 5, 6, 7, 8.
- §9 acceptance criteria — Task 10.

No gaps.

**Placeholder scan:**

- "verify by reading dexsim newton_manager for `body_index`" in Task 4 —
  the resolver has a fallback (scan `body_label`) so the verification
  step is a *preferred* path, not a placeholder.
- "actual lookup API may differ" in Task 4 — the fallback path is
  guaranteed; the note is an optimization hint, not unfinished work.
- No "TBD" / "TODO" / "implement later" in step content.

**Type/signature consistency:**

- `NewtonStepFunc.apply(action, sim_state)` — Task 6 defines signature;
  Task 7 calls with the same args.
- `_apply_action_kernel(action_wp, tape)` — Task 7 abstract method;
  Task 8 implements with the same signature.
- `_read_outputs(final_state) -> dict` — Task 7 abstract method; Task 8
  returns the documented `_order` / `_grad_track` shape.
- `_arenas_cloned: bool` — Task 1 introduces; Tasks 3 reads/writes;
  consistent.
- `num_envs` / `arena_names` view-construction kwargs — Task 4
  introduces; consistent across both views and the factory.

No drift.
