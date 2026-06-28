# Rigid Object Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ability to attach two `RigidObject`s via a fixed physics constraint and remove it, exposed both as a standalone `SimulationManager` API and as on-demand event functors triggered from a task environment.

**Architecture:** A sim-layer `RigidConstraint` batch wrapper mirrors `RigidObject`'s per-arena pattern, holding one dexsim `FixedConstraint` handle per arena (with `None` where inactive, so arena-index == list-index). `SimulationManager` owns the constraint registry and all dexsim calls. Two function-style event functors in `events.py` (`create_rigid_constraint`, `remove_rigid_constraint`) resolve `SceneEntityCfg` → `RigidObject` and delegate to the sim API, triggered via custom event modes (`"attach"`/`"detach"`).

**Tech Stack:** Python 3.11, `embodichain` package, `dexsim` (Warp/PhysX), `torch`, `numpy`, `pytest`, `black==26.3.1`.

## Global Constraints

- Every source file begins with the Apache 2.0 copyright header (see `CLAUDE.md`).
- `from __future__ import annotations` at the top of every new file.
- Use `TYPE_CHECKING` guard for `EmbodiedEnv` / `SimulationManager` imports to avoid circular imports.
- Prefer `A | B` over `Union[A, B]`.
- `@configclass` for all config objects; `MISSING` for required fields.
- `logger.log_error(msg)` raises `RuntimeError` by default — error paths call it and let it raise (matches `add_rigid_object`).
- Formatter: `black==26.3.1`. Run `black <file>` before each commit.
- Package name is `embodichain` (lowercase).
- Spec: `docs/superpowers/specs/2026-06-29-rigid-constraint-design.md`.

---

## File Structure

```
embodichain/lab/sim/cfg.py                              # MODIFY: add RigidConstraintCfg
embodichain/lab/sim/objects/constraint.py               # CREATE: RigidConstraint wrapper
embodichain/lab/sim/objects/__init__.py                  # MODIFY: export RigidConstraint
embodichain/lab/sim/sim_manager.py                       # MODIFY: +_constraints registry,
                                                          #   create/remove/get_rigid_constraint,
                                                          #   asset_uids + _deferred_destroy wiring
embodichain/lab/gym/envs/managers/events.py              # MODIFY: +2 functors (no __all__ exists)
tests/sim/objects/test_rigid_constraint.py               # CREATE: sim-layer unit tests (mocks)
tests/gym/envs/managers/test_event_rigid_constraint.py   # CREATE: functor unit tests (mocks)
tests/sim/test_rigid_constraint_integration.py           # CREATE: real-sim smoke (gpu-marked)
```

`RigidConstraintCfg` lives in `cfg.py` (where all sim cfgs live). `RigidConstraint` lives in `objects/constraint.py`. The functors live in `events.py` (no `__all__` in that file — just add the functions).

---

### Task 1: `RigidConstraintCfg` in `cfg.py`

**Files:**
- Modify: `embodichain/lab/sim/cfg.py` (append a new `@configclass` near `RigidObjectGroupCfg` ~line 900)
- Test: `tests/sim/objects/test_rigid_constraint.py` (create file, first test)

**Interfaces:**
- Produces: `RigidConstraintCfg` dataclass with fields `name: str = MISSING`, `rigid_object_a_uid: str = MISSING`, `rigid_object_b_uid: str = MISSING`, `local_frame_a: np.ndarray | None = None`, `local_frame_b: np.ndarray | None = None`, `constraint_type: Literal["fixed"] = "fixed"`.

- [ ] **Step 1: Write the failing test**

Create `tests/sim/objects/test_rigid_constraint.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Tests for the RigidConstraint sim-layer wrapper and its config."""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import MISSING

from embodichain.lab.sim.cfg import RigidConstraintCfg


def test_rigid_constraint_cfg_defaults():
    """RigidConstraintCfg requires name + both object uids; frames default None."""
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    assert cfg.name == "weld"
    assert cfg.rigid_object_a_uid == "cube"
    assert cfg.rigid_object_b_uid == "block"
    assert cfg.local_frame_a is None
    assert cfg.local_frame_b is None
    assert cfg.constraint_type == "fixed"


def test_rigid_constraint_cfg_required_fields_are_missing():
    """Required fields default to the MISSING sentinel."""
    assert RigidConstraintCfg.__dataclass_fields__["name"].default is MISSING
    assert (
        RigidConstraintCfg.__dataclass_fields__["rigid_object_a_uid"].default is MISSING
    )
    assert (
        RigidConstraintCfg.__dataclass_fields__["rigid_object_b_uid"].default is MISSING
    )


def test_rigid_constraint_cfg_accepts_frames():
    """Local frames accept 4x4 numpy arrays."""
    frame = np.eye(4, dtype=np.float32)
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
        local_frame_a=frame,
        local_frame_b=frame,
    )
    np.testing.assert_allclose(cfg.local_frame_a, frame)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v`
Expected: FAIL with `ImportError: cannot import name 'RigidConstraintCfg'`

- [ ] **Step 3: Write minimal implementation**

Append to `embodichain/lab/sim/cfg.py` (right after the `RigidObjectGroupCfg` class ends, ~line 1000, before the next `@configclass`). Note `Literal` and `Optional` are already imported at the top of `cfg.py` (lines 22): `from typing import Sequence, Union, Dict, Literal, List, Any, Optional`.

```python
@configclass
class RigidConstraintCfg:
    """Configuration for a fixed constraint between two RigidObjects.

    The constraint binds rigid_object_a's entity[i] to rigid_object_b's entity[i]
    within arena[i] (one constraint per arena).

    Args:
        name: Base constraint name. Per-arena names are derived as ``f"{name}"``
            (single env) or ``f"{name}_{i}"`` (multi env).
        rigid_object_a_uid: UID of the first RigidObject (must exist in the sim).
        rigid_object_b_uid: UID of the second RigidObject (must exist in the sim).
        local_frame_a: 4x4 joint frame in object A's local coordinates.
            ``None`` attaches at the objects' current relative pose (identity).
            Accepts a single ``(4, 4)`` matrix (shared by all envs) or an
            ``(N, 4, 4)`` array (one frame per env). Defaults to None.
        local_frame_b: As :attr:`local_frame_a`, for object B. Defaults to None.
        constraint_type: Reserved for future typed constraints (prismatic,
            revolute, spherical, d6). Only ``"fixed"`` is supported in v1.

    .. attention::
        Both objects must be :class:`RigidObject` instances and must share the
        same number of arenas.
    """

    name: str = MISSING
    """Base name of the constraint (per-arena names are derived from this)."""

    rigid_object_a_uid: str = MISSING
    """UID of the first RigidObject."""

    rigid_object_b_uid: str = MISSING
    """UID of the second RigidObject."""

    local_frame_a: np.ndarray | None = None
    """Local joint frame on object A. None -> identity (current relative pose)."""

    local_frame_b: np.ndarray | None = None
    """Local joint frame on object B. None -> identity (current relative pose)."""

    constraint_type: Literal["fixed"] = "fixed"
    """Constraint type. Only ``"fixed"`` is supported in v1."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/sim/cfg.py tests/sim/objects/test_rigid_constraint.py
git add embodichain/lab/sim/cfg.py tests/sim/objects/test_rigid_constraint.py
git commit -m "feat(sim): add RigidConstraintCfg"
```

---

### Task 2: `RigidConstraint` wrapper in `objects/constraint.py`

**Files:**
- Create: `embodichain/lab/sim/objects/constraint.py`
- Modify: `embodichain/lab/sim/objects/__init__.py` (export `RigidConstraint`)
- Test: `tests/sim/objects/test_rigid_constraint.py` (append)

**Interfaces:**
- Consumes: `RigidConstraintCfg` (Task 1), `RigidObject` (from `embodichain.lab.sim.objects`).
- Produces: `RigidConstraint` class with:
  - `__init__(cfg: RigidConstraintCfg, constraint_handles: list, rigid_object_a: RigidObject, rigid_object_b: RigidObject, device: torch.device)`
  - `@property num_envs -> int`
  - `get_relative_transform(env_ids=None) -> list[np.ndarray]`
  - `get_local_pose(actor_index: int, env_ids=None) -> list[np.ndarray]`
  - `get_name(env_id: int) -> str` (returns the base name for single env, `f"{base}_{env_id}"` for multi)
  - `is_valid(env_ids=None) -> list[bool]`
  - `destroy(self, env_ids=None, arena_resolver=None) -> None` — calls `arena.remove_constraint(name_i)` per env; `arena_resolver(i)` returns the arena for env `i`. When all handles become None, the caller (SimulationManager) drops the wrapper.

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/objects/test_rigid_constraint.py`:

```python
import torch
from unittest.mock import MagicMock

from embodichain.lab.sim.objects.constraint import RigidConstraint


def _make_handle(name="weld_0", rel_z=0.0, valid=True):
    """Build a mock dexsim constraint handle."""
    h = MagicMock()
    h.get_name.return_value = name
    h.is_valid.return_value = valid
    rel = np.eye(4, dtype=np.float32)
    rel[2, 3] = rel_z
    h.get_relative_transform.return_value = rel
    h.get_local_pose.return_value = np.eye(4, dtype=np.float32)
    return h


def _make_rigid_object(uid="cube", num_envs=4):
    """Build a mock RigidObject with a per-arena entity list."""
    obj = MagicMock()
    obj.uid = uid
    obj.num_instances = num_envs
    obj._entities = [MagicMock() for _ in range(num_envs)]
    return obj


def test_rigid_constraint_num_envs_and_init():
    """RigidConstraint exposes num_envs and stores handles + object refs."""
    cfg = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    handles = [_make_handle("weld_0"), None, _make_handle("weld_2"), None]
    obj_a = _make_rigid_object("cube", 4)
    obj_b = _make_rigid_object("block", 4)

    constraint = RigidConstraint(
        cfg=cfg,
        constraint_handles=handles,
        rigid_object_a=obj_a,
        rigid_object_b=obj_b,
        device=torch.device("cpu"),
    )
    assert constraint.num_envs == 4
    assert constraint.rigid_object_a is obj_a
    assert constraint.rigid_object_b is obj_b
    assert len(constraint.constraint_handles) == 4


def test_rigid_constraint_get_name_single_and_multi_env():
    """Single env keeps the base name; multi env appends the arena index."""
    cfg_single = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    c_single = RigidConstraint(cfg_single, [_make_handle("weld")], MagicMock(), MagicMock(), torch.device("cpu"))
    assert c_single.get_name(0) == "weld"

    cfg_multi = RigidConstraintCfg(
        name="weld",
        rigid_object_a_uid="cube",
        rigid_object_b_uid="block",
    )
    handles = [_make_handle("weld_0"), _make_handle("weld_1")]
    c_multi = RigidConstraint(cfg_multi, handles, MagicMock(), MagicMock(), torch.device("cpu"))
    assert c_multi.get_name(0) == "weld_0"
    assert c_multi.get_name(1) == "weld_1"


def test_rigid_constraint_get_relative_transform_skips_none():
    """get_relative_transform skips None handles and only returns for active envs."""
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b")
    handles = [_make_handle("weld_0", rel_z=0.1), None, _make_handle("weld_2", rel_z=0.2), None]
    constraint = RigidConstraint(cfg, handles, MagicMock(), MagicMock(), torch.device("cpu"))

    # default: all env_ids, skips None
    transforms = constraint.get_relative_transform()
    assert len(transforms) == 2
    assert transforms[0][2, 3] == pytest.approx(0.1)
    assert transforms[1][2, 3] == pytest.approx(0.2)

    # explicit subset including a None handle is skipped
    transforms_subset = constraint.get_relative_transform(env_ids=[1, 2])
    assert len(transforms_subset) == 1
    assert transforms_subset[0][2, 3] == pytest.approx(0.2)


def test_rigid_constraint_is_valid():
    """is_valid reports per-env validity, skipping None handles."""
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b")
    handles = [_make_handle(valid=True), None, _make_handle(valid=False), None]
    constraint = RigidConstraint(cfg, handles, MagicMock(), MagicMock(), torch.device("cpu"))
    assert constraint.is_valid() == [True, False]


def test_rigid_constraint_destroy_calls_arena_remove_per_env():
    """destroy calls arena.remove_constraint for each active handle in env_ids."""
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b")
    handles = [_make_handle("weld_0"), None, _make_handle("weld_2"), _make_handle("weld_3")]
    constraint = RigidConstraint(cfg, handles, MagicMock(), MagicMock(), torch.device("cpu"))

    arenas = [MagicMock() for _ in range(4)]
    arena_resolver = lambda i: arenas[i]

    constraint.destroy(env_ids=[0, 2], arena_resolver=arena_resolver)
    arenas[0].remove_constraint.assert_called_once_with("weld_0")
    arenas[2].remove_constraint.assert_called_once_with("weld_2")
    arenas[1].remove_constraint.assert_not_called()
    arenas[3].remove_constraint.assert_not_called()
    # cleared handles become None
    assert constraint.constraint_handles[0] is None
    assert constraint.constraint_handles[2] is None
    assert constraint.constraint_handles[3] is not None  # not in env_ids


def test_rigid_constraint_destroy_all_returns_all_cleared():
    """destroy with env_ids=None clears every active handle."""
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="a", rigid_object_b_uid="b")
    handles = [_make_handle("weld_0"), None, _make_handle("weld_2"), None]
    constraint = RigidConstraint(cfg, handles, MagicMock(), MagicMock(), torch.device("cpu"))
    arenas = [MagicMock() for _ in range(4)]
    constraint.destroy(env_ids=None, arena_resolver=lambda i: arenas[i])
    assert all(h is None for h in constraint.constraint_handles)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v`
Expected: FAIL with `ImportError: cannot import name 'RigidConstraint'`

- [ ] **Step 3: Write minimal implementation**

Create `embodichain/lab/sim/objects/constraint.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Rigid constraint wrapper binding two RigidObjects across arenas."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RigidConstraintCfg
    from embodichain.lab.sim.objects.rigid_object import RigidObject


@dataclass
class RigidConstraint:
    """Batch of fixed constraints linking two :class:`RigidObject` instances.

    Each entry binds ``rigid_object_a._entities[i]`` to
    ``rigid_object_b._entities[i]`` within arena ``i`` via a dexsim
    ``FixedConstraint``. The ``constraint_handles`` list has length
    ``num_envs`` with ``None`` wherever the constraint is not active in that
    arena, so arena index always equals list index.

    Args:
        cfg: The constraint configuration.
        constraint_handles: Per-arena dexsim constraint handles (None where inactive).
        rigid_object_a: The first RigidObject.
        rigid_object_b: The second RigidObject.
        device: The torch device.
    """

    cfg: RigidConstraintCfg
    constraint_handles: list[Any] = field(default_factory=list)
    rigid_object_a: RigidObject = None
    rigid_object_b: RigidObject = None
    device: torch.device = field(default_factory=torch.device("cpu"))

    @property
    def num_envs(self) -> int:
        """Number of arenas covered by this constraint."""
        return len(self.constraint_handles)

    def get_name(self, env_id: int) -> str:
        """Get the per-arena constraint name.

        For single-env constraints, returns the base name. For multi-env
        constraints, returns ``f"{base}_{env_id}"``.

        Args:
            env_id: The arena index.

        Returns:
            The constraint name registered in that arena.
        """
        if self.num_envs <= 1:
            return self.cfg.name
        return f"{self.cfg.name}_{env_id}"

    def _active_env_ids(self, env_ids: Sequence[int] | None) -> list[int]:
        """Resolve the requested env_ids, skipping handles that are None."""
        if env_ids is None:
            env_ids = range(self.num_envs)
        return [i for i in env_ids if self.constraint_handles[i] is not None]

    def get_relative_transform(self, env_ids: Sequence[int] | None = None) -> list[np.ndarray]:
        """Get the relative transform of B in A for each active env.

        Args:
            env_ids: Subset of arenas. None -> all arenas. Inactive (None)
                handles are skipped.

        Returns:
            A list of 4x4 numpy arrays, one per active env.
        """
        results = []
        for i in self._active_env_ids(env_ids):
            results.append(self.constraint_handles[i].get_relative_transform())
        return results

    def get_local_pose(
        self, actor_index: int, env_ids: Sequence[int] | None = None
    ) -> list[np.ndarray]:
        """Get the local pose of the constraint frame for the given actor.

        Args:
            actor_index: 0 for object A, 1 for object B.
            env_ids: Subset of arenas. None -> all. Inactive handles skipped.

        Returns:
            A list of 4x4 numpy arrays, one per active env.
        """
        results = []
        for i in self._active_env_ids(env_ids):
            results.append(self.constraint_handles[i].get_local_pose(actor_index))
        return results

    def is_valid(self, env_ids: Sequence[int] | None = None) -> list[bool]:
        """Check validity of each active constraint handle.

        Args:
            env_ids: Subset of arenas. None -> all. Inactive handles skipped.

        Returns:
            A list of bools, one per active env.
        """
        return [
            self.constraint_handles[i].is_valid()
            for i in self._active_env_ids(env_ids)
        ]

    def destroy(
        self,
        env_ids: Sequence[int] | None = None,
        arena_resolver: Callable[[int], Any] | None = None,
    ) -> None:
        """Remove this constraint from the specified arenas.

        Args:
            env_ids: Subset of arenas to clear. None -> all active arenas.
            arena_resolver: Callable returning the arena for a given env index.
                Required to actually remove constraints from dexsim.
        """
        for i in self._active_env_ids(env_ids):
            if arena_resolver is not None:
                arena = arena_resolver(i)
                arena.remove_constraint(self.get_name(i))
            self.constraint_handles[i] = None
```

Then modify `embodichain/lab/sim/objects/__init__.py` — add the import and keep it exported. Add after the `from .gizmo import Gizmo` line (line 29):

```python
from .constraint import RigidConstraint
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v`
Expected: PASS (all tests, including the new ones)

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/sim/objects/constraint.py embodichain/lab/sim/objects/__init__.py tests/sim/objects/test_rigid_constraint.py
git add embodichain/lab/sim/objects/constraint.py embodichain/lab/sim/objects/__init__.py tests/sim/objects/test_rigid_constraint.py
git commit -m "feat(sim): add RigidConstraint wrapper"
```

---

### Task 3: `SimulationManager` constraint registry + create API

**Files:**
- Modify: `embodichain/lab/sim/sim_manager.py` (add `_constraints` registry in `__init__`, `create_rigid_constraint`, plus helpers)
- Test: `tests/sim/objects/test_rigid_constraint.py` (append sim-layer tests using a mock sim)

**Interfaces:**
- Consumes: `RigidConstraintCfg` (Task 1), `RigidConstraint` (Task 2).
- Produces: `SimulationManager.create_rigid_constraint(cfg, env_ids=None) -> RigidConstraint`. The method resolves objects from `self._rigid_objects`, broadcasts local frames, and for each target env calls `self.get_env(i).create_fixed_constraint(name_i, obj_a._entities[i], obj_b._entities[i], frame_a, frame_b)`, stores a `RigidConstraint` in `self._constraints[cfg.name]`.

**Frame broadcasting rules** (a helper `_broadcast_frame`): `None` -> `np.eye(4)` per env; `(4,4)` -> same matrix per env; `(N,4,4)` -> index `i` (requires `N == num_envs`, else `log_error`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/objects/test_rigid_constraint.py`:

```python
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.utils import configclass  # noqa: F401  (ensures import works)


class MockArena:
    """Mock dexsim arena that records created constraints."""

    def __init__(self, fail_indices=None):
        self.created = []  # list of (name, actor0, actor1, frame_a, frame_b)
        self.removed = []   # list of names
        self.fail_indices = set(fail_indices or [])

    def create_fixed_constraint(self, name, actor0, actor1, local_frame0, local_frame1):
        self.created.append((name, actor0, actor1, local_frame0, local_frame1))
        if len(self.created) - 1 in self.fail_indices:
            return None
        h = MagicMock()
        h.get_name.return_value = name
        h.is_valid.return_value = True
        h.get_relative_transform.return_value = np.eye(4, dtype=np.float32)
        return h

    def remove_constraint(self, name):
        self.removed.append(name)


class _RigidConstraintTestSim:
    """A SimulationManager stand-in exposing only the constraint registry path.

    We avoid constructing a real dexsim World (which needs a GPU/window). Instead
    we drive create_rigid_constraint by giving it a fake `self` with the
    attributes the method touches: _rigid_objects, _arenas/_env, num_envs, device.
    """

    def __init__(self, num_envs=4, arenas=None):
        self._rigid_objects = {}
        self._constraints = {}
        self.device = torch.device("cpu")
        if num_envs == 1:
            self._arenas = []
            self._env = arenas[0] if arenas else MockArena()
        else:
            self._arenas = arenas or [MockArena() for _ in range(num_envs)]
            self._env = None

    @property
    def num_envs(self):
        return len(self._arenas) if self._arenas else 1

    def get_env(self, arena_index=-1):
        if arena_index >= 0 and self._arenas:
            return self._arenas[arena_index]
        return self._env

    # bind the real method under test
    create_rigid_constraint = SimulationManager.create_rigid_constraint
    _broadcast_frame = staticmethod(SimulationManager._broadcast_frame)


def _register_object(sim, uid, num_envs):
    obj = MagicMock()
    obj.uid = uid
    obj.num_instances = num_envs
    obj._entities = [MagicMock(name=f"{uid}_{i}") for i in range(num_envs)]
    sim._rigid_objects[uid] = obj
    return obj


def test_create_rigid_constraint_resolves_both_objects_all_envs():
    """create builds one handle per arena and stores a RigidConstraint."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    constraint = sim.create_rigid_constraint(cfg)

    assert cfg.name in sim._constraints
    assert constraint.num_envs == 4
    assert all(h is not None for h in constraint.constraint_handles)
    # each arena got exactly one create call with the right actors
    for i, arena in enumerate(sim._arenas):
        assert arena.created[i][0] == f"weld_{i}"
        assert arena.created[i][1] is sim._rigid_objects["cube"]._entities[i]
        assert arena.created[i][2] is sim._rigid_objects["block"]._entities[i]


def test_create_rigid_constraint_single_env_uses_global_env():
    """Single-env create routes through the global env and keeps the base name."""
    arena = MockArena()
    sim = _RigidConstraintTestSim(num_envs=1, arenas=[arena])
    _register_object(sim, "cube", 1)
    _register_object(sim, "block", 1)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    constraint = sim.create_rigid_constraint(cfg)
    assert constraint.constraint_handles[0] is not None
    assert arena.created[0][0] == "weld"  # base name, no suffix


def test_create_rigid_constraint_subset_env_ids():
    """env_ids subset populates only those arenas; others stay None."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)

    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    constraint = sim.create_rigid_constraint(cfg, env_ids=[0, 2])
    assert constraint.constraint_handles[0] is not None
    assert constraint.constraint_handles[1] is None
    assert constraint.constraint_handles[2] is not None
    assert constraint.constraint_handles[3] is None
    # only arenas 0 and 2 got a create call
    assert len(sim._arenas[0].created) == 1
    assert len(sim._arenas[1].created) == 0
    assert len(sim._arenas[2].created) == 1
    assert len(sim._arenas[3].created) == 0


def test_create_rigid_constraint_missing_object_raises():
    """A missing object uid raises (log_error raises RuntimeError by default)."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    # block not registered
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    with pytest.raises(RuntimeError):
        sim.create_rigid_constraint(cfg)


def test_create_rigid_constraint_duplicate_name_raises():
    """A duplicate base name raises."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    sim.create_rigid_constraint(cfg)
    with pytest.raises(RuntimeError):
        sim.create_rigid_constraint(cfg)


def test_create_rigid_constraint_failed_handle_raises():
    """If dexsim returns None for a handle, log_error raises."""
    sim = _RigidConstraintTestSim(num_envs=2, arenas=[MockArena(fail_indices=[0]), MockArena()])
    _register_object(sim, "cube", 2)
    _register_object(sim, "block", 2)
    cfg = RigidConstraintCfg(
        name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block"
    )
    with pytest.raises(RuntimeError):
        sim.create_rigid_constraint(cfg)


def test_broadcast_frame_none_to_identity():
    """None frame broadcasts to identity per env."""
    sim = _RigidConstraintTestSim(num_envs=3)
    frames = sim._broadcast_frame(None, num_envs=3, env_ids=[0, 1, 2], name="weld")
    assert len(frames) == 3
    for f in frames:
        np.testing.assert_allclose(f, np.eye(4))


def test_broadcast_frame_4x4_repeats():
    """A single 4x4 matrix repeats across all envs."""
    sim = _RigidConstraintTestSim(num_envs=3)
    frame = np.eye(4, dtype=np.float32) * 2
    frames = sim._broadcast_frame(frame, num_envs=3, env_ids=[0, 1, 2], name="weld")
    assert len(frames) == 3
    for f in frames:
        np.testing.assert_allclose(f, frame)


def test_broadcast_frame_N4x4_indexes():
    """An (N,4,4) array indexes per env and requires N == num_envs."""
    sim = _RigidConstraintTestSim(num_envs=3)
    frames_in = np.stack([np.eye(4) * i for i in range(3)], axis=0).astype(np.float32)
    frames = sim._broadcast_frame(frames_in, num_envs=3, env_ids=[0, 1, 2], name="weld")
    for i, f in enumerate(frames):
        np.testing.assert_allclose(f, frames_in[i])

    # wrong N raises
    bad = np.stack([np.eye(4)] * 2, axis=0).astype(np.float32)
    with pytest.raises(RuntimeError):
        sim._broadcast_frame(bad, num_envs=3, env_ids=[0, 1, 2], name="weld")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v -k "create or broadcast"`
Expected: FAIL with `AttributeError: 'SimulationManager' has no attribute 'create_rigid_constraint'` (and `_broadcast_frame`).

- [ ] **Step 3: Write minimal implementation**

In `embodichain/lab/sim/sim_manager.py`:

3a. Add to the imports block near the top (after the existing `from embodichain.lab.sim.cfg import ...` block, ~line 89). Add `RigidConstraintCfg` to that import:

```python
from embodichain.lab.sim.cfg import (
    RenderCfg,
    PhysicsCfg,
    MarkerCfg,
    GPUMemoryCfg,
    WindowRecordCfg,
    LightCfg,
    RigidObjectCfg,
    SoftObjectCfg,
    ClothObjectCfg,
    RigidObjectGroupCfg,
    ArticulationCfg,
    RobotCfg,
    RigidConstraintCfg,
)
```

3b. Add `RigidConstraint` to the objects import (line ~59-67):

```python
from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectGroup,
    SoftObject,
    ClothObject,
    Articulation,
    Robot,
    Light,
    RigidConstraint,
)
```

3c. In `__init__`, add the registry next to `self._rigid_objects` (line ~285):

```python
        self._rigid_objects: Dict[str, RigidObject] = dict()
        self._constraints: Dict[str, RigidConstraint] = dict()
```

3d. Add the methods. Place them right after `get_rigid_object_uid_list` (line ~1005, before `add_rigid_object_group`). 

```python
    @staticmethod
    def _broadcast_frame(
        frame: np.ndarray | None,
        num_envs: int,
        env_ids: Sequence[int],
        name: str,
    ) -> list[np.ndarray]:
        """Broadcast a local-frame spec to one matrix per target env.

        Args:
            frame: None -> identity; (4,4) -> repeated; (N,4,4) -> indexed per env.
            num_envs: Total number of arenas (used to validate (N,4,4)).
            env_ids: Target env indices to produce frames for.
            name: Constraint name (for error messages).

        Returns:
            A list of (4,4) numpy arrays, one per env in env_ids.

        Raises:
            RuntimeError: If an (N,4,4) frame's N != num_envs, or shape is invalid.
        """
        if frame is None:
            identity = np.eye(4, dtype=np.float32)
            return [identity for _ in env_ids]
        frame_np = np.asarray(frame, dtype=np.float32)
        if frame_np.shape == (4, 4):
            return [frame_np for _ in env_ids]
        if frame_np.ndim == 3 and frame_np.shape[1:] == (4, 4):
            if frame_np.shape[0] != num_envs:
                logger.log_error(
                    f"Constraint '{name}' local frame has shape {frame_np.shape} "
                    f"but num_envs is {num_envs}. Expected ({num_envs}, 4, 4)."
                )
            return [frame_np[i] for i in env_ids]
        logger.log_error(
            f"Constraint '{name}' local frame has invalid shape {frame_np.shape}. "
            "Expected None, (4, 4), or (N, 4, 4)."
        )

    def create_rigid_constraint(
        self,
        cfg: RigidConstraintCfg,
        env_ids: Sequence[int] | None = None,
    ) -> RigidConstraint:
        """Create a fixed constraint between two RigidObjects.

        Binds ``rigid_object_a``'s entity[i] to ``rigid_object_b``'s entity[i]
        within arena[i], for each env in ``env_ids``. Local frames default to
        identity (attach at the objects' current relative pose).

        Args:
            cfg: The constraint configuration.
            env_ids: Target environment indices. None -> all arenas.

        Returns:
            The created :class:`RigidConstraint`.

        Raises:
            RuntimeError: If either object is missing, the name is already in use,
                a frame shape is invalid, or dexsim fails to create a handle.
        """
        # validate constraint type (only fixed supported in v1)
        if cfg.constraint_type != "fixed":
            logger.log_error(
                f"Constraint '{cfg.name}' has unsupported type "
                f"'{cfg.constraint_type}'. Only 'fixed' is supported in v1."
            )

        # resolve objects
        if cfg.rigid_object_a_uid not in self._rigid_objects:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_a_uid}' not found for constraint "
                f"'{cfg.name}'. Available: {list(self._rigid_objects.keys())}."
            )
        if cfg.rigid_object_b_uid not in self._rigid_objects:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_b_uid}' not found for constraint "
                f"'{cfg.name}'. Available: {list(self._rigid_objects.keys())}."
            )
        rigid_object_a = self._rigid_objects[cfg.rigid_object_a_uid]
        rigid_object_b = self._rigid_objects[cfg.rigid_object_b_uid]

        # validate duplicate name
        if cfg.name in self._constraints:
            logger.log_error(
                f"Constraint '{cfg.name}' already exists. Remove it before recreating."
            )

        # validate object entity counts match num_envs
        num_envs = self.num_envs
        if rigid_object_a.num_instances != num_envs:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_a_uid}' has "
                f"{rigid_object_a.num_instances} instances but num_envs is {num_envs}."
            )
        if rigid_object_b.num_instances != num_envs:
            logger.log_error(
                f"RigidObject '{cfg.rigid_object_b_uid}' has "
                f"{rigid_object_b.num_instances} instances but num_envs is {num_envs}."
            )

        # resolve target env_ids
        if env_ids is None:
            target_env_ids = list(range(num_envs))
        else:
            target_env_ids = list(env_ids)

        # broadcast local frames
        frames_a = self._broadcast_frame(
            cfg.local_frame_a, num_envs, target_env_ids, cfg.name
        )
        frames_b = self._broadcast_frame(
            cfg.local_frame_b, num_envs, target_env_ids, cfg.name
        )

        # pre-size handles list with None, fill target envs
        handles: list = [None] * num_envs
        for idx, env_id in enumerate(target_env_ids):
            arena = self.get_env(env_id)
            name_i = cfg.name if num_envs <= 1 else f"{cfg.name}_{env_id}"
            handle = arena.create_fixed_constraint(
                name_i,
                rigid_object_a._entities[env_id],
                rigid_object_b._entities[env_id],
                frames_a[idx],
                frames_b[idx],
            )
            if handle is None:
                logger.log_error(
                    f"Failed to create constraint '{name_i}' in arena {env_id}."
                )
            handles[env_id] = handle

        constraint = RigidConstraint(
            cfg=cfg,
            constraint_handles=handles,
            rigid_object_a=rigid_object_a,
            rigid_object_b=rigid_object_b,
            device=self.device,
        )
        self._constraints[cfg.name] = constraint
        return constraint
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/sim/sim_manager.py tests/sim/objects/test_rigid_constraint.py
git add embodichain/lab/sim/sim_manager.py tests/sim/objects/test_rigid_constraint.py
git commit -m "feat(sim): add SimulationManager.create_rigid_constraint"
```

---

### Task 4: `remove_rigid_constraint` + `get_rigid_constraint` + registry wiring

**Files:**
- Modify: `embodichain/lab/sim/sim_manager.py` (add remove/get methods; wire `asset_uids` + `_deferred_destroy`)
- Test: `tests/sim/objects/test_rigid_constraint.py` (append)

**Interfaces:**
- Consumes: `RigidConstraint.destroy` (Task 2), `self._constraints` (Task 3).
- Produces:
  - `remove_rigid_constraint(name, env_ids=None) -> bool` — pops (or partially clears) the constraint; calls `constraint.destroy(env_ids, arena_resolver=self.get_env)`. When all handles are None, drops from registry. Returns True if the constraint existed (or still partially exists after subset remove), False if name unknown.
  - `get_rigid_constraint(name) -> RigidConstraint | None`
  - `get_rigid_constraint_uid_list() -> list[str]`
  - `asset_uids` extended to include constraint names.
  - `_deferred_destroy` severs `_constraints`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/objects/test_rigid_constraint.py`:

```python
def test_remove_rigid_constraint_all_envs():
    """remove with env_ids=None clears every arena and drops the registry entry."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block")
    sim.create_rigid_constraint(cfg)

    removed = sim.remove_rigid_constraint("weld")
    assert removed is True
    assert "weld" not in sim._constraints
    # each arena got remove_constraint with its per-env name
    for i, arena in enumerate(sim._arenas):
        assert f"weld_{i}" in arena.removed


def test_remove_rigid_constraint_subset_keeps_others():
    """remove with a subset env_ids clears only those arenas; registry kept."""
    sim = _RigidConstraintTestSim(num_envs=4)
    _register_object(sim, "cube", 4)
    _register_object(sim, "block", 4)
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block")
    constraint = sim.create_rigid_constraint(cfg)

    removed = sim.remove_rigid_constraint("weld", env_ids=[0, 2])
    assert removed is True
    # still in registry because envs 1,3 remain active
    assert "weld" in sim._constraints
    assert sim._constraints["weld"].constraint_handles[0] is None
    assert sim._constraints["weld"].constraint_handles[1] is not None
    assert sim._constraints["weld"].constraint_handles[2] is None
    assert sim._constraints["weld"].constraint_handles[3] is not None
    assert "weld_0" in sim._arenas[0].removed
    assert "weld_2" in sim._arenas[2].removed
    assert sim._arenas[1].removed == []


def test_remove_rigid_constraint_unknown_name_warns_false():
    """remove on an unknown name returns False without raising."""
    sim = _RigidConstraintTestSim(num_envs=4)
    removed = sim.remove_rigid_constraint("nope")
    assert removed is False


def test_get_rigid_constraint_and_uid_list():
    """get returns the constraint; uid list lists all registered names."""
    sim = _RigidConstraintTestSim(num_envs=2)
    _register_object(sim, "cube", 2)
    _register_object(sim, "block", 2)
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block")
    sim.create_rigid_constraint(cfg)
    assert sim.get_rigid_constraint("weld") is not None
    assert sim.get_rigid_constraint("nope") is None
    assert sim.get_rigid_constraint_uid_list() == ["weld"]


def test_partial_remove_then_all_drops_registry():
    """Subset remove then removing remaining envs drops the registry entry."""
    sim = _RigidConstraintTestSim(num_envs=2)
    _register_object(sim, "cube", 2)
    _register_object(sim, "block", 2)
    cfg = RigidConstraintCfg(name="weld", rigid_object_a_uid="cube", rigid_object_b_uid="block")
    sim.create_rigid_constraint(cfg)
    sim.remove_rigid_constraint("weld", env_ids=[0])
    assert "weld" in sim._constraints
    sim.remove_rigid_constraint("weld", env_ids=[1])
    assert "weld" not in sim._constraints
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v -k "remove or get_rigid"`
Expected: FAIL with `AttributeError: ... has no attribute 'remove_rigid_constraint'`.

- [ ] **Step 3: Write minimal implementation**

In `embodichain/lab/sim/sim_manager.py`, add after `create_rigid_constraint` (from Task 3):

```python
    def remove_rigid_constraint(
        self,
        name: str,
        env_ids: Sequence[int] | None = None,
    ) -> bool:
        """Remove a rigid constraint by name.

        With ``env_ids=None`` the constraint is removed from every arena and
        dropped from the registry. With a subset, only those arenas are cleared;
        the registry entry is kept until all handles become None.

        Args:
            name: The base constraint name.
            env_ids: Subset of arenas to clear. None -> all.

        Returns:
            True if the constraint was found (and removed or partially removed),
            False if the name is unknown.
        """
        constraint = self._constraints.get(name, None)
        if constraint is None:
            logger.log_warning(
                f"Constraint '{name}' not found. Nothing to remove."
            )
            return False

        constraint.destroy(env_ids=env_ids, arena_resolver=self.get_env)

        # drop from registry if no handles remain active
        if all(h is None for h in constraint.constraint_handles):
            del self._constraints[name]
        return True

    def get_rigid_constraint(self, name: str) -> RigidConstraint | None:
        """Get a rigid constraint by its base name.

        Args:
            name: The base constraint name.

        Returns:
            The constraint, or None if not found.
        """
        if name not in self._constraints:
            logger.log_warning(f"Constraint '{name}' not found.")
            return None
        return self._constraints[name]

    def get_rigid_constraint_uid_list(self) -> List[str]:
        """Get the list of registered constraint base names.

        Returns:
            List of constraint names.
        """
        return list(self._constraints.keys())
```

Then wire `asset_uids` (line ~421). Add constraints to the returned list. In `asset_uids`:

```python
    @property
    def asset_uids(self) -> List[str]:
        """Get all assets uid in the simulation.

        The assets include lights, sensors, robots, rigid objects and articulations.

        Returns:
            List[str]: list of all assets uid.
        """
        uid_list = ["default_plane"]
        uid_list.extend(list(self._lights.keys()))
        uid_list.extend(list(self._sensors.keys()))
        uid_list.extend(list(self._robots.keys()))
        uid_list.extend(list(self._rigid_objects.keys()))
        uid_list.extend(list(self._rigid_object_groups.keys()))
        uid_list.extend(list(self._soft_objects.keys()))
        uid_list.extend(list(self._cloth_objects.keys()))
        uid_list.extend(list(self._articulations.keys()))
        uid_list.extend(list(self._constraints.keys()))
        return uid_list
```

Then wire `_deferred_destroy`. In `_deferred_destroy` (~line 2271, the `_sever_wrapper_refs` calls), add:

```python
        _sever_wrapper_refs("_constraints")
```

after the `_sever_wrapper_refs("_rigid_object_groups")` line. Also clear `_constraints` in the explicit `clear()` block — find the line `self._rigid_object_groups` clear region and add:

```python
        self._constraints.clear()
```

near the other `.clear()` calls at the end of `_deferred_destroy` (after `self._arenas.clear()`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/sim/objects/test_rigid_constraint.py -v`
Expected: PASS (all sim-layer tests)

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/sim/sim_manager.py tests/sim/objects/test_rigid_constraint.py
git add embodichain/lab/sim/sim_manager.py tests/sim/objects/test_rigid_constraint.py
git commit -m "feat(sim): add remove/get_rigid_constraint and registry wiring"
```

---

### Task 5: Event functors in `events.py`

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/events.py` (add `create_rigid_constraint` + `remove_rigid_constraint` functions)
- Test: `tests/gym/envs/managers/test_event_rigid_constraint.py` (create)

**Interfaces:**
- Consumes: `SceneEntityCfg` (from `.cfg`), `RigidConstraintCfg` (from `embodichain.lab.sim.cfg`), `RigidObject` (for isinstance check), `logger`.
- Produces: two module-level functions in `events.py`:
  - `create_rigid_constraint(env, env_ids, obj_a_cfg: SceneEntityCfg, obj_b_cfg: SceneEntityCfg, name: str, local_frame_a=None, local_frame_b=None) -> None`
  - `remove_rigid_constraint(env, env_ids, name: str) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/gym/envs/managers/test_event_rigid_constraint.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Tests for the rigid-constraint event functors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import (
    create_rigid_constraint,
    remove_rigid_constraint,
)
from embodichain.lab.sim.objects.rigid_object import RigidObject


class MockRigidObjectForFunctor:
    """Stand-in for a RigidObject passing the isinstance check.

    The functor checks ``isinstance(asset, RigidObject)``. To avoid building a
    real RigidObject (needs a dexsim World), we monkeypatch the check.
    """


def _make_env(obj_a_is_rigid=True, obj_b_is_rigid=True):
    """Build a mock env with a spied sim.create/remove_rigid_constraint."""
    env = MagicMock()
    env.device = torch.device("cpu")
    env.num_envs = 4

    obj_a = MagicMock()
    obj_b = MagicMock()
    env.sim.get_asset.side_effect = lambda uid: {"cube": obj_a, "block": obj_b}[uid]

    # control the isinstance check by patching RigidObject for the test
    env._obj_a_is_rigid = obj_a_is_rigid
    env._obj_b_is_rigid = obj_b_is_rigid
    env.sim.create_rigid_constraint = MagicMock(return_value=MagicMock())
    env.sim.remove_rigid_constraint = MagicMock(return_value=True)
    return env, obj_a, obj_b


def test_create_functor_delegates_to_sim(monkeypatch):
    """create functor resolves both objects and forwards to sim.create_rigid_constraint."""
    env, obj_a, obj_b = _make_env()
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject", MockRigidObjectForFunctor
    )

    env_ids = torch.tensor([0, 2])
    create_rigid_constraint(
        env,
        env_ids,
        obj_a_cfg=SceneEntityCfg(uid="cube"),
        obj_b_cfg=SceneEntityCfg(uid="block"),
        name="weld",
    )

    env.sim.create_rigid_constraint.assert_called_once()
    call_kwargs = env.sim.create_rigid_constraint.call_args
    assert call_kwargs.kwargs["env_ids"] is env_ids
    cfg = call_kwargs.kwargs["cfg"]
    assert cfg.name == "weld"
    assert cfg.rigid_object_a_uid == "cube"
    assert cfg.rigid_object_b_uid == "block"
    assert cfg.local_frame_a is None
    assert cfg.local_frame_b is None


def test_create_functor_forwards_frames(monkeypatch):
    """create functor forwards local frames into the cfg."""
    env, _, _ = _make_env()
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject", MockRigidObjectForFunctor
    )
    frame = np.eye(4, dtype=np.float32)
    create_rigid_constraint(
        env,
        None,
        obj_a_cfg=SceneEntityCfg(uid="cube"),
        obj_b_cfg=SceneEntityCfg(uid="block"),
        name="weld",
        local_frame_a=frame,
        local_frame_b=frame,
    )
    cfg = env.sim.create_rigid_constraint.call_args.kwargs["cfg"]
    np.testing.assert_allclose(cfg.local_frame_a, frame)


def test_create_functor_rejects_non_rigid_object(monkeypatch):
    """A non-RigidObject asset raises (log_error raises RuntimeError)."""
    env, obj_a, obj_b = _make_env()
    # patch isinstance check to return False for obj_a
    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject", MagicMock(__instancecheck__=lambda self, o: False)
    )
    with pytest.raises(RuntimeError):
        create_rigid_constraint(
            env,
            None,
            obj_a_cfg=SceneEntityCfg(uid="cube"),
            obj_b_cfg=SceneEntityCfg(uid="block"),
            name="weld",
        )
    env.sim.create_rigid_constraint.assert_not_called()


def test_remove_functor_delegates():
    """remove functor forwards name + env_ids to sim.remove_rigid_constraint."""
    env, _, _ = _make_env()
    env_ids = torch.tensor([1, 3])
    remove_rigid_constraint(env, env_ids, name="weld")
    env.sim.remove_rigid_constraint.assert_called_once_with(
        "weld", env_ids=env_ids
    )


def test_remove_functor_none_env_ids():
    """remove functor forwards env_ids=None correctly."""
    env, _, _ = _make_env()
    remove_rigid_constraint(env, None, name="weld")
    env.sim.remove_rigid_constraint.assert_called_once_with("weld", env_ids=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gym/envs/managers/test_event_rigid_constraint.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_rigid_constraint'`

- [ ] **Step 3: Write minimal implementation**

In `embodichain/lab/gym/envs/managers/events.py`, add the imports near the top (the file already imports `RigidObject` at line 28, `logger`, `np`-equivalent math helpers, and `SceneEntityCfg` at line 35). Add to the existing `from embodichain.lab.sim.objects import (...)` block a new import is not needed (RigidObject already there). Add `RigidConstraintCfg` import. Find the `from embodichain.lab.sim.cfg import RigidObjectCfg, ArticulationCfg` line (line 33) and extend it:

```python
from embodichain.lab.sim.cfg import RigidObjectCfg, ArticulationCfg, RigidConstraintCfg
```

Then add `numpy` import if not present (the file uses math helpers but check). At the top, after `import random` (line 21), ensure `import numpy as np` exists. If not present, add it.

Then append the two functors at the end of the file (after `set_detached_uids_for_env_reset`):

```python
def create_rigid_constraint(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    obj_a_cfg: SceneEntityCfg,
    obj_b_cfg: SceneEntityCfg,
    name: str,
    local_frame_a: np.ndarray | None = None,
    local_frame_b: np.ndarray | None = None,
) -> None:
    """Attach two rigid objects via a fixed constraint for the given env_ids.

    Registered under a custom event mode (e.g. ``"attach"``); the task triggers it
    with ``env.event_manager.apply(mode="attach", env_ids=...)``. Delegates to
    :meth:`SimulationManager.create_rigid_constraint`.

    Args:
        env: The environment instance.
        env_ids: Target environment indices. None -> all envs.
        obj_a_cfg: SceneEntityCfg pointing at the first RigidObject.
        obj_b_cfg: SceneEntityCfg pointing at the second RigidObject.
        name: Base constraint name; per-arena names derived by the sim layer.
        local_frame_a: Local joint frame on object A. None attaches at the
            objects' current relative pose. Accepts (4,4) or (N,4,4).
        local_frame_b: Local joint frame on object B. None -> identity.

    Raises:
        RuntimeError: If either entity is not a RigidObject.
    """
    obj_a = env.sim.get_asset(obj_a_cfg.uid)
    obj_b = env.sim.get_asset(obj_b_cfg.uid)
    if not isinstance(obj_a, RigidObject) or not isinstance(obj_b, RigidObject):
        logger.log_error(
            f"Constraint '{name}' requires two RigidObjects, but got "
            f"{type(obj_a).__name__} and {type(obj_b).__name__}."
        )
    env.sim.create_rigid_constraint(
        cfg=RigidConstraintCfg(
            name=name,
            rigid_object_a_uid=obj_a_cfg.uid,
            rigid_object_b_uid=obj_b_cfg.uid,
            local_frame_a=local_frame_a,
            local_frame_b=local_frame_b,
        ),
        env_ids=env_ids,
    )


def remove_rigid_constraint(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    name: str,
) -> None:
    """Remove the named constraint for the given env_ids.

    Delegates to :meth:`SimulationManager.remove_rigid_constraint`. Idempotent:
    warns (via the sim layer) if the constraint is not found.

    Args:
        env: The environment instance.
        env_ids: Target environment indices. None -> all envs.
        name: Base constraint name to remove.
    """
    env.sim.remove_rigid_constraint(name, env_ids=env_ids)
```

Note: `EmbodiedEnv` is already imported under `TYPE_CHECKING` at the top of `events.py` (line 50). `torch` is already imported (line 19).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gym/envs/managers/test_event_rigid_constraint.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
black embodichain/lab/gym/envs/managers/events.py tests/gym/envs/managers/test_event_rigid_constraint.py
git add embodichain/lab/gym/envs/managers/events.py tests/gym/envs/managers/test_event_rigid_constraint.py
git commit -m "feat(env): add rigid-constraint event functors"
```

---

### Task 6: EventManager custom-mode wiring test

**Files:**
- Test: `tests/gym/envs/managers/test_event_rigid_constraint.py` (append)

**Interfaces:**
- Consumes: `EventManager`, `EventCfg`, the two functors (Task 5), `ManagerBase`.
- Produces: a test proving `EventManager.apply(mode="attach", env_ids)` invokes a registered custom-mode functor once with those env_ids (no env subclassing needed — the manager supports arbitrary mode strings).

- [ ] **Step 1: Write the failing test**

Append to `tests/gym/envs/managers/test_event_rigid_constraint.py`:

```python
from embodichain.lab.gym.envs.managers.event_manager import EventManager
from embodichain.lab.gym.envs.managers.cfg import EventCfg
from embodichain.utils import configclass


@configclass
class _AttachEventsCfg:
    attach: EventCfg = EventCfg(
        func=create_rigid_constraint,
        mode="attach",
        params={
            "obj_a_cfg": SceneEntityCfg(uid="cube"),
            "obj_b_cfg": SceneEntityCfg(uid="block"),
            "name": "weld",
        },
    )


def test_custom_mode_apply_invokes_functor_with_env_ids(monkeypatch):
    """EventManager.apply(mode="attach", env_ids) calls the functor once with those env_ids."""
    # Build a minimal env stand-in that EventManager needs: num_envs, device, sim.
    env = MagicMock()
    env.num_envs = 4
    env.device = torch.device("cpu")
    env.sim = MagicMock()
    env.sim.create_rigid_constraint = MagicMock()

    monkeypatch.setattr(
        "embodichain.lab.gym.envs.managers.events.RigidObject", MockRigidObjectForFunctor
    )

    manager = EventManager(cfg=_AttachEventsCfg(), env=env)

    env_ids = torch.tensor([0, 1])
    manager.apply(mode="attach", env_ids=env_ids)

    env.sim.create_rigid_constraint.assert_called_once()
    assert env.sim.create_rigid_constraint.call_args.kwargs["env_ids"] is env_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gym/envs/managers/test_event_rigid_constraint.py::test_custom_mode_apply_invokes_functor_with_env_ids -v`
Expected: FAIL — likely an import or attribute error if `EventManager` construction with the mock env fails. If the mock env is insufficient, fix the mock (the test asserts behavior, so adjust the mock to satisfy EventManager's needs). Inspect the failure and patch the mock accordingly (e.g. add `env.cfg = None` if needed).

- [ ] **Step 3: (No new implementation needed)**

`EventManager` already supports arbitrary mode strings (see `event_manager.py` `_prepare_functors` — it registers any `functor_cfg.mode` into `_mode_functor_names`). If the test fails on a missing mock attribute, fix the test's mock rather than changing `EventManager`. Document what was needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gym/envs/managers/test_event_rigid_constraint.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
black tests/gym/envs/managers/test_event_rigid_constraint.py
git add tests/gym/envs/managers/test_event_rigid_constraint.py
git commit -m "test(env): custom-mode apply invokes rigid-constraint functor"
```

---

### Task 7: Real-sim integration smoke test

**Files:**
- Create: `tests/sim/test_rigid_constraint_integration.py`

**Interfaces:**
- Consumes: `SimulationManager`, `RigidObjectCfg`, `RigidConstraintCfg`, real dexsim. Mirrors the contract in `dexsim/python/test/engine/test_constraint.py` at the EmbodiChain layer.
- Produces: a skipped-unless-gpu test that attaches two dynamic cubes, steps, asserts the relative transform stays constant, detaches, steps, asserts separation.

- [ ] **Step 1: Write the test**

Create `tests/sim/test_rigid_constraint_integration.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Real-sim integration smoke test for rigid constraints.

Skipped unless a GPU/display is available. Mirrors the dexsim
test_constraint.py contract at the EmbodiChain SimulationManager layer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RigidObjectCfg,
    RigidConstraintCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from dexsim.types import ActorType, RigidBodyShape


def _can_run_gpu_sim() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _can_run_gpu_sim(), reason="GPU simulation required for constraint integration test"
)


def _make_sim(num_envs: int = 1) -> SimulationManager:
    cfg = SimulationManagerCfg()
    cfg.headless = True
    cfg.sim_device = "cuda"
    cfg.num_envs = num_envs
    return SimulationManager(cfg)


def test_fixed_constraint_holds_relative_pose():
    """Two welded cubes keep their relative transform under physics; detach lets them separate."""
    sim = _make_sim(num_envs=1)
    try:
        attrs = RigidBodyAttributesCfg()
        attrs.mass = 0.2
        cube_cfg = RigidObjectCfg(
            uid="cube_a",
            body_type="dynamic",
            init_pos=[0.0, 0.0, 1.4],
            attrs=attrs,
            shape=MeshCfg(fpath="..."),  # placeholder; see note
        )
        block_cfg = RigidObjectCfg(
            uid="cube_b",
            body_type="dynamic",
            init_pos=[0.0, 0.0, 1.2],
            attrs=attrs,
            shape=MeshCfg(fpath="..."),
        )
        cube = sim.add_rigid_object(cube_cfg)
        block = sim.add_rigid_object(block_cfg)

        constraint = sim.create_rigid_constraint(
            RigidConstraintCfg(
                name="weld",
                rigid_object_a_uid="cube_a",
                rigid_object_b_uid="cube_b",
            )
        )
        assert constraint.is_valid() == [True]

        initial_delta_z = (
            block.get_local_pose()[0, 2, 3] - cube.get_local_pose()[0, 2, 3]
        )

        for _ in range(120):
            sim.update(step=1)

        rel = constraint.get_relative_transform()[0]
        np.testing.assert_allclose(rel[:3, 3], np.zeros(3), atol=2e-2)
        delta_z = (
            block.get_local_pose()[0, 2, 3] - cube.get_local_pose()[0, 2, 3]
        )
        assert abs(delta_z - initial_delta_z) < 2e-2

        # detach and confirm they separate
        sim.remove_rigid_constraint("weld")
        z_before = cube.get_local_pose()[0, 2, 3]
        for _ in range(120):
            sim.update(step=1)
        z_after = cube.get_local_pose()[0, 2, 3]
        assert z_after < z_before - 0.05
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
```

- [ ] **Step 2: Run test to verify it skips (or passes on GPU)**

Run: `pytest tests/sim/test_rigid_constraint_integration.py -v`
Expected: SKIPPED (no GPU) on CPU CI. On a GPU machine it should PASS.

- [ ] **Step 3: (Implementation already complete from Tasks 1-5)**

This test exercises the full sim path. If `RigidObjectCfg`/`MeshCfg` construction needs a real mesh path, replace the `fpath="..."` placeholders with a real asset path from `embodichain/data/assets` (check `SimResources` for a built-in cube mesh). Document the chosen path in the test. If the test cannot run headless on the CI runner, leave it gpu-marked and skipped — that's acceptable per the spec.

- [ ] **Step 4: Commit**

```bash
black tests/sim/test_rigid_constraint_integration.py
git add tests/sim/test_rigid_constraint_integration.py
git commit -m "test(sim): add rigid-constraint real-sim integration smoke test"
```

---

### Task 8: Run full suite, black, finalize

**Files:**
- All touched files.

- [ ] **Step 1: Run the full constraint test suite**

Run: `pytest tests/sim/objects/test_rigid_constraint.py tests/gym/envs/managers/test_event_rigid_constraint.py tests/sim/test_rigid_constraint_integration.py -v`
Expected: all unit tests PASS; integration test SKIPPED (or PASS on GPU).

- [ ] **Step 2: Run black on all changed files**

Run: `black embodichain/lab/sim/cfg.py embodichain/lab/sim/objects/constraint.py embodichain/lab/sim/objects/__init__.py embodichain/lab/sim/sim_manager.py embodichain/lab/gym/envs/managers/events.py tests/sim/objects/test_rigid_constraint.py tests/gym/envs/managers/test_event_rigid_constraint.py tests/sim/test_rigid_constraint_integration.py`
Expected: no changes (or only formatting).

- [ ] **Step 3: Run the pre-commit check skill**

Use the `/pre-commit-check` skill (it runs all local CI checks). Fix any violations it reports.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: black formatting + pre-commit fixes"
```

- [ ] **Step 5: Final summary**

The implementation is complete. Summary of what was built:
- `RigidConstraintCfg` (cfg.py)
- `RigidConstraint` wrapper (objects/constraint.py)
- `SimulationManager.create_rigid_constraint` / `remove_rigid_constraint` / `get_rigid_constraint` + `_constraints` registry
- `create_rigid_constraint` / `remove_rigid_constraint` event functors (events.py)
- Unit tests (mocks) + integration smoke test (gpu-marked)

## Self-Review

**1. Spec coverage:**
- §3 Architecture (sim layer + functor layer): Tasks 1-5.
- §4 Sim-layer API (RigidConstraint, RigidConstraintCfg, SimulationManager methods): Tasks 1-4.
- §5 Functor layer (two functors, registration/triggering): Tasks 5-6.
- §6 Data flow (create/remove, per-env selectivity, frame broadcasting, reset interaction): Tasks 3-4 + Task 6 (reset interaction is task-policy, documented in spec; no sim code needed).
- §7 Error handling: each error case is a test in Task 3-4.
- §8 Testing: Tasks 3 (sim unit), 5-6 (functor unit), 7 (integration).
- §9 File layout: matches.

**2. Placeholder scan:** The `MeshCfg(fpath="...")` in Task 7 is flagged as needing a real asset path — that's an integration-test asset-resolution detail, addressed in Task 7 Step 3. No "TBD"/"implement later" in deliverable code.

**3. Type consistency:** `create_rigid_constraint` / `remove_rigid_constraint` / `get_rigid_constraint` names match across sim layer (Task 3-4) and functor layer (Task 5). `RigidConstraint` field names (`constraint_handles`, `rigid_object_a`, `rigid_object_b`, `device`) match Task 2 init and Task 3 usage. `destroy(env_ids, arena_resolver)` signature matches Task 2 and Task 4's call. `_broadcast_frame` is a `@staticmethod` accessed as `SimulationManager._broadcast_frame` in tests (Task 3) and as `self._broadcast_frame` in `create_rigid_constraint` — both valid.
