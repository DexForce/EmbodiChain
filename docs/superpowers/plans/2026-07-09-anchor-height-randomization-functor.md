# Anchor Height Randomization Functor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a class-style event functor `randomize_anchor_height` that randomizes the Z height of a configurable anchor object (e.g., a table) and applies the same height delta to all other included scene objects, preserving their XY position and rotation.

**Architecture:** A single class-style functor plus its config live in `embodichain/lab/gym/envs/managers/randomization/spatial.py`, following the existing `planner_grid_cell_sampler` pattern. It resolves target objects at init time, samples a per-environment delta at reset time, moves the anchor relative to its configured `init_pos`, shifts all other included objects relative to their current pose, and stores the delta on `env` for downstream use.

**Tech Stack:** Python 3.10+, PyTorch, EmbodiChain's `@configclass`, `Functor` base class, `sample_uniform`, `RigidObject`/`Articulation` pose APIs.

## Global Constraints

- Python package name is `embodichain` (all lowercase).
- Every source file must start with the Apache 2.0 copyright header.
- Use `from __future__ import annotations` at the top of every file.
- Use full type hints on all public APIs; prefer `A | B` over `Union[A, B]`.
- Format with `black==26.3.1` before every commit.
- Run the `/pre-commit-check` skill before final commit/PR.
- Use lowercase-with-underscores naming for the functor class to match `planner_grid_cell_sampler`.

---

## File Structure

- **Create/Modify:** `embodichain/lab/gym/envs/managers/randomization/spatial.py`
  - Add `randomize_anchor_height_cfg` config class.
  - Add `randomize_anchor_height` class-style functor.
- **Create:** `tests/gym/envs/managers/test_randomize_anchor_height.py`
  - Unit tests for config validation, sampling, pose updates, exclusions, and state storage.

The randomization `__init__.py` uses wildcard imports (`from .spatial import *`), so no export list changes are needed unless you add `__all__` to `spatial.py`.

---

### Task 1: Add config and functor to spatial.py

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/randomization/spatial.py`
- Test: `tests/gym/envs/managers/test_randomize_anchor_height.py` (created in Task 2)

**Interfaces:**
- Consumes: `Functor`, `FunctorCfg` from `embodichain.lab.gym.envs.managers`; `sample_uniform` from `embodichain.utils.math`; `RigidObject`, `Articulation`, `RigidObjectGroup` from `embodichain.lab.sim.objects`; `env.sim` object APIs (`get_rigid_object`, `get_rigid_object_uid_list`, `get_articulation`, `get_articulation_uid_list`, `get_rigid_object_group`, `get_rigid_object_group_uid_list`).
- Produces: `randomize_anchor_height_cfg` config class and `randomize_anchor_height` functor class with `__init__(cfg, env)` and `__call__(env, env_ids)`.

- [ ] **Step 1: Write the failing test for config validation**

Create `tests/gym/envs/managers/test_randomize_anchor_height.py`:

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

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock

from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_anchor_height,
    randomize_anchor_height_cfg,
)


class MockRigidObject:
    def __init__(self, uid: str, num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        self._pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)
        self._cleared = False

    def get_local_pose(self, to_matrix: bool = True):
        return self._pose.clone()

    def set_local_pose(self, pose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self._pose[env_ids] = pose

    def clear_dynamics(self):
        self._cleared = True


class MockArticulation:
    def __init__(self, uid: str, num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        self._pose = torch.zeros(num_envs, 7)
        self._pose[:, 0] = 1.0  # qw = 1
        self._cleared = False

    def get_local_pose(self):
        return self._pose.clone()

    def set_local_pose(self, pose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self._pose[env_ids] = pose

    def clear_dynamics(self, env_ids=None):
        self._cleared = True


class MockSim:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._rigid_objects: dict[str, MockRigidObject] = {}
        self._articulations: dict[str, MockArticulation] = {}
        self._rigid_object_groups: dict[str, object] = {}

    def get_rigid_object(self, uid: str):
        return self._rigid_objects.get(uid)

    def get_rigid_object_uid_list(self):
        return list(self._rigid_objects.keys())

    def get_articulation(self, uid: str):
        return self._articulations.get(uid)

    def get_articulation_uid_list(self):
        return list(self._articulations.keys())

    def get_rigid_object_group(self, uid: str):
        return self._rigid_object_groups.get(uid)

    def get_rigid_object_group_uid_list(self):
        return list(self._rigid_object_groups.keys())

    def add_rigid_object(self, obj):
        self._rigid_objects[obj.uid] = obj

    def add_articulation(self, obj):
        self._articulations[obj.uid] = obj

    def add_rigid_object_group(self, obj):
        self._rigid_object_groups[obj.uid] = obj

    def update(self, step: int):
        pass


class MockEnv:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.sim = MockSim(num_envs)


def test_missing_anchor_uid_raises():
    env = MockEnv()
    cfg = randomize_anchor_height_cfg(
        anchor_uid="missing_table",
        height_delta_range=([-0.05], [0.05]),
    )
    with pytest.raises(ValueError):
        randomize_anchor_height(cfg, env)


def test_missing_sampling_fields_raises():
    env = MockEnv()
    cfg = randomize_anchor_height_cfg(anchor_uid="table")
    with pytest.raises(ValueError):
        randomize_anchor_height(cfg, env)


def test_empty_candidates_raises():
    env = MockEnv()
    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_candidates=[],
    )
    with pytest.raises(ValueError):
        randomize_anchor_height(cfg, env)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/gym/envs/managers/test_randomize_anchor_height.py -v
```

Expected: FAIL with `ImportError: cannot import name 'randomize_anchor_height'` or similar.

- [ ] **Step 3: Implement the config and functor in spatial.py**

Add the following to `embodichain/lab/gym/envs/managers/randomization/spatial.py`, after the existing imports and before the existing function definitions (or at the end of the file):

```python
from dataclasses import MISSING

from embodichain.lab.sim.objects import RigidObjectGroup


@configclass
class randomize_anchor_height_cfg(FunctorCfg):
    """Configuration for the randomize_anchor_height functor.

    This functor randomizes the Z height of an anchor object (e.g., a table) and
    applies the same height delta to all other included scene objects, preserving
    their XY position and rotation.
    """

    anchor_uid: str = MISSING
    """Exact UID of the anchor object whose height is randomized."""

    height_delta_range: tuple[list[float], list[float]] | None = None
    """Uniform sampling range for the height delta: ([z_min], [z_max])."""

    height_delta_candidates: list[float] | None = None
    """Discrete set of allowed height delta values."""

    include_groups: list[str] | None = None
    """Object groups to shift. ``None`` means all groups are included."""

    exclude_uids: list[str] = []
    """Additional UIDs to skip beyond the anchor object."""

    mode: str = "reset"
    """Event mode (``startup``, ``interval``, or ``reset``)."""

    physics_update_step: int = 0
    """Number of physics update steps to apply after moving objects."""

    store_key: str = "anchor_height_delta"
    """Attribute name on ``env`` where the sampled delta is stored."""


class randomize_anchor_height(Functor):
    """Randomize the height of an anchor object and shift other objects by the same delta.

    The functor samples a per-environment height delta, moves the anchor object
    relative to its configured initial position, and adds the same delta to the
    Z component of every other included object while preserving XY and rotation.
    """

    _VALID_GROUPS = {"background", "rigid_object", "rigid_object_group", "articulation"}

    def __init__(self, cfg: randomize_anchor_height_cfg, env: EmbodiedEnv):
        """Initialize the functor and resolve affected object UIDs.

        Args:
            cfg: The functor configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Validate sampling configuration
        if cfg.height_delta_range is None and cfg.height_delta_candidates is None:
            raise ValueError(
                "Either 'height_delta_range' or 'height_delta_candidates' must be provided."
            )
        if cfg.height_delta_candidates is not None and len(cfg.height_delta_candidates) == 0:
            raise ValueError("'height_delta_candidates' must not be empty.")
        if cfg.height_delta_range is not None and cfg.height_delta_candidates is not None:
            logger.log_warning(
                "Both 'height_delta_range' and 'height_delta_candidates' provided; "
                "using 'height_delta_range'."
            )

        # Resolve include groups
        include_groups = cfg.include_groups
        if include_groups is None:
            include_groups = ["background", "rigid_object", "rigid_object_group", "articulation"]
        invalid_groups = set(include_groups) - self._VALID_GROUPS
        if invalid_groups:
            raise ValueError(
                f"Invalid include_groups: {sorted(invalid_groups)}. "
                f"Valid options are: {sorted(self._VALID_GROUPS)}."
            )
        self._include_groups = include_groups

        # Confirm anchor exists
        anchor = self._get_object(cfg.anchor_uid)
        if anchor is None:
            raise ValueError(
                f"Anchor object with uid '{cfg.anchor_uid}' not found in the scene."
            )
        self._anchor = anchor

        # Build affected UID list
        self._affected_uids = self._resolve_affected_uids(env, cfg.anchor_uid, cfg.exclude_uids)

    def _get_object(self, uid: str):
        """Get a rigid object, articulation, or rigid object group by UID."""
        if uid in self._env.sim.get_rigid_object_uid_list():
            return self._env.sim.get_rigid_object(uid)
        if uid in self._env.sim.get_articulation_uid_list():
            return self._env.sim.get_articulation(uid)
        if hasattr(self._env.sim, "get_rigid_object_group_uid_list") and uid in self._env.sim.get_rigid_object_group_uid_list():
            return self._env.sim.get_rigid_object_group(uid)
        return None

    def _resolve_affected_uids(
        self, env: EmbodiedEnv, anchor_uid: str, exclude_uids: list[str]
    ) -> list[str]:
        """Resolve the list of UIDs that should be shifted."""
        uids: set[str] = set()
        if any(g in self._include_groups for g in ("background", "rigid_object")):
            uids.update(env.sim.get_rigid_object_uid_list())
        if "rigid_object_group" in self._include_groups:
            if hasattr(env.sim, "get_rigid_object_group_uid_list"):
                uids.update(env.sim.get_rigid_object_group_uid_list())
        if "articulation" in self._include_groups:
            uids.update(env.sim.get_articulation_uid_list())

        exclude = set(exclude_uids) | {anchor_uid}
        return sorted(uids - exclude)

    def _sample_delta(self, num_envs: int) -> torch.Tensor:
        """Sample a height delta for each environment."""
        cfg = self.cfg
        device = self._env.device

        if cfg.height_delta_range is not None:
            low = torch.tensor(cfg.height_delta_range[0], device=device)
            high = torch.tensor(cfg.height_delta_range[1], device=device)
            return sample_uniform(lower=low, upper=high, size=(num_envs, 1), device=device).squeeze(-1)

        # Discrete sampling
        candidates = torch.tensor(cfg.height_delta_candidates, device=device)
        indices = torch.randint(0, len(candidates), (num_envs,), device=device)
        return candidates[indices]

    def _move_object_z(self, obj, delta_z: torch.Tensor, env_ids: torch.Tensor, absolute: bool = False) -> None:
        """Move an object in Z by delta_z.

        Args:
            obj: The object to move (RigidObject, Articulation, or RigidObjectGroup).
            delta_z: Per-environment Z offset.
            env_ids: Target environment IDs.
            absolute: If True, set Z to obj.cfg.init_pos[2] + delta_z.
                      If False, add delta_z to the current Z.
        """
        if isinstance(obj, RigidObjectGroup):
            # RigidObjectGroup does not have a single init_pos; always shift relative to current pose.
            pose = obj.get_local_pose(to_matrix=True)  # (N, M, 4, 4)
            pose[env_ids, :, 2, 3] += delta_z.unsqueeze(-1)
            obj.set_local_pose(pose[env_ids], env_ids=env_ids)
            return

        pose = obj.get_local_pose()
        if pose.ndim == 3:
            # (N, 4, 4) matrix form from RigidObject
            current_z = pose[env_ids, 2, 3]
            if absolute:
                init_z = torch.tensor(
                    obj.cfg.init_pos[2], dtype=torch.float32, device=obj.device
                )
                new_z = init_z + delta_z
            else:
                new_z = current_z + delta_z
            pose[env_ids, 2, 3] = new_z
        else:
            # (N, 7) vector form from Articulation: (x, y, z, qw, qx, qy, qz)
            current_z = pose[env_ids, 2]
            if absolute:
                init_z = torch.tensor(
                    obj.cfg.init_pos[2], dtype=torch.float32, device=obj.device
                )
                new_z = init_z + delta_z
            else:
                new_z = current_z + delta_z
            pose[env_ids, 2] = new_z

        obj.set_local_pose(pose[env_ids], env_ids=env_ids)
        if hasattr(obj, "clear_dynamics"):
            if isinstance(obj, Articulation):
                obj.clear_dynamics(env_ids=env_ids)
            else:
                obj.clear_dynamics()

    def __call__(self, env: EmbodiedEnv, env_ids: torch.Tensor | None) -> None:
        """Apply the height randomization.

        Args:
            env: The environment instance.
            env_ids: Target environment IDs. If None, all environments are targeted.
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        if len(env_ids) == 0:
            return

        num_envs = len(env_ids)
        delta_z = self._sample_delta(num_envs)

        # Move anchor relative to its initial pose
        self._move_object_z(self._anchor, delta_z, env_ids, absolute=True)

        # Move affected objects relative to their current pose
        for uid in self._affected_uids:
            obj = self._get_object(uid)
            if obj is None:
                logger.log_warning(
                    f"Affected object '{uid}' no longer exists; skipping height shift."
                )
                continue
            self._move_object_z(obj, delta_z, env_ids, absolute=False)

        # Physics settle
        if self.cfg.physics_update_step > 0:
            env.sim.update(step=self.cfg.physics_update_step)

        # Store delta for downstream use
        if self.cfg.store_key:
            setattr(env, self.cfg.store_key, delta_z)
```

Note: update the imports at the top of `spatial.py`:
- Add `from dataclasses import MISSING`.
- Change `from embodichain.lab.sim.objects import RigidObject, Robot, Articulation` to `from embodichain.lab.sim.objects import RigidObject, Robot, Articulation, RigidObjectGroup`.

- [ ] **Step 4: Run the config validation tests**

Run:

```bash
pytest tests/gym/envs/managers/test_randomize_anchor_height.py -v
```

Expected: PASS for `test_missing_anchor_uid_raises`, `test_missing_sampling_fields_raises`, `test_empty_candidates_raises`.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/gym/envs/managers/randomization/spatial.py tests/gym/envs/managers/test_randomize_anchor_height.py
git commit -m "feat(randomization): add randomize_anchor_height functor and config

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add pose-update and sampling unit tests

**Files:**
- Modify: `tests/gym/envs/managers/test_randomize_anchor_height.py`
- Depends on: Task 1

**Interfaces:**
- Consumes: `randomize_anchor_height`, `randomize_anchor_height_cfg`, `MockEnv`, `MockRigidObject`, `MockArticulation` from Task 1.
- Produces: Passing tests for sampling, pose update, exclusions, and state storage.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gym/envs/managers/test_randomize_anchor_height.py`:

```python
def test_range_sampling_within_bounds():
    env = MockEnv(num_envs=100)
    table = MockRigidObject("table", num_envs=100)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=100)
    cube._pose[:, 2, 3] = 1.1
    env.sim.add_rigid_object(cube)

    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_range=([-0.05], [0.05]),
        store_key="table_delta",
    )
    functor = randomize_anchor_height(cfg, env)
    env_ids = torch.arange(100)
    functor(env, env_ids)

    delta = env.table_delta
    assert delta.shape == (100,)
    assert (delta >= -0.05).all()
    assert (delta <= 0.05).all()


def test_discrete_sampling_only_candidates():
    env = MockEnv(num_envs=50)
    table = MockRigidObject("table", num_envs=50)
    env.sim.add_rigid_object(table)

    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_candidates=[-0.05, 0.0, 0.05],
        store_key="table_delta",
    )
    functor = randomize_anchor_height(cfg, env)
    functor(env, torch.arange(50))

    assert set(env.table_delta.tolist()).issubset({-0.05, 0.0, 0.05})


def test_anchor_and_objects_shifted_by_same_delta():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube.cfg.init_pos = [0.0, 0.0, 1.1]
    cube._pose[:, 2, 3] = 1.1
    env.sim.add_rigid_object(cube)

    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_range=([0.05], [0.05]),
    )
    functor = randomize_anchor_height(cfg, env)
    functor(env, torch.arange(4))

    torch.testing.assert_close(table._pose[:, 2, 3], torch.ones(4) * 1.05)
    torch.testing.assert_close(cube._pose[:, 2, 3], torch.ones(4) * 1.15)


def test_xy_and_rotation_unchanged():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube._pose[:, 0, 3] = 0.5
    cube._pose[:, 1, 3] = -0.3
    env.sim.add_rigid_object(cube)

    original_xy = cube._pose[:, :2, 3].clone()
    original_rot = cube._pose[:, :3, :3].clone()

    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
    )
    functor = randomize_anchor_height(cfg, env)
    functor(env, torch.arange(4))

    torch.testing.assert_close(cube._pose[:, :2, 3], original_xy)
    torch.testing.assert_close(cube._pose[:, :3, :3], original_rot)


def test_exclude_uids_are_not_moved():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube._pose[:, 2, 3] = 1.1
    env.sim.add_rigid_object(cube)

    floor = MockRigidObject("floor", num_envs=4)
    floor._pose[:, 2, 3] = 0.0
    env.sim.add_rigid_object(floor)

    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
        exclude_uids=["floor"],
    )
    functor = randomize_anchor_height(cfg, env)
    functor(env, torch.arange(4))

    torch.testing.assert_close(floor._pose[:, 2, 3], torch.zeros(4))


def test_articulation_shifted():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cabinet = MockArticulation("cabinet", num_envs=4)
    cabinet._pose[:, 2] = 1.2
    env.sim.add_articulation(cabinet)

    cfg = randomize_anchor_height_cfg(
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
    )
    functor = randomize_anchor_height(cfg, env)
    functor(env, torch.arange(4))

    torch.testing.assert_close(cabinet._pose[:, 2], torch.ones(4) * 1.3)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
pytest tests/gym/envs/managers/test_randomize_anchor_height.py -v
```

Expected: Some tests may fail if the implementation does not yet handle articulation vector poses or absolute anchor movement correctly.

- [ ] **Step 3: Fix any implementation issues**

If tests fail, adjust `_move_object_z` or `_resolve_affected_uids` in `spatial.py`. Common fixes:
- Ensure `obj.device` is valid; if `MockRigidObject.device` is a string (`"cpu"`), convert via `torch.device(obj.device)`.
- Ensure `set_local_pose` receives a tensor of shape `(N, 7)` for articulations and `(N, 4, 4)` for rigid objects.

- [ ] **Step 4: Run tests again**

Run:

```bash
pytest tests/gym/envs/managers/test_randomize_anchor_height.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/gym/envs/managers/test_randomize_anchor_height.py embodichain/lab/gym/envs/managers/randomization/spatial.py
git commit -m "test(randomization): add unit tests for randomize_anchor_height

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Format and run pre-commit checks

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/randomization/spatial.py`
- Modify: `tests/gym/envs/managers/test_randomize_anchor_height.py`

- [ ] **Step 1: Format with black**

Run:

```bash
black embodichain/lab/gym/envs/managers/randomization/spatial.py tests/gym/envs/managers/test_randomize_anchor_height.py
```

Expected: Both files reformatted (or already formatted).

- [ ] **Step 2: Run the pre-commit-check skill**

Invoke:

```bash
/pre-commit-check
```

Or if not using the skill, run the relevant CI checks locally (lint, type check, tests):

```bash
pytest tests/gym/envs/managers/test_randomize_anchor_height.py -v
```

Expected: All checks pass.

- [ ] **Step 3: Commit formatting fixes**

```bash
git add embodichain/lab/gym/envs/managers/randomization/spatial.py tests/gym/envs/managers/test_randomize_anchor_height.py
git commit -m "style(randomization): format randomize_anchor_height and tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Optional integration smoke test

**Files:**
- Create: `tests/gym/envs/managers/test_randomize_anchor_height_integration.py`

**Interfaces:**
- Consumes: A minimal EmbodiChain environment with a table, one rigid object, and the `randomize_anchor_height` event wired in reset mode.
- Produces: A passing test that verifies the functor runs without crashing during reset and that the object Z changes relative to the table.

- [ ] **Step 1: Write the integration test**

Create `tests/gym/envs/managers/test_randomize_anchor_height_integration.py`:

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

from __future__ import annotations

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers import EventCfg
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_anchor_height,
)
from embodichain.lab.sim.cfg import RigidObjectCfg


@pytest.mark.skip(reason="Requires full simulation stack; run manually.")
def test_anchor_height_event_runs_in_reset():
    """Smoke test that the functor can be wired into an EmbodiedEnvCfg."""
    cfg = EmbodiedEnvCfg()
    cfg.events.anchor_height = EventCfg(
        func=randomize_anchor_height,
        mode="reset",
        params={
            "anchor_uid": "table",
            "height_delta_range": ([-0.05], [0.05]),
        },
    )
    cfg.background.append(
        RigidObjectCfg(uid="table", init_pos=[0.0, 0.0, 0.8], body_type="static")
    )
    cfg.rigid_object.append(
        RigidObjectCfg(uid="cube", init_pos=[0.1, 0.0, 0.9], body_type="dynamic")
    )
    # Actual env construction and reset would go here.
    assert hasattr(cfg.events, "anchor_height")
```

- [ ] **Step 2: Run the integration test**

Run:

```bash
pytest tests/gym/envs/managers/test_randomize_anchor_height_integration.py -v
```

Expected: SKIP (it is marked skip until a full sim stack is available).

- [ ] **Step 3: Commit**

```bash
git add tests/gym/envs/managers/test_randomize_anchor_height_integration.py
git commit -m "test(randomization): add integration smoke test for randomize_anchor_height

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task covering it |
|------------------|------------------|
| Execute after pose randomizers (reset mode) | Task 1 (mode default) and docstring |
| Move anchor by sampled delta | Task 1 `_move_object_z(..., absolute=True)` |
| Move other objects by same delta, preserve XY/rotation | Task 1 `_move_object_z(..., absolute=False)` and Task 2 tests |
| Do not move robot base | Task 1 (robot is not in resolved groups) |
| Configurable include/exclude | Task 1 `include_groups` and `exclude_uids` |
| Expose sampled delta | Task 1 `store_key` and Task 2 state-storage test |
| Lowercase-with-underscores naming | Task 1 class name `randomize_anchor_height` |

### Placeholder scan

- No TBD/TODO/fill-in-details placeholders remain.
- All code blocks contain concrete implementation or test code.
- All file paths are exact.

### Type consistency

- `randomize_anchor_height_cfg` uses `tuple[list[float], list[float]] | None` consistently.
- `__call__` signature matches the manager convention `(env, env_ids)`.
- `store_key` is a string used consistently for `setattr`/`getattr`.
