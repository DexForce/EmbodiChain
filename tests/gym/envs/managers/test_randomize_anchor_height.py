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

from embodichain.lab.gym.envs.managers import EventCfg
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_anchor_height,
)
from embodichain.lab.sim.objects import RigidObjectGroup

# ---------------------------------------------------------------------------
# Shared mock classes
# ---------------------------------------------------------------------------


class _MockObject:
    """Base mock for RigidObject / Articulation with (N, 7) pose storage."""

    def __init__(self, uid: str, num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        self._pose = torch.zeros(num_envs, 7)
        self._pose[:, 3] = 1.0  # identity quaternion
        self._cleared = False
        self._cleared_env_ids = None

    def get_local_pose(self, to_matrix: bool = False):
        if to_matrix:
            mat = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1)
            mat[:, :3, 3] = self._pose[:, :3]
            return mat
        return self._pose.clone()

    def set_local_pose(self, pose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        if pose.ndim == 3:
            self._pose[env_ids, :3] = pose[:, :3, 3]
        else:
            self._pose[env_ids] = pose

    def clear_dynamics(self, env_ids=None):
        self._cleared = True
        self._cleared_env_ids = env_ids


class _MockArticulation(_MockObject):
    """Articulation mock — identical to _MockObject in behavior."""

    pass


class _MockGroup(RigidObjectGroup):
    """Mock for RigidObjectGroup with (N, M, 4, 4) pose storage.

    Inherits from RigidObjectGroup so isinstance checks pass, but skips the
    real parent ``__init__`` to avoid heavy simulation dependencies.
    """

    def __init__(self, uid: str, num_envs: int = 4, num_objects: int = 2):
        # Skip RigidObjectGroup.__init__ — only set what the functor needs.
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        # num_objects is a property backed by self._data.num_objects
        self._data = MagicMock()
        self._data.num_objects = num_objects
        self._pose = (
            torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(num_envs, num_objects, 1, 1)
        )
        self._cleared = False
        self._cleared_env_ids = None

    def get_local_pose(self, to_matrix: bool = False):
        return self._pose.clone()

    def set_local_pose(self, pose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self._pose[env_ids] = pose

    def clear_dynamics(self, env_ids=None):
        self._cleared = True
        self._cleared_env_ids = env_ids


class _MockSim:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._objects: dict[str, _MockObject] = {}
        self._articulations: dict[str, _MockArticulation] = {}
        self._groups: dict[str, _MockGroup] = {}

    def get_rigid_object(self, uid: str):
        return self._objects.get(uid)

    def get_rigid_object_uid_list(self):
        return list(self._objects.keys())

    def get_articulation(self, uid: str):
        return self._articulations.get(uid)

    def get_articulation_uid_list(self):
        return list(self._articulations.keys())

    def get_rigid_object_group(self, uid: str):
        return self._groups.get(uid)

    def get_rigid_object_group_uid_list(self):
        return list(self._groups.keys())

    def add_rigid_object(self, obj):
        self._objects[obj.uid] = obj

    def add_articulation(self, obj):
        self._articulations[obj.uid] = obj

    def add_rigid_object_group(self, obj):
        self._groups[obj.uid] = obj

    def update(self, step: int):
        pass


class _MockEnv:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.sim = _MockSim(num_envs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    """Return a fresh MockEnv with 4 environments."""
    return _MockEnv()


@pytest.fixture
def make_functor(env):
    """Create a functor instance wired to the given env."""

    def _make(**kwargs):
        return randomize_anchor_height(
            EventCfg(func=randomize_anchor_height, mode="reset", **kwargs), env
        )

    return _make


@pytest.fixture
def env_ids():
    """Default env_ids tensor (all 4 environments)."""
    return torch.arange(4)


@pytest.fixture
def env_with_table(env):
    """Env with a table rigid object at init_pos Z=1.0."""
    table = _MockObject("table")
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)
    return env


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_anchor_uid_raises(env, make_functor, env_ids):
    functor = make_functor()
    with pytest.raises(ValueError):
        functor(
            env,
            env_ids,
            anchor_uid="missing_table",
            height_delta_range=([-0.05], [0.05]),
        )


def test_missing_sampling_fields_raises(env_with_table, make_functor, env_ids):
    functor = make_functor()
    with pytest.raises(ValueError):
        functor(env_with_table, env_ids, anchor_uid="table")


def test_empty_candidates_raises(env_with_table, make_functor, env_ids):
    functor = make_functor()
    with pytest.raises(ValueError):
        functor(
            env_with_table,
            env_ids,
            anchor_uid="table",
            height_delta_candidates=[],
        )


def test_invalid_include_groups_raises(env_with_table, make_functor, env_ids):
    functor = make_functor()
    with pytest.raises(ValueError, match="Invalid include_groups"):
        functor(
            env_with_table,
            env_ids,
            anchor_uid="table",
            height_delta_range=([-0.05], [0.05]),
            include_groups=["invalid_group", "rigid_object"],
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_envs", [100])
def test_range_sampling_within_bounds(env, make_functor, num_envs):
    env.num_envs = num_envs
    env.sim = _MockSim(num_envs)
    table = _MockObject("table", num_envs)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    functor = make_functor()
    functor(
        env,
        torch.arange(num_envs),
        anchor_uid="table",
        height_delta_range=([-0.05], [0.05]),
        store_key="table_delta",
    )

    delta = getattr(env, "table_delta")
    assert delta.shape == (num_envs,)
    assert (delta >= -0.05).all()
    assert (delta <= 0.05).all()


@pytest.mark.parametrize("num_envs", [50])
def test_discrete_sampling_only_candidates(env, make_functor, num_envs):
    env.num_envs = num_envs
    env.sim = _MockSim(num_envs)
    table = _MockObject("table", num_envs)
    env.sim.add_rigid_object(table)

    functor = make_functor()
    functor(
        env,
        torch.arange(num_envs),
        anchor_uid="table",
        height_delta_candidates=[-0.05, 0.0, 0.05],
        store_key="table_delta",
    )

    candidates = torch.tensor([-0.05, 0.0, 0.05])
    for val in getattr(env, "table_delta"):
        assert torch.any(torch.isclose(val, candidates)), f"{val} not in {candidates}"


# ---------------------------------------------------------------------------
# Shift correctness
# ---------------------------------------------------------------------------


def test_anchor_and_objects_shifted_by_same_delta(env, make_functor, env_ids):
    table = _MockObject("table")
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = _MockObject("cube")
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = make_functor()
    functor(env, env_ids, anchor_uid="table", height_delta_range=([0.05], [0.05]))

    # Anchor: absolute -> init_pos[2] + delta = 1.0 + 0.05 = 1.05
    torch.testing.assert_close(table._pose[:, 2], torch.ones(4) * 1.05)
    # Affected: relative -> current_z + delta = 1.1 + 0.05 = 1.15
    torch.testing.assert_close(cube._pose[:, 2], torch.ones(4) * 1.15)


def test_xy_and_rotation_unchanged(env, make_functor, env_ids):
    table = _MockObject("table")
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = _MockObject("cube")
    cube._pose[:, 0] = 0.5
    cube._pose[:, 1] = -0.3
    env.sim.add_rigid_object(cube)

    original_xy = cube._pose[:, :2].clone()
    original_rot = cube._pose[:, 3:7].clone()

    functor = make_functor()
    functor(env, env_ids, anchor_uid="table", height_delta_range=([0.1], [0.1]))

    torch.testing.assert_close(cube._pose[:, :2], original_xy)
    torch.testing.assert_close(cube._pose[:, 3:7], original_rot)


def test_exclude_uids_are_not_moved(env, make_functor, env_ids):
    table = _MockObject("table")
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = _MockObject("cube")
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    floor = _MockObject("floor")
    floor._pose[:, 2] = 0.0
    env.sim.add_rigid_object(floor)

    functor = make_functor()
    functor(
        env,
        env_ids,
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
        exclude_uids=["floor"],
    )

    torch.testing.assert_close(floor._pose[:, 2], torch.zeros(4))
    torch.testing.assert_close(cube._pose[:, 2], torch.ones(4) * 1.2)


def test_articulation_shifted(env, make_functor, env_ids):
    table = _MockObject("table")
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cabinet = _MockArticulation("cabinet")
    cabinet._pose[:, 2] = 1.2
    env.sim.add_articulation(cabinet)

    functor = make_functor()
    functor(env, env_ids, anchor_uid="table", height_delta_range=([0.1], [0.1]))

    torch.testing.assert_close(cabinet._pose[:, 2], torch.ones(4) * 1.3)


@pytest.mark.parametrize("num_envs", [100])
def test_asymmetric_delta_range(env, make_functor, env_ids, num_envs):
    env.num_envs = num_envs
    env.sim = _MockSim(num_envs)
    table = _MockObject("table", num_envs)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    functor = make_functor()
    functor(
        env,
        torch.arange(num_envs),
        anchor_uid="table",
        height_delta_range=([-0.1], [0.05]),
        store_key="table_delta",
    )

    delta = getattr(env, "table_delta")
    assert delta.shape == (num_envs,)
    assert (delta >= -0.1).all()
    assert (delta <= 0.05).all()


# ---------------------------------------------------------------------------
# Partial env_ids
# ---------------------------------------------------------------------------


def test_partial_env_ids(env, make_functor):
    table = _MockObject("table")
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = _MockObject("cube")
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = make_functor()
    partial_ids = torch.tensor([0, 2])
    functor(
        env,
        partial_ids,
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
    )

    # Envs 0,2 shifted; envs 1,3 unchanged
    torch.testing.assert_close(table._pose[0, 2], torch.tensor(1.1))
    torch.testing.assert_close(table._pose[2, 2], torch.tensor(1.1))
    torch.testing.assert_close(cube._pose[0, 2], torch.tensor(1.2))
    torch.testing.assert_close(cube._pose[2, 2], torch.tensor(1.2))
    torch.testing.assert_close(table._pose[1, 2], torch.tensor(0.0))
    torch.testing.assert_close(table._pose[3, 2], torch.tensor(0.0))
    torch.testing.assert_close(cube._pose[1, 2], torch.tensor(1.1))
    torch.testing.assert_close(cube._pose[3, 2], torch.tensor(1.1))

    # clear_dynamics called with targeted env_ids
    assert table._cleared and table._cleared_env_ids is not None
    torch.testing.assert_close(table._cleared_env_ids, partial_ids)
    assert cube._cleared and cube._cleared_env_ids is not None
    torch.testing.assert_close(cube._cleared_env_ids, partial_ids)


# ---------------------------------------------------------------------------
# RigidObjectGroup anchor
# ---------------------------------------------------------------------------


def test_rigid_object_group_anchor_absolute_warning(env, make_functor, env_ids, caplog):
    """When anchor is a RigidObjectGroup and absolute=True, a warning is emitted
    and the relative shift is applied."""
    group = _MockGroup("group", num_objects=2)
    group._pose[:, :, 2, 3] = 1.0
    env.sim.add_rigid_object_group(group)

    cube = _MockObject("cube")
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = make_functor()
    functor(
        env,
        env_ids,
        anchor_uid="group",
        height_delta_range=([0.1], [0.1]),
    )

    assert "absolute=True is not supported for RigidObjectGroup" in caplog.text
    # Group: relative shift -> 1.0 + 0.1 = 1.1
    torch.testing.assert_close(group._pose[:, :, 2, 3], torch.ones(4, 2) * 1.1)
    # Cube: relative shift -> 1.1 + 0.1 = 1.2
    torch.testing.assert_close(cube._pose[:, 2], torch.ones(4) * 1.2)


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires full simulation stack; run manually.")
def test_anchor_height_event_wires_into_embodied_env_cfg():
    """Smoke test: functor can be wired into an EmbodiedEnvCfg."""
    from embodichain.lab.gym.envs import EmbodiedEnvCfg
    from embodichain.lab.sim.cfg import RigidObjectCfg

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
    assert hasattr(cfg.events, "anchor_height")
