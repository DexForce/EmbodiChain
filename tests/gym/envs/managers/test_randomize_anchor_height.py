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


def _make_functor(env):
    """Create a randomize_anchor_height functor wired like the event manager."""
    return randomize_anchor_height(
        EventCfg(func=randomize_anchor_height, mode="reset"), env
    )


class MockRigidObject:
    def __init__(self, uid: str, num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        # (x, y, z, qw, qx, qy, qz)
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
            # (N, 4, 4) matrix form
            self._pose[env_ids, :3] = pose[:, :3, 3]
        else:
            # (N, 7) vector form
            self._pose[env_ids] = pose

    def clear_dynamics(self, env_ids=None):
        self._cleared = True
        self._cleared_env_ids = env_ids


class MockArticulation:
    def __init__(self, uid: str, num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        # (x, y, z, qw, qx, qy, qz)
        self._pose = torch.zeros(num_envs, 7)
        self._pose[:, 3] = 1.0  # identity quaternion
        self._cleared = False
        self._cleared_env_ids = None

    def get_local_pose(self):
        return self._pose.clone()

    def set_local_pose(self, pose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self._pose[env_ids] = pose

    def clear_dynamics(self, env_ids=None):
        self._cleared = True
        self._cleared_env_ids = env_ids


class MockRigidObjectGroup:
    def __init__(self, uid: str, num_envs: int = 4, num_objects: int = 2):
        self.uid = uid
        self.num_envs = num_envs
        self.num_objects = num_objects
        self.device = torch.device("cpu")
        self.cfg = MagicMock()
        self.cfg.init_pos = [0.0, 0.0, 0.0]
        # (num_instances, num_objects, 4, 4)
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


class MockSim:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._rigid_objects: dict[str, MockRigidObject] = {}
        self._articulations: dict[str, MockArticulation] = {}
        self._rigid_object_groups: dict[str, MockRigidObjectGroup] = {}

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


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_missing_anchor_uid_raises():
    env = MockEnv()
    functor = _make_functor(env)
    with pytest.raises(ValueError):
        functor(
            env,
            torch.arange(4),
            anchor_uid="missing_table",
            height_delta_range=([-0.05], [0.05]),
        )


def test_missing_sampling_fields_raises():
    env = MockEnv()
    table = MockRigidObject("table", num_envs=4)
    env.sim.add_rigid_object(table)

    functor = _make_functor(env)
    with pytest.raises(ValueError):
        functor(env, torch.arange(4), anchor_uid="table")


def test_empty_candidates_raises():
    env = MockEnv()
    table = MockRigidObject("table", num_envs=4)
    env.sim.add_rigid_object(table)

    functor = _make_functor(env)
    with pytest.raises(ValueError):
        functor(
            env,
            torch.arange(4),
            anchor_uid="table",
            height_delta_candidates=[],
        )


def test_invalid_include_groups_raises():
    env = MockEnv()
    table = MockRigidObject("table", num_envs=4)
    env.sim.add_rigid_object(table)

    functor = _make_functor(env)
    with pytest.raises(ValueError, match="Invalid include_groups"):
        functor(
            env,
            torch.arange(4),
            anchor_uid="table",
            height_delta_range=([-0.05], [0.05]),
            include_groups=["invalid_group", "rigid_object"],
        )


# ---------------------------------------------------------------------------
# Sampling tests
# ---------------------------------------------------------------------------


def test_range_sampling_within_bounds():
    env = MockEnv(num_envs=100)
    table = MockRigidObject("table", num_envs=100)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=100)
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = _make_functor(env)
    env_ids = torch.arange(100)
    functor(
        env,
        env_ids,
        anchor_uid="table",
        height_delta_range=([-0.05], [0.05]),
        store_key="table_delta",
    )

    delta = env.table_delta
    assert delta.shape == (100,)
    assert (delta >= -0.05).all()
    assert (delta <= 0.05).all()


def test_discrete_sampling_only_candidates():
    env = MockEnv(num_envs=50)
    table = MockRigidObject("table", num_envs=50)
    env.sim.add_rigid_object(table)

    functor = _make_functor(env)
    functor(
        env,
        torch.arange(50),
        anchor_uid="table",
        height_delta_candidates=[-0.05, 0.0, 0.05],
        store_key="table_delta",
    )

    candidates = torch.tensor([-0.05, 0.0, 0.05])
    for val in env.table_delta:
        assert torch.any(torch.isclose(val, candidates)), f"{val} not in {candidates}"


# ---------------------------------------------------------------------------
# Shift correctness tests
# ---------------------------------------------------------------------------


def test_anchor_and_objects_shifted_by_same_delta():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube.cfg.init_pos = [0.0, 0.0, 1.1]
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = _make_functor(env)
    functor(
        env,
        torch.arange(4),
        anchor_uid="table",
        height_delta_range=([0.05], [0.05]),
    )

    # Anchor: absolute mode -> init_pos[2] + delta = 1.0 + 0.05 = 1.05
    torch.testing.assert_close(table._pose[:, 2], torch.ones(4) * 1.05)
    # Affected: relative mode -> current_z + delta = 1.1 + 0.05 = 1.15
    torch.testing.assert_close(cube._pose[:, 2], torch.ones(4) * 1.15)


def test_xy_and_rotation_unchanged():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube._pose[:, 0] = 0.5
    cube._pose[:, 1] = -0.3
    env.sim.add_rigid_object(cube)

    original_xy = cube._pose[:, :2].clone()
    original_rot = cube._pose[:, 3:7].clone()

    functor = _make_functor(env)
    functor(
        env,
        torch.arange(4),
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
    )

    torch.testing.assert_close(cube._pose[:, :2], original_xy)
    torch.testing.assert_close(cube._pose[:, 3:7], original_rot)


def test_exclude_uids_are_not_moved():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    floor = MockRigidObject("floor", num_envs=4)
    floor._pose[:, 2] = 0.0
    env.sim.add_rigid_object(floor)

    functor = _make_functor(env)
    functor(
        env,
        torch.arange(4),
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
        exclude_uids=["floor"],
    )

    torch.testing.assert_close(floor._pose[:, 2], torch.zeros(4))
    torch.testing.assert_close(cube._pose[:, 2], torch.ones(4) * 1.2)


def test_articulation_shifted():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cabinet = MockArticulation("cabinet", num_envs=4)
    cabinet._pose[:, 2] = 1.2
    env.sim.add_articulation(cabinet)

    functor = _make_functor(env)
    functor(
        env,
        torch.arange(4),
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
    )

    torch.testing.assert_close(cabinet._pose[:, 2], torch.ones(4) * 1.3)


def test_asymmetric_delta_range():
    env = MockEnv(num_envs=100)
    table = MockRigidObject("table", num_envs=100)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    functor = _make_functor(env)
    env_ids = torch.arange(100)
    functor(
        env,
        env_ids,
        anchor_uid="table",
        height_delta_range=([-0.1], [0.05]),
        store_key="table_delta",
    )

    delta = env.table_delta
    assert delta.shape == (100,)
    assert (delta >= -0.1).all()
    assert (delta <= 0.05).all()


# ---------------------------------------------------------------------------
# Partial env_ids and clear_dynamics tests
# ---------------------------------------------------------------------------


def test_partial_env_ids():
    env = MockEnv(num_envs=4)
    table = MockRigidObject("table", num_envs=4)
    table.cfg.init_pos = [0.0, 0.0, 1.0]
    env.sim.add_rigid_object(table)

    cube = MockRigidObject("cube", num_envs=4)
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = _make_functor(env)

    # Only apply to envs 0 and 2
    partial_ids = torch.tensor([0, 2])
    functor(
        env,
        partial_ids,
        anchor_uid="table",
        height_delta_range=([0.1], [0.1]),
    )

    # Envs 0 and 2 should be shifted
    torch.testing.assert_close(table._pose[0, 2], torch.tensor(1.1))
    torch.testing.assert_close(table._pose[2, 2], torch.tensor(1.1))
    torch.testing.assert_close(cube._pose[0, 2], torch.tensor(1.2))
    torch.testing.assert_close(cube._pose[2, 2], torch.tensor(1.2))

    # Envs 1 and 3 should remain unchanged
    torch.testing.assert_close(table._pose[1, 2], torch.tensor(0.0))
    torch.testing.assert_close(table._pose[3, 2], torch.tensor(0.0))
    torch.testing.assert_close(cube._pose[1, 2], torch.tensor(1.1))
    torch.testing.assert_close(cube._pose[3, 2], torch.tensor(1.1))

    # clear_dynamics should have been called only for the targeted env_ids
    assert table._cleared
    assert table._cleared_env_ids is not None
    torch.testing.assert_close(
        table._cleared_env_ids, partial_ids
    ), "clear_dynamics should receive the targeted env_ids"

    assert cube._cleared
    assert cube._cleared_env_ids is not None
    torch.testing.assert_close(
        cube._cleared_env_ids, partial_ids
    ), "clear_dynamics should receive the targeted env_ids"


# ---------------------------------------------------------------------------
# RigidObjectGroup anchor tests
# ---------------------------------------------------------------------------


def test_rigid_object_group_anchor_absolute_warning(caplog):
    """When the anchor is a RigidObjectGroup and absolute=True, a warning is
    emitted and the relative shift is applied.
    """
    env = MockEnv(num_envs=4)
    group = MockRigidObjectGroup("group", num_envs=4, num_objects=2)
    group._pose[:, :, 2, 3] = 1.0  # set initial Z to 1.0
    env.sim.add_rigid_object_group(group)

    cube = MockRigidObject("cube", num_envs=4)
    cube._pose[:, 2] = 1.1
    env.sim.add_rigid_object(cube)

    functor = _make_functor(env)
    functor(
        env,
        torch.arange(4),
        anchor_uid="group",
        height_delta_range=([0.1], [0.1]),
    )

    # Verify that the warning log message was emitted
    assert "absolute=True is not supported for RigidObjectGroup" in caplog.text

    # Group: relative shift applied -> 1.0 + 0.1 = 1.1
    torch.testing.assert_close(group._pose[:, :, 2, 3], torch.ones(4, 2) * 1.1)
    # Cube: relative shift -> 1.1 + 0.1 = 1.2
    torch.testing.assert_close(cube._pose[:, 2], torch.ones(4) * 1.2)
