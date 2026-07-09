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
