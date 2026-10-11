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
from copy import deepcopy
from types import SimpleNamespace
import torch
from embodichain.utils.math import quat_from_matrix, matrix_from_quat


class Rigid:
    body_type = "dynamic"

    def __init__(self):
        self.value = torch.tensor(
            [[0.2, 0, 0.1, 0, 0, 0, 1, 0.3, 0, 0, 0, 0, 0]], dtype=torch.float32
        )
        self.writes = 0

    @property
    def body_state(self):
        return self.value

    def get_local_pose(self, *, to_matrix=True):
        if not to_matrix:
            return self.value[:, :7]
        matrix = torch.eye(4).unsqueeze(0)
        matrix[:, :3, :3] = matrix_from_quat(self.value[:, 3:7])
        matrix[:, :3, 3] = self.value[:, :3]
        return matrix

    def clear_dynamics(self, *, env_ids):
        self.writes += 1
        self.value[:, 7:] = 0

    def set_local_pose(self, value, *, env_ids):
        self.writes += 1
        if value.shape == (1, 4, 4):
            self.value[:, :3] = value[:, :3, 3]
            self.value[:, 3:7] = quat_from_matrix(value[:, :3, :3])
        else:
            self.value[:, :7] = value

    def set_velocity(self, *, lin_vel, ang_vel, env_ids):
        self.value[:, 7:10] = lin_vel
        self.value[:, 10:] = ang_vel


class Articulation:
    joint_names = ("hinge",)

    def __init__(self):
        self.state = {
            "root_pose": torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]]),
            "root_lin_vel": torch.zeros(1, 3),
            "root_ang_vel": torch.zeros(1, 3),
            "qpos": torch.tensor([[0.2]]),
            "qvel": torch.tensor([[0.1]]),
            "target_qpos": torch.tensor([[0.3]]),
            "target_qvel": torch.tensor([[0.4]]),
            "qf": torch.tensor([[0.5]]),
        }
        self.body_data = SimpleNamespace(fetch_state=lambda: self.state)

    def get_qpos(self, *, target=False):
        return self.state["target_qpos" if target else "qpos"]

    def get_qvel(self, *, target=False):
        return self.state["target_qvel" if target else "qvel"]

    def get_qf(self):
        return self.state["qf"]

    def set_state(self, *, env_ids, clear_dynamics, root_velocity, **kwargs):
        self.state.update(deepcopy(kwargs))
        self.state["root_lin_vel"] = root_velocity[:, :3].clone()
        self.state["root_ang_vel"] = root_velocity[:, 3:].clone()


class Sim:
    num_envs = 1
    device = torch.device("cpu")
    physics_backend = "default"

    def __init__(self):
        self.rigid = {"cube": Rigid(), "other": Rigid()}
        self.articulations = {"drawer": Articulation()}
        self.steps = 0
        self.syncs = 0

    def get_rigid_object_uid_list(self):
        return list(self.rigid)

    def get_robot_uid_list(self):
        return []

    def get_articulation_uid_list(self):
        return list(self.articulations)

    def get_rigid_object(self, uid):
        return self.rigid.get(uid)

    def get_robot(self, uid):
        return None

    def get_articulation(self, uid):
        return self.articulations.get(uid)

    def sync_render_state(self):
        self.syncs += 1

    def step(self, count):
        self.steps += count
