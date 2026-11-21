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
from types import SimpleNamespace
import torch
from embodichain.utils.math import quat_from_matrix, matrix_from_quat


class Rigid:
    body_type = "dynamic"

    def __init__(self, num_envs=1):
        self.value = torch.tensor(
            [[0.2, 0, 0.1, 0, 0, 0, 1, 0.3, 0, 0, 0, 0, 0]], dtype=torch.float32
        )
        self.value = self.value.expand(num_envs, -1).clone()
        self.writes = 0

    @property
    def body_state(self):
        return self.value

    def get_local_pose(self, *, to_matrix=True):
        if not to_matrix:
            return self.value[:, :7]
        matrix = torch.eye(4).repeat(len(self.value), 1, 1)
        matrix[:, :3, :3] = matrix_from_quat(self.value[:, 3:7])
        matrix[:, :3, 3] = self.value[:, :3]
        return matrix

    def clear_dynamics(self, *, env_ids):
        self.writes += 1
        self.value[env_ids, 7:] = 0

    def set_local_pose(self, value, *, env_ids):
        self.writes += 1
        if value.ndim == 3:
            self.value[env_ids, :3] = value[:, :3, 3]
            self.value[env_ids, 3:7] = quat_from_matrix(value[:, :3, :3])
        else:
            self.value[env_ids, :7] = value

    def set_velocity(self, *, lin_vel, ang_vel, env_ids):
        self.value[env_ids, 7:10] = lin_vel
        self.value[env_ids, 10:] = ang_vel


class Visual:
    """Render-only facade deliberately lacking rigid-body state and dynamics."""

    def __init__(self, num_envs=1):
        self.value = torch.eye(4).repeat(num_envs, 1, 1)
        self.writes = 0

    def get_local_pose(self, *, to_matrix=True):
        if to_matrix:
            return self.value
        return torch.cat(
            (self.value[:, :3, 3], quat_from_matrix(self.value[:, :3, :3])), -1
        )

    def set_local_pose(self, value, *, env_ids):
        self.writes += 1
        self.value[env_ids] = value


class Articulation:
    joint_names = ("hinge",)

    def __init__(self, num_envs=1):
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
        self.state = {
            key: value.expand(num_envs, -1).clone() for key, value in self.state.items()
        }
        self.body_data = SimpleNamespace(
            fetch_state=lambda: self.state,
            qpos_limits=torch.tensor([[[-1.0, 1.0]]]).expand(num_envs, -1, -1).clone(),
        )

    def get_qpos(self, *, target=False):
        return self.state["target_qpos" if target else "qpos"]

    def get_qvel(self, *, target=False):
        return self.state["target_qvel" if target else "qvel"]

    def get_qf(self):
        return self.state["qf"]

    def set_state(self, *, env_ids, clear_dynamics, root_velocity, **kwargs):
        for key, value in kwargs.items():
            self.state[key][env_ids] = value
        self.state["root_lin_vel"][env_ids] = root_velocity[:, :3]
        self.state["root_ang_vel"][env_ids] = root_velocity[:, 3:]


class Sim:
    num_envs = 1
    device = torch.device("cpu")
    physics_backend = "default"

    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.rigid = {"cube": Rigid(num_envs), "other": Rigid(num_envs)}
        self.articulations = {"drawer": Articulation(num_envs)}
        self.visuals = {}
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

    def get_visual_object_uid_list(self):
        return list(self.visuals)

    def get_visual_object(self, uid):
        return self.visuals.get(uid)

    def sync_render_state(self):
        self.syncs += 1

    def step(self, count):
        self.steps += count
