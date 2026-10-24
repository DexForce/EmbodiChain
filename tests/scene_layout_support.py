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

"""Authored scene and small simulation ports shared by expansion tests."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from embodichain.lab.sim.scene.expansion import SceneLayoutInput


def make_source() -> SceneLayoutInput:
    meshes = {
        "workbench": trimesh.creation.box([1.5, 1.2, 0.1]),
        "tray": trimesh.creation.box([0.3, 0.3, 0.02]),
        "mug": trimesh.creation.box([0.04, 0.04, 0.08]),
        "z_target": trimesh.creation.box([0.1, 0.1, 0.1]),
    }
    poses = {uid: np.eye(4) for uid in meshes}
    poses["tray"][:3, 3] = [-0.3, 0, 0.061]
    poses["mug"][:3, 3] = [-0.3, 0, 0.111]
    poses["z_target"][:3, 3] = [0.4, 0.3, 0.101]
    return SceneLayoutInput(
        "authored-test",
        "workbench",
        poses,
        {uid: mesh.vertices for uid, mesh in meshes.items()},
        {uid: mesh.faces for uid, mesh in meshes.items()},
        {"tray": "workbench", "mug": "tray", "z_target": "workbench"},
    )


def make_policy(**kwargs: object) -> dict:
    return {
        "group_targets": ["tray"],
        "batch_size": 7,
        "intra_group": {"targets": {"mug": {}}},
        **kwargs,
    }


class RigidPort:
    def __init__(self, uid: str, source: SceneLayoutInput, count: int) -> None:
        self.uid = uid
        self.source = source
        self.poses = (
            torch.tensor(np.array(source.poses[uid])).expand(count, 4, 4).clone()
        )
        self.velocity = torch.ones(count, 3)
        self.writes = []

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix
        return self.poses.clone()

    def get_vertices(self, env_ids: list[int], scale: bool = False) -> torch.Tensor:
        assert scale
        return torch.tensor(np.array(self.source.vertices[self.uid])).expand(
            len(env_ids), -1, -1
        )

    def get_triangles(self, env_ids: list[int]) -> torch.Tensor:
        return torch.tensor(np.array(self.source.faces[self.uid])).expand(
            len(env_ids), -1, -1
        )

    def get_body_scale(self) -> torch.Tensor:
        return torch.ones(len(self.poses), 3)

    def clear_dynamics(self, env_ids: list[int]) -> None:
        self.velocity[env_ids] = 0

    def set_local_pose(self, poses: torch.Tensor, env_ids: list[int]) -> None:
        self.writes.append(tuple(env_ids))
        self.poses[env_ids] = poses.to(self.poses.dtype)


class SimPort:
    def __init__(self, source: SceneLayoutInput, count: int = 4) -> None:
        self.num_envs = count
        self.device = torch.device("cpu")
        self.asset_uids = list(source.poses)
        self.entities = {uid: RigidPort(uid, source, count) for uid in source.poses}
        self.time = 0
        self.syncs = 0

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.entities)

    def get_rigid_object(self, uid: str) -> RigidPort:
        return self.entities[uid]

    def sync_render_state(self) -> None:
        self.syncs += 1
