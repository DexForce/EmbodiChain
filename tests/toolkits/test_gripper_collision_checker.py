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

from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
    GripperCollisionChecker,
)


def test_query_chunks_and_keeps_ground_collision():
    call_sizes: list[int] = []

    checker = GripperCollisionChecker.__new__(GripperCollisionChecker)
    checker.cfg = GripperCollisionCfg(query_batch_size=2)
    checker.device = torch.device("cpu")

    def fake_query_batch_points(batch_points, collision_threshold=0.0, is_visual=False):
        call_sizes.append(batch_points.shape[0])
        is_colliding = torch.zeros(
            batch_points.shape[0], batch_points.shape[1], dtype=torch.bool
        )
        distances = torch.ones(batch_points.shape[0], batch_points.shape[1])
        return is_colliding, distances

    def fake_get_gripper_pc(grasp_poses, open_lengths):
        batch_size = grasp_poses.shape[0]
        point_cloud = torch.zeros(batch_size, 4, 3)
        point_cloud[:, :, 2] = grasp_poses[:, 0, 3].unsqueeze(-1)
        return point_cloud

    checker._checker = SimpleNamespace(query_batch_points=fake_query_batch_points)
    checker._get_gripper_pc = fake_get_gripper_pc
    checker.get_ground_height = lambda obj_pose: 0.5

    obj_pose = torch.eye(4)
    grasp_poses = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)
    grasp_poses[:, 0, 3] = torch.tensor([0.0, 0.4, 0.6, 0.8, 1.0])
    open_lengths = torch.full((5,), 0.02)

    is_colliding, distances = checker.query(
        obj_pose=obj_pose,
        grasp_poses=grasp_poses,
        open_lengths=open_lengths,
        collision_threshold=0.0,
        is_filter_ground_collision=True,
        is_visual=False,
    )

    assert call_sizes == [2, 2, 1]
    assert is_colliding.tolist() == [True, True, False, False, False]
    assert torch.allclose(
        distances, torch.tensor([-0.5, -0.1, 0.1, 0.3, 0.5], dtype=torch.float32)
    )
