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

from unittest.mock import Mock

import torch

from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
)


def test_runtime_pose_cost_is_applied_before_top_k_selection() -> None:
    generator = GraspGenerator.__new__(GraspGenerator)
    generator.device = torch.device("cpu")
    generator.vertices = torch.tensor(
        [[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]],
        dtype=torch.float32,
    )
    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.2, 0.0]],
        dtype=torch.float32,
    )
    half_width = torch.tensor([0.02, 0.0, 0.0])
    generator._hit_point_pairs = torch.stack(
        [centers - half_width, centers + half_width],
        dim=1,
    )
    generator.cfg = GraspGeneratorCfg(
        max_deviation_angle=torch.pi / 6,
        n_deviated_approach_directions=1,
        n_top_grasps=1,
    )
    generator._collision_checker = Mock()

    def collision_query(_object_pose, grasp_poses, _open_lengths, **_kwargs):
        count = grasp_poses.shape[0]
        return torch.zeros(count, dtype=torch.bool), torch.zeros(count)

    generator._collision_checker.query.side_effect = collision_query

    def prefer_highest_y(grasp_poses, costs):
        adjusted = costs.clone()
        adjusted[torch.argmax(grasp_poses[:, 1, 3])] = -1.0
        return adjusted

    success, poses, _, costs = generator.get_valid_grasp_poses(
        torch.eye(4),
        torch.tensor([0.0, 0.0, -1.0]),
        pose_cost_fn=prefer_highest_y,
    )

    assert success
    assert poses.shape == (1, 4, 4)
    assert torch.isclose(poses[0, 1, 3], torch.tensor(0.2))
    assert costs.tolist() == [-1.0]
