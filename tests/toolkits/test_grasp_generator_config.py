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

"""Tests for GraspGenerator configuration defaults."""

from __future__ import annotations

import math
from unittest.mock import Mock

import torch

from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
)


def test_default_grasp_generator_uses_only_requested_approach_direction() -> None:
    assert GraspGeneratorCfg().n_deviated_approach_directions == 1


def test_grasp_generator_filters_closing_axes_by_approach_alignment() -> None:
    generator = object.__new__(GraspGenerator)
    generator.device = torch.device("cpu")
    generator.vertices = torch.tensor(
        [[-0.2, -0.1, -0.1], [0.2, 0.1, 0.1]], dtype=torch.float32
    )
    generator.cfg = GraspGeneratorCfg(n_deviated_approach_directions=1)
    horizontal_pair = torch.tensor(
        [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=torch.float32
    )
    tilt_angle = math.radians(20.0)
    tilted_axis = torch.tensor(
        [math.cos(tilt_angle), 0.0, math.sin(tilt_angle)], dtype=torch.float32
    )
    tilted_pair = torch.stack(
        [
            torch.tensor([0.1, 0.0, 0.0]),
            torch.tensor([0.1, 0.0, 0.0]) + tilted_axis * 0.1,
        ]
    )
    generator._hit_point_pairs = torch.stack([horizontal_pair, tilted_pair])
    generator._collision_checker = Mock()
    generator._collision_checker.query.side_effect = (
        lambda object_pose, poses, open_lengths, **kwargs: (
            torch.zeros(poses.shape[0], dtype=torch.bool),
            torch.zeros(poses.shape[0]),
        )
    )

    success, grasp_poses, _, _ = generator.get_valid_grasp_poses(
        torch.eye(4),
        torch.tensor([0.0, 0.0, -1.0]),
        max_approach_alignment_angle=math.radians(5.0),
    )

    assert success is True
    assert grasp_poses.shape[0] == 1
    assert torch.allclose(grasp_poses[0, :3, 2], torch.tensor([0.0, 0.0, -1.0]))
