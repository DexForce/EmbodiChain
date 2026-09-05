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

from embodichain.gen_sim.action_engine.runtime.coordinated_safety import (
    _canonicalize_parallel_jaw_poses,
    _minimum_interarm_capsule_clearance,
    _rank_non_crossing_grasp_pairs,
    _segments_intersect_2d,
    _trajectory_safety_report,
)


def _pose(x: float, y: float, z: float = 0.75) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor([x, y, z])
    return pose


def test_parallel_jaw_pose_chooses_equivalent_half_turn_nearest_live_eef() -> None:
    pose = torch.eye(4).unsqueeze(0)
    pose[0, 0, 0] = -1.0
    pose[0, 1, 1] = -1.0

    result = _canonicalize_parallel_jaw_poses(pose, torch.eye(4))

    torch.testing.assert_close(result.poses[0], torch.eye(4))
    assert result.flipped.tolist() == [True]
    assert result.selected_rotation_radians.tolist() == [0.0]
    assert result.alternative_rotation_radians.tolist() == pytest.approx([torch.pi])


def test_pair_ranking_rejects_reversal_overlap_and_xy_crossing() -> None:
    left = torch.stack(
        (
            _pose(0.0, -0.20),
            _pose(1.0, 0.20),
            _pose(0.0, 0.18),
        )
    )
    right = torch.stack(
        (
            _pose(0.0, 0.20),
            _pose(-1.0, -0.20),
            _pose(0.0, 0.19),
        )
    )
    result = _rank_non_crossing_grasp_pairs(
        left,
        right,
        left_costs=torch.tensor([0.0, 10.0, 10.0]),
        right_costs=torch.tensor([0.0, 10.0, 10.0]),
        left_rotation_costs=torch.zeros(3),
        right_rotation_costs=torch.zeros(3),
        left_base=_pose(-1.0, -0.30),
        right_base=_pose(1.0, 0.30),
        left_to_right_direction=torch.tensor([0.0, 1.0, 0.0]),
        minimum_separation=0.08,
        minimum_lateral_gap=0.05,
    )

    assert result.ranked_pairs[0] == (0, 0)
    assert (1, 1) not in result.ranked_pairs  # XY paths intersect.
    assert (2, 2) not in result.ranked_pairs  # Distinct poses are too close.
    assert result.rejection_counts["reversed"] > 0
    assert result.rejection_counts["too_close"] > 0
    assert result.rejection_counts["path_crossing"] > 0


def test_xy_route_intersection_includes_touching_and_collinear_overlap() -> None:
    assert _segments_intersect_2d(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.5, 0.0, 0.0]),
        torch.tensor([1.5, 0.0, 0.0]),
    )
    assert _segments_intersect_2d(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([2.0, 1.0, 0.0]),
    )


def test_capsule_clearance_covers_every_left_right_link_segment() -> None:
    left = torch.tensor(
        [[[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    crossing = torch.tensor(
        [[[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    clear = crossing.clone()
    clear[..., 2] = 1.0

    crossing_clearance = _minimum_interarm_capsule_clearance(
        left,
        crossing,
        capsule_radius=0.05,
    )
    safe_clearance = _minimum_interarm_capsule_clearance(
        left,
        clear,
        capsule_radius=0.05,
    )

    torch.testing.assert_close(crossing_clearance, torch.tensor([[-0.10]]))
    torch.testing.assert_close(safe_clearance, torch.tensor([[0.90]]))


def test_trajectory_safety_rejects_orientation_jump_order_and_capsule_collision() -> (
    None
):
    left_qpos = torch.zeros(1, 3, 2)
    right_qpos = torch.zeros(1, 3, 2)
    left_qpos[:, 1, 0] = 0.40
    left_eef = torch.eye(4).repeat(1, 3, 1, 1)
    right_eef = torch.eye(4).repeat(1, 3, 1, 1)
    desired_left = left_eef.clone()
    desired_right = right_eef.clone()
    left_eef[:, 1, 0, 0] = -1.0
    left_eef[:, 1, 1, 1] = -1.0
    left_eef[:, :, 1, 3] = torch.tensor([-0.2, 0.2, -0.2])
    right_eef[:, :, 1, 3] = torch.tensor([0.2, -0.2, 0.2])
    left_links = torch.tensor(
        [
            [
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        ]
    )
    right_links = torch.tensor(
        [
            [
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        ]
    )

    report = _trajectory_safety_report(
        left_qpos=left_qpos,
        right_qpos=right_qpos,
        left_eef=left_eef,
        right_eef=right_eef,
        desired_left_eef=desired_left,
        desired_right_eef=desired_right,
        left_link_points=left_links,
        right_link_points=right_links,
        left_to_right_direction=torch.tensor([0.0, 1.0, 0.0]),
        maximum_joint_step=0.25,
        maximum_orientation_error=0.20,
        minimum_lateral_gap=0.05,
        capsule_radius=0.05,
        minimum_capsule_clearance=0.0,
    )

    assert report.success.tolist() == [False]
    assert report.failed_checks == {
        "joint_step": [True],
        "orientation": [True],
        "lateral_order": [True],
        "capsule_collision": [True],
    }
