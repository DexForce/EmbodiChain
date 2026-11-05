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

"""Analytic frame, symmetry and rigid-grasp contracts."""

from __future__ import annotations

import math

import pytest
import torch

from embodichain.compute.geometry import nearest_half_turn_pose, rotate_rigid_grasp
from embodichain.utils.math import pose_inv


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_half_turn_selection_preserves_translation_and_symmetric_ties(
    dtype: torch.dtype,
) -> None:
    target = torch.eye(4, dtype=dtype).repeat(3, 1, 1)
    target[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    reference = target.clone()
    reference[1, :2, :2] *= -1
    reference[2, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
    original = target.clone()

    result = nearest_half_turn_pose(target, reference)

    expected = target.clone()
    expected[1:, :2, :2] *= -1
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(target, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rigid_rotation_uses_link_frame_and_retains_grasp_transform(
    dtype: torch.dtype,
) -> None:
    link = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    link[:, :3, 3] = torch.tensor([[2.0, 3.0, 0.0], [-2.0, 1.0, 0.0]], dtype=dtype)
    link[1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=dtype)
    local_grasp = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    local_grasp[:, 0, 3] = 2.0
    grasp = link @ local_grasp
    original_link, original_grasp = link.clone(), grasp.clone()
    angles = torch.tensor([[0.0, math.pi / 2], [0.0, -math.pi / 2]], dtype=dtype)

    rotated, carried = rotate_rigid_grasp(
        link, grasp, torch.tensor([0.0, 0.0, 4.0]), [1.0, 0.0, 0.0], angles
    )

    # A grasp at local x=2 swings around local x=1. The second link's
    # quarter-turn world frame maps its local (1,-1) point to world (1,1).
    expected_final = torch.tensor([[3.0, 4.0, 0.0], [-1.0, 2.0, 0.0]], dtype=dtype)
    torch.testing.assert_close(carried[:, -1, :3, 3], expected_final)
    torch.testing.assert_close(carried[:, 0], grasp)
    torch.testing.assert_close(
        pose_inv(rotated) @ carried, local_grasp[:, None].expand(-1, 2, -1, -1)
    )
    torch.testing.assert_close(link, original_link)
    torch.testing.assert_close(grasp, original_grasp)


def test_shared_angle_samples_broadcast_over_rigid_grasp_batch() -> None:
    poses = torch.eye(4).repeat(3, 1, 1)
    angles = torch.tensor([0.0, math.pi])

    rotated, carried = rotate_rigid_grasp(
        poses, poses, torch.tensor([0.0, 0.0, 1.0]), [0.0, 0.0, 0.0], angles
    )

    assert rotated.shape == (3, 2, 4, 4)
    torch.testing.assert_close(rotated, carried)
    torch.testing.assert_close(rotated[:, 0], poses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_support_mesh_envelope_preserves_world_axes_and_explicit_inset(dtype):
    from embodichain.compute.geometry.rigid import support_mesh_envelope

    corners = torch.tensor(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=dtype
    ).repeat(2, 1, 1)
    object_pose = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    object_pose[:, 2, 3] = torch.tensor([5, 7], dtype=dtype)
    reference_pose = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
    reference_pose[:, :3, :3] = torch.tensor(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=dtype
    )
    reference_pose[:, :3, 3] = torch.tensor([[10, 20, 1], [30, 40, 2]], dtype=dtype)
    bottom, top, lower, upper = support_mesh_envelope(
        corners,
        corners * corners.new_tensor([2, 3, 1]),
        object_pose,
        reference_pose,
        inset_xy=0.002,
    )
    torch.testing.assert_close(bottom, corners.new_tensor([4, 6]))
    torch.testing.assert_close(top, corners.new_tensor([2, 3]))
    torch.testing.assert_close(
        lower, corners.new_tensor([[7.002, 18.002], [27.002, 38.002]])
    )
    torch.testing.assert_close(
        upper, corners.new_tensor([[12.998, 21.998], [32.998, 41.998]])
    )
