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

"""Shared placement geometry retains grasp and support-clearance semantics."""

from __future__ import annotations

import pytest
import torch

from embodichain_benchmarks.motion_generation.scenarios._placement_geometry import (
    build_flat_object_grasp_pose,
    build_pan_handle_grasp_pose,
    build_placing_object_target_pose,
    build_support_object_target_pose,
    get_pan_basin_vertices,
)

pytestmark = pytest.mark.no_sim

_CPU = torch.device("cpu")


def test_flat_grasp_uses_transformed_center_and_world_top() -> None:
    vertices = torch.tensor([[-0.1, -0.05, 0.0], [0.1, 0.05, 0.02]])
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    grasp = build_flat_object_grasp_pose(
        pose,
        vertices,
        vertices.min(0).values,
        vertices.max(0).values,
        _CPU,
        z_clearance=0.018,
    )
    torch.testing.assert_close(grasp[:3, 3], torch.tensor([1.0, 2.0, 3.038]))
    torch.testing.assert_close(pose[:3, 3], torch.tensor([1.0, 2.0, 3.0]))


def test_handle_grasp_points_down_and_moves_toward_handle_root() -> None:
    pan = torch.tensor(
        [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.02], [0.0, 0.3, 0.06], [0.0, 0.5, 0.08]]
    )
    grasp = build_pan_handle_grasp_pose(torch.eye(4), pan, _CPU)
    torch.testing.assert_close(grasp[:3, 3], torch.tensor([0.0, 0.365, 0.086]))
    torch.testing.assert_close(grasp[:3, 2], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(grasp[:3, :3].T @ grasp[:3, :3], torch.eye(3))
    assert torch.linalg.det(grasp[:3, :3]).item() == pytest.approx(1.0)


def test_placing_target_clears_basin_and_ignores_tall_handle() -> None:
    bread = torch.tensor([[-0.1, -0.05, 0.0], [0.1, 0.05, 0.02]])
    pan = torch.tensor([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.02], [0.0, 0.5, 0.08]])
    initial = torch.eye(4)
    support = build_support_object_target_pose(initial, _CPU)
    placing = build_placing_object_target_pose(initial, bread, pan, support, _CPU)
    # The basin top is 0.731 m; the bread bottom retains 6 mm clearance.
    torch.testing.assert_close(support[:3, 3], torch.tensor([-0.06, 0.0, 0.711]))
    torch.testing.assert_close(placing[:3, 3], torch.tensor([-0.12, -0.16, 0.737]))
    torch.testing.assert_close(initial, torch.eye(4))


def test_empty_basin_selection_preserves_full_mesh() -> None:
    pan = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.2, 0.2]])
    torch.testing.assert_close(get_pan_basin_vertices(pan), pan)
