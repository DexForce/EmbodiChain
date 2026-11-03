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

import math

import pytest
import torch

from embodichain.compute.geometry.mesh_bounds import (
    aabb_sphere_cover,
    transform_vertex_bounds,
)


def test_bounds_project_vertices_not_the_larger_local_aabb():
    points = torch.tensor([[[-1.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]])
    c = math.sqrt(0.5)
    pose = torch.tensor(
        [[[c, -c, 0, 1.0], [c, c, 0, 2.0], [0, 0, 1.0, 3.0], [0, 0, 0, 1.0]]],
        dtype=torch.float64,
    )
    lower, upper = transform_vertex_bounds(points, pose)
    torch.testing.assert_close(
        lower, torch.tensor([[1 - 2 * c, 2 - c, 3]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        upper, torch.tensor([[1 + c, 2 + 2 * c, 4]], dtype=torch.float64)
    )
    assert lower.dtype == pose.dtype


def test_sphere_cover_preserves_grid_order_and_circumscribed_cell_radius():
    spheres = aabb_sphere_cover(torch.tensor([[0.0, 0, 0], [3.0, 6.0, 9.0]]))
    assert spheres.shape == (27, 4)
    torch.testing.assert_close(
        spheres[:3, :3],
        torch.tensor([[0.5, 1.0, 1.5], [0.5, 1.0, 4.5], [0.5, 1.0, 7.5]]),
    )
    torch.testing.assert_close(spheres[-1, :3], torch.tensor([2.5, 5.0, 7.5]))
    torch.testing.assert_close(spheres[:, 3], torch.full((27,), math.sqrt(14) / 2))


def test_mesh_bounds_reject_unavailable_geometry():
    with pytest.raises(ValueError, match="unavailable"):
        transform_vertex_bounds(torch.empty(1, 0, 3), torch.eye(4)[None])
    with pytest.raises(ValueError, match="finite"):
        aabb_sphere_cover(torch.tensor([[float("nan"), 0.0, 0.0]]))
