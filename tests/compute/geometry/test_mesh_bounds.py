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

import numpy as np
import pytest
import torch
import trimesh

from embodichain.compute.geometry.mesh_bounds import (
    aabb_sphere_cover,
    input_mesh_bounds,
    input_mesh_radius,
    iter_clearance_translations,
    maximum_halfspace_violation,
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


def test_input_mesh_group_bounds_and_radius_keep_all_components():
    first = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    second = first.copy()
    second.apply_translation([4.0, 0.0, 0.0])
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    low, high = input_mesh_bounds([first, second], pose)
    np.testing.assert_array_equal(low, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(high, [6.0, 3.0, 4.0])
    assert input_mesh_radius([first, second], pose) == pytest.approx(math.sqrt(61))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_halfspace_measurement_includes_vertices_after_the_chunk_boundary(dtype):
    # The only outside point follows the 256-row multiplication chunk.
    points = np.zeros((257, 3), dtype=dtype)
    points[-1] = [4.0, -3.0, 0.0]
    normals = np.array([[1.0, 0, 0], [0, -1.0, 0]], dtype=dtype)
    offsets = np.array([1.0, 1.0], dtype=dtype)
    assert maximum_halfspace_violation(points, normals, offsets) == 3.0
    points[-1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        maximum_halfspace_violation(points, normals, offsets)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clearance_candidates_preserve_order_and_round_outward(dtype):
    reference = torch.zeros((), dtype=dtype)
    envelope = (np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0]))
    candidates = list(
        iter_clearance_translations(
            np.zeros(3),
            np.ones(3),
            [(envelope, 0.1)],
            [],
            horizontal=0,
            reference=reference,
        )
    )
    assert [label for _, label, _ in candidates] == ["+x", "-x", "+z"]
    outward = torch.nextafter(reference.new_tensor(2.1), reference.new_tensor(math.inf))
    expected = (
        torch.stack(
            [
                torch.tensor([[1, 0, 0]]),
                torch.tensor([[-1, 0, 0]]),
                torch.tensor([[0, 0, 1]]),
            ]
        ).to(dtype)
        * outward
    )
    torch.testing.assert_close(
        torch.stack([delta for delta, _, _ in candidates]), expected
    )
    assert all(value.dtype == dtype and value == outward for _, _, value in candidates)

    # An upward-facing plane disqualifies horizontal directions, retaining +Z.
    plane_only = list(
        iter_clearance_translations(
            np.zeros(3),
            np.ones(3),
            [],
            [(np.array([0.0, 0.0, 1.0]), 0.0, 0.1)],
            horizontal=0,
            reference=reference,
        )
    )
    assert [label for _, label, _ in plane_only] == ["+z"]
