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

"""Independent polygon-distance examples and batched tensor contracts."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.geometry import signed_distance_to_polygon


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polygon_distances_preserve_noncontiguous_batch_axes_and_chunk_boundaries(
    dtype,
) -> None:
    polygon = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=dtype)
    points = torch.tensor(
        [[[-2, 0], [0, 0], [1, 0]], [[0.5, 1.5], [2, 2], [-1, -1]]], dtype=dtype
    ).transpose(0, 1)
    expected = torch.tensor([[1, -1, 0], [0.5, 2**0.5, 0]], dtype=dtype).transpose(0, 1)
    assert not points.is_contiguous()
    for chunk in [1, 4, 16384]:
        torch.testing.assert_close(
            signed_distance_to_polygon(points, polygon, max_points_per_chunk=chunk),
            expected,
        )


def test_polygon_distance_uses_even_odd_interior_for_concave_outline() -> None:
    polygon = torch.tensor([[0.0, 0.0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    points = torch.tensor([[0.5, 0.5], [1.5, 1.5], [1.5, 0.5], [-1, 1]])
    expected = torch.tensor([-0.5, 0.5, -0.5, 1.0])
    torch.testing.assert_close(signed_distance_to_polygon(points, polygon), expected)
    torch.testing.assert_close(
        signed_distance_to_polygon(points, polygon.flip(0)), expected
    )


def test_polygon_distance_supports_empty_queries_and_point_gradients() -> None:
    polygon = torch.tensor([[-1.0, -1], [1, -1], [1, 1], [-1, 1]])
    assert signed_distance_to_polygon(torch.empty(0, 4, 2), polygon).shape == (0, 4)
    point = torch.tensor([[3.0, 0.0]], requires_grad=True)
    result = signed_distance_to_polygon(point, polygon)
    torch.testing.assert_close(result, torch.tensor([2.0]))
    result.sum().backward()
    torch.testing.assert_close(point.grad, torch.tensor([[1.0, 0.0]]))


@pytest.mark.parametrize("case", ["point_shape", "vertex_count", "dtype", "chunk"])
def test_polygon_distance_rejects_invalid_tensor_contracts(case: str) -> None:
    points = torch.zeros(2, 2)
    polygon = torch.tensor([[0.0, 0.0], [1, 0], [0, 1]])
    if case == "point_shape":
        points = torch.zeros(2, 3)
    if case == "vertex_count":
        polygon = polygon[:2]
    if case == "dtype":
        polygon = polygon.double()
    with pytest.raises(ValueError):
        signed_distance_to_polygon(
            points, polygon, max_points_per_chunk=0 if case == "chunk" else 16384
        )
