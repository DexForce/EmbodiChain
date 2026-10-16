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

"""Batched geometry contracts on CPU and explicitly enabled CUDA devices."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from embodichain.compute.geometry import (
    aabb_containment_violation,
    aabb_overlap_depth,
    group_height_offset,
    transform_aabbs,
    triangle_geometry,
)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return (
        torch.device("cuda", torch.cuda.current_device())
        if request.param == "cuda"
        else torch.device("cpu")
    )


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def test_overlap_broadcasts_independent_batches_and_pairs(device, dtype):
    boxes = torch.tensor(
        [
            [[[0.0, 0.0], [2.0, 2.0]], [[1.0, 1.0], [3.0, 3.0]]],
            [[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]],
        ],
        device=device,
        dtype=dtype,
    )
    actual = aabb_overlap_depth(boxes[:, :, None], boxes[:, None, :])
    expected = torch.tensor(
        [
            [[[2.0, 2.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]],
            [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]],
        ],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.device == device and actual.dtype == dtype


def test_overlap_reports_intersection_length_for_containment_and_touching(
    device, dtype
):
    outer = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], device=device, dtype=dtype)
    inside = torch.tensor([[-0.5, -0.25], [0.5, 0.25]], device=device, dtype=dtype)
    torch.testing.assert_close(aabb_overlap_depth(outer, inside), inside[1] - inside[0])
    touch = torch.tensor([[2.0, -1.0], [3.0, 1.0]], device=device, dtype=dtype)
    torch.testing.assert_close(
        aabb_overlap_depth(outer, touch),
        torch.tensor([0.0, 2.0], device=device, dtype=dtype),
    )


def test_containment_measures_larger_violation_for_each_axis(device, dtype):
    bounds = torch.tensor(
        [
            [[-0.2, 0.1], [1.4, 0.7]],
            [[0.1, -0.3], [0.9, 1.2]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        device=device,
        dtype=dtype,
    )
    region = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype)
    expected = torch.tensor(
        [[0.4, 0.0], [0.0, 0.3], [0.0, 0.0]], device=device, dtype=dtype
    )
    torch.testing.assert_close(aabb_containment_violation(bounds, region), expected)


def test_pair_kernels_mark_invalid_rows_without_hiding_them_as_nonoverlap(
    device, dtype
):
    boxes = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2.0, 0.0], [1.0, 1.0]],
            [[float("nan"), 0.0], [1.0, 1.0]],
        ],
        device=device,
        dtype=dtype,
    )
    for kernel in (aabb_overlap_depth, aabb_containment_violation):
        result = kernel(boxes, boxes[0])
        assert torch.isfinite(result[0]).all()
        assert torch.isnan(result[1:]).all()


def test_group_height_broadcasts_batches_ignoring_invalid_padding(device, dtype):
    # Two groups, each padded to three objects; the second has no valid members.
    bounds = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.4], [1.0, 1.0, 0.8]],
                [[0.0, 0.0, 0.1], [1.0, 1.0, 0.2]],
                [[float("nan")] * 3] * 2,
            ],
            [[[float("nan")] * 3] * 2] * 3,
        ],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor([[True, True, False], [False, False, False]], device=device)
    support = torch.tensor([[1.0], [2.0]], device=device, dtype=dtype)
    offset, valid = group_height_offset(
        bounds, support, valid_mask=mask, clearance=0.05
    )
    torch.testing.assert_close(
        offset, torch.tensor([[0.95, 0.0], [1.95, 0.0]], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        valid, torch.tensor([[True, False], [True, False]], device=device)
    )
    assert offset.dtype == dtype and offset.device == device


def test_group_height_rejects_unmasked_bad_box_and_nonfinite_support_in_device_mask(
    device, dtype
):
    bounds = torch.tensor(
        [
            [[[0.0, 0.0, 0.2], [1.0, 1.0, 0.5]]],
            [[[0.0, 0.0, 0.8], [1.0, 1.0, 0.5]]],
            [[[0.0, 0.0, 0.2], [1.0, 1.0, 0.5]]],
        ],
        device=device,
        dtype=dtype,
    )
    offset, valid = group_height_offset(
        bounds, torch.tensor([1.0, 1.0, float("inf")], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        offset, torch.tensor([0.8, 0.0, 0.0], device=device, dtype=dtype)
    )
    torch.testing.assert_close(valid, torch.tensor([True, False, False], device=device))


def test_group_height_handles_empty_objects_and_zero_batch_rows(device, dtype):
    offset, valid = group_height_offset(
        torch.empty((2, 0, 2, 3), device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )
    assert offset.shape == (2,) and valid.shape == (2,)
    assert not offset.any() and not valid.any()
    offset, valid = group_height_offset(
        torch.empty((0, 4, 2, 3), device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )
    assert offset.shape == (0,) and valid.shape == (0,)


def test_group_mask_can_broadcast_new_batch_dimensions(device, dtype):
    bounds = torch.tensor(
        [[[0.0, 0.0, 0.2], [1.0, 1.0, 0.4]], [[0.0, 0.0, 0.5], [1.0, 1.0, 0.8]]],
        device=device,
        dtype=dtype,
    )
    mask = torch.tensor([[True, False], [False, True]], device=device)
    offset, valid = group_height_offset(
        bounds, torch.tensor(1.0, device=device, dtype=dtype), valid_mask=mask
    )
    torch.testing.assert_close(
        offset, torch.tensor([0.8, 0.5], device=device, dtype=dtype)
    )
    assert valid.all()


def test_transform_matches_eight_corners_with_rotation_scale_shear_and_reflection(
    device, dtype
):
    bounds = torch.tensor(
        [[[-1.0, -0.2, 0.1], [0.3, 0.7, 1.6]], [[0.2, 0.4, -0.7], [0.8, 1.4, 0.9]]],
        device=device,
        dtype=dtype,
    )
    matrices = torch.tensor(
        [
            [
                [0.36, -0.96, 0.8, 0.3],
                [0.48, 0.72, -0.6, -0.4],
                [-0.8, 0.0, 0.6, 1.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [-1.0, 0.5, 0.2, 2.0],
                [0.0, 3.0, 0.1, 1.0],
                [0.0, 0.0, 2.0, -0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    # Explicit leading singleton axes produce each transform x each box.
    transformed = transform_aabbs(bounds[None], matrices[:, None])
    selectors = torch.tensor(
        list(product([0, 1], repeat=3)), device=device, dtype=torch.bool
    )
    corners = torch.where(selectors, bounds[:, None, 1], bounds[:, None, 0])
    corner_result = (
        torch.einsum("tij,bkj->tbki", matrices[:, :3, :3], corners)
        + matrices[:, None, None, :3, 3]
    )
    expected = torch.stack(
        (corner_result.amin(dim=-2), corner_result.amax(dim=-2)), dim=-2
    )
    torch.testing.assert_close(transformed, expected)
    assert transformed.shape == (2, 2, 2, 3)
    assert transformed.dtype == dtype and transformed.device == device


def test_transform_preserves_dtype_at_extreme_same_sign_bounds(device, dtype):
    maximum = torch.finfo(dtype).max
    bounds = torch.tensor(
        [[maximum * 0.6] * 3, [maximum * 0.8] * 3], device=device, dtype=dtype
    )
    result = transform_aabbs(bounds, torch.eye(4, device=device, dtype=dtype))
    torch.testing.assert_close(result, bounds)


def test_transform_invalid_affine_and_box_inputs_yield_nan_rows(device, dtype):
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype
    )
    matrices = torch.eye(4, device=device, dtype=dtype).repeat(3, 1, 1)
    matrices[1, 3, 0] = 0.1
    matrices[2, 0, 0] = float("inf")
    result = transform_aabbs(bounds, matrices)
    torch.testing.assert_close(result[0], bounds)
    assert torch.isnan(result[1:]).all()


def test_triangle_geometry_handles_winding_degeneracy_and_invalid_padding(
    device, dtype
):
    triangles = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[float("nan")] * 3] * 3,
        ],
        device=device,
        dtype=dtype,
    ).reshape(2, 2, 3, 3)
    normals, areas, valid = triangle_geometry(triangles)
    expected_normals = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(normals, expected_normals)
    torch.testing.assert_close(
        areas, torch.tensor([[3.0, 3.0], [0.0, 0.0]], device=device, dtype=dtype)
    )
    torch.testing.assert_close(
        valid, torch.tensor([[True, True], [False, False]], device=device)
    )
    assert normals.dtype == dtype and normals.device == device
    assert areas.dtype == dtype and areas.device == device


def test_triangle_geometry_handles_empty_batches(device, dtype):
    normals, areas, valid = triangle_geometry(
        torch.empty((2, 0, 3, 3), device=device, dtype=dtype)
    )
    assert normals.shape == (2, 0, 3)
    assert areas.shape == valid.shape == (2, 0)


def test_kernels_do_not_read_tensor_values_on_host(device, dtype, monkeypatch):
    bounds = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype
    )
    transform = torch.eye(4, device=device, dtype=dtype)
    support = torch.tensor(1.0, device=device, dtype=dtype)
    triangles = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A tensor value was read by host Python")

    with monkeypatch.context() as patch:
        for name in ("item", "__bool__", "tolist", "numpy", "cpu"):
            patch.setattr(torch.Tensor, name, forbidden)
        aabb_overlap_depth(bounds, bounds)
        aabb_containment_violation(bounds, bounds)
        group_height_offset(bounds[None], support)
        transform_aabbs(bounds, transform)
        triangle_geometry(triangles)


def test_shape_dtype_and_mask_metadata_fail_before_arithmetic():
    box = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="same dtype"):
        aabb_overlap_depth(box, box.double())
    with pytest.raises(ValueError, match="float32"):
        aabb_overlap_depth(box.long(), box.long())
    with pytest.raises(ValueError, match="shape"):
        aabb_containment_violation(torch.zeros(3), box)
    with pytest.raises(ValueError, match="dimension"):
        aabb_overlap_depth(box, torch.zeros(2, 2))
    with pytest.raises(ValueError, match="boolean"):
        group_height_offset(box[None], torch.tensor(0.0), valid_mask=torch.ones(1))
    with pytest.raises(ValueError, match="shape"):
        group_height_offset(
            box[None], torch.tensor(0.0), valid_mask=torch.ones(2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="shape"):
        triangle_geometry(box)
    with pytest.raises(ValueError, match="Expected"):
        transform_aabbs(box, torch.eye(3))


@pytest.mark.gpu
def test_mixed_device_inputs_are_rejected():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    box = torch.zeros((2, 3), device="cuda")
    with pytest.raises(ValueError, match="same dtype and device"):
        aabb_overlap_depth(box, box.cpu())
    with pytest.raises(ValueError, match="device"):
        group_height_offset(
            box[None], box.new_tensor(0.0), valid_mask=torch.ones(1, dtype=torch.bool)
        )


def test_triangle_normal_does_not_underflow_when_squaring_small_cross_product(
    device, dtype
):
    scale = 1e-10 if dtype == torch.float32 else 1e-100
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0]],
        device=device,
        dtype=dtype,
    )
    normals, area, valid = triangle_geometry(vertices)
    assert valid
    torch.testing.assert_close(normals, vertices.new_tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(area / (scale * scale), area.new_tensor(0.5))


def test_random_repeated_vertices_never_normalize_cancellation_residue(device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(927)
    triangles = torch.randn((128, 3, 3), generator=generator, dtype=dtype).to(device)
    triangles[:, 2] = triangles[:, 1]
    normals, areas, valid = triangle_geometry(triangles)
    assert not valid.any()
    assert not normals.any()
    assert not areas.any()


def test_random_collinear_directions_do_not_become_valid_from_product_cancellation(
    device, dtype
):
    generator = torch.Generator(device="cpu").manual_seed(187)
    directions = torch.randn((128, 3), generator=generator, dtype=dtype).to(device)
    triangles = torch.stack(
        (torch.zeros_like(directions), directions, directions * 3.0), dim=-2
    )
    normals, areas, valid = triangle_geometry(triangles)
    assert not valid.any()
    assert not normals.any()
    assert not areas.any()


def test_resolvable_thin_triangles_remain_valid(device, dtype):
    height = 32 * torch.finfo(dtype).eps
    triangles = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0 + height, 0.0]],
        device=device,
        dtype=dtype,
    )
    normals, areas, valid = triangle_geometry(triangles)
    assert valid
    torch.testing.assert_close(normals, normals.new_tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(areas / height, areas.new_tensor(0.5))
