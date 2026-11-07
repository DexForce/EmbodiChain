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

import numpy as np
import pytest
import torch

from embodichain.compute.geometry._hierarchical import (
    _counter_units,
    _fixed_prefix_sum,
    _sample_free_aabb_region,
)


def _propose(device: str, *, start: int, count: int):
    dtype = torch.float64
    region = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0]]], dtype=dtype, device=device
    ).expand(count, 2, 2)
    half = torch.tensor([[0.08, 0.12]], dtype=dtype, device=device).expand(count, 2)
    obstacles = torch.tensor(
        [[[[-0.6, -0.4], [-0.1, 0.3]], [[0.1, -0.1], [0.5, 0.6]]]],
        dtype=dtype,
        device=device,
    ).expand(count, 2, 2, 2)
    unit = _counter_units(
        key=5729, start=start, count=count, width=3, device=torch.device(device)
    )
    return _sample_free_aabb_region(region, half, obstacles, unit, 0.02)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_free_rectangles_cover_only_independently_valid_centers(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    xy, valid = _propose(device, start=0, count=4096)
    xy, valid = xy.cpu().numpy(), valid.cpu().numpy()
    assert valid.all()
    assert (np.abs(xy) <= 1).all()
    # Independent complete-AABB pair predicate, including the requested margin.
    half = np.array([0.08, 0.12])
    obstacles = np.array([[[-0.6, -0.4], [-0.1, 0.3]], [[0.1, -0.1], [0.5, 0.6]]])
    for obstacle in obstacles:
        gaps = np.maximum(xy - half - obstacle[1], obstacle[0] - (xy + half))
        assert (gaps.max(axis=1) >= 0.02 - 1e-12).all()
    # Both disjoint free sides must remain reachable rather than projecting
    # rejected proposals onto a single edge.
    assert (xy[:, 0] < -0.8).any() and (xy[:, 0] > 0.8).any()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_free_region_prefix_does_not_depend_on_batch_partition(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    complete, valid = _propose(device, start=19, count=257)
    for size in (1, 7, 64):
        pieces = [
            _propose(device, start=19 + start, count=min(size, 257 - start))
            for start in range(0, 257, size)
        ]
        assert torch.equal(complete, torch.cat([p[0] for p in pieces]))
        assert torch.equal(valid, torch.cat([p[1] for p in pieces]))


def test_free_region_handles_exact_fit_and_impossible_obstacles():
    region = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]],
        dtype=torch.float64,
    )
    half = torch.tensor([[0.1, 0.1]] * 3, dtype=torch.float64)
    # First point touches at the exact clearance and is valid. The second line
    # intersects the obstacle on its whole domain; the third square is blocked.
    obstacles = torch.tensor(
        [
            [[[0.12, -0.1], [0.3, 0.1]]],
            [[[-0.2, -0.2], [0.2, 1.2]]],
            [[[-1.0, -1.0], [2.0, 2.0]]],
        ],
        dtype=torch.float64,
    )
    unit = torch.full((3, 3), 0.5, dtype=torch.float64)
    xy, valid = _sample_free_aabb_region(region, half, obstacles, unit, 0.02)
    assert valid.tolist() == [True, False, False]
    np.testing.assert_allclose(xy[0].numpy(), [0, 0], atol=1e-16)


def test_free_region_respects_per_candidate_footprint_sizes():
    region = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]], dtype=torch.float64).expand(
        2, 2, 2
    )
    half = torch.tensor([[0.05, 0.05], [2.0, 2.0]], dtype=torch.float64)
    obstacle = torch.tensor([[[[-0.1, -0.1], [0.1, 0.1]]]], dtype=torch.float64).expand(
        2, 1, 2, 2
    )
    xy, valid = _sample_free_aabb_region(
        region, half, obstacle, torch.full((2, 3), 0.25, dtype=torch.float64), 0
    )
    assert valid.tolist() == [True, False]
    assert np.max(np.abs(xy[0].numpy()) - half[0].numpy() - 0.1) >= -1e-12


def test_obstacle_reduction_uses_bounded_chunks_for_large_group_counts():
    from torch.utils._python_dispatch import TorchDispatchMode

    observed = []

    class AllocationShapes(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            if (
                isinstance(result, torch.Tensor)
                and result.dtype == torch.bool
                and result.ndim == 4
            ):
                observed.append(tuple(result.shape))
            return result

    count, obstacles_count = 2, 33
    region = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]], dtype=torch.float64).expand(
        count, 2, 2
    )
    half = torch.full((count, 2), 0.05, dtype=torch.float64)
    obstacles = torch.tensor(
        [[[[-0.1, -0.1], [0.1, 0.1]]]], dtype=torch.float64
    ).expand(count, obstacles_count, 2, 2)
    with AllocationShapes():
        _, valid = _sample_free_aabb_region(
            region,
            half,
            obstacles,
            torch.full((count, 3), 0.25, dtype=torch.float64),
            0.02,
        )
    assert valid.all()
    assert observed and max(shape[-1] for shape in observed) <= 8


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_area_prefix_reduction_is_identical_for_single_and_multiple_rows(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    weights = (
        (torch.arange(4489, dtype=torch.float64, device=device) * 0.0137) % 1 + 0.000001
    ).reshape(1, -1)
    one = _fixed_prefix_sum(weights)
    many = _fixed_prefix_sum(weights.expand(7, -1).contiguous())
    assert torch.equal(one, many[:1])
    # Separate CPU serial oracle checks numerical meaning without assuming
    # the implementation's parallel summation order.
    np.testing.assert_allclose(
        one.cpu().numpy(),
        np.cumsum(weights.cpu().numpy(), axis=1),
        rtol=2e-14,
        atol=1e-12,
    )
