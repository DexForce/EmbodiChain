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

"""Tests for distance-based trajectory interpolation utilities."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from embodichain.lab.sim.utility.action_utils import (
    interpolate_with_distance,
    resample_with_distance,
)


def test_interpolate_with_distance_preserves_intermediate_keyframe() -> None:
    keyframes = torch.tensor([[[0.0], [1.0], [3.0]]])

    result = interpolate_with_distance(keyframes, interp_num=5, device="cpu")

    # One interval is reserved per segment; the two remaining intervals are
    # split evenly by largest remainder for segment lengths one and two.
    expected = torch.tensor([[[0.0], [0.5], [1.0], [2.0], [3.0]]])
    assert torch.equal(result, expected)


def test_interpolate_with_distance_allocates_each_batch_independently() -> None:
    keyframes = torch.tensor(
        [
            [[0.0], [1.0], [4.0]],
            [[0.0], [3.0], [4.0]],
        ]
    )

    result = interpolate_with_distance(keyframes, interp_num=6, device="cpu")

    expected = torch.tensor(
        [
            [[0.0], [0.5], [1.0], [2.0], [3.0], [4.0]],
            [[0.0], [1.0], [2.0], [3.0], [3.5], [4.0]],
        ]
    )
    assert torch.equal(result, expected)


def test_interpolate_with_distance_retains_duplicate_keyframes() -> None:
    keyframes = torch.tensor([[[0.0], [0.0], [1.0]]])

    result = interpolate_with_distance(keyframes, interp_num=4, device="cpu")

    expected = torch.tensor([[[0.0], [0.0], [0.5], [1.0]]])
    assert torch.equal(result, expected)


def test_interpolate_with_distance_rejects_insufficient_samples() -> None:
    keyframes = torch.tensor([[[0.0], [1.0], [3.0]]])

    with pytest.raises(ValueError, match="at least the number of keyframes"):
        interpolate_with_distance(keyframes, interp_num=2, device="cpu")


def test_interpolate_with_distance_repeats_single_keyframe() -> None:
    keyframes = torch.tensor([[[2.0, 3.0]]])

    result = interpolate_with_distance(keyframes, interp_num=3, device="cpu")

    expected = keyframes.expand(-1, 3, -1)
    assert torch.equal(result, expected)


@pytest.mark.gpu
def test_interpolate_with_distance_preserves_keyframes_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    keyframes = torch.tensor([[[0.0], [1.0], [3.0]]], device="cuda:0")

    result = interpolate_with_distance(keyframes, interp_num=5, device="cuda:0")

    expected = torch.tensor([[[0.0], [0.5], [1.0], [2.0], [3.0]]], device="cuda:0")
    assert torch.equal(result, expected)


def test_resample_with_distance_allows_path_downsampling() -> None:
    wp.init()
    path = torch.tensor([[[0.0], [1.0], [3.0]]])

    result = resample_with_distance(path, interp_num=2, device="cpu")

    expected = torch.tensor([[[0.0], [3.0]]])
    assert torch.equal(result, expected)
