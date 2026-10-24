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

"""Independent bounded-layout checks for CPU and CUDA counter sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embodichain.compute.geometry._sampling import _sample_aabb_layouts


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_sampling_masks_match_independent_aabb_checks(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    half = torch.tensor(
        [[0.12, 0.08], [0.20, 0.15], [0.06, 0.10]], dtype=torch.float64, device=device
    )
    region = torch.tensor(
        [[-1.0, -0.6], [1.0, 0.6]], dtype=torch.float64, device=device
    )
    xy, mask = _sample_aabb_layouts(
        half, region, seed=42, start=0, count=4096, margin=0.02
    )
    boxes = np.stack(
        (xy.cpu().numpy() - half.cpu().numpy(), xy.cpu().numpy() + half.cpu().numpy()),
        axis=-2,
    )
    expected = np.ones(len(boxes), dtype=bool)
    for index, scene in enumerate(boxes):
        for i in range(len(scene)):
            assert np.all(scene[i, 0] >= [-1.0, -0.6])
            assert np.all(scene[i, 1] <= [1.0, 0.6])
            for j in range(i):
                expected[index] &= (
                    np.max(
                        np.maximum(scene[i, 0] - scene[j, 1], scene[j, 0] - scene[i, 1])
                    )
                    >= 0.02
                )
    np.testing.assert_array_equal(mask.cpu(), expected)
    assert 0 < expected.sum() < len(expected)
    second, _ = _sample_aabb_layouts(
        half, region, seed=42, start=3000, count=1096, margin=0.02
    )
    assert torch.equal(xy[3000:], second)


def test_single_group_has_no_pairwise_rejections() -> None:
    half = torch.tensor([[0.2, 0.1]], dtype=torch.float64)
    region = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
    xy, valid = _sample_aabb_layouts(
        half, region, seed=7, start=0, count=100, margin=1.0
    )
    assert xy.shape == (100, 1, 2)
    assert valid.all()
