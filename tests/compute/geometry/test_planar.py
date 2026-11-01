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

"""Arena planar group transform oracles and device/dtype contracts."""

from __future__ import annotations

import math
import pytest
import torch

from embodichain.compute.geometry.planar import compose_planar_offsets


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_planar_yaw_preserves_tilt_mounting_height_and_inputs(dtype):
    reference = torch.tensor(
        [[[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]]], dtype=dtype
    )
    before = reference.clone()
    offsets = torch.tensor([[[0.5, -0.5, math.pi / 2]], [[0, 0, 0]]], dtype=dtype)
    result = compose_planar_offsets(reference, offsets)
    expected = torch.tensor(
        [[0, 0, 1, 1.5], [1, 0, 0, 1.5], [0, 1, 0, 3], [0, 0, 0, 1]], dtype=dtype
    )
    assert torch.allclose(result[0, 0], expected, atol=1e-6, rtol=0)
    assert torch.equal(result[1, 0], before[0]) and torch.equal(reference, before)
    result.zero_()
    assert torch.equal(reference, before)


@pytest.mark.gpu
def test_planar_cuda_matches_cpu_with_noncontiguous_offsets():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    reference = torch.eye(4).repeat(2, 1, 1)
    offsets = torch.tensor([[[0.1, 0.2, 0.3], [0, 0, 0], [0.4, 0.5, 0.6], [0, 0, 0]]])[
        :, ::2
    ]
    assert not offsets.is_contiguous()
    actual = compose_planar_offsets(reference.cuda(), offsets.cuda()).cpu()
    assert torch.allclose(
        actual, compose_planar_offsets(reference, offsets), atol=1e-6, rtol=0
    )
