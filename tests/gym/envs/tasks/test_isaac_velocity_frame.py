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
"""Compare fused reference vectors with the established Torch mathematics."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from embodichain.utils.math import quat_apply_inverse
from embodichain_tasks.locomotion.velocity.isaac._frame import _reference_frame_vectors

__all__: list[str] = []
pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_frame_vectors_preserve_math_strides_and_snapshot_ownership(
    device: str,
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    wp.init()
    generator = torch.Generator(device=device).manual_seed(42)
    pose = torch.randn((2048, 7), generator=generator, device=device)
    quaternion = pose[:, 3:]
    quaternion /= quaternion.norm(dim=-1, keepdim=True)
    quaternion[:4] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
    )
    linear = torch.randn((2048, 3), generator=generator, device=device) * 10
    angular = torch.randn((2048, 3), generator=generator, device=device) * 10
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    yaw_quaternion = torch.zeros_like(quaternion)
    yaw_quaternion[:, 2] = torch.sin(yaw / 2)
    yaw_quaternion[:, 3] = torch.cos(yaw / 2)
    gravity = torch.zeros_like(linear)
    gravity[:, 2] = -1
    expected = (
        quat_apply_inverse(quaternion, linear),
        quat_apply_inverse(quaternion, angular),
        quat_apply_inverse(quaternion, gravity),
        quat_apply_inverse(yaw_quaternion, linear),
    )
    before = [value.clone() for value in (pose, linear, angular)]
    actual = _reference_frame_vectors(quaternion, linear, angular)
    wp.synchronize_device(device)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-6)
    saved = [value.clone() for value in actual]
    _reference_frame_vectors(
        quaternion, torch.zeros_like(linear), torch.zeros_like(angular)
    )
    wp.synchronize_device(device)
    for a, b in zip(actual, saved, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip((pose, linear, angular), before, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
