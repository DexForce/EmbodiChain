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
"""Fuse reference frame vectors while retaining independently owned snapshots."""

from __future__ import annotations

import torch
import warp as wp

from embodichain.utils.device_utils import current_warp_stream

__all__: list[str] = []

wp.set_module_options({"enable_fp_fusion": False, "enable_backward": False})


@wp.func
def _inverse(q: wp.vec3, w: float, value: wp.vec3) -> wp.vec3:
    tangent = wp.cross(q, value) * 2.0
    return value - w * tangent + wp.cross(q, tangent)


@wp.kernel(enable_backward=False)
def _frame_vectors(
    quaternion: wp.array2d(dtype=float),
    linear: wp.array2d(dtype=float),
    angular: wp.array2d(dtype=float),
    output: wp.array3d(dtype=float),
):
    row = wp.tid()
    x, y, z, w = (
        quaternion[row, 0],
        quaternion[row, 1],
        quaternion[row, 2],
        quaternion[row, 3],
    )
    xyz = wp.vec3(x, y, z)
    velocity = wp.vec3(linear[row, 0], linear[row, 1], linear[row, 2])
    omega = wp.vec3(angular[row, 0], angular[row, 1], angular[row, 2])
    yaw = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    v0 = _inverse(xyz, w, velocity)
    v1 = _inverse(xyz, w, omega)
    v2 = _inverse(xyz, w, wp.vec3(0.0, 0.0, -1.0))
    v3 = _inverse(wp.vec3(0.0, 0.0, wp.sin(yaw / 2.0)), wp.cos(yaw / 2.0), velocity)
    for column in range(3):
        output[row, 0, column] = v0[column]
        output[row, 1, column] = v1[column]
        output[row, 2, column] = v2[column]
        output[row, 3, column] = v3[column]


def _reference_frame_vectors(
    quaternion: torch.Tensor, linear: torch.Tensor, angular: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fresh body-frame vectors and yaw-frame linear velocity."""
    output = linear.new_empty((linear.shape[0], 4, 3))
    wp.launch(
        _frame_vectors,
        dim=linear.shape[0],
        inputs=[
            wp.from_torch(quaternion),
            wp.from_torch(linear),
            wp.from_torch(angular),
            wp.from_torch(output),
        ],
        device=str(linear.device),
        stream=current_warp_stream(linear.device),
    )
    return tuple(output.unbind(dim=1))
