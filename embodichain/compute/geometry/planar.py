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

"""Batched arena-Z planar displacement of rigid group frames."""

from __future__ import annotations

import torch

__all__ = ["compose_planar_offsets"]


def compose_planar_offsets(
    reference: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Compose arena XY translations and yaw rotations around group origins.

    Args:
        reference: Group-to-arena matrices, shape ``(V, 4, 4)``.
        offsets: XY metre/yaw radian displacements, shape ``(..., V, 3)``.
    Returns:
        Independent transformed groups, shape ``(..., V, 4, 4)``. Height and
        authored tilt are preserved. Inputs share floating dtype/device.
    """
    if reference.ndim != 3 or reference.shape[-2:] != (4, 4):
        raise ValueError("reference must have shape (V,4,4)")
    if offsets.ndim < 2 or offsets.shape[-2:] != (reference.shape[0], 3):
        raise ValueError("offsets must have shape (...,V,3)")
    if (
        not reference.is_floating_point()
        or reference.dtype != offsets.dtype
        or reference.device != offsets.device
    ):
        raise ValueError("reference and offsets must share floating dtype/device")
    result = reference.expand(*offsets.shape[:-2], *reference.shape).clone()
    c, s = offsets[..., 2].cos(), offsets[..., 2].sin()
    first, second = result[..., 0, :3].clone(), result[..., 1, :3].clone()
    result[..., 0, :3] = c[..., None] * first - s[..., None] * second
    result[..., 1, :3] = s[..., None] * first + c[..., None] * second
    result[..., :2, 3] = result[..., :2, 3] + offsets[..., :2]
    return result
