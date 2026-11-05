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


"""Pure carrier-envelope geometry, independent of simulation and tasks."""

from __future__ import annotations

import torch

__all__ = ["inside_carrier_envelope"]


def inside_carrier_envelope(
    vertices: torch.Tensor,
    object_pose: torch.Tensor,
    carrier_pose: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Check horizontal mesh extent and its lowest point against a carrier AABB.

    Args:
        vertices: Object-local points, shape ``(V, 3)`` or ``(B, V, 3)``.
        object_pose: World object transforms, shape ``(B, 4, 4)``.
        carrier_pose: World carrier transforms, shape ``(B, 4, 4)``.
        bounds: Carrier-local lower/upper bounds, shape ``(2, 3)``.

    Returns:
        Boolean mask per row, using the original two-millimetre tolerance.
        The upper extent is deliberately unconstrained: this is an envelope
        condition, not an inner-cavity or physical retention certificate.
    """
    relative = torch.linalg.solve(carrier_pose, object_pose)
    local = vertices @ relative[:, :3, :3].transpose(-1, -2) + relative[:, None, :3, 3]
    minimum, maximum = local.amin(1), local.amax(1)
    margin = 0.002
    return (
        torch.isfinite(local).all(dim=-1).all(dim=-1)
        & (minimum[:, :2] >= bounds[0, :2] - margin).all(-1)
        & (maximum[:, :2] <= bounds[1, :2] + margin).all(-1)
        & (minimum[:, 2] >= bounds[0, 2] - margin)
        & (minimum[:, 2] <= bounds[1, 2] + margin)
    )
