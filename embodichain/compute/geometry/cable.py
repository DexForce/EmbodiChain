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

"""Cable material coordinates and geometric observations, independent of simulation."""

from __future__ import annotations

import torch

__all__ = ["sample_cable_material", "cable_endpoints", "cable_grasp_slip"]


def cable_endpoints(
    state: torch.Tensor, rest_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return proximal/distal capsule ends from xyz+xyzw segment states.

    Args:
        state: World segment states with shape ``(..., S, 13)``.
        rest_lengths: Positive capsule axis lengths with shape ``(S,)``.

    Returns:
        Two tensors of shape ``(..., S, 3)``, in material order.
    """
    if (
        state.ndim < 2
        or state.shape[-1] != 13
        or state.shape[-2] != rest_lengths.numel()
    ):
        raise ValueError("Cable state and rest lengths must agree on segment count.")
    if (
        rest_lengths.ndim != 1
        or rest_lengths.numel() == 0
        or not bool((torch.isfinite(rest_lengths) & (rest_lengths > 0)).all())
    ):
        raise ValueError("Cable rest lengths must be a positive vector.")
    q = state[..., 3:7]
    # The capsule's local +Z axis in world coordinates; quaternions are XYZW.
    x, y, z, w = q.unbind(-1)
    axis = torch.stack(
        (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), -1
    )
    half_axis = 0.5 * rest_lengths.to(state)[..., None] * axis
    return state[..., :3] - half_axis, state[..., :3] + half_axis


def sample_cable_material(
    state: torch.Tensor, rest_lengths: torch.Tensor, distance: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Observe a fixed material coordinate, rather than a nearest world point.

    Args:
        state: World states with shape ``(B,S,13)``.
        rest_lengths: Positive capsule axis lengths ``(S,)``.
        distance: Material arc length from the proximal cable end, in metres.

    Returns:
        World point, unit material tangent, and stable segment index.
    """
    if state.ndim != 3:
        raise ValueError("Material sampling requires batched cable state (B,S,13).")
    starts, ends = cable_endpoints(state, rest_lengths)
    cumulative = rest_lengths.to(state).cumsum(0)
    if not 0 <= distance <= float(cumulative[-1]):
        raise ValueError("Material distance is outside this cable.")
    index = min(
        int(torch.searchsorted(cumulative, cumulative.new_tensor(distance))),
        state.shape[-2] - 1,
    )
    previous = cumulative[index - 1] if index else cumulative.new_tensor(0)
    fraction = (distance - previous) / rest_lengths.to(state)[index]
    direction = ends[:, index] - starts[:, index]
    return (
        starts[:, index] + fraction * direction,
        torch.nn.functional.normalize(direction, dim=-1),
        index,
    )


def cable_grasp_slip(
    point_in_tool: torch.Tensor, initial_point_in_tool: torch.Tensor
) -> torch.Tensor:
    """Measure material-point slip in the tool frame, invariant to wrist motion.

    Args:
        point_in_tool: Current material point in the TCP frame, ``(B,3)``.
        initial_point_in_tool: Independent snapshot when the fingers closed.

    Returns:
        Slip distance in metres for each environment.
    """
    return torch.linalg.vector_norm(point_in_tool - initial_point_in_tool, dim=-1)
