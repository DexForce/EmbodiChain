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

"""Tensor interpolation of sampled gripper surfaces."""

from __future__ import annotations

import torch

__all__ = ["interpolate_hand_samples"]


def interpolate_hand_samples(
    body: torch.Tensor,
    pads: torch.Tensor,
    gaps: torch.Tensor,
    widths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate closing samples at candidate pad gaps, clamping at endpoints.

    Args:
        body: Body samples of shape ``(G, V, 3)``.
        pads: Two pad surfaces of shape ``(G, 2, P, 3)``.
        gaps: Strictly descending sample gaps of shape ``(G,)``, with G >= 2.
        widths: Candidate pad gaps of shape ``(K,)``; owns output dtype/device.

    Returns:
        Interpolated body and pad samples, with leading candidate dimension K.
        Inputs are not modified. Numeric validity and monotonicity belong to the
        geometry-construction boundary, avoiding repeated device synchronization.
    """
    if body.ndim != 3 or pads.ndim != 4 or gaps.ndim != 1 or widths.ndim != 1:
        raise ValueError("Invalid sampled hand tensor dimensions.")
    if len(gaps) < 2 or body.shape[0] != len(gaps) or pads.shape[0] != len(gaps):
        raise ValueError("Hand surfaces must share at least two gap samples.")
    if body.shape[-1] != 3 or pads.shape[1] != 2 or pads.shape[-1] != 3:
        raise ValueError("Hand geometry requires XYZ samples and two pads.")
    gaps = gaps.to(widths)
    high = torch.searchsorted(-gaps.contiguous(), -widths).clamp(1, len(gaps) - 1)
    low = high - 1
    alpha = ((gaps[low] - widths) / (gaps[low] - gaps[high])).clamp(0, 1)
    body, pads = body.to(widths), pads.to(widths)
    return (
        torch.lerp(body[low], body[high], alpha[:, None, None]),
        torch.lerp(pads[low], pads[high], alpha[:, None, None, None]),
    )
