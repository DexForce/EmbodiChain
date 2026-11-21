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

"""Batched contact facts for measured manipulation objectives."""

from __future__ import annotations

import torch

__all__ = ["manipulation_contacts"]


def manipulation_contacts(
    pairs: torch.Tensor,
    normals: torch.Tensor,
    impulses: torch.Tensor,
    present: torch.Tensor,
    impulse_valid: torch.Tensor,
    object_ids: torch.Tensor,
    pad_ids: torch.Tensor,
    support_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce native contact rows into bilateral grip and upward support facts.

    Args:
        pairs: Actor pairs, shaped [B, C, 2].
        normals: Actor-zero-to-one normals, shaped [B, C, 3].
        impulses: Nonnegative normal impulses, shaped [B, C].
        present: Populated contact slots, shaped [B, C].
        impulse_valid: Native impulse availability, shaped [B, C].
        object_ids: Target actor per row, shaped [B].
        pad_ids: Explicit opposing pads per row, shaped [B, 2].
        support_ids: Explicit receiving support per row, shaped [B].

    Returns:
        Grip, hand contact, upward support, and evidence validity, each [B].
        Opposed impulse-weighted pad normals must have cosine at most -0.5.
        Missing quantities invalidate evidence; empty complete contact rows are
        a measured negative, not missing data. No temporal verdict is made here.
    """
    batch, contacts = present.shape
    if pairs.shape != (batch, contacts, 2) or normals.shape != (batch, contacts, 3):
        raise ValueError("Contact pairs/normals must share [batch, contacts] axes")
    if impulses.shape != present.shape or impulse_valid.shape != present.shape:
        raise ValueError("Contact impulse and validity layouts must match")
    if (
        object_ids.shape != (batch,)
        or support_ids.shape != (batch,)
        or pad_ids.shape != (batch, 2)
    ):
        raise ValueError("Actor bindings must match the contact batch")
    finite = (
        torch.isfinite(normals).all(-1) & torch.isfinite(impulses) & (impulses >= 0)
    )
    valid = (~present | (finite & impulse_valid)).all(-1)
    normal = torch.nan_to_num(normals)
    object_first = pairs[..., 0] == object_ids[:, None]
    object_second = pairs[..., 1] == object_ids[:, None]
    other = torch.where(object_first, pairs[..., 1], pairs[..., 0])
    outward = torch.where(object_first[..., None], normal, -normal)
    active = (
        present
        & impulse_valid
        & finite
        & (impulses > 0)
        & (object_first | object_second)
    )
    lengths = torch.linalg.vector_norm(outward, dim=-1)
    valid &= (~active | (lengths > 1e-8)).all(-1)
    unit = outward / lengths.clamp_min(1e-8)[..., None]
    pad = (other[..., None] == pad_ids[:, None, :]) & active[..., None]
    weighted = torch.where(
        pad[..., None],
        unit[:, :, None] * torch.nan_to_num(impulses)[:, :, None, None],
        0,
    ).sum(1)
    magnitudes = torch.linalg.vector_norm(weighted, dim=-1)
    opposed = (weighted[:, 0] * weighted[:, 1]).sum(-1) / magnitudes.prod(-1).clamp_min(
        1e-12
    )
    grip = (magnitudes > 1e-8).all(-1) & (opposed <= -0.5)
    # When the object is actor zero, its outward normal points downward to support.
    support = (active & (other == support_ids[:, None]) & (unit[..., 2] <= -0.9)).any(
        -1
    )
    return grip & valid, pad.any(dim=(1, 2)) & valid, support & valid, valid
