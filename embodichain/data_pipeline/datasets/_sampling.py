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

"""Pure selection, temporal windows and padding shared by data sources."""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__: list[str] = []


def _context_indices(
    valid: torch.Tensor,
    continuity: torch.Tensor | None,
    rows: torch.Tensor,
    anchors: torch.Tensor,
    history: int,
    future: int,
    segments: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = torch.arange(valid.shape[1], device=valid.device)
    compatible = valid[rows]
    if continuity is not None:
        compatible &= continuity[rows] == continuity[rows, anchors, None]
    if segments is not None:
        compatible &= segments[rows] == segments[rows, anchors, None]
    lower = torch.where(
        (~compatible) & (steps[None] < anchors[:, None]), steps[None] + 1, 0
    ).amax(-1)
    upper = torch.where(
        (~compatible) & (steps[None] > anchors[:, None]),
        steps[None] - 1,
        valid.shape[1] - 1,
    ).amin(-1)
    offsets = torch.arange(1 - history, future, device=valid.device)
    indices = anchors[:, None] + offsets[None]
    mask = (indices >= lower[:, None]) & (indices <= upper[:, None])
    return indices.clamp(0, valid.shape[1] - 1), mask, mask & (offsets[None] >= 0)


def _pad_context(
    batch: TensorDict, valid: torch.Tensor, loss_mask: torch.Tensor
) -> TensorDict:
    for key, value in batch.items(include_nested=True, leaves_only=True):
        mask = valid.reshape(*valid.shape, *((1,) * (value.ndim - valid.ndim)))
        identifier = isinstance(key, str) and (
            key.startswith("ods_")
            or key
            in {
                "frame_ids",
                "generations",
                "task_index",
                "subtask_index",
                "accepted_at_ns",
                "published_at_monotonic_ns",
            }
        )
        value.masked_fill_(~mask, -1 if identifier else 0)
    batch["valid"] = valid
    batch["loss_mask"] = loss_mask
    return batch


def _draw_budgeted_rows(
    candidate_rows: torch.Tensor,
    remaining: torch.Tensor,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw episode-uniform rows with depletion, using O(batch + cache) memory.

    Vector proposals retain the normal uniform distribution; exhausted proposals
    are rejected locally. Quotas are never expanded into one entry per credit.
    The usual nondepleting batch uses one randint and bincount without a row loop.
    """
    available = remaining.tolist()
    active = [index for index, count in enumerate(available) if count > 0]
    selected: list[int] = []
    while len(selected) < batch_size:
        proposals = torch.randint(
            len(active),
            (batch_size - len(selected),),
            generator=generator,
            device=candidate_rows.device,
        )
        active_tensor = torch.tensor(
            active, dtype=torch.long, device=candidate_rows.device
        )
        indices = active_tensor[proposals]
        counts = torch.bincount(indices, minlength=len(available))
        if not selected and bool((counts <= remaining).all()):
            return candidate_rows[indices]
        for index in indices.tolist():
            if available[index] > 0:
                selected.append(index)
                available[index] -= 1
        active = [index for index, count in enumerate(available) if count > 0]
    return candidate_rows[
        torch.tensor(selected, dtype=torch.long, device=candidate_rows.device)
    ]
