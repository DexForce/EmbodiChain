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

"""Private selection shared by live and resident source-budget samplers."""

from __future__ import annotations

import torch

__all__: list[str] = []


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
