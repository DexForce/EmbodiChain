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

"""Preserve Python row selection semantics at native render-batch boundaries."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import torch

__all__: list[str] = []


def _normalize_render_rows(
    env_ids: Sequence[int] | torch.Tensor | None, count: int
) -> tuple[list[int] | None, list[int] | None]:
    """Resolve negative rows and retain each duplicate row's final write."""
    if env_ids is None:
        return None, None
    if torch.is_tensor(env_ids):
        env_ids = env_ids.detach().cpu().tolist()
    rows = [operator.index(index) for index in env_ids]
    rows = [index + count if index < 0 else index for index in rows]
    if any(index < 0 or index >= count for index in rows):
        raise IndexError("Render environment index out of range.")
    last = {row: offset for offset, row in enumerate(rows)}
    if len(last) == len(rows):
        return rows, None
    return list(last), list(last.values())


def _select_render_values(
    values: torch.Tensor, offsets: list[int] | None, broadcast_ndim: int
) -> torch.Tensor:
    """Select final per-row payloads while leaving legacy broadcasts intact."""
    if offsets is None or values.ndim <= broadcast_ndim:
        return values
    if broadcast_ndim == 0 and values.shape == (1,):
        return values
    return values.index_select(
        0, torch.tensor(offsets, dtype=torch.long, device=values.device)
    )
