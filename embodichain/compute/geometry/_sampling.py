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

"""Counter-addressed, device-resident rectangular AABB proposal batches."""

from __future__ import annotations

import hashlib

import torch

__all__: list[str] = []


def _sample_aabb_layouts(
    half_extents: torch.Tensor,
    region: torch.Tensor,
    *,
    seed: int,
    start: int,
    count: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent XY layouts and mask those satisfying every clearance.

    Integer counters address each proposal independently of batch boundaries.
    The 32-bit mixing products fit signed int64, avoiding device-dependent
    overflow. Float64 coordinates use an open uniform grid of 2**32 values.
    This is rejection sampling, not an optimizer or a physical validator.
    """
    groups = half_extents.shape[0]
    device = half_extents.device
    counter = torch.arange(
        start * groups * 2,
        (start + count) * groups * 2,
        dtype=torch.int64,
        device=device,
    )
    key = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:4], "little")
    word = (counter & 0xFFFFFFFF) ^ (counter >> 32) ^ key
    word = ((word ^ (word >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    word = ((word ^ (word >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    unit = ((word ^ (word >> 16)).to(torch.float64) + 0.5) / 2**32
    low, high = region[0] + half_extents, region[1] - half_extents
    xy = low + unit.reshape(count, groups, 2) * (high - low)
    first, second = torch.triu_indices(groups, groups, offset=1, device=device)
    gap = (
        (xy[:, first] - xy[:, second]).abs()
        - half_extents[first]
        - half_extents[second]
    )
    valid = torch.isfinite(xy).all(dim=(-1, -2))
    valid &= ((xy - half_extents >= region[0]) & (xy + half_extents <= region[1])).all(
        dim=(-1, -2)
    )
    valid &= (gap.amax(dim=-1) >= margin).all(dim=-1)
    return xy, valid
