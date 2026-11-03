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

"""Pure sampled-trajectory limit checks."""

from __future__ import annotations

import torch

__all__ = ["joint_velocity_validity"]


def joint_velocity_validity(
    positions: torch.Tensor, dt: torch.Tensor, limits: torch.Tensor
) -> torch.Tensor:
    """Check every finite interval against explicit per-row joint speeds.

    Args:
        positions: Joint positions with shape ``(B, T, D)``.
        dt: Arrival intervals with shape ``(B, T)``; the first is not a motion.
        limits: Nonnegative speed limits with shape ``(B, D)``.

    Returns:
        Boolean validity for each environment row. A zero interval permits only
        displacement within the numerical tolerance of ``1e-5`` joint units.

    Raises:
        ValueError: If shapes disagree or limits are negative/nonfinite.
    """
    if (
        positions.ndim != 3
        or dt.shape != positions.shape[:2]
        or limits.shape != (positions.shape[0], positions.shape[2])
    ):
        raise ValueError(
            "Motion velocity checks require matching batched positions, dt and limits."
        )
    if not torch.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("Joint velocity limits must be finite and non-negative.")
    delta = (positions[:, 1:] - positions[:, :-1]).abs()
    allowed = limits[:, None] * dt[:, 1:, None]
    return (
        torch.isfinite(positions).all(-1).all(-1)
        & torch.isfinite(dt).all(-1)
        & (dt >= 0).all(-1)
        & (delta <= allowed + 1e-5).all(-1).all(-1)
    )
