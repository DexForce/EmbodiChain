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

import math
from typing import Any

import torch

__all__ = ["joint_velocity_validity", "joint_velocity_diagnostics"]


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


def joint_velocity_diagnostics(
    positions: torch.Tensor, dt: torch.Tensor, limits: torch.Tensor
) -> list[dict[str, Any]]:
    """Report the largest finite-interval speed ratio per environment.

    Args:
        positions: Joint samples (B, T, D).
        dt: Arrival intervals (B, T).
        limits: Per-row joint speed caps (B, D).

    Returns:
        Per-row largest-ratio frame/joint and original scalar operands. Nonfinite
        diagnostic values are represented by None; this does not relax validity.
    """
    records = []
    for env_id in range(positions.shape[0]):
        delta = (positions[env_id, 1:] - positions[env_id, :-1]).abs()
        allowed = dt[env_id, 1:, None] * limits[env_id, None]
        ratio = torch.where(
            allowed > 0, delta / allowed, torch.where(delta == 0, 0.0, float("inf"))
        )
        if not ratio.numel():
            continue
        index = int(torch.nan_to_num(ratio, nan=float("inf")).argmax())
        frame, joint = divmod(index, positions.shape[2])
        values = {
            "previous_qpos": float(positions[env_id, frame, joint]),
            "next_qpos": float(positions[env_id, frame + 1, joint]),
            "dt": float(dt[env_id, frame + 1]),
            "velocity_limit": float(limits[env_id, joint]),
            "speed_ratio": float(ratio[frame, joint]),
        }
        records.append(
            {
                "env_id": env_id,
                "frame": frame + 1,
                "control_joint_index": joint,
                **{
                    key: value if math.isfinite(value) else None
                    for key, value in values.items()
                },
            }
        )
    return records
