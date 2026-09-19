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

"""Robot-independent tensor helpers for Unitree velocity tasks."""

from __future__ import annotations

import math
import re

import torch

__all__ = [
    "command_active",
    "constant",
    "joint_limit_cost",
    "phase_signal",
    "required",
    "resolve_pattern_values",
]


def constant(values: tuple[float, ...], like: torch.Tensor) -> torch.Tensor:
    """Create a constant tensor on the same device and dtype as ``like``."""
    return torch.tensor(values, device=like.device, dtype=like.dtype)


def phase_signal(
    episode_step: torch.Tensor,
    control_dt: float,
    period: float,
    command: torch.Tensor,
    *,
    zero_when_standing: bool,
) -> torch.Tensor:
    """Return the two-channel sinusoidal gait phase."""
    global_phase = (episode_step * control_dt).remainder(period) / period
    signal = torch.stack(
        (
            torch.sin(2.0 * math.pi * global_phase),
            torch.cos(2.0 * math.pi * global_phase),
        ),
        dim=-1,
    )
    if zero_when_standing:
        standing = torch.linalg.vector_norm(command, dim=-1) < 0.1
        signal = torch.where(standing.unsqueeze(-1), torch.zeros_like(signal), signal)
    return signal


def required(value: torch.Tensor | None, name: str) -> torch.Tensor:
    """Return a required tensor or report the missing state field."""
    if value is None:
        raise ValueError(f"Required task state is missing: {name}")
    return value


def joint_limit_cost(
    joint_pos: torch.Tensor,
    soft_joint_lower: torch.Tensor,
    soft_joint_upper: torch.Tensor,
) -> torch.Tensor:
    """Return the summed soft joint-limit violation."""
    below = (soft_joint_lower - joint_pos).clamp_min(0.0)
    above = (joint_pos - soft_joint_upper).clamp_min(0.0)
    return (below + above).sum(dim=-1)


def command_active(command: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return whether planar or yaw command magnitude exceeds ``threshold``."""
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
    return speed > threshold


def resolve_pattern_values(
    patterns: dict[str, float], names: tuple[str, ...]
) -> list[float]:
    """Resolve the final matching regular-expression value for each joint."""
    values: list[float] = []
    for name in names:
        matched = [
            float(value)
            for pattern, value in patterns.items()
            if re.fullmatch(pattern, name)
        ]
        if not matched:
            raise ValueError(f"No pattern value resolves joint {name}")
        values.append(matched[-1])
    return values
