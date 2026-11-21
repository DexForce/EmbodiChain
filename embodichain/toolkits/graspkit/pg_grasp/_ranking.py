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

"""Shared numerical ranking for antipodal candidates and unqualified proposals."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__: list[str] = []

_DEFAULT_RANKING_WEIGHTS = (0.5, 0.5, 0.0)


def _normalized_ranking_weights(
    angle_weight: float, center_weight: float, opening_weight: float
) -> tuple[float, float, float]:
    """Validate relative weights and normalize without overflowing their sum."""
    weights = (angle_weight, center_weight, opening_weight)
    for name, value in zip(
        ("angle_weight", "center_weight", "opening_weight"), weights
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    scale = max(weights)
    if scale == 0:
        raise ValueError("At least one ranking weight must be positive.")
    angle, center, opening = (value / scale for value in weights)
    total = angle + center + opening
    return angle / total, center / total, opening / total


def _grasp_costs(
    poses: torch.Tensor,
    widths: torch.Tensor,
    centers: torch.Tensor,
    approach_direction: torch.Tensor,
    mesh_center: torch.Tensor,
    weights: tuple[float, float, float],
) -> torch.Tensor:
    """Score qualified geometry using normalized angle, center and opening weights."""
    axis = F.normalize(poses[:, :3, 0], dim=-1)
    angle = torch.acos((axis * approach_direction).sum(-1).clamp(-1, 1)).abs()
    angle_cost = (angle - 0.5 * torch.pi).abs() / (0.5 * torch.pi)
    distance = torch.norm(centers - mesh_center, dim=-1)
    eps = torch.finfo(distance.dtype).eps
    center_cost = distance / distance.max().clamp_min(eps)
    angle_weight, center_weight, opening_weight = weights
    if opening_weight:
        opening_cost = 1 - widths / widths.max().clamp_min(eps)
        return (
            angle_weight * angle_cost
            + opening_weight * opening_cost
            + center_weight * center_cost
        )
    return angle_weight * angle_cost + center_weight * center_cost
