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

"""Measured geometric conditions shared by environment and expert adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import torch

__all__ = ["ConditionResult", "evaluate_near_target", "evaluate_below_height"]


@dataclass(frozen=True)
class ConditionResult:
    """Owned per-row physical truth, validity and diagnostic measurements.

    Args:
        satisfied: Boolean vector of physical predicate values.
        valid: Boolean vector identifying available finite evidence.
        metrics: Named floating measurements with the same leading batch size.
    """

    satisfied: torch.Tensor
    valid: torch.Tensor
    metrics: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.satisfied.dtype != torch.bool or self.satisfied.ndim != 1:
            raise ValueError("satisfied must be a one-dimensional boolean tensor.")
        if self.valid.dtype != torch.bool or self.valid.shape != self.satisfied.shape:
            raise ValueError("valid must be a boolean vector matching satisfied.")
        if self.valid.device != self.satisfied.device:
            raise ValueError("Condition masks must share a device.")
        owned = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Condition metric names must be nonempty strings.")
            if value.ndim < 1 or value.shape[0] != self.satisfied.shape[0]:
                raise ValueError("Condition metrics must match the batch size.")
            if value.device != self.satisfied.device:
                raise ValueError("Condition metrics and masks must share a device.")
            owned[name] = value.clone()
        object.__setattr__(self, "satisfied", (self.satisfied & self.valid).clone())
        object.__setattr__(self, "valid", self.valid.clone())
        object.__setattr__(self, "metrics", owned)


def _positions(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("Positions must have shape (num_envs, 3).")
    if not value.is_floating_point():
        raise TypeError("Positions must be floating tensors.")


def evaluate_near_target(
    position: torch.Tensor,
    target: torch.Tensor,
    tolerance: float,
    *,
    axes: Sequence[int] = (0, 1, 2),
    inclusive: bool = True,
) -> ConditionResult:
    """Measure Euclidean distance in selected environment-local axes.

    Args:
        position: Object positions with shape ``(B, 3)`` in metres.
        target: Target positions with shape ``(3,)`` or ``(B, 3)``.
        tolerance: Positive finite distance bound in metres.
        axes: Nonempty, distinct indices from X=0, Y=1 and Z=2.
        inclusive: Whether equality with the bound passes.

    Returns:
        Physical truth, validity and the measured ``distance`` vector.
    """
    _positions(position)
    axes = tuple(axes)
    if (
        not axes
        or any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes)
        or len(set(axes)) != len(axes)
    ):
        raise ValueError("axes must select distinct indices from 0, 1, 2.")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be positive and finite.")
    if type(inclusive) is not bool:
        raise TypeError("inclusive must be a bool.")
    target = target.to(device=position.device, dtype=position.dtype)
    if target.shape == (3,):
        target = target.unsqueeze(0).expand_as(position)
    _positions(target)
    if target.shape != position.shape:
        raise ValueError("Target batch must match object positions.")
    valid = torch.isfinite(position).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    distance = torch.linalg.vector_norm((position - target)[:, axes], dim=-1)
    satisfied = distance <= tolerance if inclusive else distance < tolerance
    return ConditionResult(satisfied, valid, {"distance": distance})


def evaluate_below_height(position: torch.Tensor, height: float) -> ConditionResult:
    """Check whether an object is strictly below a local height in metres.

    Args:
        position: Object positions with shape ``(B, 3)``.
        height: Finite local Z threshold.

    Returns:
        Physical truth, validity and measured ``height``.
    """
    _positions(position)
    if (
        isinstance(height, bool)
        or not isinstance(height, (int, float))
        or not math.isfinite(height)
    ):
        raise ValueError("height must be finite.")
    valid = torch.isfinite(position).all(dim=-1)
    return ConditionResult(position[:, 2] < height, valid, {"height": position[:, 2]})
