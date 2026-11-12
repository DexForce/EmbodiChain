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

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from embodichain.compute.geometry._distance import _measure_near_target, _positions
from embodichain.lab.evaluation._predicates import _HEIGHT, _near_params

if TYPE_CHECKING:
    from embodichain.lab.evaluation import RuleResult

__all__ = ["ConditionResult", "evaluate_near_target", "evaluate_below_height"]


@dataclass(frozen=True)
class ConditionResult:
    """Owned per-row physical truth, validity and diagnostic measurements.

    Args:
        satisfied: Boolean vector of physical predicate values.
        valid: Boolean vector identifying available finite evidence.
        metrics: Named floating measurements with the same leading batch size.
        rule_results: Optional diagnostics from a compiled rule graph.
        evaluation_identity: Content identity of the compiled rule graph, if supplied.
    """

    satisfied: torch.Tensor
    valid: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    rule_results: Mapping[str, RuleResult] = field(default_factory=dict)
    evaluation_identity: str | None = None

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
    params = _near_params(
        {"tolerance": tolerance, "axes": axes, "inclusive": inclusive}
    )
    _positions(position)
    passed, valid, distance = _measure_near_target(position, target, **params)
    return ConditionResult(passed, valid, {"distance": distance})


def evaluate_below_height(position: torch.Tensor, height: float) -> ConditionResult:
    """Check whether an object is strictly below a local height in metres.

    Args:
        position: Object positions with shape ``(B, 3)``.
        height: Finite local Z threshold.

    Returns:
        Physical truth, validity and measured ``height``.
    """
    result = _HEIGHT.measure(
        {"position": position}, _HEIGHT.normalize({"height": height})
    )
    return ConditionResult(result.satisfied, result.valid, result.metrics)
