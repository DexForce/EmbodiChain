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

"""Bounded observed continuation of one atomic invocation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch

if TYPE_CHECKING:
    from .plans import SkillPlan


@runtime_checkable
class PlanContinuation(Protocol):
    """Independently owned progress within one skill invocation.

    ``position`` advances strictly within ``limit`` (at most 256 chunks).
    ``phase_name`` identifies the current chunk for audit and host observations.
    Snapshots must retain no live controller or simulator handles.
    """

    position: int
    limit: int
    phase_name: str

    def snapshot(self) -> PlanContinuation:
        """Copy the bounded progress and its observation-derived data."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class PlanContinuationResult:
    """Next plan or terminal per-row acceptance from fresh observations.

    Args:
        accepted: Owned boolean subset of the invocation's active rows.
        plan: Next bounded phase plan, or ``None`` at the terminal boundary.
        messages: Physical rejection details suitable for execution event logs.
    """

    accepted: torch.Tensor
    plan: SkillPlan | None = None
    messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.accepted, torch.Tensor)
            or self.accepted.dtype != torch.bool
            or self.accepted.ndim != 1
        ):
            raise ValueError("Continuation acceptance must be a boolean row mask.")
        if not all(isinstance(message, str) for message in self.messages):
            raise TypeError("Continuation messages must be strings.")
        object.__setattr__(self, "accepted", self.accepted.clone())
        object.__setattr__(self, "messages", tuple(self.messages))


__all__ = ["PlanContinuation", "PlanContinuationResult"]
