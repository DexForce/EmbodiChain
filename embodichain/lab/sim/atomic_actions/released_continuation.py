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

"""Explicit released-hand cleanup and following Park planning contracts."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING, runtime_checkable
from .policies import PlanningRecovery

if TYPE_CHECKING:
    from .bindings import ActionBinding
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext
    from .plans import ActionPlan
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "ReleasedHandClearanceRecovery",
    "ReleasedHandParkRecovery",
    "ReleasedHandContinuationSession",
    "ReleasedHandContinuationProvider",
]


@dataclass(frozen=True, slots=True)
class ReleasedHandClearanceRecovery(PlanningRecovery):
    """Fixed source retreat/height search, qualified with one or two exact Parks."""

    kind: str = field(default="released_clearance", init=False)
    skill_id: str = field(default="move_end_effector", init=False)
    revision: int = field(default=1, init=False)


@dataclass(frozen=True, slots=True)
class ReleasedHandParkRecovery(PlanningRecovery):
    """Qualify one explicitly bound released-hand Park and its source joint vias."""

    kind: str = field(default="released_park", init=False)
    skill_id: str = field(default="move_joints", init=False)
    revision: int = field(default=1, init=False)


@runtime_checkable
class ReleasedHandContinuationSession(Protocol):
    """Borrowed exact suffix authority; previews never issue physical commands."""

    @property
    def park_bindings(self) -> tuple[ActionBinding, ...]:
        """Return immutable endpoint bindings in real following-call order."""
        ...

    def resolve_park(
        self, index: int, context: PlanningContext
    ) -> ResolvedActionRequest:
        """Ground the exact suffix occurrence against this private projection."""
        ...

    def plan_park(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        """Build through the same ordinary engine and complete pipeline."""
        ...


@runtime_checkable
class ReleasedHandContinuationProvider(Protocol):
    """Engine-owned authorization for source cleanup and Park occurrences."""

    robot: Robot

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ReleasedHandContinuationSession | None:
        """Return the selected declared suffix or None only when unbound."""
        ...

    def require_park(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> None:
        """Reject a foreign or undeclared released Park before its planning."""
        ...
