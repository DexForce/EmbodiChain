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

"""Grounded action invocations consumed by the deterministic skill layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .bindings import ActionBinding
from .goals import ActionGoal
from .policies import MotionPolicy, RecoveryPolicy

GoalT = TypeVar("GoalT", bound=ActionGoal)


@dataclass(frozen=True, slots=True)
class ActionInvocation(Generic[GoalT]):
    """One fully typed and embodiment-bound atomic skill request.

    This is a runtime-domain object, not the JSON protocol emitted by an MLLM.
    An action compiler is responsible for converting a semantic ``SkillCallSpec``
    into this grounded representation.
    """

    skill_id: str
    """Stable registered skill identifier."""

    goal: GoalT
    """Action-specific goal value object."""

    binding: ActionBinding
    """Semantic-role bindings for the selected robot embodiment."""

    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)
    """Reusable motion-generation settings."""

    recovery_policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    """Bounded local execution recovery settings."""

    invocation_id: str | None = None
    """Optional correlation identifier propagated into execution traces."""

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string.")
        goal_kind = getattr(type(self.goal), "goal_kind", None)
        if not isinstance(goal_kind, str) or not goal_kind:
            raise TypeError(
                "goal must implement the ActionGoal protocol with a non-empty "
                "goal_kind class variable."
            )
        if not isinstance(self.binding, ActionBinding):
            raise TypeError("binding must be an ActionBinding.")
        if not isinstance(self.motion_policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(self.recovery_policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id.strip()
        ):
            raise ValueError("invocation_id must be a non-empty string when set.")


__all__ = ["ActionInvocation", "GoalT"]
