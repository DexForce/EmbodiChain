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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .bindings import ActionBinding, ResolvedActionBinding
from .control import ActionControlOverrides
from .goals import ActionGoal
from .policies import MotionPolicy, RecoveryPolicy

GoalT = TypeVar("GoalT", bound=ActionGoal)


@dataclass(frozen=True, slots=True, eq=False)
class ActionOptions:
    """Marker base for immutable, skill-specific runtime options.

    Subclasses belong to action modules and contain only behavior that may vary
    between invocations. Robot resources and semantic targets do not belong in
    this object.
    """


OptionsT = TypeVar("OptionsT", bound=ActionOptions)


@dataclass(frozen=True, slots=True)
class ActionInvocation(Generic[GoalT, OptionsT]):
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
    """Semantic-role bindings to keys in the selected robot's control parts."""

    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)
    """Reusable motion-generation settings."""

    recovery_policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    """Bounded local execution recovery settings."""

    skill_options: OptionsT | None = None
    """Optional per-invocation behavior override for the selected skill."""

    control_overrides: ActionControlOverrides = field(
        default_factory=ActionControlOverrides
    )
    """Optional semantic control commands for this invocation revision."""

    invocation_id: str | None = None
    """Optional correlation identifier propagated into execution traces."""

    revision: int = 0
    """Monotonic revision used when replacing a runtime invocation."""

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
        if self.skill_options is not None and not isinstance(
            self.skill_options, ActionOptions
        ):
            raise TypeError("skill_options must be an ActionOptions instance.")
        if not isinstance(self.control_overrides, ActionControlOverrides):
            raise TypeError("control_overrides must be an ActionControlOverrides.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id.strip()
        ):
            raise ValueError("invocation_id must be a non-empty string when set.")
        if not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class ResolvedActionRequest(Generic[GoalT, OptionsT]):
    """Engine-owned immutable planning snapshot for one invocation revision.

    Recovery replans reuse this object verbatim and vary only the
    :class:`PlanningContext`. Deep-copying policies and skill options severs
    references to caller-owned runtime objects before planning starts.
    """

    skill_id: str
    goal: GoalT
    binding: ResolvedActionBinding
    motion_policy: MotionPolicy
    recovery_policy: RecoveryPolicy
    skill_options: OptionsT
    invocation_id: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string.")
        if not isinstance(self.binding, ResolvedActionBinding):
            raise TypeError("binding must be a ResolvedActionBinding.")
        if not isinstance(self.motion_policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(self.recovery_policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        if not isinstance(self.skill_options, ActionOptions):
            raise TypeError("skill_options must be an ActionOptions instance.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id.strip()
        ):
            raise ValueError("invocation_id must be a non-empty string when set.")
        if not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be a non-negative integer.")
        object.__setattr__(self, "motion_policy", deepcopy(self.motion_policy))
        object.__setattr__(self, "recovery_policy", deepcopy(self.recovery_policy))
        object.__setattr__(self, "skill_options", deepcopy(self.skill_options))


__all__ = [
    "ActionInvocation",
    "ActionOptions",
    "GoalT",
    "OptionsT",
    "ResolvedActionRequest",
]
