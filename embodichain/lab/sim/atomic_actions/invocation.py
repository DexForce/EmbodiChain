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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Generic, TypeVar

from embodichain.lab.sim.common import BatchEntity

from .bindings import ActionBinding
from .control import ActionControlOverrides
from .policies import MotionPolicy, RecoveryPolicy
from .tracking import TrackingPolicy

GoalT = TypeVar("GoalT")


@dataclass(frozen=True, slots=True, eq=False)
class ActionOptions:
    """Marker base for immutable, skill-specific runtime options.

    Subclasses belong to action modules and contain only behavior that may vary
    between invocations. Robot resources and semantic targets do not belong in
    this object.
    """


OptionsT = TypeVar("OptionsT", bound=ActionOptions)


@dataclass(frozen=True, slots=True)
class PhaseEffectGateRequirement:
    """Require physical-effect evidence before one trajectory segment starts.

    The requirement carries only stable core correlation data. Semantic
    integrations own the corresponding observation specification and monitor;
    the execution session owns blocking, timeout, and action-retry behavior.

    Args:
        gate_id: Invocation-local stable gate identifier.
        segment_name: Exact named trajectory segment blocked by this gate.
    """

    gate_id: str
    segment_name: str

    def __post_init__(self) -> None:
        for name in ("gate_id", "segment_name"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"{name} must be a non-empty string without outer whitespace."
                )

    def snapshot(self) -> PhaseEffectGateRequirement:
        """Return an independently constructed immutable requirement."""
        return PhaseEffectGateRequirement(
            gate_id=self.gate_id,
            segment_name=self.segment_name,
        )


def _goal_snapshot_memo(goal: object) -> dict[int, object]:
    """Return deepcopy memo entries for live goal references and runtime caches."""
    memo: dict[int, object] = {}
    visited: set[int] = set()

    def visit(value: object) -> None:
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if isinstance(value, BatchEntity):
            memo[value_id] = value
            return
        if is_dataclass(value) and not isinstance(value, type):
            for data_field in fields(value):
                nested = getattr(value, data_field.name)
                if not data_field.init and nested is not None:
                    memo[id(nested)] = nested
                else:
                    visit(nested)
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(key)
                visit(nested)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                visit(nested)

    visit(goal)
    return memo


@dataclass(frozen=True, slots=True)
class ActionInvocation(Generic[GoalT, OptionsT]):
    """One fully typed and endpoint-bound atomic skill request.

    This is a runtime-domain object, not the JSON protocol emitted by an MLLM.
    An action compiler is responsible for converting a semantic ``SkillCallSpec``
    into this grounded representation.
    """

    skill_id: str
    """Stable registered skill identifier."""

    goal: GoalT
    """Action-specific goal value object."""

    binding: ActionBinding
    """Generic skill endpoint bindings owned by the selected engine."""

    motion_policy: MotionPolicy = field(default_factory=MotionPolicy)
    """Reusable motion-generation settings."""

    tracking_policy: TrackingPolicy = field(
        default_factory=TrackingPolicy.joint_position
    )
    """Typed in-flight tracking and terminal-acceptance settings."""

    recovery_policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    """Bounded local execution recovery settings."""

    phase_effect_gates: tuple[PhaseEffectGateRequirement, ...] = ()
    """Physical-effect gates enforced at named trajectory-segment entries."""

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
        if not isinstance(self.binding, ActionBinding):
            raise TypeError("binding must be an ActionBinding.")
        if not isinstance(self.motion_policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(self.tracking_policy, TrackingPolicy):
            raise TypeError("tracking_policy must be a TrackingPolicy.")
        if not isinstance(self.recovery_policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        phase_effect_gates = tuple(self.phase_effect_gates)
        if not all(
            type(value) is PhaseEffectGateRequirement for value in phase_effect_gates
        ):
            raise TypeError(
                "phase_effect_gates must contain exact "
                "PhaseEffectGateRequirement values."
            )
        gate_ids = [value.gate_id for value in phase_effect_gates]
        segment_names = [value.segment_name for value in phase_effect_gates]
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("Phase-effect gate IDs must be unique per invocation.")
        if len(set(segment_names)) != len(segment_names):
            raise ValueError(
                "At most one phase-effect gate may block each trajectory segment."
            )
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
        object.__setattr__(
            self,
            "phase_effect_gates",
            tuple(value.snapshot() for value in phase_effect_gates),
        )


@dataclass(frozen=True, slots=True)
class ResolvedActionRequest(Generic[GoalT, OptionsT]):
    """Engine-owned immutable planning snapshot for one invocation revision.

    Recovery replans reuse this object verbatim and vary only the
    :class:`PlanningContext`. Deep-copying goal value payloads, policies, and
    skill options severs caller-owned mutable data before planning starts while
    retaining simulator-backed entity handles and private runtime caches.
    """

    skill_id: str
    goal: GoalT
    binding: ActionBinding
    motion_policy: MotionPolicy
    tracking_policy: TrackingPolicy
    recovery_policy: RecoveryPolicy
    skill_options: OptionsT
    phase_effect_gates: tuple[PhaseEffectGateRequirement, ...] = ()
    invocation_id: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string.")
        if not isinstance(self.binding, ActionBinding):
            raise TypeError("binding must be an ActionBinding.")
        if not isinstance(self.motion_policy, MotionPolicy):
            raise TypeError("motion_policy must be a MotionPolicy.")
        if not isinstance(self.tracking_policy, TrackingPolicy):
            raise TypeError("tracking_policy must be a TrackingPolicy.")
        if not isinstance(self.recovery_policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be a RecoveryPolicy.")
        phase_effect_gates = tuple(self.phase_effect_gates)
        if not all(
            type(value) is PhaseEffectGateRequirement for value in phase_effect_gates
        ):
            raise TypeError(
                "phase_effect_gates must contain exact "
                "PhaseEffectGateRequirement values."
            )
        gate_ids = [value.gate_id for value in phase_effect_gates]
        segment_names = [value.segment_name for value in phase_effect_gates]
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("Phase-effect gate IDs must be unique per request.")
        if len(set(segment_names)) != len(segment_names):
            raise ValueError(
                "At most one phase-effect gate may block each trajectory segment."
            )
        if not isinstance(self.skill_options, ActionOptions):
            raise TypeError("skill_options must be an ActionOptions instance.")
        if self.invocation_id is not None and (
            not isinstance(self.invocation_id, str) or not self.invocation_id.strip()
        ):
            raise ValueError("invocation_id must be a non-empty string when set.")
        if not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("revision must be a non-negative integer.")
        object.__setattr__(
            self,
            "goal",
            deepcopy(self.goal, _goal_snapshot_memo(self.goal)),
        )
        object.__setattr__(
            self,
            "binding",
            ActionBinding(
                owner_id=self.binding.owner_id,
                endpoints=self.binding.endpoints,
            ),
        )
        object.__setattr__(self, "motion_policy", deepcopy(self.motion_policy))
        object.__setattr__(self, "tracking_policy", deepcopy(self.tracking_policy))
        object.__setattr__(self, "recovery_policy", deepcopy(self.recovery_policy))
        object.__setattr__(
            self,
            "phase_effect_gates",
            tuple(value.snapshot() for value in phase_effect_gates),
        )
        object.__setattr__(self, "skill_options", deepcopy(self.skill_options))

    def snapshot(self) -> ResolvedActionRequest[GoalT, OptionsT]:
        """Return an independently owned resolved-request snapshot."""
        return ResolvedActionRequest(
            skill_id=self.skill_id,
            goal=self.goal,
            binding=self.binding,
            motion_policy=self.motion_policy,
            tracking_policy=self.tracking_policy,
            recovery_policy=self.recovery_policy,
            phase_effect_gates=self.phase_effect_gates,
            skill_options=self.skill_options,
            invocation_id=self.invocation_id,
            revision=self.revision,
        )


__all__ = [
    "ActionInvocation",
    "ActionOptions",
    "GoalT",
    "OptionsT",
    "PhaseEffectGateRequirement",
    "ResolvedActionRequest",
]
