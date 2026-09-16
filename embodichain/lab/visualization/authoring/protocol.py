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

"""Immutable protocol objects for browser skill-sequence authoring.

Every value defined here is a frozen, self-validating snapshot safe to pass
between the simulation thread and a browser-facing UI layer. The module is
pure Python and does not import viser or any simulation package, mirroring
:mod:`embodichain.lab.visualization.protocol`.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from enum import Enum, unique
from types import MappingProxyType
from typing import Mapping, Union

__all__ = [
    "AddCard",
    "AuthoringCommand",
    "CompileSequence",
    "ExecuteSequence",
    "ExecutionProgress",
    "MoveCard",
    "RemoveCard",
    "SUPPORTED_SKILL_IDS",
    "SequenceSnapshot",
    "SkillCard",
    "SkillCardState",
    "UpdateCard",
    "freeze_params",
]

SUPPORTED_SKILL_IDS: tuple[str, ...] = ("move_end_effector", "pick_up", "place")
"""Atomic Skills currently supported by the authoring session compiler."""


@unique
class SkillCardState(Enum):
    """Lifecycle state of one skill card inside an authored sequence."""

    UNCONFIGURED = "unconfigured"
    """The card is missing required configuration and cannot be compiled."""

    READY = "ready"
    """The card is fully configured and eligible for compilation."""

    RUNNING = "running"
    """The card's trajectory segment is currently being executed."""

    SUCCEEDED = "succeeded"
    """The card's trajectory segment finished executing."""

    FAILED = "failed"
    """Compilation or execution of the card failed."""


def _require_non_empty(value: object, name: str) -> None:
    """Require ``value`` to be a non-empty string."""
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _require_optional_non_empty(value: object, name: str) -> None:
    """Require ``value`` to be ``None`` or a non-empty string."""
    if value is not None:
        _require_non_empty(value, name)


def _freeze_value(value: object) -> object:
    """Recursively convert mappings and sequences into immutable views."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def freeze_params(params: Mapping[str, object]) -> Mapping[str, object]:
    """Return an immutable, recursively frozen view of a parameter mapping.

    Args:
        params: Parameter dictionary keyed by non-empty string names.

    Returns:
        A read-only mapping whose list and dict values were converted to
        tuples and read-only mappings.

    Raises:
        TypeError: If ``params`` is not a mapping.
        ValueError: If a key is not a non-empty string.
    """
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping of string keys to values.")
    frozen: dict[str, object] = {}
    for key, value in params.items():
        _require_non_empty(key, "params key")
        frozen[key] = _freeze_value(value)
    return MappingProxyType(frozen)


def _validate_optional_segment(start: object, stop: object) -> None:
    """Validate a paired optional half-open waypoint range."""
    if (start is None) != (stop is None):
        raise ValueError("segment_start and segment_stop must be set together.")
    if start is None:
        return
    for name, value in (("segment_start", start), ("segment_stop", stop)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
    if start < 0:
        raise ValueError("segment_start must be non-negative.")
    if stop <= start:
        raise ValueError("segment_stop must be greater than segment_start.")


@dataclass(frozen=True)
class SkillCard:
    """Immutable snapshot of one skill invocation card.

    A card references an installed Atomic Skill, an optional target scene
    entity, and a frozen parameter dictionary. Segment metadata is back-filled
    after a successful compilation and describes the card's half-open waypoint
    range ``[segment_start, segment_stop)`` inside the compiled trajectory.
    """

    card_id: str
    skill_id: str
    entity_uid: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)
    state: SkillCardState = SkillCardState.UNCONFIGURED
    failure_message: str | None = None
    segment_start: int | None = None
    segment_stop: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.card_id, "SkillCard.card_id")
        _require_non_empty(self.skill_id, "SkillCard.skill_id")
        _require_optional_non_empty(self.entity_uid, "SkillCard.entity_uid")
        if not isinstance(self.state, SkillCardState):
            raise TypeError("SkillCard.state must be a SkillCardState.")
        object.__setattr__(self, "params", freeze_params(self.params))
        if self.state is SkillCardState.FAILED:
            _require_non_empty(self.failure_message, "SkillCard.failure_message")
        elif self.failure_message is not None:
            raise ValueError("failure_message is only allowed on FAILED cards.")
        _validate_optional_segment(self.segment_start, self.segment_stop)

    @property
    def segment_waypoint_count(self) -> int:
        """Number of compiled waypoints owned by this card, or zero."""
        if self.segment_start is None or self.segment_stop is None:
            return 0
        return self.segment_stop - self.segment_start


@dataclass(frozen=True)
class SequenceSnapshot:
    """Read-only sequence state published to browser-facing UI layers."""

    cards: tuple[SkillCard, ...]
    compiled: bool
    trajectory_waypoint_count: int = 0

    def __post_init__(self) -> None:
        cards = tuple(self.cards)
        if not all(isinstance(card, SkillCard) for card in cards):
            raise TypeError("SequenceSnapshot cards must be SkillCard instances.")
        card_ids = [card.card_id for card in cards]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("SequenceSnapshot contains duplicate card IDs.")
        if type(self.compiled) is not bool:
            raise TypeError("SequenceSnapshot.compiled must be a bool.")
        count = self.trajectory_waypoint_count
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("trajectory_waypoint_count must be an integer.")
        if count < 0:
            raise ValueError("trajectory_waypoint_count must be non-negative.")
        if not self.compiled and count != 0:
            raise ValueError(
                "trajectory_waypoint_count must be zero when the sequence is "
                "not compiled."
            )
        object.__setattr__(self, "cards", cards)


@dataclass(frozen=True)
class AddCard:
    """Append or insert one new skill card into the sequence."""

    skill_id: str
    card_id: str | None = None
    entity_uid: str | None = None
    params: Mapping[str, object] = field(default_factory=dict)
    index: int | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.skill_id, "AddCard.skill_id")
        _require_optional_non_empty(self.card_id, "AddCard.card_id")
        _require_optional_non_empty(self.entity_uid, "AddCard.entity_uid")
        object.__setattr__(self, "params", freeze_params(self.params))
        if self.index is not None:
            if isinstance(self.index, bool) or not isinstance(self.index, int):
                raise TypeError("AddCard.index must be an integer or None.")
            if self.index < 0:
                raise ValueError("AddCard.index must be non-negative.")


@dataclass(frozen=True)
class RemoveCard:
    """Remove one skill card from the sequence."""

    card_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.card_id, "RemoveCard.card_id")


@dataclass(frozen=True)
class MoveCard:
    """Move one skill card to a new position in the sequence."""

    card_id: str
    new_index: int

    def __post_init__(self) -> None:
        _require_non_empty(self.card_id, "MoveCard.card_id")
        if isinstance(self.new_index, bool) or not isinstance(self.new_index, int):
            raise TypeError("MoveCard.new_index must be an integer.")
        if self.new_index < 0:
            raise ValueError("MoveCard.new_index must be non-negative.")


@dataclass(frozen=True)
class UpdateCard:
    """Update the target entity or parameters of one skill card.

    ``params`` entries are merged over the card's existing parameters. A
    ``None`` ``entity_uid`` leaves the target entity unchanged; set
    ``clear_entity_uid`` to remove it instead.
    """

    card_id: str
    params: Mapping[str, object] | None = None
    entity_uid: str | None = None
    clear_entity_uid: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.card_id, "UpdateCard.card_id")
        _require_optional_non_empty(self.entity_uid, "UpdateCard.entity_uid")
        if type(self.clear_entity_uid) is not bool:
            raise TypeError("UpdateCard.clear_entity_uid must be a bool.")
        if self.entity_uid is not None and self.clear_entity_uid:
            raise ValueError("UpdateCard cannot both set and clear the entity UID.")
        if self.params is not None:
            object.__setattr__(self, "params", freeze_params(self.params))
        if (
            self.params is None
            and self.entity_uid is None
            and not self.clear_entity_uid
        ):
            raise ValueError("UpdateCard must change parameters or the entity UID.")


@dataclass(frozen=True)
class CompileSequence:
    """Compile every configured card into one executable trajectory."""


@dataclass(frozen=True)
class ExecuteSequence:
    """Execute the compiled trajectory on the simulation thread."""

    hold_steps: int = 0
    """Number of final-pose hold updates after the trajectory finishes."""

    def __post_init__(self) -> None:
        if isinstance(self.hold_steps, bool) or not isinstance(self.hold_steps, int):
            raise TypeError("ExecuteSequence.hold_steps must be an integer.")
        if self.hold_steps < 0:
            raise ValueError("ExecuteSequence.hold_steps must be non-negative.")


AuthoringCommand = Union[
    AddCard,
    RemoveCard,
    MoveCard,
    UpdateCard,
    CompileSequence,
    ExecuteSequence,
]
"""Union of every command accepted by ``AuthoringSession.handle_command``."""


@dataclass(frozen=True)
class ExecutionProgress:
    """Immutable progress report of one stepwise execution tick.

    One value is produced after every simulation update performed by
    :meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.execute_stepwise`.
    Trajectory ticks and final-pose hold ticks count separately, so
    ``step_index`` is always an index into ``total_steps`` of the reported
    phase.
    """

    step_index: int
    total_steps: int
    holding: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("step_index", self.step_index),
            ("total_steps", self.total_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ExecutionProgress.{name} must be an integer.")
        if self.total_steps <= 0:
            raise ValueError("ExecutionProgress.total_steps must be positive.")
        if not 0 <= self.step_index < self.total_steps:
            raise ValueError(
                "ExecutionProgress.step_index must be inside "
                f"[0, {self.total_steps})."
            )
        if type(self.holding) is not bool:
            raise TypeError("ExecutionProgress.holding must be a bool.")

    @property
    def fraction(self) -> float:
        """Completed share of the reported phase in ``(0, 1]``."""
        return (self.step_index + 1) / self.total_steps


def is_finite_number(value: object) -> bool:
    """Return whether ``value`` is a real, finite, non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    number = float(value)
    return number == number and number not in (float("inf"), float("-inf"))
