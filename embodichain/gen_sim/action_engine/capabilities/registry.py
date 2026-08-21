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

"""Capability registry shared by planning metadata and compilation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from embodichain.gen_sim.action_engine.domain.motion import validate_motion_policy

__all__ = [
    "ActionCapability",
    "ActionTemplate",
    "CapabilityRegistry",
    "OperatorCapability",
    "PhaseTemplate",
]


@dataclass(frozen=True)
class ActionCapability:
    """Describe one public AtomicAction class exposed to the compiler."""

    class_name: str
    target_binding_kinds: frozenset[str]
    controls: frozenset[str]


@dataclass(frozen=True)
class ActionTemplate:
    """Describe one symbolic atomic action before actor materialization."""

    atomic_action_class: str
    target_binding: Mapping[str, Any]
    motion_policy: Mapping[str, Any]
    control: str = "arm"
    actor: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_binding",
            MappingProxyType(dict(self.target_binding)),
        )
        object.__setattr__(
            self,
            "motion_policy",
            MappingProxyType(validate_motion_policy(self.motion_policy)),
        )
        if self.actor is not None:
            object.__setattr__(self, "actor", MappingProxyType(dict(self.actor)))


@dataclass(frozen=True)
class PhaseTemplate:
    """Group atomic actions that execute on one graph edge."""

    name: str
    state_semantic: str
    actions: tuple[ActionTemplate, ...]


ExpandOperator = Callable[[Mapping[str, Any]], list[dict[str, Any]]]
BuildPhases = Callable[[Mapping[str, Any]], Sequence[PhaseTemplate]]


@dataclass(frozen=True)
class OperatorCapability:
    """Bind a semantic operator to deterministic expansion and lowering."""

    name: str
    description: str
    expand: ExpandOperator
    build_phases: BuildPhases
    expansion_topology: str = "serial"
    lifecycle: str = "release"
    planner_visible: bool = True

    def __post_init__(self) -> None:
        if self.expansion_topology not in {"serial", "parallel_children"}:
            raise ValueError(
                "Operator expansion_topology must be 'serial' or "
                "'parallel_children'."
            )
        if self.lifecycle not in {"release", "terminal_hold"}:
            raise ValueError("Operator lifecycle must be 'release' or 'terminal_hold'.")


class CapabilityRegistry:
    """Store explicit operator and atomic-action capabilities.

    Registration is intentionally strict. Replacing a capability by accident
    would silently change compilation semantics, so callers must construct a
    new registry when they need a different definition.
    """

    def __init__(self) -> None:
        self._operators: dict[str, OperatorCapability] = {}
        self._actions: dict[str, ActionCapability] = {}

    def register_operator(self, capability: OperatorCapability) -> None:
        """Register one semantic operator."""
        if capability.name in self._operators:
            raise ValueError(f"Operator {capability.name!r} is already registered.")
        self._operators[capability.name] = capability

    def register_action(self, capability: ActionCapability) -> None:
        """Register one public AtomicAction contract."""
        if capability.class_name in self._actions:
            raise ValueError(
                f"Atomic action {capability.class_name!r} is already registered."
            )
        self._actions[capability.class_name] = capability

    def operator(self, name: str) -> OperatorCapability:
        """Return an operator or raise a capability-focused error."""
        try:
            return self._operators[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown semantic operator {name!r}; available operators are "
                f"{sorted(self._operators)}."
            ) from exc

    def action(self, class_name: str) -> ActionCapability:
        """Return an atomic-action contract or raise a focused error."""
        try:
            return self._actions[class_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown atomic action {class_name!r}; available actions are "
                f"{sorted(self._actions)}."
            ) from exc

    def operator_names(self) -> tuple[str, ...]:
        """Return only the semantic skills exposed to the LLM planner."""
        return tuple(
            sorted(
                name
                for name, capability in self._operators.items()
                if capability.planner_visible
            )
        )

    def operator_descriptions(self) -> dict[str, str]:
        """Return JSON-safe operator descriptions."""
        return {
            name: self._operators[name].description for name in self.operator_names()
        }

    def validate_action_template(self, template: ActionTemplate) -> None:
        """Validate one compiler-produced action against its registered API."""
        capability = self.action(template.atomic_action_class)
        kind = template.target_binding.get("kind")
        if kind not in capability.target_binding_kinds:
            raise ValueError(
                f"{template.atomic_action_class} does not accept target binding "
                f"kind {kind!r}; expected one of "
                f"{sorted(capability.target_binding_kinds)}."
            )
        if template.control not in capability.controls:
            raise ValueError(
                f"{template.atomic_action_class} does not support control "
                f"{template.control!r}; expected one of "
                f"{sorted(capability.controls)}."
            )
