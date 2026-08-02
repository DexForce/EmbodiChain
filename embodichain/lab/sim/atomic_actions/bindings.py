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

"""Semantic-role to robot-resource bindings for atomic actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def _normalize_resource_map(
    values: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    """Validate and freeze a semantic-role resource mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized: dict[str, str] = {}
    for role, resource in values.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{field_name} roles must be non-empty strings.")
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError(f"{field_name} resources must be non-empty strings.")
        normalized[role] = resource
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ActionBinding:
    """Bind semantic action roles to embodiment-specific control resources.

    The action and an agent-facing request refer to roles such as ``primary``,
    ``source`` and ``destination``. Only the compiler or application binding
    layer needs to know concrete robot resources such as ``left_arm``.
    """

    manipulators: Mapping[str, str] = field(default_factory=dict)
    """Manipulator resources keyed by semantic role."""

    end_effectors: Mapping[str, str] = field(default_factory=dict)
    """End-effector resources keyed by semantic role."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manipulators",
            _normalize_resource_map(self.manipulators, field_name="manipulators"),
        )
        object.__setattr__(
            self,
            "end_effectors",
            _normalize_resource_map(self.end_effectors, field_name="end_effectors"),
        )

    def manipulator(self, role: str = "primary") -> str:
        """Return the manipulator resource bound to ``role``.

        Args:
            role: Semantic manipulator role.

        Returns:
            Concrete robot control-resource name.

        Raises:
            KeyError: If the requested role is not bound.
        """
        try:
            return self.manipulators[role]
        except KeyError as exc:
            raise KeyError(f"No manipulator is bound to role {role!r}.") from exc

    def end_effector(self, role: str = "primary") -> str:
        """Return the end-effector resource bound to ``role``.

        Args:
            role: Semantic end-effector role.

        Returns:
            Concrete robot control-resource name.

        Raises:
            KeyError: If the requested role is not bound.
        """
        try:
            return self.end_effectors[role]
        except KeyError as exc:
            raise KeyError(f"No end effector is bound to role {role!r}.") from exc


__all__ = ["ActionBinding"]
