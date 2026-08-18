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

"""Semantic-role to robot control-part bindings for atomic actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

from .control import ControlCommand, JointPositionCommand


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
    """Bind semantic action roles to names from ``Robot.control_parts``.

    A role such as ``primary``, ``source`` or ``destination`` is an
    action-defined semantic participant slot. It describes the responsibility
    a resource has within that action and is not itself a robot resource.
    Actions publish their required slots through ``manipulator_roles`` and
    ``end_effector_roles``. Role names are scoped independently to those two
    maps, so matching names associate an arm and hand/tool with the same
    functional participant without making the maps interchangeable.

    ``primary`` has no inherent left/right, ordering, or default-control-part
    meaning. Only the compiler or application binding layer needs to map it to
    concrete robot control-part names such as ``left_arm`` and ``left_hand``.

    Every mapping value is a key from the current robot's ``control_parts``
    configuration. This value object validates the mapping shape; the
    :class:`~embodichain.lab.sim.atomic_actions.AtomicActionEngine` validates
    the names against its owned robot before planning. ``end_effectors`` refers
    to actuated tool/hand control parts, not TCP or kinematic frame names.
    """

    manipulators: Mapping[str, str] = field(default_factory=dict)
    """Manipulator control-part names keyed by semantic role."""

    end_effectors: Mapping[str, str] = field(default_factory=dict)
    """Tool or hand control-part names keyed by semantic role."""

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
        """Return the manipulator control-part name bound to ``role``.

        Args:
            role: Semantic manipulator role.

        Returns:
            Key from the current robot's ``control_parts`` mapping.

        Raises:
            KeyError: If the requested role is not bound.
        """
        try:
            return self.manipulators[role]
        except KeyError as exc:
            raise KeyError(f"No manipulator is bound to role {role!r}.") from exc

    def end_effector(self, role: str = "primary") -> str:
        """Return the tool/hand control-part name bound to ``role``.

        Args:
            role: Semantic end-effector role.

        Returns:
            Key from the current robot's ``control_parts`` mapping.

        Raises:
            KeyError: If the requested role is not bound.
        """
        try:
            return self.end_effectors[role]
        except KeyError as exc:
            raise KeyError(f"No end effector is bound to role {role!r}.") from exc


@dataclass(frozen=True, slots=True)
class ResolvedControlPart:
    """One engine-validated robot control part.

    Instances are produced by engine-owned planning services. They keep
    robot-specific indices out of :class:`ActionBinding` and agent-facing
    invocation schemas.
    """

    name: str
    """Key from ``Robot.control_parts``."""

    joint_ids: tuple[int, ...]
    """Full-robot joint indices belonging to this control part."""

    commands: Mapping[str, ControlCommand] = field(default_factory=dict)
    """Engine-profile commands, including invocation-level overrides."""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ResolvedControlPart.name must be a non-empty string.")
        joint_ids = tuple(self.joint_ids)
        if not joint_ids or not all(
            isinstance(joint_id, int) and joint_id >= 0 for joint_id in joint_ids
        ):
            raise ValueError(
                "ResolvedControlPart.joint_ids must contain non-negative integers."
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("ResolvedControlPart.joint_ids must be unique.")
        object.__setattr__(self, "joint_ids", joint_ids)
        if not isinstance(self.commands, Mapping):
            raise TypeError("ResolvedControlPart.commands must be a mapping.")
        commands: dict[str, ControlCommand] = {}
        for name, command in self.commands.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Control command names must be non-empty strings.")
            if not isinstance(command, ControlCommand):
                raise TypeError(
                    "ResolvedControlPart.commands values must be ControlCommand "
                    "instances."
                )
            commands[name] = command.snapshot()
        object.__setattr__(self, "commands", MappingProxyType(commands))

    @property
    def dof(self) -> int:
        """Return the number of joints in this control part."""
        return len(self.joint_ids)

    def with_command_overrides(
        self,
        overrides: Mapping[str, ControlCommand],
    ) -> ResolvedControlPart:
        """Return a snapshot with role-local semantic command overrides."""
        merged = dict(self.commands)
        merged.update(overrides)
        return ResolvedControlPart(
            name=self.name,
            joint_ids=self.joint_ids,
            commands=merged,
        )

    def command(self, name: str) -> ControlCommand:
        """Return an owned semantic command snapshot.

        Args:
            name: Semantic command name, for example ``open`` or ``grasp``.

        Raises:
            KeyError: If this control part does not define ``name``.
        """
        try:
            command = self.commands[name]
        except KeyError as exc:
            raise KeyError(
                f"Control part {self.name!r} has no command {name!r}. "
                f"Available commands: {sorted(self.commands)}."
            ) from exc
        return command.snapshot()

    def joint_positions(
        self,
        name: str,
        *,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Resolve a named joint-position command for a planning batch."""
        try:
            command = self.commands[name]
        except KeyError as exc:
            raise KeyError(
                f"Control part {self.name!r} has no command {name!r}. "
                f"Available commands: {sorted(self.commands)}."
            ) from exc
        if not isinstance(command, JointPositionCommand):
            raise TypeError(
                f"Control command {name!r} on {self.name!r} is "
                f"{type(command).__name__}, not JointPositionCommand."
            )
        return command.resolve(
            num_envs=num_envs,
            control_dof=self.dof,
            device=device,
            dtype=dtype,
        )


def _normalize_resolved_map(
    values: Mapping[str, ResolvedControlPart],
    *,
    field_name: str,
) -> Mapping[str, ResolvedControlPart]:
    """Validate and freeze a resolved semantic-role mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized: dict[str, ResolvedControlPart] = {}
    for role, resource in values.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{field_name} roles must be non-empty strings.")
        if not isinstance(resource, ResolvedControlPart):
            raise TypeError(
                f"{field_name} values must be ResolvedControlPart instances."
            )
        normalized[role] = resource
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ResolvedActionBinding:
    """Runtime control parts resolved from an :class:`ActionBinding`."""

    manipulators: Mapping[str, ResolvedControlPart] = field(default_factory=dict)
    end_effectors: Mapping[str, ResolvedControlPart] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manipulators",
            _normalize_resolved_map(
                self.manipulators, field_name="resolved manipulators"
            ),
        )
        object.__setattr__(
            self,
            "end_effectors",
            _normalize_resolved_map(
                self.end_effectors, field_name="resolved end_effectors"
            ),
        )

    def manipulator(self, role: str = "primary") -> ResolvedControlPart:
        """Return the resolved manipulator for ``role``."""
        try:
            return self.manipulators[role]
        except KeyError as exc:
            raise KeyError(f"No manipulator is bound to role {role!r}.") from exc

    def end_effector(self, role: str = "primary") -> ResolvedControlPart:
        """Return the resolved tool/hand control part for ``role``."""
        try:
            return self.end_effectors[role]
        except KeyError as exc:
            raise KeyError(f"No end effector is bound to role {role!r}.") from exc


__all__ = ["ActionBinding", "ResolvedActionBinding", "ResolvedControlPart"]
