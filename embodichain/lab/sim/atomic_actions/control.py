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

"""Semantic command profiles for robot control parts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

OPEN_COMMAND = "open"
"""Conventional semantic command for an open end effector."""

GRASP_COMMAND = "grasp"
"""Conventional semantic command for an object-holding end effector."""


class ControlCommand(ABC):
    """Immutable-by-ownership command associated with one control part.

    Command subclasses own their payload and must return another owned value
    from :meth:`snapshot`. This keeps engine profiles and resolved invocation
    requests isolated from caller-owned mutable tensors.
    """

    @abstractmethod
    def snapshot(self) -> ControlCommand:
        """Return an independently owned copy of this command."""

    @abstractmethod
    def equivalent_to(self, other: ControlCommand) -> bool:
        """Return whether ``other`` has exactly the same command semantics."""


@dataclass(frozen=True, slots=True, eq=False, init=False)
class JointPositionCommand(ControlCommand):
    """A semantic command represented by one or batched joint positions.

    ``positions`` has shape ``(control_dof,)`` or
    ``(n_envs, control_dof)``. A one-dimensional command is broadcast to the
    planning batch when resolved.
    """

    _positions: torch.Tensor

    def __init__(self, positions: torch.Tensor) -> None:
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor.")
        if positions.dim() not in (1, 2) or positions.shape[-1] == 0:
            raise ValueError(
                "positions must have shape (control_dof,) or "
                "(n_envs, control_dof), got "
                f"{tuple(positions.shape)}."
            )
        if not torch.isfinite(positions).all().item():
            raise ValueError("positions must contain only finite values.")
        object.__setattr__(self, "_positions", positions.detach().clone())

    @property
    def positions(self) -> torch.Tensor:
        """Return an owned copy of the command payload."""
        return self._positions.clone()

    def snapshot(self) -> JointPositionCommand:
        """Return an independently owned command snapshot."""
        return JointPositionCommand(self._positions)

    def equivalent_to(self, other: ControlCommand) -> bool:
        """Return whether ``other`` owns identical joint positions."""
        return isinstance(other, JointPositionCommand) and self._positions.equal(
            other._positions
        )

    def resolve(
        self,
        *,
        n_envs: int,
        control_dof: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Validate, move, and broadcast this command for a planning batch.

        Args:
            n_envs: Number of selected environments.
            control_dof: Joint count of the resolved control part.
            device: Target planning device.
            dtype: Optional target dtype.

        Returns:
            Independently owned tensor with shape ``(n_envs, control_dof)``.

        Raises:
            ValueError: If the command shape does not match the control part or
                selected environment batch.
        """
        if not isinstance(n_envs, int) or n_envs < 1:
            raise ValueError("n_envs must be a positive integer.")
        if not isinstance(control_dof, int) or control_dof < 1:
            raise ValueError("control_dof must be a positive integer.")
        if self._positions.shape[-1] != control_dof:
            raise ValueError(
                f"Joint-position command has {self._positions.shape[-1]} joints, "
                f"but the resolved control part has {control_dof}."
            )
        resolved = self._positions.to(device=device, dtype=dtype)
        if resolved.dim() == 1:
            return resolved.unsqueeze(0).expand(n_envs, -1).clone()
        if resolved.shape[0] != n_envs:
            raise ValueError(
                f"Batched joint-position command has {resolved.shape[0]} "
                f"environments, expected {n_envs}."
            )
        return resolved.clone()


def _snapshot_commands(
    commands: Mapping[str, ControlCommand],
    *,
    field_name: str,
) -> Mapping[str, ControlCommand]:
    """Validate and freeze a semantic command mapping."""
    if not isinstance(commands, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    snapshots: dict[str, ControlCommand] = {}
    for name, command in commands.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(
                f"{field_name} keys must be non-empty strings without outer "
                "whitespace."
            )
        if not isinstance(command, ControlCommand):
            raise TypeError(f"{field_name} values must be ControlCommand instances.")
        snapshot = command.snapshot()
        if type(snapshot) is not type(command) or snapshot is command:
            raise TypeError(
                f"{field_name}[{name!r}].snapshot() must return an independently "
                "owned value of the same ControlCommand type."
            )
        snapshots[name] = snapshot
    return MappingProxyType(snapshots)


@dataclass(frozen=True, slots=True)
class ControlPartCommandProfile:
    """Reusable semantic commands for one named robot control part.

    Profiles are registered once on :class:`AtomicActionEngine`, keyed by
    names from ``Robot.control_parts``. They describe embodiment-specific
    meanings such as ``open``, ``grasp`` or ``ready`` without coupling those
    values to an action implementation.
    """

    commands: Mapping[str, ControlCommand] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "commands",
            _snapshot_commands(self.commands, field_name="commands"),
        )

    @classmethod
    def joint_positions(
        cls,
        **commands: torch.Tensor,
    ) -> ControlPartCommandProfile:
        """Build a profile whose entries are joint-position commands."""
        return cls(
            commands={
                name: JointPositionCommand(positions)
                for name, positions in commands.items()
            }
        )

    def snapshot(self) -> ControlPartCommandProfile:
        """Return an independently owned profile snapshot."""
        return ControlPartCommandProfile(commands=self.commands)


def _snapshot_endpoint_commands(
    values: Mapping[str, Mapping[str, Mapping[str, ControlCommand]]],
    *,
    field_name: str,
) -> Mapping[str, Mapping[str, Mapping[str, ControlCommand]]]:
    """Validate and freeze slot/endpoint-scoped command overrides."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    slots: dict[str, Mapping[str, Mapping[str, ControlCommand]]] = {}
    for slot_id, endpoints in values.items():
        if not isinstance(slot_id, str) or not slot_id or slot_id != slot_id.strip():
            raise ValueError(
                f"{field_name} slot IDs must be non-empty strings without outer "
                "whitespace."
            )
        if not isinstance(endpoints, Mapping):
            raise TypeError(f"{field_name}[{slot_id!r}] must be a mapping.")
        endpoint_snapshots: dict[str, Mapping[str, ControlCommand]] = {}
        for endpoint_id, commands in endpoints.items():
            if (
                not isinstance(endpoint_id, str)
                or not endpoint_id
                or endpoint_id != endpoint_id.strip()
            ):
                raise ValueError(
                    f"{field_name} endpoint IDs must be non-empty strings without "
                    "outer whitespace."
                )
            endpoint_snapshots[endpoint_id] = _snapshot_commands(
                commands,
                field_name=f"{field_name}[{slot_id!r}][{endpoint_id!r}]",
            )
        slots[slot_id] = MappingProxyType(endpoint_snapshots)
    return MappingProxyType(slots)


@dataclass(frozen=True, slots=True)
class ActionControlOverrides:
    """Per-invocation semantic commands keyed by slot and endpoint.

    The first two keys match a skill's ``(slot_id, endpoint_id)`` contract.
    The innermost mapping contains semantic command names. Overrides are
    captured in the invocation revision's immutable planning snapshot.
    """

    endpoints: Mapping[
        str,
        Mapping[str, Mapping[str, ControlCommand]],
    ] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "endpoints",
            _snapshot_endpoint_commands(
                self.endpoints,
                field_name="endpoints",
            ),
        )

    @property
    def is_empty(self) -> bool:
        """Whether this invocation defines no command overrides."""
        return not self.endpoints

    def as_flat_mapping(
        self,
    ) -> Mapping[tuple[str, str], Mapping[str, ControlCommand]]:
        """Return immutable overrides keyed by ``(slot_id, endpoint_id)``."""
        return MappingProxyType(
            {
                (slot_id, endpoint_id): commands
                for slot_id, endpoints in self.endpoints.items()
                for endpoint_id, commands in endpoints.items()
            }
        )


__all__ = [
    "ActionControlOverrides",
    "ControlCommand",
    "ControlPartCommandProfile",
    "GRASP_COMMAND",
    "JointPositionCommand",
    "OPEN_COMMAND",
]
