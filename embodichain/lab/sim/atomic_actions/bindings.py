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

"""Generic runtime endpoint bindings consumed by atomic actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeVar

import torch

from .control import ControlCommand


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Validate and return one strict identifier."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _normalize_identifiers(
    values: frozenset[str],
    *,
    field_name: str,
) -> frozenset[str]:
    """Validate and freeze an identifier set."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of strings.")
    try:
        normalized = frozenset(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    for value in normalized:
        _validate_identifier(value, field_name=field_name)
    return normalized


def _snapshot_commands(
    values: Mapping[str, ControlCommand],
) -> Mapping[str, ControlCommand]:
    """Validate semantic endpoint commands and own their snapshots."""
    if not isinstance(values, Mapping):
        raise TypeError("EndpointBinding.commands must be a mapping.")
    commands: dict[str, ControlCommand] = {}
    for name, command in values.items():
        _validate_identifier(name, field_name="EndpointBinding command names")
        if not isinstance(command, ControlCommand):
            raise TypeError(
                "EndpointBinding.commands values must be ControlCommand instances."
            )
        snapshot = command.snapshot()
        if type(snapshot) is not type(command) or snapshot is command:
            raise TypeError(
                "ControlCommand.snapshot() must return an independently owned "
                "value of the same command type."
            )
        commands[name] = snapshot
    return MappingProxyType(commands)


def _validate_target_fingerprint(
    target: RuntimeEndpointTarget,
    *,
    field_name: str,
) -> Hashable:
    """Return one hashable, snapshot-stable target address fingerprint."""
    fingerprint = target.address_fingerprint
    try:
        hash(fingerprint)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be hashable.") from exc
    return fingerprint


class RuntimeEndpointTarget(ABC):
    """Stable controller destination produced by an endpoint adapter.

    Targets contain immutable addressing data only. Live controllers, sockets,
    simulator entities, and other process-owned handles belong to an
    endpoint-command transport rather than this value.
    """

    @property
    @abstractmethod
    def transport_id(self) -> str:
        """Return the registered transport kind used by this target."""

    @property
    @abstractmethod
    def target_id(self) -> str:
        """Return the destination identifier within its transport."""

    @property
    def address_fingerprint(self) -> Hashable:
        """Return the stable controller-address and safe-hold fingerprint.

        The default covers the exact target type and transport-scoped
        destination. Target types whose hold footprint depends on additional
        immutable addressing fields must override this property and include
        those fields. Replans and explicit revisions may replace payloads, but
        they may not change this fingerprint in place.
        """
        return type(self), self.transport_id, self.target_id

    def snapshot(self) -> RuntimeEndpointTarget:
        """Return an independently owned target snapshot."""
        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class JointPositionTarget(RuntimeEndpointTarget):
    """Joint-position destination backed by one robot control part."""

    TRANSPORT_ID = "robot.joint_position"

    control_part: str
    joint_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_identifier(
            self.control_part,
            field_name="JointPositionTarget.control_part",
        )
        joint_ids = tuple(self.joint_ids)
        if not joint_ids or not all(
            isinstance(joint_id, int)
            and not isinstance(joint_id, bool)
            and joint_id >= 0
            for joint_id in joint_ids
        ):
            raise ValueError(
                "JointPositionTarget.joint_ids must contain non-negative integers."
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("JointPositionTarget.joint_ids must be unique.")
        object.__setattr__(self, "joint_ids", joint_ids)

    @property
    def transport_id(self) -> str:
        """Return the built-in joint-position transport identifier."""
        return self.TRANSPORT_ID

    @property
    def target_id(self) -> str:
        """Return the robot control-part destination."""
        return self.control_part

    @property
    def address_fingerprint(self) -> Hashable:
        """Return the destination plus the joints that must remain holdable."""
        return (
            type(self),
            self.transport_id,
            self.target_id,
            self.joint_ids,
        )


TargetT = TypeVar("TargetT", bound=RuntimeEndpointTarget)


@dataclass(frozen=True, slots=True)
class EndpointBinding:
    """One action-local endpoint resolved to a runtime controller target."""

    slot_id: str
    endpoint_id: str
    resource_id: str
    adapter_id: str
    target: RuntimeEndpointTarget
    capabilities: frozenset[str] = frozenset()
    commands: Mapping[str, ControlCommand] = field(default_factory=dict)
    claim_tokens: frozenset[str] = frozenset()
    joint_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.slot_id, field_name="EndpointBinding.slot_id")
        _validate_identifier(
            self.endpoint_id,
            field_name="EndpointBinding.endpoint_id",
        )
        _validate_identifier(
            self.resource_id,
            field_name="EndpointBinding.resource_id",
        )
        _validate_identifier(self.adapter_id, field_name="EndpointBinding.adapter_id")
        if not isinstance(self.target, RuntimeEndpointTarget):
            raise TypeError("EndpointBinding.target must be a RuntimeEndpointTarget.")
        target = self.target.snapshot()
        if type(target) is not type(self.target) or target is self.target:
            raise TypeError(
                "RuntimeEndpointTarget.snapshot() must return an independently "
                "owned value of the same target type."
            )
        _validate_identifier(
            target.transport_id,
            field_name="RuntimeEndpointTarget.transport_id",
        )
        _validate_identifier(
            target.target_id,
            field_name="RuntimeEndpointTarget.target_id",
        )
        source_fingerprint = _validate_target_fingerprint(
            self.target,
            field_name="RuntimeEndpointTarget.address_fingerprint",
        )
        target_fingerprint = _validate_target_fingerprint(
            target,
            field_name="RuntimeEndpointTarget.snapshot().address_fingerprint",
        )
        if target_fingerprint != source_fingerprint:
            raise ValueError(
                "RuntimeEndpointTarget.snapshot() must preserve its address "
                "fingerprint."
            )
        object.__setattr__(self, "target", target)
        object.__setattr__(
            self,
            "capabilities",
            _normalize_identifiers(
                self.capabilities,
                field_name="EndpointBinding.capabilities",
            ),
        )
        object.__setattr__(self, "commands", _snapshot_commands(self.commands))
        object.__setattr__(
            self,
            "claim_tokens",
            _normalize_identifiers(
                self.claim_tokens,
                field_name="EndpointBinding.claim_tokens",
            ),
        )
        joint_ids = tuple(self.joint_ids)
        if not all(
            isinstance(joint_id, int)
            and not isinstance(joint_id, bool)
            and joint_id >= 0
            for joint_id in joint_ids
        ):
            raise ValueError(
                "EndpointBinding.joint_ids must contain non-negative integers."
            )
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError("EndpointBinding.joint_ids must be unique.")
        if isinstance(target, JointPositionTarget):
            if joint_ids and joint_ids != target.joint_ids:
                raise ValueError(
                    "EndpointBinding.joint_ids must match its JointPositionTarget."
                )
            joint_ids = target.joint_ids
        object.__setattr__(self, "joint_ids", joint_ids)

    @property
    def key(self) -> tuple[str, str]:
        """Return the action-local ``(slot, endpoint)`` key."""
        return self.slot_id, self.endpoint_id

    @property
    def destination_key(self) -> tuple[str, str]:
        """Return the transport-scoped physical destination key."""
        return self.target.transport_id, self.target.target_id

    def require_target(self, target_type: type[TargetT]) -> TargetT:
        """Return the runtime target after an explicit type check."""
        if not isinstance(target_type, type) or not issubclass(
            target_type, RuntimeEndpointTarget
        ):
            raise TypeError("target_type must be a RuntimeEndpointTarget subclass.")
        if not isinstance(self.target, target_type):
            raise TypeError(
                f"Endpoint {self.slot_id}.{self.endpoint_id} uses "
                f"{type(self.target).__name__}, expected {target_type.__name__}."
            )
        return self.target.snapshot()

    def command(self, name: str) -> ControlCommand:
        """Return one owned semantic-command snapshot."""
        try:
            command = self.commands[name]
        except KeyError as exc:
            raise KeyError(
                f"Endpoint {self.slot_id}.{self.endpoint_id} has no command "
                f"{name!r}; available commands are {sorted(self.commands)}."
            ) from exc
        return command.snapshot()

    def joint_positions(
        self,
        name: str,
        *,
        n_envs: int,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Resolve a named joint-position command for a planning batch."""
        from .control import JointPositionCommand

        target = self.require_target(JointPositionTarget)
        command = self.command(name)
        if not isinstance(command, JointPositionCommand):
            raise TypeError(
                f"Endpoint command {name!r} is {type(command).__name__}, not "
                "JointPositionCommand."
            )
        return command.resolve(
            n_envs=n_envs,
            control_dof=len(target.joint_ids),
            device=device,
            dtype=dtype,
        )

    def with_commands(
        self,
        overrides: Mapping[str, ControlCommand],
    ) -> EndpointBinding:
        """Return an endpoint snapshot with semantic-command overrides."""
        merged = dict(self.commands)
        merged.update(overrides)
        return EndpointBinding(
            slot_id=self.slot_id,
            endpoint_id=self.endpoint_id,
            resource_id=self.resource_id,
            adapter_id=self.adapter_id,
            target=self.target,
            capabilities=self.capabilities,
            commands=merged,
            claim_tokens=self.claim_tokens,
            joint_ids=self.joint_ids,
        )

    def snapshot(self) -> EndpointBinding:
        """Return an independently owned endpoint-binding snapshot."""
        return EndpointBinding(
            slot_id=self.slot_id,
            endpoint_id=self.endpoint_id,
            resource_id=self.resource_id,
            adapter_id=self.adapter_id,
            target=self.target,
            capabilities=self.capabilities,
            commands=self.commands,
            claim_tokens=self.claim_tokens,
            joint_ids=self.joint_ids,
        )


@dataclass(frozen=True, slots=True)
class ActionBinding:
    """Engine-owned generic endpoint bindings for one atomic action call."""

    owner_id: str
    endpoints: tuple[EndpointBinding, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.owner_id, field_name="ActionBinding.owner_id")
        if isinstance(self.endpoints, (str, bytes)):
            raise TypeError("ActionBinding.endpoints must be an iterable.")
        try:
            endpoints = tuple(self.endpoints)
        except TypeError as exc:
            raise TypeError("ActionBinding.endpoints must be an iterable.") from exc
        if not all(isinstance(endpoint, EndpointBinding) for endpoint in endpoints):
            raise TypeError(
                "ActionBinding.endpoints values must be EndpointBinding instances."
            )
        keys = [endpoint.key for endpoint in endpoints]
        if len(set(keys)) != len(keys):
            raise ValueError("ActionBinding endpoint keys must be unique.")
        snapshots = tuple(endpoint.snapshot() for endpoint in endpoints)
        object.__setattr__(self, "endpoints", snapshots)

    @property
    def endpoint_keys(self) -> tuple[tuple[str, str], ...]:
        """Return action-local endpoint keys in binding order."""
        return tuple(endpoint.key for endpoint in self.endpoints)

    @property
    def targets(self) -> tuple[RuntimeEndpointTarget, ...]:
        """Return unique owned runtime targets in binding order."""
        targets: list[RuntimeEndpointTarget] = []
        seen: set[tuple[str, str]] = set()
        for endpoint in self.endpoints:
            if endpoint.destination_key in seen:
                continue
            seen.add(endpoint.destination_key)
            targets.append(endpoint.target.snapshot())
        return tuple(targets)

    def endpoint(
        self,
        slot_id: str,
        endpoint_id: str,
    ) -> EndpointBinding:
        """Return one action-local resolved endpoint."""
        key = (slot_id, endpoint_id)
        for endpoint in self.endpoints:
            if endpoint.key == key:
                return endpoint.snapshot()
        raise KeyError(
            f"No endpoint is bound to {slot_id}.{endpoint_id}; available endpoints "
            f"are {list(self.endpoint_keys)}."
        )

    def with_command_overrides(
        self,
        overrides: Mapping[tuple[str, str], Mapping[str, ControlCommand]],
    ) -> ActionBinding:
        """Return a binding snapshot with endpoint-scoped command overrides."""
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be a mapping.")
        unknown = set(overrides).difference(self.endpoint_keys)
        if unknown:
            raise KeyError(
                f"Command overrides reference unbound endpoints {sorted(unknown)}."
            )
        return ActionBinding(
            owner_id=self.owner_id,
            endpoints=tuple(
                (
                    endpoint.with_commands(overrides[endpoint.key])
                    if endpoint.key in overrides
                    else endpoint
                )
                for endpoint in self.endpoints
            ),
        )


__all__ = [
    "ActionBinding",
    "EndpointBinding",
    "JointPositionTarget",
    "RuntimeEndpointTarget",
]
