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

"""Transport-neutral runtime command values for atomic actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from .bindings import JointPositionTarget, RuntimeEndpointTarget


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Validate and return one strict identifier."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _snapshot_target(target: RuntimeEndpointTarget) -> RuntimeEndpointTarget:
    """Validate and own one runtime target snapshot."""
    if not isinstance(target, RuntimeEndpointTarget):
        raise TypeError("target must be a RuntimeEndpointTarget.")
    snapshot = target.snapshot()
    if type(snapshot) is not type(target) or snapshot is target:
        raise TypeError(
            "RuntimeEndpointTarget.snapshot() must return an independently owned "
            "value of the same target type."
        )
    _validate_identifier(
        snapshot.transport_id,
        field_name="RuntimeEndpointTarget.transport_id",
    )
    _validate_identifier(
        snapshot.target_id,
        field_name="RuntimeEndpointTarget.target_id",
    )
    source_fingerprint = target.address_fingerprint
    snapshot_fingerprint = snapshot.address_fingerprint
    try:
        hash(source_fingerprint)
        hash(snapshot_fingerprint)
    except TypeError as exc:
        raise TypeError(
            "RuntimeEndpointTarget.address_fingerprint must be hashable."
        ) from exc
    if snapshot_fingerprint != source_fingerprint:
        raise ValueError(
            "RuntimeEndpointTarget.snapshot() must preserve its address fingerprint."
        )
    return snapshot


class RuntimeCommandPayload(ABC):
    """Immutable-by-ownership payload submitted to one runtime transport."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Return the number of environment rows in this payload."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device shared by this payload's batched values."""

    @property
    @abstractmethod
    def transport_id(self) -> str:
        """Return the transport kind that accepts this payload."""

    @abstractmethod
    def snapshot(self) -> RuntimeCommandPayload:
        """Return an independently owned payload snapshot."""


def _validate_payload_metadata(payload: RuntimeCommandPayload) -> None:
    """Validate transport-neutral payload metadata."""
    if (
        not isinstance(payload.batch_size, int)
        or isinstance(payload.batch_size, bool)
        or payload.batch_size < 1
    ):
        raise ValueError("RuntimeCommandPayload.batch_size must be a positive integer.")
    if not isinstance(payload.device, torch.device):
        raise TypeError("RuntimeCommandPayload.device must be a torch.device.")
    _validate_identifier(
        payload.transport_id,
        field_name="RuntimeCommandPayload.transport_id",
    )


def _snapshot_payload(payload: RuntimeCommandPayload) -> RuntimeCommandPayload:
    """Validate and own one runtime payload snapshot."""
    if not isinstance(payload, RuntimeCommandPayload):
        raise TypeError("payload must be a RuntimeCommandPayload.")
    snapshot = payload.snapshot()
    if type(snapshot) is not type(payload) or snapshot is payload:
        raise TypeError(
            "RuntimeCommandPayload.snapshot() must return an independently owned "
            "value of the same payload type."
        )
    _validate_payload_metadata(snapshot)
    return snapshot


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionPayload(RuntimeCommandPayload):
    """Batched joint-position targets for the built-in robot transport.

    Args:
        positions: Joint positions with shape ``(batch_size, control_dof)``.
        velocities: Optional joint velocities with the same shape and device.
    """

    TRANSPORT_ID: ClassVar[str] = JointPositionTarget.TRANSPORT_ID

    positions: torch.Tensor
    velocities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor.")
        if (
            self.positions.dim() != 2
            or self.positions.shape[0] < 1
            or self.positions.shape[1] < 1
        ):
            raise ValueError(
                "positions must have shape (batch_size, control_dof) with non-zero "
                "dimensions."
            )
        if not torch.isfinite(self.positions).all().item():
            raise ValueError("positions must contain only finite values.")
        if self.velocities is not None:
            if not isinstance(self.velocities, torch.Tensor):
                raise TypeError("velocities must be a torch.Tensor or None.")
            if self.velocities.shape != self.positions.shape:
                raise ValueError("velocities must match positions shape.")
            if self.velocities.device != self.positions.device:
                raise ValueError("velocities must share the positions device.")
            if not torch.isfinite(self.velocities).all().item():
                raise ValueError("velocities must contain only finite values.")
        object.__setattr__(self, "positions", self.positions.clone())
        if self.velocities is not None:
            object.__setattr__(self, "velocities", self.velocities.clone())

    @property
    def batch_size(self) -> int:
        """Return the number of environment rows."""
        return int(self.positions.shape[0])

    @property
    def dof(self) -> int:
        """Return the number of controlled joints."""
        return int(self.positions.shape[1])

    @property
    def device(self) -> torch.device:
        """Return the tensor device."""
        return self.positions.device

    @property
    def transport_id(self) -> str:
        """Return the built-in joint-position transport identifier."""
        return self.TRANSPORT_ID

    def snapshot(self) -> JointPositionPayload:
        """Return an independently owned joint payload."""
        return JointPositionPayload(
            positions=self.positions,
            velocities=self.velocities,
        )


@dataclass(frozen=True, slots=True, eq=False)
class EndpointCommand:
    """One transport-compatible payload addressed to one runtime target.

    Args:
        target: Immutable destination resolved from an action endpoint.
        payload: Batched command value accepted by the target transport.
    """

    target: RuntimeEndpointTarget
    payload: RuntimeCommandPayload

    def __post_init__(self) -> None:
        target = _snapshot_target(self.target)
        payload = _snapshot_payload(self.payload)
        if target.transport_id != payload.transport_id:
            raise ValueError(
                f"Target transport {target.transport_id!r} does not accept payload "
                f"transport {payload.transport_id!r}."
            )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "payload", payload)

    @property
    def transport_id(self) -> str:
        """Return the common target and payload transport identifier."""
        return self.target.transport_id

    @property
    def destination_key(self) -> tuple[str, str]:
        """Return the transport-scoped destination identifier."""
        return self.transport_id, self.target.target_id

    @property
    def batch_size(self) -> int:
        """Return the payload batch size."""
        return self.payload.batch_size

    @property
    def device(self) -> torch.device:
        """Return the payload device."""
        return self.payload.device

    def snapshot(self) -> EndpointCommand:
        """Return an independently owned endpoint command."""
        return EndpointCommand(target=self.target, payload=self.payload)


@dataclass(frozen=True, slots=True, eq=False)
class RuntimeCommandFrame:
    """Synchronized endpoint commands for one batched runtime instant.

    Args:
        commands: Commands dispatched together for this frame.
        active_mask: Boolean environment rows allowed to execute commands.
            Transports must actively neutralize addressed targets for false
            rows rather than leaving a previously persistent command running.
        env_ids: Stable environment identifiers for the batch rows.
        hold_duration: Per-row delay before advancing to the next frame.
    """

    commands: tuple[EndpointCommand, ...]
    active_mask: torch.Tensor
    env_ids: torch.Tensor
    hold_duration: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.commands, (str, bytes)):
            raise TypeError("commands must be an iterable of EndpointCommand values.")
        try:
            commands = tuple(self.commands)
        except TypeError as exc:
            raise TypeError(
                "commands must be an iterable of EndpointCommand values."
            ) from exc
        if not all(isinstance(command, EndpointCommand) for command in commands):
            raise TypeError("commands values must be EndpointCommand instances.")

        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if (
            self.env_ids.dtype != torch.long
            or self.env_ids.dim() != 1
            or self.env_ids.shape[0] < 1
        ):
            raise ValueError("env_ids must be int64 with shape (batch_size,).")
        batch_size = int(self.env_ids.shape[0])
        if torch.unique(self.env_ids).numel() != batch_size:
            raise ValueError("env_ids must be unique.")
        if not isinstance(self.active_mask, torch.Tensor):
            raise TypeError("active_mask must be a torch.Tensor.")
        if self.active_mask.dtype != torch.bool or self.active_mask.shape != (
            batch_size,
        ):
            raise ValueError(f"active_mask must be bool with shape ({batch_size},).")
        if not isinstance(self.hold_duration, torch.Tensor):
            raise TypeError("hold_duration must be a torch.Tensor.")
        if self.hold_duration.shape != (batch_size,):
            raise ValueError(f"hold_duration must have shape ({batch_size},).")
        if (
            not torch.isfinite(self.hold_duration).all().item()
            or (self.hold_duration < 0.0).any().item()
        ):
            raise ValueError("hold_duration must contain finite non-negative values.")
        if self.active_mask.device != self.env_ids.device:
            raise ValueError("active_mask and env_ids must share a device.")
        if self.hold_duration.device != self.env_ids.device:
            raise ValueError("hold_duration and env_ids must share a device.")

        snapshots = tuple(command.snapshot() for command in commands)
        destinations: set[tuple[str, str]] = set()
        joint_owners: dict[int, tuple[str, str]] = {}
        for command in snapshots:
            if command.batch_size != batch_size:
                raise ValueError(
                    f"Payload for destination {command.destination_key} has batch "
                    f"size {command.batch_size}, expected {batch_size}."
                )
            if command.device != self.env_ids.device:
                raise ValueError(
                    f"Payload for destination {command.destination_key} must share "
                    "the frame device."
                )
            if command.destination_key in destinations:
                raise ValueError(
                    f"RuntimeCommandFrame contains duplicate destination "
                    f"{command.destination_key}."
                )
            destinations.add(command.destination_key)

            if isinstance(command.target, JointPositionTarget):
                if not isinstance(command.payload, JointPositionPayload):
                    raise TypeError(
                        "JointPositionTarget requires a JointPositionPayload."
                    )
                expected_dof = len(command.target.joint_ids)
                if command.payload.dof != expected_dof:
                    raise ValueError(
                        f"Joint payload for destination {command.destination_key} has "
                        f"DOF {command.payload.dof}, expected {expected_dof}."
                    )
                overlaps = sorted(
                    joint_id
                    for joint_id in command.target.joint_ids
                    if joint_id in joint_owners
                )
                if overlaps:
                    owners = sorted({joint_owners[joint_id] for joint_id in overlaps})
                    raise ValueError(
                        f"Joint destination {command.destination_key} overlaps joint "
                        f"IDs {overlaps} already owned by {owners}."
                    )
                for joint_id in command.target.joint_ids:
                    joint_owners[joint_id] = command.destination_key

        object.__setattr__(self, "commands", snapshots)
        object.__setattr__(self, "active_mask", self.active_mask.clone())
        object.__setattr__(self, "env_ids", self.env_ids.clone())
        object.__setattr__(self, "hold_duration", self.hold_duration.clone())

    @property
    def batch_size(self) -> int:
        """Return the number of environment rows."""
        return int(self.env_ids.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the shared frame device."""
        return self.env_ids.device

    @property
    def targets(self) -> tuple[RuntimeEndpointTarget, ...]:
        """Return owned targets in command order."""
        return tuple(_snapshot_target(command.target) for command in self.commands)

    def with_active_mask(self, active_mask: torch.Tensor) -> RuntimeCommandFrame:
        """Return a frame snapshot with a replacement active-row mask.

        Args:
            active_mask: Boolean mask with one value per environment row.

        Returns:
            Independently owned frame with unchanged commands and timing.
        """
        return RuntimeCommandFrame(
            commands=self.commands,
            active_mask=active_mask,
            env_ids=self.env_ids,
            hold_duration=self.hold_duration,
        )

    def snapshot(self) -> RuntimeCommandFrame:
        """Return an independently owned command frame."""
        return RuntimeCommandFrame(
            commands=self.commands,
            active_mask=self.active_mask,
            env_ids=self.env_ids,
            hold_duration=self.hold_duration,
        )


@dataclass(frozen=True, slots=True, eq=False)
class TimedCommandSequence:
    """Ordered runtime command frames for one stable environment batch.

    ``env_ids`` is authoritative even when ``frames`` is empty, preserving the
    batch size and device needed by compilation and execution boundaries.

    Args:
        frames: Ordered command frames in execution order.
        env_ids: Stable environment identifiers retained for empty sequences.
    """

    frames: tuple[RuntimeCommandFrame, ...]
    env_ids: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if (
            self.env_ids.dtype != torch.long
            or self.env_ids.dim() != 1
            or self.env_ids.shape[0] < 1
        ):
            raise ValueError("env_ids must be int64 with shape (batch_size,).")
        if torch.unique(self.env_ids).numel() != self.env_ids.numel():
            raise ValueError("env_ids must be unique.")
        if isinstance(self.frames, (str, bytes)):
            raise TypeError("frames must be an iterable of RuntimeCommandFrame values.")
        try:
            frames = tuple(self.frames)
        except TypeError as exc:
            raise TypeError(
                "frames must be an iterable of RuntimeCommandFrame values."
            ) from exc
        if not all(isinstance(frame, RuntimeCommandFrame) for frame in frames):
            raise TypeError("frames values must be RuntimeCommandFrame instances.")
        snapshots: list[RuntimeCommandFrame] = []
        for index, frame in enumerate(frames):
            if frame.device != self.env_ids.device:
                raise ValueError(f"Frame {index} must share the sequence device.")
            if not torch.equal(frame.env_ids, self.env_ids):
                raise ValueError(f"Frame {index} env_ids do not match the sequence.")
            snapshots.append(frame.snapshot())
        object.__setattr__(self, "frames", tuple(snapshots))
        object.__setattr__(self, "env_ids", self.env_ids.clone())

    @property
    def batch_size(self) -> int:
        """Return the preserved environment batch size."""
        return int(self.env_ids.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the preserved batch device."""
        return self.env_ids.device

    @property
    def frame_count(self) -> int:
        """Return the number of command frames."""
        return len(self.frames)

    @property
    def targets(self) -> tuple[RuntimeEndpointTarget, ...]:
        """Return unique owned destinations in first-use order."""
        targets: list[RuntimeEndpointTarget] = []
        seen: set[tuple[str, str]] = set()
        for frame in self.frames:
            for command in frame.commands:
                if command.destination_key in seen:
                    continue
                seen.add(command.destination_key)
                targets.append(_snapshot_target(command.target))
        return tuple(targets)

    def snapshot(self) -> TimedCommandSequence:
        """Return an independently owned timed sequence."""
        return TimedCommandSequence(frames=self.frames, env_ids=self.env_ids)


__all__ = [
    "EndpointCommand",
    "JointPositionPayload",
    "RuntimeCommandFrame",
    "RuntimeCommandPayload",
    "TimedCommandSequence",
]
