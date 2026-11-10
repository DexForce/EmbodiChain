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

"""Endpoint-command transport contracts and deterministic routing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .bindings import RuntimeEndpointTarget
from .runtime_commands import (
    EndpointCommand,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
)

if TYPE_CHECKING:
    from .runner import CommandAcknowledgement
    from .state import PlanningContext


def _validate_identifier(value: str, *, field_name: str) -> str:
    """Validate and return one strict identifier."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _validate_timeout(timeout: float) -> float:
    """Validate and normalize one acknowledgement timeout."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a real number.")
    normalized = float(timeout)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("timeout must be finite and greater than zero.")
    return normalized


@runtime_checkable
class EndpointCommandTransport(Protocol):
    """Backend that owns one kind of runtime endpoint command.

    Implementations own live simulator entities, device clients, or controller
    handles. Runtime command values retain only immutable addressing and payload
    data, so they remain independent of those process-owned resources.
    """

    @property
    def transport_id(self) -> str:
        """Return the exact identifier used to register this transport."""

    @property
    def payload_type(self) -> type[RuntimeCommandPayload]:
        """Return the runtime payload type accepted by :meth:`send`."""

    def send(
        self,
        frame: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Submit one transport-local command frame.

        Implementations must actively neutralize every inactive environment
        row for every addressed target. Silently skipping an inactive row is
        unsafe for persistent controllers such as base-velocity transports.
        """

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Hold transport-local targets at their observed state."""

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Cancel outstanding commands for transport-local targets."""


class EndpointCommandRouter:
    """Route generic endpoint operations to exact registered transports.

    The router implements :class:`~.runner.CommandSink` structurally while
    avoiding a module-load dependency on ``runner``. Acknowledgement types are
    imported only when an operation is executed, which keeps the transport
    boundary safe to import while the runner imports this module.

    Args:
        transports: Either an exact ``transport_id -> transport`` mapping or an
            iterable of transports from which that mapping is built. Mapping
            keys must exactly equal each value's declared ``transport_id``.

    Raises:
        TypeError: If a registration does not implement the transport contract.
        ValueError: If an identifier is invalid, a mapping key is not exact, or
            the same transport identifier is registered more than once.
    """

    def __init__(
        self,
        transports: (
            Mapping[str, EndpointCommandTransport] | Iterable[EndpointCommandTransport]
        ),
    ) -> None:
        registrations = self._registrations(transports)
        registered: dict[str, EndpointCommandTransport] = {}
        payload_types: dict[str, type[RuntimeCommandPayload]] = {}
        for map_key, transport in registrations:
            if not isinstance(transport, EndpointCommandTransport):
                raise TypeError(
                    "Registered values must implement EndpointCommandTransport."
                )
            transport_id = _validate_identifier(
                transport.transport_id,
                field_name="EndpointCommandTransport.transport_id",
            )
            if map_key is not None and map_key != transport_id:
                raise ValueError(
                    f"Transport mapping key {map_key!r} must exactly match declared "
                    f"transport_id {transport_id!r}."
                )
            if transport_id in registered:
                raise ValueError(
                    f"Endpoint transport {transport_id!r} is registered more than once."
                )
            payload_type = transport.payload_type
            if not isinstance(payload_type, type) or not issubclass(
                payload_type, RuntimeCommandPayload
            ):
                raise TypeError(
                    f"Transport {transport_id!r} payload_type must be a "
                    "RuntimeCommandPayload subclass."
                )
            registered[transport_id] = transport
            payload_types[transport_id] = payload_type
        self._transports: Mapping[str, EndpointCommandTransport] = MappingProxyType(
            registered
        )
        self._payload_types: Mapping[str, type[RuntimeCommandPayload]] = (
            MappingProxyType(payload_types)
        )

    @staticmethod
    def _registrations(
        transports: (
            Mapping[str, EndpointCommandTransport] | Iterable[EndpointCommandTransport]
        ),
    ) -> tuple[tuple[str | None, EndpointCommandTransport], ...]:
        """Normalize mapping and iterable registration forms."""
        if isinstance(transports, Mapping):
            registrations: list[tuple[str | None, EndpointCommandTransport]] = []
            for key, transport in transports.items():
                _validate_identifier(key, field_name="Transport mapping keys")
                registrations.append((key, transport))
            return tuple(registrations)
        if isinstance(transports, (str, bytes)):
            raise TypeError("transports must be a mapping or iterable of transports.")
        try:
            return tuple((None, transport) for transport in transports)
        except TypeError as exc:
            raise TypeError(
                "transports must be a mapping or iterable of transports."
            ) from exc

    @property
    def transports(self) -> Mapping[str, EndpointCommandTransport]:
        """Return the immutable exact transport registry."""
        return self._transports

    def send(
        self,
        frame: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Route one synchronized command frame by transport identifier.

        Dispatch is preflighted before any transport is called. An unknown
        transport or incompatible payload therefore rejects the whole frame
        without creating a partially dispatched operation.

        Args:
            frame: Generic runtime command frame to split by transport.
            timeout: Maximum acknowledgement latency for each transport.

        Returns:
            Aggregated acknowledgement. It is accepted only when every
            addressed transport accepts its local frame.
        """
        if not isinstance(frame, RuntimeCommandFrame):
            raise TypeError("frame must be a RuntimeCommandFrame.")
        normalized_timeout = _validate_timeout(timeout)
        grouped: dict[str, list[EndpointCommand]] = {}
        for command in frame.commands:
            grouped.setdefault(command.transport_id, []).append(command)

        unknown = tuple(
            transport_id
            for transport_id in grouped
            if transport_id not in self._transports
        )
        if unknown:
            return self._unknown_acknowledgement("send", unknown)

        incompatibilities: list[str] = []
        for transport_id, commands in grouped.items():
            payload_type = self._payload_types[transport_id]
            for command in commands:
                if not isinstance(command.payload, payload_type):
                    incompatibilities.append(
                        f"transport {transport_id!r} expects "
                        f"{payload_type.__name__}, got "
                        f"{type(command.payload).__name__} for target "
                        f"{command.target.target_id!r}"
                    )
        if incompatibilities:
            return self._rejected_acknowledgement(
                "send rejected: " + "; ".join(incompatibilities)
            )

        acknowledgements: list[tuple[str, CommandAcknowledgement]] = []
        for transport_id, commands in grouped.items():
            subframe = RuntimeCommandFrame(
                commands=tuple(commands),
                active_mask=frame.active_mask,
                env_ids=frame.env_ids,
                hold_duration=frame.hold_duration,
            )
            transport = self._transports[transport_id]
            acknowledgement = self._invoke_transport(
                transport_id,
                "send",
                lambda transport=transport, subframe=subframe: transport.send(
                    subframe,
                    timeout=normalized_timeout,
                ),
            )
            acknowledgements.append((transport_id, acknowledgement))
        return self._aggregate_acknowledgements("send", acknowledgements)

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Route an observed-state hold request by target transport.

        Args:
            targets: Runtime destinations to hold.
            context: Fresh observation used by each transport to form its hold.
            timeout: Maximum acknowledgement latency for each transport.

        Returns:
            Aggregated acknowledgement. It is accepted only when every
            addressed transport accepts its hold.
        """
        normalized_timeout = _validate_timeout(timeout)
        grouped = self._group_targets(targets)
        unknown = tuple(
            transport_id
            for transport_id in grouped
            if transport_id not in self._transports
        )
        if unknown:
            return self._unknown_acknowledgement("hold", unknown)

        acknowledgements: list[tuple[str, CommandAcknowledgement]] = []
        for transport_id, local_targets in grouped.items():
            transport = self._transports[transport_id]
            acknowledgement = self._invoke_transport(
                transport_id,
                "hold",
                lambda transport=transport, local_targets=local_targets: transport.hold(
                    local_targets,
                    context,
                    timeout=normalized_timeout,
                ),
            )
            acknowledgements.append((transport_id, acknowledgement))
        return self._aggregate_acknowledgements("hold", acknowledgements)

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Route cancellation by target transport.

        Args:
            targets: Runtime destinations whose outstanding commands are
                cancelled.
            timeout: Maximum acknowledgement latency for each transport.

        Returns:
            Aggregated acknowledgement. It is accepted only when every
            addressed transport accepts cancellation.
        """
        normalized_timeout = _validate_timeout(timeout)
        grouped = self._group_targets(targets)
        unknown = tuple(
            transport_id
            for transport_id in grouped
            if transport_id not in self._transports
        )
        if unknown:
            return self._unknown_acknowledgement("cancel", unknown)

        acknowledgements: list[tuple[str, CommandAcknowledgement]] = []
        for transport_id, local_targets in grouped.items():
            transport = self._transports[transport_id]
            acknowledgement = self._invoke_transport(
                transport_id,
                "cancel",
                lambda transport=transport, local_targets=local_targets: transport.cancel(
                    local_targets,
                    timeout=normalized_timeout,
                ),
            )
            acknowledgements.append((transport_id, acknowledgement))
        return self._aggregate_acknowledgements("cancel", acknowledgements)

    @staticmethod
    def _group_targets(
        targets: tuple[RuntimeEndpointTarget, ...],
    ) -> dict[str, tuple[RuntimeEndpointTarget, ...]]:
        """Validate, snapshot, and group runtime targets in first-use order."""
        if isinstance(targets, (str, bytes)):
            raise TypeError(
                "targets must be an iterable of RuntimeEndpointTarget values."
            )
        try:
            source_targets = tuple(targets)
        except TypeError as exc:
            raise TypeError(
                "targets must be an iterable of RuntimeEndpointTarget values."
            ) from exc

        grouped: dict[str, list[RuntimeEndpointTarget]] = {}
        for target in source_targets:
            if not isinstance(target, RuntimeEndpointTarget):
                raise TypeError(
                    "targets values must be RuntimeEndpointTarget instances."
                )
            snapshot = target.snapshot()
            if type(snapshot) is not type(target) or snapshot is target:
                raise TypeError(
                    "RuntimeEndpointTarget.snapshot() must return an independently "
                    "owned value of the same target type."
                )
            transport_id = _validate_identifier(
                snapshot.transport_id,
                field_name="RuntimeEndpointTarget.transport_id",
            )
            _validate_identifier(
                snapshot.target_id,
                field_name="RuntimeEndpointTarget.target_id",
            )
            grouped.setdefault(transport_id, []).append(snapshot)
        return {
            transport_id: tuple(local_targets)
            for transport_id, local_targets in grouped.items()
        }

    @staticmethod
    def _validate_acknowledgement(
        transport_id: str,
        operation: str,
        acknowledgement: object,
    ) -> CommandAcknowledgement:
        """Require transports to return the runner acknowledgement value."""
        from .runner import CommandAcknowledgement

        if not isinstance(acknowledgement, CommandAcknowledgement):
            raise TypeError(
                f"Transport {transport_id!r} {operation}() must return "
                f"CommandAcknowledgement, got {type(acknowledgement).__name__}."
            )
        return acknowledgement

    @staticmethod
    def _invoke_transport(
        transport_id: str,
        operation: str,
        invoke: Callable[[], object],
    ) -> CommandAcknowledgement:
        """Convert one transport-local failure without blocking other transports."""
        from .runner import CommandAcknowledgement, CommandAckStatus

        try:
            acknowledgement = invoke()
            return EndpointCommandRouter._validate_acknowledgement(
                transport_id,
                operation,
                acknowledgement,
            )
        except Exception as exc:
            return CommandAcknowledgement(
                CommandAckStatus.REJECTED,
                f"Transport {transport_id!r} {operation}() failed with "
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _aggregate_acknowledgements(
        operation: str,
        acknowledgements: list[tuple[str, CommandAcknowledgement]],
    ) -> CommandAcknowledgement:
        """Aggregate transport acknowledgements with deterministic precedence."""
        from .runner import CommandAcknowledgement, CommandAckStatus

        failures = [
            (transport_id, acknowledgement)
            for transport_id, acknowledgement in acknowledgements
            if not acknowledgement.accepted
        ]
        if not failures:
            diagnostics = "; ".join(
                f"{transport_id}: {acknowledgement.message}"
                for transport_id, acknowledgement in acknowledgements
                if acknowledgement.message
            )
            return CommandAcknowledgement.accepted_ack(diagnostics)

        status = (
            CommandAckStatus.TIMED_OUT
            if any(
                acknowledgement.status is CommandAckStatus.TIMED_OUT
                for _, acknowledgement in failures
            )
            else CommandAckStatus.REJECTED
        )
        diagnostics = "; ".join(
            f"transport {transport_id!r} {acknowledgement.status.value}: "
            f"{acknowledgement.message or 'no diagnostic'}"
            for transport_id, acknowledgement in failures
        )
        return CommandAcknowledgement(
            status,
            f"{operation} failed: {diagnostics}",
        )

    @staticmethod
    def _unknown_acknowledgement(
        operation: str,
        transport_ids: tuple[str, ...],
    ) -> CommandAcknowledgement:
        """Build a rejection for unregistered exact transport identifiers."""
        identifiers = ", ".join(repr(transport_id) for transport_id in transport_ids)
        return EndpointCommandRouter._rejected_acknowledgement(
            f"{operation} rejected: no transport is registered for {identifiers}."
        )

    @staticmethod
    def _rejected_acknowledgement(message: str) -> CommandAcknowledgement:
        """Build one rejected runner acknowledgement without an import cycle."""
        from .runner import CommandAcknowledgement, CommandAckStatus

        return CommandAcknowledgement(CommandAckStatus.REJECTED, message)


__all__ = ["EndpointCommandRouter", "EndpointCommandTransport"]
