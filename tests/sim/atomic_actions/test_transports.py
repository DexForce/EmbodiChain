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

"""Pure routing tests for endpoint-command transports."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from embodichain.lab.sim.atomic_actions.bindings import RuntimeEndpointTarget
from embodichain.lab.sim.atomic_actions.runner import (
    CommandAcknowledgement,
    CommandAckStatus,
    CommandSink,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
)
from embodichain.lab.sim.atomic_actions.transports import (
    EndpointCommandRouter,
    EndpointCommandTransport,
)


@dataclass(frozen=True, slots=True)
class _Target(RuntimeEndpointTarget):
    """Test-only runtime target."""

    _transport_id: str
    _target_id: str

    @property
    def transport_id(self) -> str:
        """Return the addressed transport."""
        return self._transport_id

    @property
    def target_id(self) -> str:
        """Return the local destination."""
        return self._target_id


@dataclass(frozen=True, slots=True)
class _Payload(RuntimeCommandPayload):
    """Test-only payload with transport-neutral scalar data."""

    _transport_id: str
    values: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Return the payload batch size."""
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the payload device."""
        return self.values.device

    @property
    def transport_id(self) -> str:
        """Return the addressed transport."""
        return self._transport_id

    def snapshot(self) -> _Payload:
        """Return an independently owned payload."""
        return _Payload(self._transport_id, self.values.clone())


@dataclass(frozen=True, slots=True)
class _OtherPayload(RuntimeCommandPayload):
    """Different payload type used to exercise compatibility checks."""

    _transport_id: str
    values: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Return the payload batch size."""
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the payload device."""
        return self.values.device

    @property
    def transport_id(self) -> str:
        """Return the addressed transport."""
        return self._transport_id

    def snapshot(self) -> _OtherPayload:
        """Return an independently owned payload."""
        return _OtherPayload(self._transport_id, self.values.clone())


class _FakeTransport:
    """Recording transport with configurable acknowledgements."""

    def __init__(
        self,
        transport_id: str,
        *,
        payload_type: type[RuntimeCommandPayload] = _Payload,
    ) -> None:
        self._transport_id = transport_id
        self._payload_type = payload_type
        self.send_ack: object = CommandAcknowledgement.accepted_ack()
        self.hold_ack: object = CommandAcknowledgement.accepted_ack()
        self.cancel_ack: object = CommandAcknowledgement.accepted_ack()
        self.send_error: Exception | None = None
        self.hold_error: Exception | None = None
        self.cancel_error: Exception | None = None
        self.send_calls: list[tuple[RuntimeCommandFrame, float]] = []
        self.hold_calls: list[
            tuple[tuple[RuntimeEndpointTarget, ...], object, float]
        ] = []
        self.cancel_calls: list[tuple[tuple[RuntimeEndpointTarget, ...], float]] = []

    @property
    def transport_id(self) -> str:
        """Return the fake registration identifier."""
        return self._transport_id

    @property
    def payload_type(self) -> type[RuntimeCommandPayload]:
        """Return the accepted fake payload type."""
        return self._payload_type

    def send(
        self,
        frame: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record one transport-local frame."""
        self.send_calls.append((frame, timeout))
        if self.send_error is not None:
            raise self.send_error
        return self.send_ack  # type: ignore[return-value]

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: object,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record one transport-local hold."""
        self.hold_calls.append((targets, context, timeout))
        if self.hold_error is not None:
            raise self.hold_error
        return self.hold_ack  # type: ignore[return-value]

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Record one transport-local cancellation."""
        self.cancel_calls.append((targets, timeout))
        if self.cancel_error is not None:
            raise self.cancel_error
        return self.cancel_ack  # type: ignore[return-value]


def _command(
    transport_id: str,
    target_id: str,
    *,
    payload_type: type[RuntimeCommandPayload] = _Payload,
) -> EndpointCommand:
    """Build one two-row endpoint command."""
    return EndpointCommand(
        target=_Target(transport_id, target_id),
        payload=payload_type(  # type: ignore[call-arg]
            transport_id,
            torch.tensor([[1.0], [2.0]]),
        ),
    )


def _frame(*commands: EndpointCommand) -> RuntimeCommandFrame:
    """Build one two-row command frame."""
    return RuntimeCommandFrame(
        commands=commands,
        active_mask=torch.tensor([True, False]),
        env_ids=torch.tensor([3, 8]),
        hold_duration=torch.tensor([0.1, 0.2]),
    )


def test_transport_protocol_is_runtime_checkable() -> None:
    assert isinstance(_FakeTransport("alpha"), EndpointCommandTransport)
    assert not isinstance(object(), EndpointCommandTransport)


def test_router_structurally_implements_command_sink() -> None:
    assert isinstance(EndpointCommandRouter([]), CommandSink)


def test_router_builds_owned_exact_registry_from_mapping() -> None:
    alpha = _FakeTransport("alpha")
    registrations = {"alpha": alpha}
    router = EndpointCommandRouter(registrations)

    registrations.clear()
    assert dict(router.transports) == {"alpha": alpha}
    with pytest.raises(TypeError):
        router.transports["beta"] = _FakeTransport("beta")  # type: ignore[index]


def test_router_rejects_non_exact_mapping_key() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        EndpointCommandRouter({"alias": _FakeTransport("alpha")})


def test_router_rejects_duplicate_transport_registration() -> None:
    with pytest.raises(ValueError, match="more than once"):
        EndpointCommandRouter([_FakeTransport("alpha"), _FakeTransport("alpha")])


def test_router_rejects_invalid_transport_contract_and_payload_type() -> None:
    with pytest.raises(TypeError, match="EndpointCommandTransport"):
        EndpointCommandRouter([object()])  # type: ignore[list-item]

    invalid_payload = _FakeTransport("alpha")
    invalid_payload._payload_type = str  # type: ignore[assignment]
    with pytest.raises(TypeError, match="payload_type"):
        EndpointCommandRouter([invalid_payload])


def test_send_groups_subframes_and_preserves_frame_metadata() -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    router = EndpointCommandRouter({"alpha": alpha, "beta": beta})
    frame = _frame(
        _command("alpha", "a0"),
        _command("beta", "b0"),
        _command("alpha", "a1"),
    )

    acknowledgement = router.send(frame, timeout=0.75)

    assert acknowledgement.accepted
    assert len(alpha.send_calls) == 1
    assert len(beta.send_calls) == 1
    alpha_frame, alpha_timeout = alpha.send_calls[0]
    beta_frame, beta_timeout = beta.send_calls[0]
    assert [command.target.target_id for command in alpha_frame.commands] == [
        "a0",
        "a1",
    ]
    assert [command.target.target_id for command in beta_frame.commands] == ["b0"]
    assert torch.equal(alpha_frame.active_mask, frame.active_mask)
    assert torch.equal(alpha_frame.env_ids, frame.env_ids)
    assert torch.equal(alpha_frame.hold_duration, frame.hold_duration)
    assert alpha_frame.active_mask.data_ptr() != frame.active_mask.data_ptr()
    assert alpha_timeout == beta_timeout == 0.75


def test_send_unknown_transport_rejects_before_any_dispatch() -> None:
    alpha = _FakeTransport("alpha")
    router = EndpointCommandRouter([alpha])

    acknowledgement = router.send(
        _frame(_command("alpha", "a0"), _command("missing", "x0")),
        timeout=1.0,
    )

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "missing" in acknowledgement.message
    assert alpha.send_calls == []


def test_send_incompatible_payload_rejects_before_dispatch() -> None:
    alpha = _FakeTransport("alpha", payload_type=_Payload)
    router = EndpointCommandRouter([alpha])

    acknowledgement = router.send(
        _frame(_command("alpha", "a0", payload_type=_OtherPayload)),
        timeout=1.0,
    )

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "alpha" in acknowledgement.message
    assert "_Payload" in acknowledgement.message
    assert "_OtherPayload" in acknowledgement.message
    assert alpha.send_calls == []


def test_send_aggregates_partial_rejection_with_transport_id() -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    alpha.send_ack = CommandAcknowledgement.accepted_ack("queued")
    beta.send_ack = CommandAcknowledgement(
        CommandAckStatus.REJECTED,
        "controller busy",
    )
    router = EndpointCommandRouter([alpha, beta])

    acknowledgement = router.send(
        _frame(_command("alpha", "a0"), _command("beta", "b0")),
        timeout=1.0,
    )

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "beta" in acknowledgement.message
    assert "controller busy" in acknowledgement.message
    assert len(alpha.send_calls) == len(beta.send_calls) == 1


def test_send_timed_out_status_takes_failure_precedence() -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    alpha.send_ack = CommandAcknowledgement(CommandAckStatus.REJECTED, "rejected")
    beta.send_ack = CommandAcknowledgement(CommandAckStatus.TIMED_OUT, "late")
    router = EndpointCommandRouter([alpha, beta])

    acknowledgement = router.send(
        _frame(_command("alpha", "a0"), _command("beta", "b0")),
        timeout=1.0,
    )

    assert acknowledgement.status is CommandAckStatus.TIMED_OUT
    assert "alpha" in acknowledgement.message
    assert "beta" in acknowledgement.message


def test_send_converts_transport_exception_and_continues_dispatch() -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    alpha.send_error = RuntimeError("send exploded")
    router = EndpointCommandRouter([alpha, beta])

    acknowledgement = router.send(
        _frame(_command("alpha", "a0"), _command("beta", "b0")),
        timeout=1.0,
    )

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "alpha" in acknowledgement.message
    assert "RuntimeError" in acknowledgement.message
    assert "send exploded" in acknowledgement.message
    assert len(alpha.send_calls) == len(beta.send_calls) == 1


def test_hold_groups_targets_and_forwards_observation_context() -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    router = EndpointCommandRouter([alpha, beta])
    context = object()

    acknowledgement = router.hold(
        (
            _Target("alpha", "a0"),
            _Target("beta", "b0"),
            _Target("alpha", "a1"),
        ),
        context,  # type: ignore[arg-type]
        timeout=0.4,
    )

    assert acknowledgement.accepted
    alpha_targets, alpha_context, alpha_timeout = alpha.hold_calls[0]
    beta_targets, beta_context, beta_timeout = beta.hold_calls[0]
    assert [target.target_id for target in alpha_targets] == ["a0", "a1"]
    assert [target.target_id for target in beta_targets] == ["b0"]
    assert alpha_context is beta_context is context
    assert alpha_timeout == beta_timeout == 0.4


def test_cancel_groups_targets_and_aggregates_partial_failure() -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    beta.cancel_ack = CommandAcknowledgement(CommandAckStatus.TIMED_OUT, "late")
    router = EndpointCommandRouter([alpha, beta])

    acknowledgement = router.cancel(
        (
            _Target("beta", "b0"),
            _Target("alpha", "a0"),
            _Target("beta", "b1"),
        ),
        timeout=0.2,
    )

    assert acknowledgement.status is CommandAckStatus.TIMED_OUT
    assert "beta" in acknowledgement.message
    assert [target.target_id for target in beta.cancel_calls[0][0]] == ["b0", "b1"]
    assert [target.target_id for target in alpha.cancel_calls[0][0]] == ["a0"]


@pytest.mark.parametrize("operation", ["hold", "cancel"])
def test_safe_stop_transport_exception_does_not_block_later_transport(
    operation: str,
) -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    setattr(alpha, f"{operation}_error", RuntimeError(f"{operation} exploded"))
    router = EndpointCommandRouter([alpha, beta])
    targets = (_Target("alpha", "a0"), _Target("beta", "b0"))

    if operation == "hold":
        acknowledgement = router.hold(
            targets,
            object(),  # type: ignore[arg-type]
            timeout=1.0,
        )
        alpha_calls = alpha.hold_calls
        beta_calls = beta.hold_calls
    else:
        acknowledgement = router.cancel(targets, timeout=1.0)
        alpha_calls = alpha.cancel_calls
        beta_calls = beta.cancel_calls

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "alpha" in acknowledgement.message
    assert "RuntimeError" in acknowledgement.message
    assert f"{operation} exploded" in acknowledgement.message
    assert len(alpha_calls) == len(beta_calls) == 1


@pytest.mark.parametrize("operation", ["hold", "cancel"])
def test_target_operation_unknown_transport_rejects_before_dispatch(
    operation: str,
) -> None:
    alpha = _FakeTransport("alpha")
    router = EndpointCommandRouter([alpha])
    if operation == "hold":
        acknowledgement = router.hold(
            (_Target("missing", "x0"),),
            object(),  # type: ignore[arg-type]
            timeout=1.0,
        )
        calls = alpha.hold_calls
    else:
        acknowledgement = router.cancel(
            (_Target("missing", "x0"),),
            timeout=1.0,
        )
        calls = alpha.cancel_calls

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "missing" in acknowledgement.message
    assert calls == []


@pytest.mark.parametrize("operation", ["send", "hold", "cancel"])
def test_router_converts_invalid_return_type_and_continues_dispatch(
    operation: str,
) -> None:
    alpha = _FakeTransport("alpha")
    beta = _FakeTransport("beta")
    setattr(alpha, f"{operation}_ack", object())
    router = EndpointCommandRouter([alpha, beta])

    if operation == "send":
        acknowledgement = router.send(
            _frame(_command("alpha", "a0"), _command("beta", "b0")),
            timeout=1.0,
        )
        beta_calls = beta.send_calls
    elif operation == "hold":
        acknowledgement = router.hold(
            (_Target("alpha", "a0"), _Target("beta", "b0")),
            object(),  # type: ignore[arg-type]
            timeout=1.0,
        )
        beta_calls = beta.hold_calls
    else:
        acknowledgement = router.cancel(
            (_Target("alpha", "a0"), _Target("beta", "b0")),
            timeout=1.0,
        )
        beta_calls = beta.cancel_calls

    assert acknowledgement.status is CommandAckStatus.REJECTED
    assert "alpha" in acknowledgement.message
    assert "CommandAcknowledgement" in acknowledgement.message
    assert len(beta_calls) == 1


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("operation", ["send", "hold", "cancel"])
def test_router_rejects_invalid_timeout(operation: str, timeout: float) -> None:
    router = EndpointCommandRouter([])

    with pytest.raises(ValueError, match="timeout"):
        if operation == "send":
            router.send(_frame(), timeout=timeout)
        elif operation == "hold":
            router.hold((), object(), timeout=timeout)  # type: ignore[arg-type]
        else:
            router.cancel((), timeout=timeout)


def test_empty_operations_are_accepted() -> None:
    router = EndpointCommandRouter([])

    assert router.send(_frame(), timeout=1.0).accepted
    assert router.hold((), object(), timeout=1.0).accepted  # type: ignore[arg-type]
    assert router.cancel((), timeout=1.0).accepted
