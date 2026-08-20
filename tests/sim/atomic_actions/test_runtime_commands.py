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

"""Pure value-object tests for transport-neutral runtime commands."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from embodichain.lab.sim.atomic_actions.bindings import (
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
    TimedCommandSequence,
)


@dataclass(frozen=True, slots=True)
class _TestTarget(RuntimeEndpointTarget):
    """Small target used to exercise custom transports."""

    _transport_id: str
    _target_id: str

    @property
    def transport_id(self) -> str:
        """Return the test transport identifier."""
        return self._transport_id

    @property
    def target_id(self) -> str:
        """Return the test destination identifier."""
        return self._target_id


@dataclass(frozen=True, slots=True)
class _OpaquePayload(RuntimeCommandPayload):
    """Metadata-only payload used for transport and device validation."""

    rows: int
    payload_device: torch.device
    payload_transport: str

    @property
    def batch_size(self) -> int:
        """Return the configured row count."""
        return self.rows

    @property
    def device(self) -> torch.device:
        """Return the configured device."""
        return self.payload_device

    @property
    def transport_id(self) -> str:
        """Return the configured transport identifier."""
        return self.payload_transport

    def snapshot(self) -> _OpaquePayload:
        """Return an independently owned payload."""
        return _OpaquePayload(
            rows=self.rows,
            payload_device=self.payload_device,
            payload_transport=self.payload_transport,
        )


class _SelfSnapshotPayload(RuntimeCommandPayload):
    """Invalid payload whose snapshot aliases the source."""

    @property
    def batch_size(self) -> int:
        """Return one row."""
        return 1

    @property
    def device(self) -> torch.device:
        """Return the CPU device."""
        return torch.device("cpu")

    @property
    def transport_id(self) -> str:
        """Return the test transport."""
        return "test.transport"

    def snapshot(self) -> _SelfSnapshotPayload:
        """Incorrectly return this same payload."""
        return self


def _joint_command(
    control_part: str,
    joint_ids: tuple[int, ...],
    positions: torch.Tensor,
) -> EndpointCommand:
    """Build one joint endpoint command for a test."""
    return EndpointCommand(
        target=JointPositionTarget(control_part, joint_ids),
        payload=JointPositionPayload(positions),
    )


def _frame(
    commands: tuple[EndpointCommand, ...],
    *,
    active_mask: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
    hold_duration: torch.Tensor | None = None,
) -> RuntimeCommandFrame:
    """Build a two-row CPU frame with optional field replacements."""
    return RuntimeCommandFrame(
        commands=commands,
        active_mask=(
            torch.tensor([True, False]) if active_mask is None else active_mask
        ),
        env_ids=torch.tensor([4, 9]) if env_ids is None else env_ids,
        hold_duration=(
            torch.tensor([0.0, 0.1]) if hold_duration is None else hold_duration
        ),
    )


def test_runtime_command_payload_is_abstract() -> None:
    with pytest.raises(TypeError):
        RuntimeCommandPayload()  # type: ignore[abstract]


def test_joint_position_payload_owns_tensors_and_snapshots() -> None:
    positions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    velocities = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    payload = JointPositionPayload(positions, velocities)

    positions.fill_(9.0)
    velocities.fill_(8.0)
    snapshot = payload.snapshot()
    snapshot.positions.fill_(7.0)
    assert payload.positions.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert payload.velocities is not None
    assert torch.allclose(
        payload.velocities,
        torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    )
    assert payload.batch_size == 2
    assert payload.dof == 2
    assert payload.device == torch.device("cpu")
    assert payload.transport_id == JointPositionTarget.TRANSPORT_ID


@pytest.mark.parametrize(
    "positions, message",
    [
        (torch.empty(0, 2), "non-zero"),
        (torch.empty(2, 0), "non-zero"),
        (torch.zeros(2), "shape"),
        (torch.tensor([[float("nan")]]), "finite"),
    ],
)
def test_joint_position_payload_rejects_invalid_positions(
    positions: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JointPositionPayload(positions)


def test_joint_position_payload_validates_velocities() -> None:
    positions = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="match positions shape"):
        JointPositionPayload(positions, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="finite"):
        JointPositionPayload(
            positions,
            torch.tensor([[0.0, float("inf")], [0.0, 0.0]]),
        )


def test_endpoint_command_requires_matching_transport() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        EndpointCommand(
            target=_TestTarget("test.target", "base"),
            payload=_OpaquePayload(2, torch.device("cpu"), "test.payload"),
        )


def test_endpoint_command_owns_target_and_payload_snapshots() -> None:
    target = _TestTarget("test.transport", "base")
    payload = _OpaquePayload(2, torch.device("cpu"), "test.transport")
    command = EndpointCommand(target=target, payload=payload)

    assert command.target is not target
    assert command.payload is not payload
    assert command.transport_id == "test.transport"
    assert command.destination_key == ("test.transport", "base")
    assert command.batch_size == 2
    assert command.device == torch.device("cpu")
    assert command.snapshot().payload is not command.payload


def test_endpoint_command_rejects_aliased_payload_snapshot() -> None:
    with pytest.raises(TypeError, match="independently owned"):
        EndpointCommand(
            target=_TestTarget("test.transport", "base"),
            payload=_SelfSnapshotPayload(),
        )


def test_runtime_command_frame_accepts_disjoint_joint_destinations() -> None:
    frame = _frame(
        (
            _joint_command("left", (0, 2), torch.zeros(2, 2)),
            _joint_command("right", (1, 3), torch.ones(2, 2)),
        )
    )

    assert frame.batch_size == 2
    assert frame.device == torch.device("cpu")
    assert [target.target_id for target in frame.targets] == ["left", "right"]
    assert frame.active_mask.tolist() == [True, False]
    assert frame.env_ids.tolist() == [4, 9]


def test_runtime_command_frame_rejects_payload_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="batch size 1, expected 2"):
        _frame((_joint_command("arm", (0,), torch.zeros(1, 1)),))


def test_runtime_command_frame_rejects_payload_device_mismatch() -> None:
    command = EndpointCommand(
        target=_TestTarget("test.transport", "base"),
        payload=_OpaquePayload(2, torch.device("meta"), "test.transport"),
    )
    with pytest.raises(ValueError, match="share the frame device"):
        _frame((command,))


def test_runtime_command_frame_rejects_duplicate_destination() -> None:
    target = _TestTarget("test.transport", "base")
    command = EndpointCommand(
        target=target,
        payload=_OpaquePayload(2, torch.device("cpu"), "test.transport"),
    )
    with pytest.raises(ValueError, match="duplicate destination"):
        _frame((command, command))


def test_runtime_command_frame_requires_joint_payload_for_joint_target() -> None:
    command = EndpointCommand(
        target=JointPositionTarget("arm", (0,)),
        payload=_OpaquePayload(
            2,
            torch.device("cpu"),
            JointPositionTarget.TRANSPORT_ID,
        ),
    )
    with pytest.raises(TypeError, match="requires a JointPositionPayload"):
        _frame((command,))


def test_runtime_command_frame_rejects_joint_target_dof_mismatch() -> None:
    with pytest.raises(ValueError, match="DOF 1, expected 2"):
        _frame((_joint_command("arm", (0, 1), torch.zeros(2, 1)),))


def test_runtime_command_frame_rejects_overlapping_joint_ids() -> None:
    with pytest.raises(ValueError, match=r"overlaps joint IDs \[2\]"):
        _frame(
            (
                _joint_command("left", (0, 2), torch.zeros(2, 2)),
                _joint_command("right", (2, 3), torch.zeros(2, 2)),
            )
        )


def test_runtime_command_frame_validates_batch_metadata() -> None:
    command = _joint_command("arm", (0,), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="active_mask"):
        _frame((command,), active_mask=torch.tensor([1, 0]))
    with pytest.raises(ValueError, match="env_ids"):
        _frame((command,), env_ids=torch.tensor([4.0, 9.0]))
    with pytest.raises(ValueError, match="hold_duration"):
        _frame((command,), hold_duration=torch.tensor([0.0, float("nan")]))
    with pytest.raises(ValueError, match="non-negative"):
        _frame((command,), hold_duration=torch.tensor([0.0, -0.1]))
    with pytest.raises(ValueError, match="unique"):
        _frame((command,), env_ids=torch.tensor([4, 4]))


def test_runtime_command_frame_with_active_mask_returns_owned_frame() -> None:
    frame = _frame((_joint_command("arm", (0,), torch.zeros(2, 1)),))
    replacement = torch.tensor([False, True])
    updated = frame.with_active_mask(replacement)

    replacement.fill_(False)
    updated.commands[0].payload.positions.fill_(4.0)
    assert updated.active_mask.tolist() == [False, True]
    assert frame.active_mask.tolist() == [True, False]
    assert isinstance(frame.commands[0].payload, JointPositionPayload)
    assert frame.commands[0].payload.positions.tolist() == [[0.0], [0.0]]


def test_timed_command_sequence_preserves_empty_batch_and_device() -> None:
    env_ids = torch.tensor([3, 7], dtype=torch.long)
    sequence = TimedCommandSequence(frames=(), env_ids=env_ids)

    env_ids.fill_(0)
    assert sequence.frame_count == 0
    assert sequence.batch_size == 2
    assert sequence.device == torch.device("cpu")
    assert sequence.env_ids.tolist() == [3, 7]
    assert sequence.targets == ()


def test_timed_command_sequence_requires_matching_frame_env_ids() -> None:
    frame = _frame((_joint_command("arm", (0,), torch.zeros(2, 1)),))
    with pytest.raises(ValueError, match="env_ids do not match"):
        TimedCommandSequence(
            frames=(frame,),
            env_ids=torch.tensor([4, 8], dtype=torch.long),
        )


def test_timed_command_sequence_owns_frames_and_returns_unique_targets() -> None:
    first = _frame(
        (
            _joint_command("left", (0,), torch.zeros(2, 1)),
            _joint_command("right", (1,), torch.ones(2, 1)),
        )
    )
    second = _frame((_joint_command("left", (0,), torch.full((2, 1), 2.0)),))
    sequence = TimedCommandSequence(
        frames=(first, second),
        env_ids=torch.tensor([4, 9]),
    )
    snapshot = sequence.snapshot()

    snapshot.frames[0].active_mask.fill_(False)
    targets = sequence.targets
    assert sequence.frame_count == 2
    assert sequence.frames[0].active_mask.tolist() == [True, False]
    assert [target.target_id for target in targets] == ["left", "right"]
    assert targets[0] is not sequence.frames[0].commands[0].target


def test_timed_command_sequence_rejects_invalid_frame_values() -> None:
    with pytest.raises(TypeError, match="RuntimeCommandFrame"):
        TimedCommandSequence(
            frames=(object(),),  # type: ignore[arg-type]
            env_ids=torch.tensor([0], dtype=torch.long),
        )


def test_timed_command_sequence_requires_nonempty_int64_batch() -> None:
    with pytest.raises(ValueError, match="int64"):
        TimedCommandSequence(frames=(), env_ids=torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="int64"):
        TimedCommandSequence(frames=(), env_ids=torch.tensor([0.0]))
    with pytest.raises(ValueError, match="unique"):
        TimedCommandSequence(frames=(), env_ids=torch.tensor([2, 2]))
