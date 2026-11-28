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


"""Validate and copy batches of built-in joint command frames."""

from __future__ import annotations

from collections import defaultdict
import torch
from .bindings import JointPositionTarget
from .runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    TimedCommandSequence,
    _canonical_cpu_command,
    _unique_env_ids,
)

__all__: list[str] = []


def _transfer(tensors, device):
    """Own one slab per dtype; preserve bits, shapes and independent field spans."""
    import torch

    grouped = defaultdict(dict)
    for tensor in tensors:
        grouped[tensor.dtype][id(tensor)] = tensor
    copied = {}
    for group in grouped.values():
        values = tuple(group.values())
        packed = torch.cat([value.detach().reshape(-1) for value in values]).to(device)
        for value, part in zip(values, packed.split([v.numel() for v in values])):
            copied[id(value)] = part.reshape(value.shape)
    return copied


def leaves(frame):
    if type(frame) is not RuntimeCommandFrame or type(frame.commands) not in (
        list,
        tuple,
    ):
        return None
    tensors = [frame.active_mask, frame.env_ids, frame.hold_duration]
    for command in frame.commands:
        if (
            type(command) is not EndpointCommand
            or type(command.target) is not JointPositionTarget
            or type(command.payload) is not JointPositionPayload
        ):
            return None
        tensors.append(command.payload.positions)
        if command.payload.velocities is not None:
            tensors.append(command.payload.velocities)
    if any(type(value) is not torch.Tensor or value.requires_grad for value in tensors):
        return None
    if frame.env_ids.device.type != "cuda" or any(
        value.device != frame.env_ids.device for value in tensors
    ):
        return None
    return tensors


def cpu_frame(frame, copied):
    commands = tuple(
        EndpointCommand(
            target=command.target,
            payload=JointPositionPayload(
                positions=copied[id(command.payload.positions)],
                velocities=(
                    copied[id(command.payload.velocities)]
                    if command.payload.velocities is not None
                    else None
                ),
            ),
        )
        for command in frame.commands
    )
    return RuntimeCommandFrame(
        commands=commands,
        active_mask=copied[id(frame.active_mask)],
        env_ids=copied[id(frame.env_ids)],
        hold_duration=copied[id(frame.hold_duration)],
    )


def own(cls, **fields):
    value = object.__new__(cls)
    for name, field in fields.items():
        object.__setattr__(value, name, field)
    return value


def lift_frame(frame, copied):
    return own(
        RuntimeCommandFrame,
        commands=tuple(
            own(
                EndpointCommand,
                target=command.target,
                payload=own(
                    JointPositionPayload,
                    positions=copied[id(command.payload.positions)],
                    velocities=(
                        copied[id(command.payload.velocities)]
                        if command.payload.velocities is not None
                        else None
                    ),
                ),
            )
            for command in frame.commands
        ),
        active_mask=copied[id(frame.active_mask)],
        env_ids=copied[id(frame.env_ids)],
        hold_duration=copied[id(frame.hold_duration)],
    )


def numeric(frame):
    # Already passed all native CPU checks; collect owned validated fields.
    values = [frame.active_mask, frame.env_ids, frame.hold_duration]
    for command in frame.commands:
        values.append(command.payload.positions)
        if command.payload.velocities is not None:
            values.append(command.payload.velocities)
    return values


def validate_frame(frame):
    if (
        type(frame) is not RuntimeCommandFrame
        or type(frame.env_ids) is not torch.Tensor
        or frame.env_ids.device.type != "cuda"
    ):
        return False
    values = leaves(frame)
    if values is None:
        return False
    checked = cpu_frame(frame, _transfer(values, "cpu"))
    result = lift_frame(checked, _transfer(numeric(checked), frame.env_ids.device))
    for name in ("commands", "active_mask", "env_ids", "hold_duration"):
        object.__setattr__(frame, name, getattr(result, name))
    return True


def _plain_cpu_tensor(value: object) -> bool:
    """Identify dense CPU fields with ordinary cloning and stacking semantics."""
    return (
        type(value) is torch.Tensor
        and value.device.type == "cpu"
        and value.layout == torch.strided
        and not value.is_quantized
        and not value.is_nested
        and not value.requires_grad
        and not value.has_names()
    )


def _cpu_field(value: torch.Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> bool:
    return _plain_cpu_tensor(value) and value.dtype == dtype and value.shape == shape


def _copy_joint_command(
    target: JointPositionTarget,
    positions: torch.Tensor,
    velocities: torch.Tensor | None,
) -> EndpointCommand:
    """Copy plain joint fields whose layout and values the caller validates."""
    return own(
        EndpointCommand,
        target=own(
            JointPositionTarget,
            control_part=target.control_part,
            joint_ids=target.joint_ids,
        ),
        payload=own(
            JointPositionPayload,
            positions=positions.clone(),
            velocities=None if velocities is None else velocities.clone(),
        ),
    )


def _copy_joint_frame(
    frame: RuntimeCommandFrame,
    expected_commands: tuple[EndpointCommand, ...],
    shape: torch.Size,
) -> RuntimeCommandFrame | None:
    """Copy ordinary joint fields sharing the expected endpoint and velocity layout."""
    if (
        type(frame.commands) is not tuple
        or len(frame.commands) != len(expected_commands)
        or not _cpu_field(frame.env_ids, torch.long, shape)
        or not _cpu_field(frame.active_mask, torch.bool, shape)
        or not _cpu_field(frame.hold_duration, torch.float32, shape)
    ):
        return None

    commands = []
    for command, expected in zip(frame.commands, expected_commands):
        if (
            not _canonical_cpu_command(command)
            or vars(command.payload)
            or command.target != expected.target
        ):
            return None
        payload = command.payload
        joint_shape = (shape[0], len(command.target.joint_ids))
        if not _cpu_field(payload.positions, torch.float32, joint_shape):
            return None
        velocity = payload.velocities
        if (velocity is None) != (expected.payload.velocities is None):
            return None
        if velocity is not None and not _cpu_field(
            velocity, torch.float32, joint_shape
        ):
            return None
        commands.append(
            _copy_joint_command(expected.target, payload.positions, velocity)
        )
    return own(
        RuntimeCommandFrame,
        commands=tuple(commands),
        env_ids=frame.env_ids.clone(),
        active_mask=frame.active_mask.clone(),
        hold_duration=frame.hold_duration.clone(),
    )


def _valid_joint_chunk(
    frames: list[RuntimeCommandFrame], env_ids: torch.Tensor
) -> bool:
    """Check the newly owned numeric fields across time before publishing them."""
    if not (torch.stack([frame.env_ids for frame in frames]) == env_ids).all():
        return False
    holds = torch.stack([frame.hold_duration for frame in frames])
    if not (torch.isfinite(holds) & (holds >= 0)).all():
        return False
    for i, command in enumerate(frames[0].commands):
        payloads = [frame.commands[i].payload for frame in frames]
        if not torch.isfinite(torch.stack([p.positions for p in payloads])).all():
            return False
        if (
            command.payload.velocities is not None
            and not torch.isfinite(torch.stack([p.velocities for p in payloads])).all()
        ):
            return False
    return True


def snapshot_cpu_sequence(
    sequence: TimedCommandSequence,
) -> TimedCommandSequence | None:
    """Batch repeated joint-frame checks, falling back to ordered public validation."""
    frames = sequence.frames
    ids = sequence.env_ids
    if (
        type(sequence) is not TimedCommandSequence
        or type(frames) is not tuple
        or len(frames) < 2
        or type(ids) is not torch.Tensor
        or ids.is_nested
        or ids.ndim != 1
        or not _cpu_field(ids, torch.long, ids.shape)
        or not ids.numel()
        or not all(type(frame) is RuntimeCommandFrame for frame in frames)
    ):
        return None

    first = frames[0]
    if type(first.commands) is not tuple or not first.commands:
        return None
    ids = ids.clone()
    template = _copy_joint_frame(first, first.commands, ids.shape)
    if (
        template is None
        or not torch.equal(template.env_ids, ids)
        or not _unique_env_ids(ids)
    ):
        return None
    # The public constructor checks all destination conflicts and target invariants
    # once. Subsequent frames must match these exact built-in targets and shapes.
    template = template.snapshot()
    snapshots = [template]
    # Bound the extra stack used for a single field to 64K elements (or one
    # unusually large field), including for trajectories with many waypoints.
    max_field = max(cmd.payload.positions.numel() for cmd in template.commands)
    chunk_size = max(1, min(128, 65536 // max_field))
    for start in range(1, len(frames), chunk_size):
        chunk = []
        for frame in frames[start : start + chunk_size]:
            copied = _copy_joint_frame(frame, template.commands, ids.shape)
            if copied is None:
                return None
            chunk.append(copied)
        if not _valid_joint_chunk(chunk, ids):
            return None
        snapshots.extend(chunk)
    return own(TimedCommandSequence, frames=tuple(snapshots), env_ids=ids)


def validate_sequence(sequence):
    if (
        type(sequence.frames) not in (list, tuple)
        or type(sequence.env_ids) is not torch.Tensor
        or sequence.env_ids.device.type != "cuda"
        or sequence.env_ids.requires_grad
    ):
        return False
    tensors = [sequence.env_ids]
    for frame in sequence.frames:
        values = leaves(frame)
        if values is None or frame.env_ids.device != sequence.env_ids.device:
            return False
        tensors.extend(values)
    copied = _transfer(tensors, "cpu")
    # Preserve native sequence-before-frame rejection order, including
    # malformed mutable IDs in an otherwise previously valid snapshot.
    TimedCommandSequence(frames=(), env_ids=copied[id(sequence.env_ids)])
    for index, frame in enumerate(sequence.frames):
        if not torch.equal(copied[id(frame.env_ids)], copied[id(sequence.env_ids)]):
            raise ValueError(f"Frame {index} env_ids do not match the sequence.")
    checked = TimedCommandSequence(
        frames=tuple(cpu_frame(frame, copied) for frame in sequence.frames),
        env_ids=copied[id(sequence.env_ids)],
    )
    tensors = [checked.env_ids]
    for frame in checked.frames:
        tensors.extend(numeric(frame))
    copied = _transfer(tensors, sequence.env_ids.device)
    object.__setattr__(
        sequence,
        "frames",
        tuple(lift_frame(frame, copied) for frame in checked.frames),
    )
    object.__setattr__(sequence, "env_ids", copied[id(checked.env_ids)])
    return True
