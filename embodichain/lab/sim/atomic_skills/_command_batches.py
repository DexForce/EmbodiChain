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


"""Validate owned CUDA command snapshots in bounded CPU slabs."""

from __future__ import annotations

from collections import defaultdict
import torch
from .bindings import JointPositionTarget
from .runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    TimedCommandSequence,
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
    values = leaves(frame)
    if values is None:
        return False
    checked = cpu_frame(frame, _transfer(values, "cpu"))
    result = lift_frame(checked, _transfer(numeric(checked), frame.env_ids.device))
    for name in ("commands", "active_mask", "env_ids", "hold_duration"):
        object.__setattr__(frame, name, getattr(result, name))
    return True


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
