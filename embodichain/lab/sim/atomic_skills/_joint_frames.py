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

"""Lower full-robot trajectories to owned per-endpoint command frames."""

from __future__ import annotations

import torch

from ._command_batches import _copy_joint_command, _plain_cpu_tensor, own
from .bindings import JointPositionTarget
from .plans import TimedTrajectory
from .runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    TimedCommandSequence,
    _canonical_joint_target,
    _frame_from_fresh_commands,
    _sequence_from_fresh_frames,
)

__all__: list[str] = []


def _frame_at(
    targets: tuple[JointPositionTarget, ...],
    trajectory: TimedTrajectory,
    active_mask: torch.Tensor,
    index: int,
) -> RuntimeCommandFrame:
    """Construct one frame through the ordinary target and payload contracts."""
    commands = []
    for target in targets:
        joints = list(target.joint_ids)
        velocities = (
            None
            if trajectory.velocities is None
            else trajectory.velocities[:, index, joints]
        )
        commands.append(
            EndpointCommand(
                target,
                JointPositionPayload(
                    trajectory.positions[:, index, joints], velocities
                ),
            )
        )
    # dt records arrival intervals. The terminal sample holds its own interval.
    arrival = min(index + 1, trajectory.waypoint_count - 1)
    return _frame_from_fresh_commands(
        commands,
        active_mask=active_mask,
        env_ids=trajectory.env_ids,
        hold_duration=trajectory.dt[:, arrival],
    )


def _can_batch(
    targets: tuple[JointPositionTarget, ...],
    trajectory: TimedTrajectory,
    active_mask: torch.Tensor,
) -> bool:
    if type(trajectory) is not TimedTrajectory:
        return False
    fields = (active_mask, trajectory.env_ids, trajectory.dt, trajectory.positions)
    if trajectory.velocities is not None:
        fields += (trajectory.velocities,)
    return (
        all(_plain_cpu_tensor(t) for t in fields)
        and trajectory.positions.ndim == 3
        and trajectory.dt.shape == trajectory.positions.shape[:2]
        and (
            trajectory.velocities is None
            or trajectory.velocities.shape == trajectory.positions.shape
        )
        and all(_canonical_joint_target(t) for t in targets)
    )


def _batch_frames(
    first: RuntimeCommandFrame, trajectory: TimedTrajectory
) -> TimedCommandSequence | None:
    """Reuse the first frame's validated joint layout for subsequent waypoints."""
    indices = [
        torch.tensor(
            c.target.joint_ids, dtype=torch.long, device=trajectory.positions.device
        )
        for c in first.commands
    ]
    count = trajectory.waypoint_count
    frames = [first]
    # Bound the gathered positions to 64K elements (or one wide waypoint).
    elements_per_waypoint = sum(c.payload.positions.numel() for c in first.commands)
    block_size = max(1, min(128, 65536 // elements_per_waypoint))
    for start in range(1, count, block_size):
        stop = min(start + block_size, count)
        arrivals = trajectory.dt[:, min(start + 1, count - 1) : stop + 1].clone()
        if not torch.isfinite(arrivals).all() or (arrivals < 0).any():
            return None

        holds = arrivals.unbind(1)
        blocks = []
        for command, joints in zip(first.commands, indices):
            positions = trajectory.positions[:, start:stop].index_select(2, joints)
            velocities = None
            if trajectory.velocities is not None:
                velocities = trajectory.velocities[:, start:stop].index_select(
                    2, joints
                )
            if not torch.isfinite(positions).all() or (
                velocities is not None and not torch.isfinite(velocities).all()
            ):
                return None
            blocks.append(
                (
                    command.target,
                    positions.unbind(1),
                    None if velocities is None else velocities.unbind(1),
                )
            )

        for offset in range(stop - start):
            commands = tuple(
                _copy_joint_command(
                    target,
                    positions[offset],
                    None if velocities is None else velocities[offset],
                )
                for target, positions, velocities in blocks
            )
            frames.append(
                own(
                    RuntimeCommandFrame,
                    commands=commands,
                    active_mask=first.active_mask.clone(),
                    env_ids=first.env_ids.clone(),
                    hold_duration=holds[min(offset, len(holds) - 1)].clone(),
                )
            )
    return own(
        TimedCommandSequence, frames=tuple(frames), env_ids=first.env_ids.clone()
    )


def lower_joint_frames(
    targets: tuple[JointPositionTarget, ...],
    trajectory: TimedTrajectory,
    active_mask: torch.Tensor,
) -> TimedCommandSequence:
    """Lower CPU joint data in time blocks while preserving validation and ownership."""
    count = trajectory.waypoint_count
    if not count:
        return _sequence_from_fresh_frames([], trajectory.env_ids)

    first = _frame_at(targets, trajectory, active_mask, 0)
    if count > 1 and _can_batch(targets, trajectory, active_mask):
        sequence = _batch_frames(first, trajectory)
        if sequence is not None:
            return sequence

    # Extensions keep their snapshot callbacks. Invalid numerical data also
    # follows the original per-frame order and field-specific diagnostics.
    frames = [first]
    frames.extend(
        _frame_at(targets, trajectory, active_mask, i) for i in range(1, count)
    )
    return _sequence_from_fresh_frames(frames, trajectory.env_ids)
