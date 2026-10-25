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

"""Shared conservative phase permissions for the Atomic and Gym sample hosts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.motion.expansion import TrajectoryPhase

# All other phases, including hand motion, contact paths and Pour's interior
# tilt target, remain exact. MoveJoints targets must be separate invocations.
FREE_PHASES: dict[str, frozenset[str]] = {
    "pick_up": frozenset({"approach", "lift"}),
    "place": frozenset({"approach", "retract"}),
    "pour": frozenset(),
    "press": frozenset({"approach", "retract"}),
    "slide": frozenset({"approach", "return"}),
    "twist": frozenset({"approach", "retract"}),
    # Only the empty-hand motion to the pickup pre-grasp is free. Every
    # subsequent phase coordinates a grasp or carries the object.
    "hand_over": frozenset({"pickup_approach"}),
    "move_joints": frozenset({"move_joints"}),
    "axis_align": frozenset({"approach", "manipulate"}),
    "open_door": frozenset({"approach", "retract"}),
    "move_held_object": frozenset({"transport"}),
    "push_object": frozenset({"approach", "retract"}),
}

__all__: list[str] = []


def _free_joint_indices(
    positions: torch.Tensor,
    phases: tuple[TrajectoryPhase, ...],
    motion_joint_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Keep joints that actually move in the declared free phases.

    A dual-arm binding includes both arms, but its stationary participant
    must retain the reference command during the other arm's approach.
    """
    active = torch.zeros(positions.shape[1], dtype=torch.bool, device=positions.device)
    for phase in phases:
        if phase.allowed_operators:
            samples = positions[phase.start_index : phase.stop_index]
            active |= (samples != samples[:1]).any(dim=0)
    return tuple(index for index in motion_joint_indices if bool(active[index]))


def _canonical_samples(
    positions: torch.Tensor,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove zero-time joins for expansion and retain their exact source map."""
    dt = dt.detach().cpu().clone()
    positions = positions.detach().cpu()
    dt[0] = 0
    keep = torch.ones(len(dt), dtype=torch.bool)
    for index in range(1, len(dt)):
        if float(dt[index]) == 0:
            if not torch.allclose(
                positions[index], positions[index - 1], atol=1e-5, rtol=0
            ):
                raise ValueError("Atomic action has a position change in zero time")
            keep[index] = False
    counts = torch.cumsum(keep.long(), 0)
    boundaries = torch.cat((torch.zeros(1, dtype=torch.long), counts))
    return positions[keep], dt[keep], boundaries, counts - 1
