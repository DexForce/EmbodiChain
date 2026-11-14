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

"""Explicit waypoint phase paths composed through the ordinary motion facade."""

from __future__ import annotations

from dataclasses import replace
from itertools import groupby
from typing import TYPE_CHECKING
import torch
from .planners.utils import MoveType, PlanResult, RejectedMotionCandidate

if TYPE_CHECKING:
    from .motion_generator import MotionGenerator, MotionGenOptions
    from .planners.utils import PlanState

__all__: list[str] = []


def generate_phase_paths(
    generator: MotionGenerator,
    targets: list[PlanState],
    options: MotionGenOptions,
    *,
    defer_recovery: bool,
) -> PlanResult:
    """Retain one declared grid while grouping consecutive path modes."""
    modes = options.waypoint_modes
    if len(modes) != len(targets) or not all(
        t.move_type is MoveType.EEF_MOVE for t in targets
    ):
        raise ValueError(
            f"waypoint_modes must describe every Cartesian waypoint: modes={modes}, target_types={[state.move_type for state in targets]}"
        )
    if (
        options.start_qpos is None
        or options.sample_count is None
        or options.sample_count <= len(targets)
    ):
        raise ValueError(
            "phase paths require a start and at least one interval per waypoint"
        )
    if type(defer_recovery) is not bool or (
        defer_recovery and "cartesian" not in modes
    ):
        raise ValueError("deferred phase recovery requires a Cartesian phase")
    pieces, start, offset = [], options.start_qpos, 0
    for mode, group in groupby(modes):
        stop = offset + len(tuple(group))
        intervals = (options.sample_count - 1) * stop // len(targets) - (
            options.sample_count - 1
        ) * offset // len(targets)
        result = generator.generate(
            targets[offset:stop],
            options=replace(
                options,
                waypoint_modes=(),
                start_qpos=start,
                sample_count=intervals + 1,
                is_linear=mode == "cartesian",
            ),
            defer_recovery=defer_recovery and mode == "cartesian",
        )
        if result.positions is None or result.dt is None:
            raise ValueError("phase path returned no timed trajectory")
        pieces.append(result)
        start = result.positions[:, -1]
        offset = stop
    success = torch.stack([piece.success for piece in pieces]).all(dim=0)
    positions = torch.cat(
        [p.positions if i == 0 else p.positions[:, 1:] for i, p in enumerate(pieces)],
        dim=1,
    )
    dt = torch.cat(
        [p.dt if i == 0 else p.dt[:, 1:] for i, p in enumerate(pieces)], dim=1
    )
    rejected = None
    if any(p.rejected_candidate is not None for p in pieces):
        mask = torch.zeros_like(success)
        candidates = []
        for i, p in enumerate(pieces):
            points = p.positions
            if p.rejected_candidate is not None:
                mask |= p.rejected_candidate.env_mask
                points = p.rejected_candidate.positions
            candidates.append(points if i == 0 else points[:, 1:])
        rejected = RejectedMotionCandidate(torch.cat(candidates, dim=1), dt, mask)
    return PlanResult(
        success=success,
        positions=torch.where(
            success[:, None, None], positions, options.start_qpos[:, None]
        ),
        dt=dt,
        rejected_candidate=rejected,
        recovery=pieces[-1].recovery,
    )
