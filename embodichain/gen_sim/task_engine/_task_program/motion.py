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

"""Cartesian approach sampling as a task-owned motion-planning policy."""

from __future__ import annotations

from dataclasses import replace

import torch

from embodichain.lab.sim.planners.motion_generator import (
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.planners.utils import (
    MoveType,
    PlanResult,
    PlanState,
    interpolate_xpos_batched,
)

__all__: list[str] = []


def _cartesian_samples(
    start: torch.Tensor,
    targets: list[PlanState],
    count: int,
) -> list[PlanState]:
    """Include every supplied waypoint without changing the output time budget."""
    if count - 1 < len(targets):
        raise ValueError("Cartesian approach sampling needs one sample per target.")
    result: list[PlanState] = []
    previous = start
    remaining = count - 1
    for index, target in enumerate(targets):
        assert target.xpos is not None
        pose = target.xpos.to(device=start.device, dtype=start.dtype)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0).expand(start.shape[0], -1, -1)
        if pose.shape != start.shape:
            raise ValueError("Cartesian approach targets must match the start batch.")
        intervals = remaining // (len(targets) - index)
        interpolated = interpolate_xpos_batched(previous, pose, intervals + 1)
        result.extend(
            PlanState(move_type=MoveType.EEF_MOVE, xpos=interpolated[:, point])
            for point in range(1, intervals + 1)
        )
        previous = pose
        remaining -= intervals
    return result


class ApproachMotionGenerator(MotionGenerator):
    """Keep multi-waypoint EEF approaches Cartesian; the core solves every sample."""

    def generate(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions | None = None,
    ) -> PlanResult:
        if (
            options is not None
            and options.strategy == "ik_interp"
            and not options.preserve_cartesian_samples
            and len(target_states) > 1
            and all(state.move_type is MoveType.EEF_MOVE for state in target_states)
        ):
            if (
                options.start_qpos is None
                or options.control_part is None
                or options.sample_count is None
            ):
                raise ValueError(
                    "Approach planning requires a bound start and sample count."
                )
            start = self.robot.compute_fk(
                qpos=options.start_qpos,
                name=options.control_part,
                to_matrix=True,
            )
            target_states = _cartesian_samples(
                start, target_states, options.sample_count
            )
            options = replace(options, preserve_cartesian_samples=True, is_linear=True)
        return super().generate(target_states, options=options)
