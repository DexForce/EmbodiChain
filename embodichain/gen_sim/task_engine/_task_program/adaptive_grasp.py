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

"""GenSim grasp geometry shared by pickup and continuation handover."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable

import torch

from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
)

__all__: list[str] = []

ADAPTIVE_GRASP = "gen_sim.adaptive_grasp"
ADAPTIVE_GRASP_REVISION = 1


@dataclass(frozen=True)
class GraspStrategy:
    direction: torch.Tensor
    axis: torch.Tensor | None
    positive: torch.Tensor


def observed_axis(affordance: Any, pose: torch.Tensor) -> torch.Tensor:
    local = getattr(affordance, "internal_axis", None)
    if local is None:
        axis = affordance.get_object_longest_axis(pose, max_points=1000)
    else:
        axis = (pose[:, :3, :3] @ local.to(pose)[..., None]).squeeze(-1)
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if not torch.isfinite(axis).all() or (norm <= 1e-6).any():
        raise ValueError("Adaptive grasp requires a finite observed object axis.")
    return axis / norm


def opposite_grasp_end(
    axis: torch.Tensor,
    pose: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
) -> torch.Tensor:
    projection = ((source[:, :3, 3] - pose[:, :3, 3]) * axis).sum(-1)
    near_destination = ((destination[:, :3, 3] - pose[:, :3, 3]) * axis).sum(-1) >= 0
    return torch.where(projection.abs() > 1e-5, projection < 0, near_destination)


def grasp_strategies(
    axis: torch.Tensor,
    pose: torch.Tensor,
    tcp: torch.Tensor,
    *,
    ends: bool,
    positive: torch.Tensor | None = None,
) -> tuple[GraspStrategy, ...]:
    """Prefer tutorial-style approaches without assuming an arm or axis sign."""
    down = torch.zeros_like(axis)
    down[:, 2] = -1
    toward = pose[:, :3, 3] - tcp[:, :3, 3]
    toward[:, 2] = 0
    length = torch.linalg.vector_norm(toward, dim=-1, keepdim=True)
    diagonal = toward / length.clamp_min(1e-6) * math.sin(math.pi / 3)
    diagonal[:, 2] = -0.5
    diagonal = torch.where(length > 1e-6, diagonal, down)
    vertical = axis[:, 2:3].abs() >= math.sqrt(0.5)
    directions = (
        torch.where(vertical, diagonal, down),
        torch.where(vertical, down, diagonal),
    )
    nearest = ((tcp[:, :3, 3] - pose[:, :3, 3]) * axis).sum(-1) >= 0
    parts = (
        (positive,)
        if positive is not None
        else ((nearest, ~nearest) if ends else (nearest,))
    )
    return tuple(
        GraspStrategy(direction, axis if ends else None, part)
        for direction in directions
        for part in parts
    )


def select_grasp(
    affordance: Any,
    generator: Any,
    pose: torch.Tensor,
    choices: tuple[GraspStrategy, ...],
    context: Any,
    key: str,
    feasible: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
) -> tuple[Any, torch.Tensor]:
    """Keep the first feasible strategy per row and the normal branch sampler."""
    covered = torch.zeros(pose.shape[0], dtype=torch.bool, device=pose.device)
    pools, directions, strategies, counts = [], [], [], []
    for index, choice in enumerate(choices):
        candidates = affordance.get_grasp_candidates(
            generator,
            pose,
            choice.direction,
            obj_longest_axis=choice.axis,
            is_positive_part=choice.positive,
        )
        poses, valid = (
            (candidates.poses, candidates.valid)
            if feasible is None
            else feasible(candidates.poses, choice.direction)
        )
        valid = candidates.valid & valid & ~covered[:, None]
        counts.append(valid.sum(-1).tolist())
        covered |= valid.any(-1)
        width = valid.shape[1]
        pools.append(
            AffordancePoseCandidates(poses=poses, costs=candidates.costs, valid=valid)
        )
        directions.append(choice.direction[:, None].expand(-1, width, -1))
        strategies.append(
            torch.full(valid.shape, index, dtype=torch.long, device=pose.device)
        )
        if covered.all():
            break
    combined = AffordancePoseCandidates(
        poses=torch.cat([p.poses for p in pools], 1),
        costs=torch.cat([p.costs for p in pools], 1),
        valid=torch.cat([p.valid for p in pools], 1),
    )
    sample = affordance.sample_candidates(
        combined,
        sampling=context.affordance_sampling,
        env_ids=context.env_ids,
        key=key,
        reference_poses=pose,
    )
    selected = torch.tensor(
        sample.metadata["candidate_ids"], device=pose.device
    ).clamp_min(0)
    rows = torch.arange(pose.shape[0], device=pose.device)
    direction = torch.cat(directions, 1)[rows, selected]
    strategy = torch.cat(strategies, 1)[rows, selected]
    return (
        replace(
            sample,
            metadata={
                **sample.metadata,
                "adaptive_grasp": {
                    "strategy_indices": torch.where(
                        sample.success, strategy, -1
                    ).tolist(),
                    "approach_directions": direction.tolist(),
                    "feasible_counts": counts,
                },
            },
        ),
        direction,
    )
