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

"""Shared Atomic affordance-axis adaptation and ordered grasp sampling."""

from __future__ import annotations
from dataclasses import replace
from typing import Any, Literal
from collections.abc import Callable
import torch
from embodichain.compute.geometry.gripper import GraspApproach
from ..affordance_sampling import AffordancePoseCandidates

__all__: list[str] = []


def observed_axis(affordance: Any, pose: torch.Tensor) -> torch.Tensor:
    local = getattr(affordance, "internal_axis", None)
    axis = (
        affordance.get_object_longest_axis(pose, max_points=1000)
        if local is None
        else (pose[:, :3, :3] @ local.to(pose)[..., None]).squeeze(-1)
    )
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    if not torch.isfinite(axis).all() or (norm <= 1e-6).any():
        raise ValueError("Adaptive grasp requires a finite observed object axis.")
    return axis / norm


def select_grasp(
    affordance: Any,
    generator: Any,
    pose: torch.Tensor,
    choices: tuple[GraspApproach, ...],
    context: Any,
    key: str,
    feasible: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
    candidate_observer: Callable[[AffordancePoseCandidates], None] | None = None,
    raw_candidate_observer: Callable[[AffordancePoseCandidates], None] | None = None,
    axis_region_policy: Literal["mesh_extent", "supported_end"] = "mesh_extent",
    candidate_pool: AffordancePoseCandidates | None = None,
) -> tuple[Any, torch.Tensor]:
    """Keep the first feasible strategy per row and the normal branch sampler."""
    if axis_region_policy not in ("mesh_extent", "supported_end"):
        raise ValueError("Unknown axis_region_policy.")
    if candidate_pool is not None and len(choices) != 1:
        raise ValueError("A cached candidate pool belongs to exactly one strategy.")
    from embodichain.toolkits.graspkit.pg_grasp.regions import for_axis_region

    scoped_generator = for_axis_region(generator, axis_region_policy)
    covered = torch.zeros(pose.shape[0], dtype=torch.bool, device=pose.device)
    pools, directions, strategies, counts = [], [], [], []
    for index, choice in enumerate(choices):
        candidates = candidate_pool
        if candidates is None:
            candidates = affordance.get_grasp_candidates(
                scoped_generator,
                pose,
                choice.direction,
                obj_longest_axis=choice.axis,
                is_positive_part=choice.positive,
            )
        if raw_candidate_observer is not None:
            raw_candidate_observer(candidates)
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
    if candidate_observer is not None:
        candidate_observer(combined)
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
                    **(
                        {"axis_region_policy": axis_region_policy}
                        if axis_region_policy != "mesh_extent"
                        else {}
                    ),
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
