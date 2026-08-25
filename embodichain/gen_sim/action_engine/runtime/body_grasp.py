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

"""Pure body-grasp candidate filtering for elongated rigid objects."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from collections.abc import Callable

import torch

from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    AxisAlignGoal,
)
from .geometry_axes import LocalGeometryAxes
from .geometry_axes import analyze_local_geometry_axes

__all__ = [
    "AxisAlignBodyGraspAdapter",
    "BodyGraspAdaptation",
    "BodyGraspSelection",
    "select_body_grasp_candidates",
]


@dataclass(frozen=True, slots=True)
class BodyGraspSelection:
    """One selected body grasp per environment row."""

    success: torch.Tensor
    grasp_xpos: torch.Tensor
    candidate_indices: torch.Tensor
    body_candidate_counts: torch.Tensor
    central_candidate_counts: torch.Tensor
    radial_candidate_counts: torch.Tensor
    minimum_normalized_axial_offset: torch.Tensor
    minimum_long_axis_opening_cosine: torch.Tensor
    reachable_candidate_counts: torch.Tensor
    ranked_grasp_xpos: torch.Tensor
    ranked_candidate_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class BodyGraspAdaptation:
    """AxisAlign goal plus auditable body-grasp selection metadata."""

    goal: AxisAlignGoal
    alternative_goals: tuple[AxisAlignGoal, ...]
    alternative_rank_indices: tuple[int, ...]
    axes: LocalGeometryAxes
    selection: BodyGraspSelection


class AxisAlignBodyGraspAdapter:
    """Lower elongated-object semantics into an explicit mainline grasp goal."""

    def __init__(
        self,
        *,
        body_band_fraction: float = 0.80,
        maximum_long_axis_opening_cosine: float = 0.50,
    ) -> None:
        self.body_band_fraction = body_band_fraction
        self.maximum_long_axis_opening_cosine = maximum_long_axis_opening_cosine

    def adapt(
        self,
        goal: AxisAlignGoal,
        *,
        object_pose: torch.Tensor,
        grasp_generator: object,
        approach_direction: torch.Tensor,
        target_axis: torch.Tensor,
        seed: int,
        candidate_feasibility: Callable[[torch.Tensor], torch.Tensor] | None = None,
        maximum_adaptations: int = 12,
    ) -> BodyGraspAdaptation:
        affordance = goal.semantics.affordance
        if not isinstance(affordance, AxisAlignAffordance):
            raise TypeError("AxisAlign body grasp requires AxisAlignAffordance.")
        if affordance.mesh_vertices is None or affordance.mesh_triangles is None:
            raise ValueError("AxisAlign body grasp requires indexed mesh geometry.")
        axes = analyze_local_geometry_axes(affordance.mesh_vertices)
        sampled = self._sample(
            grasp_generator,
            seed=seed,
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=object_pose,
            approach_direction=approach_direction,
            obj_longest_axis=None,
            is_positive_part=True,
        )
        candidates, costs = self._pack(sampled, object_pose)
        feasible = (
            None if candidate_feasibility is None else candidate_feasibility(candidates)
        )
        selection = select_body_grasp_candidates(
            candidates,
            costs,
            object_pose,
            axes,
            body_band_fraction=self.body_band_fraction,
            maximum_long_axis_opening_cosine=(self.maximum_long_axis_opening_cosine),
            feasible=feasible,
        )
        if not bool(selection.success.all().item()):
            failed = (
                torch.nonzero(~selection.success, as_tuple=False).flatten().tolist()
            )
            raise ValueError(
                "No central radial body grasp is available for rows "
                f"{failed}; central_counts="
                f"{selection.central_candidate_counts.tolist()}, radial_counts="
                f"{selection.radial_candidate_counts.tolist()}, min_axial="
                f"{selection.minimum_normalized_axial_offset.tolist()}, "
                "min_opening_cos="
                f"{selection.minimum_long_axis_opening_cosine.tolist()}, "
                "reachable_counts="
                f"{selection.reachable_candidate_counts.tolist()}."
            )
        adaptation_count = min(
            maximum_adaptations,
            selection.ranked_grasp_xpos.shape[1],
        )
        alternative_ranks = tuple(
            int(value)
            for value in torch.linspace(
                0,
                selection.ranked_grasp_xpos.shape[1] - 1,
                adaptation_count,
            )
            .round()
            .to(torch.int64)
            .tolist()
        )
        goals = tuple(
            replace(goal, grasp_xpos=selection.ranked_grasp_xpos[:, rank])
            for rank in alternative_ranks
        )
        del target_axis
        return BodyGraspAdaptation(
            goal=goals[0],
            alternative_goals=goals,
            alternative_rank_indices=alternative_ranks,
            axes=axes,
            selection=selection,
        )

    @staticmethod
    def _sample(
        generator: object,
        *,
        seed: int,
        **kwargs: object,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        poses = kwargs.get("obj_poses")
        if not isinstance(poses, torch.Tensor):
            raise TypeError("obj_poses must be a torch.Tensor.")
        devices: list[int] = []
        if poses.device.type == "cuda":
            devices.append(
                torch.cuda.current_device()
                if poses.device.index is None
                else poses.device.index
            )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return generator.get_valid_grasp_poses(  # type: ignore[attr-defined]
                **kwargs
            )

    @staticmethod
    def _pack(
        sampled: list[tuple[torch.Tensor, torch.Tensor]],
        object_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(sampled) != object_pose.shape[0]:
            raise ValueError("Grasp generator must return one result per object row.")
        count = max((poses.shape[0] for poses, _ in sampled), default=0)
        if count == 0:
            raise ValueError("Grasp generator returned no candidates.")
        candidates = torch.eye(
            4,
            dtype=torch.float32,
            device=object_pose.device,
        ).repeat(object_pose.shape[0], count, 1, 1)
        costs = torch.full(
            (object_pose.shape[0], count),
            torch.inf,
            dtype=torch.float32,
            device=object_pose.device,
        )
        for env_index, (poses, values) in enumerate(sampled):
            row_count = poses.shape[0]
            if row_count == 0:
                continue
            candidates[env_index, :row_count] = poses.to(
                device=object_pose.device,
                dtype=torch.float32,
            )
            costs[env_index, :row_count] = values.to(
                device=object_pose.device,
                dtype=torch.float32,
            )
        return candidates, costs


def select_body_grasp_candidates(
    candidates: torch.Tensor,
    costs: torch.Tensor,
    object_pose: torch.Tensor,
    axes: LocalGeometryAxes,
    *,
    body_band_fraction: float = 0.80,
    maximum_long_axis_opening_cosine: float = 0.50,
    feasible: torch.Tensor | None = None,
) -> BodyGraspSelection:
    """Keep central radial grasps and reject cap/end grasps."""
    if candidates.ndim != 4 or candidates.shape[-2:] != (4, 4):
        raise ValueError("candidates must have shape (B, N, 4, 4).")
    if costs.shape != candidates.shape[:2]:
        raise ValueError("costs must have shape (B, N).")
    if object_pose.shape != (candidates.shape[0], 4, 4):
        raise ValueError("object_pose must have shape (B, 4, 4).")
    if not 0.0 < body_band_fraction <= 1.0:
        raise ValueError("body_band_fraction must be in (0, 1].")
    if not 0.0 <= maximum_long_axis_opening_cosine < 1.0:
        raise ValueError("maximum_long_axis_opening_cosine must be in [0, 1).")
    if feasible is None:
        feasible = torch.ones_like(costs, dtype=torch.bool)
    if feasible.dtype != torch.bool or feasible.shape != costs.shape:
        raise ValueError("feasible must be a bool tensor shaped (B, N).")

    rotation = object_pose[:, :3, :3]
    translation = object_pose[:, :3, 3]
    local_centers = torch.matmul(
        candidates[..., :3, 3] - translation[:, None],
        rotation,
    )
    center = axes.bounds_center.to(
        device=candidates.device,
        dtype=candidates.dtype,
    )
    long_axis = axes.long_axis.to(
        device=candidates.device,
        dtype=candidates.dtype,
    )
    axial_offset = torch.abs(torch.sum((local_centers - center) * long_axis, dim=-1))
    normalized_axial = axial_offset / max(axes.long_half_extent, 1.0e-8)
    within_body = normalized_axial <= body_band_fraction

    world_opening = torch.nn.functional.normalize(candidates[..., :3, 0], dim=-1)
    local_opening = torch.matmul(world_opening, rotation)
    long_axis_opening = torch.abs(torch.sum(local_opening * long_axis, dim=-1))
    radial = long_axis_opening <= maximum_long_axis_opening_cosine
    valid = within_body & radial & feasible & torch.isfinite(costs)

    ranked = torch.where(valid, costs, torch.inf)
    best_cost, best_index = ranked.min(dim=1)
    env_index = torch.arange(candidates.shape[0], device=candidates.device)
    valid_counts = valid.sum(dim=1)
    rank_count = int(valid_counts.min().item())
    ranked_indices = torch.argsort(ranked, dim=1)[:, :rank_count]
    ranked_grasps = candidates[
        env_index[:, None],
        ranked_indices,
    ].clone()
    return BodyGraspSelection(
        success=torch.isfinite(best_cost),
        grasp_xpos=candidates[env_index, best_index].clone(),
        candidate_indices=best_index.clone(),
        body_candidate_counts=valid.sum(dim=1),
        central_candidate_counts=within_body.sum(dim=1),
        radial_candidate_counts=radial.sum(dim=1),
        minimum_normalized_axial_offset=normalized_axial.min(dim=1).values,
        minimum_long_axis_opening_cosine=long_axis_opening.min(dim=1).values,
        reachable_candidate_counts=feasible.sum(dim=1),
        ranked_grasp_xpos=ranked_grasps,
        ranked_candidate_indices=ranked_indices.clone(),
    )
