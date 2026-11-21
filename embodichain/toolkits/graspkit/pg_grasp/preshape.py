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

"""Source-compatible full-hand tabletop candidate filtering without robot state."""

from __future__ import annotations
from dataclasses import dataclass
import json, math
from typing import Any
import torch
from embodichain.utils import logger

__all__ = [
    "PreshapeHandClearance",
    "HandleGraspCandidates",
    "generate_handle_grasp_candidates",
]


@dataclass(frozen=True, slots=True)
class PreshapeHandClearance:
    """Sampled closing sweep in TCP coordinates, not a collision-free certificate.

    Args:
        points: Whole-hand points from the source seven-state closing sweep.
        aperture: Nominal pad gap minus the original 4 mm contact reserve.
        max_retreat: Minimum pad tip depth minus the original 4 mm reserve.
        table_z: Measured world-frame support height.
    """

    points: torch.Tensor
    aperture: float
    max_retreat: float
    table_z: float

    def filter(
        self, poses: torch.Tensor, widths: torch.Tensor, costs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep sampler ranking; reject aperture/table violations without fallback.

        Args:
            poses: Candidate world TCP transforms, shaped ``(K,4,4)``.
            widths: Matching opening widths, shaped ``(K,)``.
            costs: Original matching costs, with infinity marking invalid rows.

        Returns:
            Shifted poses, unchanged widths and original costs masked by full-hand
            table and aperture qualification. Only original 5 mm retreat increments
            and the 7 mm table reserve are evaluated.
        """
        result = poses.clone()
        if self.max_retreat < 0.020:
            return result, widths, torch.full_like(costs, torch.inf)
        accepted = torch.zeros_like(costs, dtype=torch.bool)
        best_z = torch.full_like(costs, -torch.inf)
        finite = torch.isfinite(costs) & torch.isfinite(widths)
        finite &= torch.isfinite(poses).all(dim=-1).all(dim=-1)
        finite &= (widths >= 0.0) & (widths <= self.aperture)
        points = self.points.to(poses)
        # Reserve pad depth for contact rather than moving arbitrarily far out.
        # Seven mm includes the five-mm geometric margin and tracking allowance.
        for retreat in torch.arange(0.020, self.max_retreat + 1.0e-7, 0.005):
            shifted = poses.clone()
            shifted[:, :3, 3] -= float(retreat) * poses[:, :3, 2]
            min_z = (points @ shifted[:, 2, :3].T).amin(dim=0) + shifted[:, 2, 3]
            best_z = torch.maximum(best_z, min_z)
            clear = finite & ~accepted & (min_z >= self.table_z + 0.007)
            result[clear] = shifted[clear]
            accepted |= clear
        if not accepted.any():
            raw = torch.isfinite(costs) & torch.isfinite(widths) & (widths >= 0)
            raw &= torch.isfinite(poses).all(dim=-1).all(dim=-1)
            logger.log_info(
                "GenSim E6 grasp screen: "
                + json.dumps(
                    {
                        "raw_count": costs.numel(),
                        "finite_source_candidates": int(raw.sum()),
                        "opening_pass": int(finite.sum()),
                        "table_pass": int(accepted.sum()),
                        "required_width_min_m": (
                            float(widths[raw].min()) if raw.any() else None
                        ),
                        "usable_aperture_m": self.aperture,
                        "max_retreat_m": self.max_retreat,
                        "best_hand_min_z_m": (
                            float(best_z[finite].max()) if finite.any() else None
                        ),
                        "required_hand_min_z_m": self.table_z + 0.007,
                        "scope": "geometric_rejection_counts_not_physical_success",
                    }
                )
            )
        return result, widths, torch.where(accepted, costs, torch.inf)


@dataclass(frozen=True, slots=True)
class HandleGraspCandidates:
    """Filtered rows plus original finite-candidate aperture facts."""

    rows: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    required_width: float
    width_fit: bool


def generate_handle_grasp_candidates(
    generator: Any,
    *,
    mesh_vertices: torch.Tensor,
    mesh_triangles: torch.Tensor,
    obj_poses: torch.Tensor,
    approach_direction: torch.Tensor,
    clearance: PreshapeHandClearance,
) -> HandleGraspCandidates:
    """Keep source diagonal approach and fallback cross-axis handle candidates.

    Args:
        generator: Existing configured parallel-jaw candidate service.
        mesh_vertices: Handle-local measured geometry.
        mesh_triangles: Ordered handle triangle indices.
        obj_poses: Observed handle poses with an environment dimension.
        approach_direction: Explicit bound joint translation axes in world space.
        clearance: Owned seven-state hand sweep and measured table height.

    Returns:
        Filtered candidate rows retaining source costs, finite masks and fallback
        ordering, plus required-width facts for the Atomic aperture recovery.
        No IK, physical observation or success decision is performed.
    """
    if approach_direction.ndim == 1:
        direction = approach_direction.unsqueeze(0)
        squeeze = True
    elif approach_direction.ndim == 2 and approach_direction.shape[-1] == 3:
        direction = approach_direction
        squeeze = False
    else:
        raise ValueError("Handle approach direction must have shape (3,) or (B,3).")
    horizontal = direction.clone()
    horizontal[:, 2] = 0.0
    norm = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True)
    valid = norm.squeeze(-1) > 1.0e-6
    if valid.any():
        diagonal = horizontal[valid] / norm[valid]
        diagonal = diagonal * math.sqrt(0.5)
        diagonal[:, 2] = -math.sqrt(0.5)
        direction = direction.clone()
        direction[valid] = diagonal
    direction = direction[0] if squeeze else direction
    rows = generator.get_grasp_candidates(
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        obj_poses=obj_poses,
        approach_direction=direction,
        obj_longest_axis=None,
        is_positive_part=True,
    )
    required_width = math.inf
    width_fit = False

    def screened(poses, widths, costs):
        nonlocal required_width, width_fit
        finite = torch.isfinite(costs) & torch.isfinite(widths) & (widths >= 0)
        finite &= torch.isfinite(poses).all(dim=-1).all(dim=-1)
        if finite.any():
            required_width = min(required_width, float(widths[finite].min()))
            width_fit |= bool((widths[finite] <= clearance.aperture).any())
        return clearance.filter(poses, widths, costs)

    rotation = obj_poses.new_tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    augmented = []
    for row, (poses, widths, costs) in enumerate(rows):
        filtered = screened(poses, widths, costs)
        if torch.isfinite(filtered[2]).any():
            augmented.append(filtered)
            continue
        centered = mesh_vertices.to(poses)
        centered = centered - centered.mean(dim=0, keepdim=True)
        _, singular_values, principal_axes = torch.linalg.svd(
            centered, full_matrices=False
        )
        if (
            singular_values.shape[0] < 2
            or singular_values[0] <= 2.0 * singular_values[1]
        ):
            augmented.append(filtered)
            continue
        longest_axis = principal_axes[0]
        stock_closing_local = (obj_poses[row, :3, :3].to(poses).T @ poses[:, :3, 0].T).T
        long_axis_candidates = torch.abs(
            stock_closing_local @ longest_axis
        ) >= math.cos(math.radians(20.0))
        if not long_axis_candidates.any():
            augmented.append(filtered)
            continue
        rotated = poses.clone()
        rotated[:, :3, :3] = poses[:, :3, :3] @ rotation
        closing_local = (obj_poses[row, :3, :3].to(rotated).T @ rotated[:, :3, 0].T).T
        projections = mesh_vertices.to(rotated) @ closing_local.T
        rotated_widths = projections.amax(dim=0) - projections.amin(dim=0)
        rotated_costs = torch.where(
            long_axis_candidates, costs, torch.full_like(costs, torch.inf)
        )
        augmented.append(screened(rotated, rotated_widths, rotated_costs))
    return HandleGraspCandidates(tuple(augmented), required_width, width_fit)
