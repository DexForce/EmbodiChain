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

"""Standalone post-ranking opening-envelope qualification for parallel jaws."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import torch

from embodichain.compute.geometry.gripper import full_object_opening_mask
from embodichain.toolkits.graspkit.pose_generator import ParallelJawGraspPoseGenerator

if TYPE_CHECKING:
    from .pose_generator import GraspRegionSelector

__all__ = ["OpeningEnvelopeGraspPoseGenerator"]


class OpeningEnvelopeGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Keep the delegate's candidate order while requiring full opening clearance.

    Args:
        delegate: Calibrated parallel-jaw service. Its ranking, NMS and top-K
            decisions happen before this source-compatible envelope filter.

    Empty rows retain a finite padding pose with infinite cost. Paired and
    annotation protocols delegate unchanged; this service owns no task state.
    """

    def __init__(self, delegate: ParallelJawGraspPoseGenerator) -> None:
        if not isinstance(delegate, ParallelJawGraspPoseGenerator):
            raise TypeError("Opening envelope requires a parallel-jaw delegate.")
        super().__init__(delegate.gripper_model)
        self._delegate = delegate

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
        axis_region_selector: GraspRegionSelector | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Filter complete object opening after the delegate's ordinary selection.

        Args:
            mesh_vertices: Local source vertices (V, 3).
            mesh_triangles: Source triangles (F, 3).
            obj_poses: Current world object poses (B, 4, 4).
            approach_direction: Shared or per-row world approach vectors.
            obj_longest_axis: Optional region axis passed through unchanged.
            is_positive_part: Explicit region sign, shared or per row.
            axis_region_selector: Optional existing numeric region selector.

        Returns:
            Ordered owned pose/cost rows with unchanged failure-padding semantics.
        """
        rows = self._delegate.get_valid_grasp_poses(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            obj_poses=obj_poses,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
            **(
                {"axis_region_selector": axis_region_selector}
                if axis_region_selector is not None
                else {}
            ),
        )
        if obj_poses.shape != (len(rows), 4, 4):
            raise ValueError("Opening filter rows must match observed object poses.")
        filtered = []
        for row, (poses, costs) in enumerate(rows):
            if (
                poses.ndim != 3
                or poses.shape[1:] != (4, 4)
                or costs.shape != poses.shape[:1]
            ):
                raise ValueError("Grasp proposals require matching poses and costs.")
            costs = costs.to(poses.device)
            accepted = torch.isfinite(costs)
            accepted &= full_object_opening_mask(
                mesh_vertices,
                poses,
                obj_poses[row].to(poses),
                self.gripper_model.max_opening_width,
            )
            if accepted.any():
                filtered.append((poses[accepted].clone(), costs[accepted].clone()))
            else:
                padding = (
                    poses[:1] if len(poses) else obj_poses[row : row + 1].to(poses)
                )
                filtered.append((padding.clone(), poses.new_full((1,), torch.inf)))
        return filtered

    def get_best_grasp_poses(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delegate the independent articulation best-grasp protocol."""
        return self._delegate.get_best_grasp_poses(**kwargs)

    def get_grasp_candidates(
        self, **kwargs: Any
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Delegate annotation candidates without changing paired metadata."""
        return self._delegate.get_grasp_candidates(**kwargs)

    def get_dual_arm_valid_grasp_poses(
        self, **kwargs: Any
    ) -> list[dict[str, dict[str, object]] | None]:
        """Delegate the separate coordinated grasp protocol unchanged."""
        return self._delegate.get_dual_arm_valid_grasp_poses(**kwargs)
