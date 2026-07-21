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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from embodichain.toolkits.graspkit.pg_grasp import (
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger


@dataclass
class Affordance:
    """Base class for affordance data.

    Represents an object's interaction possibilities. Subclasses carry whatever
    typed fields they need (mesh tensors, interaction points, etc.); the base
    class only carries an object label and a free-form custom_config dict.
    """

    object_label: str = ""
    """Label of the object this affordance belongs to."""

    custom_config: dict[str, Any] = field(default_factory=dict)
    """User-defined configuration payload."""

    def set_custom_config(self, key: str, value: Any) -> None:
        """Set a custom affordance configuration value."""
        self.custom_config[key] = value

    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get a custom affordance configuration value."""
        return self.custom_config.get(key, default)

    def get_batch_size(self) -> int:
        """Return the batch size of this affordance data."""
        return 1


@dataclass
class AntipodalAffordance(Affordance):
    """Antipodal grasp affordance for parallel-jaw grippers."""

    mesh_vertices: torch.Tensor | None = None
    """Object mesh vertices, shape [N, 3]."""

    mesh_triangles: torch.Tensor | None = None
    """Object mesh triangle indices, shape [M, 3]."""

    generator_cfg: GraspGeneratorCfg | None = None
    """Optional grasp-generator configuration."""

    gripper_collision_cfg: GripperCollisionCfg | None = None
    """Optional gripper-collision configuration."""

    force_reannotate: bool = False
    """If True, recompute the grasp annotation on each access."""

    _generator: GraspGenerator | None = field(default=None, init=False, repr=False)

    def _init_generator(self) -> None:
        if self.mesh_vertices is None or self.mesh_triangles is None:
            logger.log_error(
                "mesh_vertices and mesh_triangles must be provided to initialize "
                "AntipodalAffordance.",
                ValueError,
            )
        self._generator = GraspGenerator(
            vertices=self.mesh_vertices,
            triangles=self.mesh_triangles,
            cfg=self.generator_cfg,
            gripper_collision_cfg=self.gripper_collision_cfg,
        )
        if self.force_reannotate or self._generator._hit_point_pairs is None:
            self._generator.annotate()

    def _resolve_approach_direction(
        self, approach_direction: torch.Tensor
    ) -> torch.Tensor:
        """Move the approach direction to the grasp generator device."""
        return approach_direction.to(
            device=self._generator.device,
            dtype=torch.float32,
        )

    def get_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
        *,
        max_approach_alignment_angle: float | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        results = []
        for i, obj_pose in enumerate(obj_poses):
            generator_kwargs = {}
            if max_approach_alignment_angle is not None:
                generator_kwargs["max_approach_alignment_angle"] = (
                    max_approach_alignment_angle
                )
            is_success, grasp_poses, _, costs = self._generator.get_valid_grasp_poses(
                obj_pose, approach_direction, **generator_kwargs
            )
            if grasp_poses.shape == (4, 4):
                grasp_poses = grasp_poses.unsqueeze(0)
            if costs.dim() == 0:
                costs = costs.unsqueeze(0)
            if not is_success:
                logger.log_warning(
                    f"Failed to find valid grasp poses for {i}-th object."
                )
                costs = torch.full(
                    (grasp_poses.shape[0],),
                    torch.inf,
                    dtype=torch.float32,
                    device=grasp_poses.device,
                )
            results.append((grasp_poses, costs))
        return results

    def get_best_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        grasp_xpos_list: list[torch.Tensor] = []
        is_success_list: list[bool] = []
        open_length_list: list[float] = []
        grasp_pose_bias = self.get_custom_config("grasp_pose_bias")
        for i, obj_pose in enumerate(obj_poses):
            biased_result = self._get_biased_grasp_pose(
                obj_pose,
                approach_direction,
                grasp_pose_bias,
            )
            if biased_result is None:
                is_success, grasp_xpos, open_length = self._generator.get_grasp_poses(
                    obj_pose, approach_direction
                )
            else:
                is_success, grasp_xpos, open_length = biased_result
            if is_success:
                grasp_xpos_list.append(grasp_xpos.unsqueeze(0))
            else:
                logger.log_warning(f"No valid grasp pose found for {i}-th object.")
                grasp_xpos_list.append(
                    torch.eye(
                        4, dtype=torch.float32, device=self._generator.device
                    ).unsqueeze(0)
                )
            is_success_list.append(is_success)
            open_length_list.append(open_length)
        is_success_t = torch.tensor(
            is_success_list, dtype=torch.bool, device=self._generator.device
        )
        grasp_xpos = torch.concatenate(grasp_xpos_list, dim=0)
        open_length_t = torch.tensor(
            open_length_list, dtype=torch.float32, device=self._generator.device
        )
        return is_success_t, grasp_xpos, open_length_t

    def _get_biased_grasp_pose(
        self,
        obj_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        grasp_pose_bias: Any,
    ) -> tuple[bool, torch.Tensor, float] | None:
        if not _is_upright_bottle_side_grasp_bias(grasp_pose_bias):
            return None
        is_success, grasp_poses, open_lengths, costs = (
            self._generator.get_valid_grasp_poses(obj_pose, approach_direction)
        )
        if not is_success:
            return None
        grasp_poses = _ensure_grasp_pose_batch(grasp_poses)
        if grasp_poses.shape[0] == 0:
            return None
        open_lengths = _ensure_vector(
            open_lengths,
            grasp_poses.shape[0],
            grasp_poses.device,
        )
        costs = _ensure_vector(costs, grasp_poses.shape[0], grasp_poses.device)
        scores = self._upright_bottle_side_grasp_scores(
            obj_pose,
            grasp_poses,
            costs,
            grasp_pose_bias,
        )
        if not torch.isfinite(scores).any():
            return None
        best_idx = torch.argmin(scores)
        return True, grasp_poses[best_idx], float(open_lengths[best_idx].item())

    def _upright_bottle_side_grasp_scores(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        costs: torch.Tensor,
        grasp_pose_bias: Mapping[str, Any],
    ) -> torch.Tensor:
        if self.mesh_vertices is None or self.mesh_vertices.numel() == 0:
            return _normalized_costs(costs)

        device = grasp_poses.device
        vertices = self.mesh_vertices.to(device=device, dtype=torch.float32)
        extents = vertices.max(dim=0).values - vertices.min(dim=0).values
        long_axis_index = int(torch.argmax(extents).item())
        long_extent = torch.clamp(extents[long_axis_index], min=1e-6)
        axis_min = vertices[:, long_axis_index].min()

        obj_pose = obj_pose.to(device=device, dtype=torch.float32)
        centers_world = grasp_poses[:, :3, 3]
        centers_local = (centers_world - obj_pose[:3, 3]) @ obj_pose[:3, :3]
        height_fraction = (centers_local[:, long_axis_index] - axis_min) / long_extent

        lower, upper = _preferred_height_fraction(grasp_pose_bias)
        interval = max(float(upper - lower), 1e-6)
        below_cost = torch.clamp(float(lower) - height_fraction, min=0.0) / interval
        above_cost = torch.clamp(height_fraction - float(upper), min=0.0) / interval
        height_cost = below_cost + above_cost

        closing_axes_local = grasp_poses[:, :3, 0] @ obj_pose[:3, :3]
        side_cost = torch.abs(closing_axes_local[:, long_axis_index])
        side_weight = (
            1.5 if bool(grasp_pose_bias.get("prefer_side_grasp", True)) else 0.5
        )
        scores = _normalized_costs(costs) + 2.0 * height_cost + side_weight * side_cost

        preferred_mask = (height_fraction >= float(lower)) & (
            height_fraction <= float(upper)
        )
        if bool(grasp_pose_bias.get("prefer_side_grasp", True)):
            preferred_mask = preferred_mask & (side_cost <= 0.65)
        if preferred_mask.any():
            scores = scores.clone()
            scores[~preferred_mask] = torch.inf
        return scores


def _is_upright_bottle_side_grasp_bias(value: Any) -> bool:
    return (
        isinstance(value, Mapping) and value.get("mode") == "upright_bottle_side_grasp"
    )


def _preferred_height_fraction(value: Mapping[str, Any]) -> tuple[float, float]:
    raw = value.get("preferred_height_fraction", [0.35, 0.75])
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 2:
        return 0.35, 0.75
    lower = min(max(float(raw[0]), 0.0), 1.0)
    upper = min(max(float(raw[1]), 0.0), 1.0)
    if upper <= lower:
        return 0.35, 0.75
    return lower, upper


def _ensure_grasp_pose_batch(grasp_poses: torch.Tensor) -> torch.Tensor:
    if grasp_poses.shape == (4, 4):
        return grasp_poses.unsqueeze(0)
    return grasp_poses


def _ensure_vector(value: Any, count: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.numel() == count:
        return tensor.reshape(count)
    if tensor.numel() == 1:
        return tensor.repeat(count)
    flattened = tensor.flatten()
    if flattened.numel() < count:
        padding = flattened[-1:].repeat(count - flattened.numel())
        flattened = torch.cat([flattened, padding], dim=0)
    return flattened[:count]


def _normalized_costs(costs: torch.Tensor) -> torch.Tensor:
    finite_mask = torch.isfinite(costs)
    if not finite_mask.any():
        return torch.zeros_like(costs)
    finite_costs = costs[finite_mask]
    min_cost = finite_costs.min()
    max_cost = finite_costs.max()
    if float(max_cost - min_cost) < 1e-6:
        normalized = torch.zeros_like(costs)
    else:
        normalized = (costs - min_cost) / (max_cost - min_cost)
    normalized = normalized.clone()
    normalized[~finite_mask] = torch.inf
    return normalized


@dataclass
class InteractionPoints(Affordance):
    """Batch of 3D interaction points on an object surface."""

    points: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 3))
    """Batch of 3D interaction points with shape [B, 3]."""

    normals: torch.Tensor | None = None
    """Optional surface normals at each interaction point with shape [B, 3]."""

    point_types: list[str] = field(default_factory=list)
    """Optional labels for each point's interaction type."""

    def get_points_by_type(self, point_type: str) -> torch.Tensor | None:
        """Get points by their interaction type."""
        if point_type in self.point_types:
            indices = [i for i, t in enumerate(self.point_types) if t == point_type]
            return self.points[indices]
        return None

    def get_batch_size(self) -> int:
        """Return the number of interaction points in this affordance."""
        return self.points.shape[0]

    def get_approach_direction(self, point_idx: int) -> torch.Tensor:
        """Get recommended approach direction for a given point."""
        if self.normals is not None:
            return -self.normals[point_idx]
        return torch.tensor(
            [0, 0, 1], dtype=self.points.dtype, device=self.points.device
        )


__all__ = ["Affordance", "AntipodalAffordance", "InteractionPoints"]
