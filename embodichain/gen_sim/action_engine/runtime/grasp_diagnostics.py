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

"""Stage-separated diagnostics for GenSim antipodal grasp generation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

from .coordinated_safety import (
    _canonicalize_parallel_jaw_poses,
    _rank_non_crossing_grasp_pairs,
)

__all__: list[str] = []

_UPRIGHT_SIDE_GRASP_MAX_AXIS_ALIGNMENT = 0.65
_UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION = 0.35
_UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION = 0.75
_UPRIGHT_SIDE_GRASP_HEIGHT_COST_WEIGHT = 2.0
_UPRIGHT_SIDE_GRASP_CANDIDATE_LIMIT = 50


@dataclass(frozen=True, slots=True)
class _UprightGraspSelectionContext:
    local_axis: torch.Tensor


@dataclass(frozen=True, slots=True)
class _DualGraspSelectionContext:
    left_eef: torch.Tensor
    right_eef: torch.Tensor
    left_base: torch.Tensor
    right_base: torch.Tensor
    left_to_right_direction: torch.Tensor
    pair_rank: int
    minimum_separation: float
    minimum_lateral_gap: float


class _TracingAntipodalGraspPoseGenerator(AntipodalGraspPoseGenerator):
    """Retain compact S1-S5 evidence from the concrete GenSim grasp backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_dual_trace: dict[str, Any] | None = None
        self._last_upright_trace: dict[str, Any] | None = None
        self._selection_context: _DualGraspSelectionContext | None = None
        self._upright_selection_context: _UprightGraspSelectionContext | None = None

    @property
    def last_dual_trace(self) -> dict[str, Any] | None:
        """Return an owned snapshot of the most recent dual-grasp trace."""
        return deepcopy(self._last_dual_trace)

    @property
    def last_upright_trace(self) -> dict[str, Any] | None:
        """Return an owned snapshot of the most recent upright-grasp trace."""
        return deepcopy(self._last_upright_trace)

    @contextmanager
    def upright_selection_context(
        self,
        *,
        local_axis: torch.Tensor,
    ) -> Iterator[None]:
        """Install one invocation-local side-grasp selection policy."""
        if self._upright_selection_context is not None:
            raise RuntimeError("Upright grasp selection context cannot be nested.")
        axis = torch.as_tensor(local_axis, dtype=torch.float32).reshape(-1)
        norm = torch.linalg.vector_norm(axis)
        if axis.shape != (3,) or not torch.isfinite(axis).all() or norm <= 1.0e-6:
            raise ValueError("Upright grasp local_axis must be one finite 3-vector.")
        self._last_upright_trace = None
        self._upright_selection_context = _UprightGraspSelectionContext(
            local_axis=(axis / norm).clone()
        )
        try:
            yield
        finally:
            self._upright_selection_context = None

    @contextmanager
    def dual_arm_selection_context(
        self,
        *,
        left_eef: torch.Tensor,
        right_eef: torch.Tensor,
        left_base: torch.Tensor,
        right_base: torch.Tensor,
        left_to_right_direction: torch.Tensor,
        pair_rank: int,
        minimum_separation: float,
        minimum_lateral_gap: float,
    ) -> Iterator[None]:
        """Install one invocation-local arm context for pair-aware selection."""
        if self._selection_context is not None:
            raise RuntimeError("Dual grasp selection context cannot be nested.")
        if type(pair_rank) is not int or pair_rank < 0:
            raise ValueError("pair_rank must be a non-negative integer.")
        self._selection_context = _DualGraspSelectionContext(
            left_eef=torch.as_tensor(left_eef, dtype=torch.float32).clone(),
            right_eef=torch.as_tensor(right_eef, dtype=torch.float32).clone(),
            left_base=torch.as_tensor(left_base, dtype=torch.float32).clone(),
            right_base=torch.as_tensor(right_base, dtype=torch.float32).clone(),
            left_to_right_direction=torch.as_tensor(
                left_to_right_direction, dtype=torch.float32
            ).clone(),
            pair_rank=pair_rank,
            minimum_separation=float(minimum_separation),
            minimum_lateral_gap=float(minimum_lateral_gap),
        )
        try:
            yield
        finally:
            self._selection_context = None

    @staticmethod
    def _failed_arm_result(reference: torch.Tensor) -> dict[str, Any]:
        return {
            "is_success": False,
            "grasp_poses": torch.eye(
                4,
                dtype=torch.float32,
                device=reference.device,
            ),
            "open_lengths": 0.0,
            "total_cost": torch.zeros(1, device=reference.device),
        }

    def _select_pair(
        self,
        result: dict[str, dict[str, Any]] | None,
        *,
        row_index: int,
    ) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any] | None]:
        context = self._selection_context
        if context is None or result is None:
            return result, None
        left = result["left"]
        right = result["right"]
        if not left.get("is_success", False) or not right.get("is_success", False):
            return result, {
                "requested_pair_rank": context.pair_rank,
                "valid_pair_count": 0,
                "selected": False,
                "reason": "one_or_both_arms_have_no_candidates",
            }
        left_poses = torch.as_tensor(left["grasp_poses"], dtype=torch.float32)
        right_poses = torch.as_tensor(right["grasp_poses"], dtype=torch.float32)
        if left_poses.ndim == 2:
            left_poses = left_poses.unsqueeze(0)
        if right_poses.ndim == 2:
            right_poses = right_poses.unsqueeze(0)
        left_canonical = _canonicalize_parallel_jaw_poses(
            left_poses,
            context.left_eef[row_index],
        )
        right_canonical = _canonicalize_parallel_jaw_poses(
            right_poses,
            context.right_eef[row_index],
        )
        ranking = _rank_non_crossing_grasp_pairs(
            left_canonical.poses,
            right_canonical.poses,
            left_costs=torch.as_tensor(left["total_cost"], dtype=torch.float32),
            right_costs=torch.as_tensor(right["total_cost"], dtype=torch.float32),
            left_rotation_costs=left_canonical.selected_rotation_radians,
            right_rotation_costs=right_canonical.selected_rotation_radians,
            left_base=context.left_base[row_index],
            right_base=context.right_base[row_index],
            left_to_right_direction=context.left_to_right_direction,
            minimum_separation=context.minimum_separation,
            minimum_lateral_gap=context.minimum_lateral_gap,
        )
        trace: dict[str, Any] = {
            "requested_pair_rank": context.pair_rank,
            "valid_pair_count": len(ranking.ranked_pairs),
            "rejection_counts": dict(ranking.rejection_counts),
            "left_half_turn_count": int(left_canonical.flipped.sum().item()),
            "right_half_turn_count": int(right_canonical.flipped.sum().item()),
            "selected": context.pair_rank < len(ranking.ranked_pairs),
        }
        if context.pair_rank >= len(ranking.ranked_pairs):
            trace["reason"] = "requested_pair_rank_unavailable"
            return {
                "left": self._failed_arm_result(left_poses),
                "right": self._failed_arm_result(right_poses),
            }, trace
        left_index, right_index = ranking.ranked_pairs[context.pair_rank]
        left_pose = left_canonical.poses[left_index]
        right_pose = right_canonical.poses[right_index]
        trace.update(
            {
                "selected_left_index": left_index,
                "selected_right_index": right_index,
                "selected_pair_score": ranking.scores[context.pair_rank],
                "selected_left_half_turn": bool(left_canonical.flipped[left_index]),
                "selected_right_half_turn": bool(right_canonical.flipped[right_index]),
                "selected_left_rotation_radians": float(
                    left_canonical.selected_rotation_radians[left_index]
                ),
                "selected_right_rotation_radians": float(
                    right_canonical.selected_rotation_radians[right_index]
                ),
                "selected_left_pose": left_pose.detach().cpu().tolist(),
                "selected_right_pose": right_pose.detach().cpu().tolist(),
                "selected_separation": float(
                    torch.linalg.vector_norm(left_pose[:3, 3] - right_pose[:3, 3])
                ),
            }
        )

        def selected_arm(
            arm: dict[str, Any],
            poses: torch.Tensor,
            index: int,
        ) -> dict[str, Any]:
            open_lengths = torch.as_tensor(arm["open_lengths"])
            costs = torch.as_tensor(arm["total_cost"], dtype=torch.float32)
            return {
                "is_success": True,
                "grasp_poses": poses[index : index + 1],
                "open_lengths": open_lengths[index : index + 1],
                "total_cost": costs.new_zeros(1),
            }

        return {
            "left": selected_arm(left, left_canonical.poses, left_index),
            "right": selected_arm(right, right_canonical.poses, right_index),
        }, trace

    @staticmethod
    def _transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        return points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]

    @classmethod
    def _filter_counts(
        cls,
        backend: Any,
        *,
        mesh_vertices: torch.Tensor,
        object_pose: torch.Tensor,
        arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float,
    ) -> dict[str, int]:
        pairs = backend.antipodal_pairs.to(dtype=torch.float32)
        origin = cls._transform_points(pairs[:, 0], object_pose)
        hit = cls._transform_points(pairs[:, 1], object_pose)
        world_vertices = cls._transform_points(mesh_vertices, object_pose)
        projection = torch.matmul(world_vertices, arm_direction)
        extent = projection.max() - projection.min()
        left_threshold = projection.min() + extent * (0.5 - middle_empty_ratio * 0.5)
        right_threshold = projection.max() - extent * (0.5 - middle_empty_ratio * 0.5)
        origin_projection = torch.matmul(origin, arm_direction)
        hit_projection = torch.matmul(hit, arm_direction)
        masks = {
            "left": (origin_projection < left_threshold)
            | (hit_projection < left_threshold),
            "right": (origin_projection > right_threshold)
            | (hit_projection > right_threshold),
        }
        counts: dict[str, int] = {}
        for side, mask in masks.items():
            grasp_x = F.normalize(hit[mask] - origin[mask], dim=1)
            cosine = torch.clamp(
                torch.sum(grasp_x * approach_direction, dim=1), -1.0, 1.0
            )
            angle = torch.abs(torch.acos(cosine))
            angle_valid = torch.abs(angle - torch.pi * 0.5) <= float(
                backend._max_deviation_angle
            )
            counts[f"{side}_partition_pair_count"] = int(mask.sum().item())
            counts[f"{side}_angle_valid_pair_count"] = int(angle_valid.sum().item())
        return counts

    @staticmethod
    def _candidate_count(result: dict[str, Any]) -> int:
        poses = result.get("grasp_poses")
        if not result.get("is_success", False) or not isinstance(poses, torch.Tensor):
            return 0
        return int(poses.shape[0]) if poses.ndim == 3 else 0

    def get_valid_grasp_poses(
        self,
        **kwargs: Any,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Apply v14's side and mid-body preference for upright pickup."""
        results = super().get_valid_grasp_poses(**kwargs)
        context = self._upright_selection_context
        if context is None:
            return results

        vertices = torch.as_tensor(kwargs["mesh_vertices"], dtype=torch.float32)
        object_poses = torch.as_tensor(kwargs["obj_poses"], dtype=torch.float32)
        local_axis = context.local_axis.to(
            device=vertices.device,
            dtype=vertices.dtype,
        )
        vertex_positions = torch.matmul(vertices, local_axis)
        axis_min = vertex_positions.min()
        axis_extent = vertex_positions.max() - axis_min
        if float(axis_extent) <= 1.0e-6:
            raise ValueError("Upright grasp axis must span non-zero object geometry.")

        ranked_results: list[tuple[torch.Tensor, torch.Tensor]] = []
        row_traces: list[dict[str, Any]] = []
        for row_index, (result, object_pose) in enumerate(
            zip(results, object_poses, strict=True)
        ):
            grasp_poses, costs = result
            grasp_poses = torch.as_tensor(grasp_poses, dtype=torch.float32)
            costs = torch.as_tensor(
                costs,
                device=grasp_poses.device,
                dtype=torch.float32,
            )
            if grasp_poses.ndim == 2:
                grasp_poses = grasp_poses.unsqueeze(0)
            object_pose = object_pose.to(
                device=grasp_poses.device,
                dtype=grasp_poses.dtype,
            )
            axis = local_axis.to(
                device=grasp_poses.device,
                dtype=grasp_poses.dtype,
            )
            world_upright = torch.matmul(object_pose[:3, :3], axis)
            closing_axes = F.normalize(grasp_poses[:, :3, 0], dim=1)
            axis_alignment = torch.abs(
                torch.sum(closing_axes * world_upright[None], dim=1)
            )
            side_compatible = axis_alignment <= _UPRIGHT_SIDE_GRASP_MAX_AXIS_ALIGNMENT
            relative_centers = grasp_poses[:, :3, 3] - object_pose[None, :3, 3]
            center_axis_positions = torch.sum(
                relative_centers * world_upright[None],
                dim=1,
            )
            center_fractions = (
                center_axis_positions - axis_min.to(grasp_poses.device)
            ) / axis_extent.to(grasp_poses.device)
            central_band = (
                center_fractions >= _UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION
            ) & (center_fractions <= _UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION)
            interval = (
                _UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION
                - _UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION
            )
            height_penalty = (
                torch.clamp(
                    _UPRIGHT_SIDE_GRASP_MIN_AXIS_FRACTION - center_fractions,
                    min=0.0,
                )
                + torch.clamp(
                    center_fractions - _UPRIGHT_SIDE_GRASP_MAX_AXIS_FRACTION,
                    min=0.0,
                )
            ) / interval
            adjusted_costs = (
                torch.where(
                    side_compatible,
                    costs,
                    torch.full_like(costs, torch.inf),
                )
                + _UPRIGHT_SIDE_GRASP_HEIGHT_COST_WEIGHT * height_penalty
            )
            ranked = torch.argsort(adjusted_costs)[:_UPRIGHT_SIDE_GRASP_CANDIDATE_LIMIT]
            ranked_results.append((grasp_poses[ranked], adjusted_costs[ranked]))
            finite_ranked = torch.isfinite(adjusted_costs[ranked])
            best_index = int(ranked[0].item()) if bool(finite_ranked.any()) else None
            row_traces.append(
                {
                    "environment_index": row_index,
                    "local_axis": axis.detach().cpu().tolist(),
                    "candidate_count": int(grasp_poses.shape[0]),
                    "side_compatible_count": int(side_compatible.sum().item()),
                    "central_band_count": int(central_band.sum().item()),
                    "side_and_central_count": int(
                        (side_compatible & central_band).sum().item()
                    ),
                    "retained_count": int(finite_ranked.sum().item()),
                    "best_candidate_axis_alignment": (
                        None
                        if best_index is None
                        else float(axis_alignment[best_index].item())
                    ),
                    "best_candidate_axis_fraction": (
                        None
                        if best_index is None
                        else float(center_fractions[best_index].item())
                    ),
                }
            )
        self._last_upright_trace = (
            row_traces[0] if len(row_traces) == 1 else {"environment_rows": row_traces}
        )
        return ranked_results

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> list[dict | None]:
        """Run the standard generator while observing its NMS/collision boundary."""
        vertices = kwargs["mesh_vertices"]
        triangles = kwargs["mesh_triangles"]
        backend = self._backend(vertices, triangles)
        poses = self._object_poses(kwargs["obj_poses"], device=backend.device)
        directions = self._approach_directions(
            kwargs["approach_direction"],
            batch_size=poses.shape[0],
            device=backend.device,
        )
        arm_direction = self._approach_directions(
            kwargs["left_to_right_arm_direction"],
            batch_size=1,
            device=backend.device,
        )[0]
        ratio = float(kwargs.get("middle_empty_ratio", 0.4))
        collision_records: list[dict[str, Any]] = []
        checker = backend._collision_checker
        original_query = checker.query

        def traced_query(*args: Any, **query_kwargs: Any):
            colliding, distance = original_query(*args, **query_kwargs)
            distance = torch.as_tensor(distance, dtype=torch.float32)
            collision_records.append(
                {
                    "nms_candidate_count": int(colliding.numel()),
                    "noncolliding_candidate_count": int((~colliding).sum().item()),
                    "minimum_signed_distance": float(distance.min().item()),
                    "maximum_signed_distance": float(distance.max().item()),
                }
            )
            return colliding, distance

        checker.query = traced_query
        try:
            results = super().get_dual_arm_valid_grasp_poses(**kwargs)
        finally:
            checker.query = original_query

        row_traces: list[dict[str, Any]] = []
        selected_results: list[dict[str, dict[str, Any]] | None] = []
        for row_index, (object_pose, approach, result) in enumerate(
            zip(poses, directions, results, strict=True)
        ):
            filter_counts = self._filter_counts(
                backend,
                mesh_vertices=vertices.to(device=backend.device, dtype=torch.float32),
                object_pose=object_pose,
                arm_direction=arm_direction,
                approach_direction=approach,
                middle_empty_ratio=ratio,
            )
            record_offset = 2 * row_index
            records = collision_records[record_offset : record_offset + 2]
            left_record = records[0] if len(records) >= 1 else {}
            right_record = records[1] if len(records) >= 2 else {}
            left = {} if result is None else result["left"]
            right = {} if result is None else result["right"]
            left_final = self._candidate_count(left)
            right_final = self._candidate_count(right)
            selected_result, pair_trace = self._select_pair(
                result,
                row_index=row_index,
            )
            selected_results.append(selected_result)
            row_traces.append(
                {
                    "environment_index": row_index,
                    "approach_direction": approach.detach().cpu().tolist(),
                    "middle_empty_ratio": ratio,
                    "S1_grasp_pair_generation": {
                        "antipodal_pair_count": int(backend.antipodal_pairs.shape[0]),
                    },
                    "S2_approach_angle_filtering": filter_counts,
                    "S3_nms": {
                        "left_candidate_count": int(
                            left_record.get("nms_candidate_count", 0)
                        ),
                        "right_candidate_count": int(
                            right_record.get("nms_candidate_count", 0)
                        ),
                    },
                    "S4_collision_filtering": {
                        "left_candidate_count": int(
                            left_record.get("noncolliding_candidate_count", 0)
                        ),
                        "right_candidate_count": int(
                            right_record.get("noncolliding_candidate_count", 0)
                        ),
                        "left_minimum_signed_distance": left_record.get(
                            "minimum_signed_distance"
                        ),
                        "left_maximum_signed_distance": left_record.get(
                            "maximum_signed_distance"
                        ),
                        "right_minimum_signed_distance": right_record.get(
                            "minimum_signed_distance"
                        ),
                        "right_maximum_signed_distance": right_record.get(
                            "maximum_signed_distance"
                        ),
                    },
                    "S5_left_right_pairing": {
                        "left_final_count": left_final,
                        "right_final_count": right_final,
                        "paired": left_final > 0 and right_final > 0,
                    },
                    "pair_selection": pair_trace,
                }
            )
        self._last_dual_trace = (
            row_traces[0] if len(row_traces) == 1 else {"environment_rows": row_traces}
        )
        return selected_results
