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

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

__all__: list[str] = []


class _TracingAntipodalGraspPoseGenerator(AntipodalGraspPoseGenerator):
    """Retain compact S1-S5 evidence from the concrete GenSim grasp backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_dual_trace: dict[str, Any] | None = None

    @property
    def last_dual_trace(self) -> dict[str, Any] | None:
        """Return an owned snapshot of the most recent dual-grasp trace."""
        return deepcopy(self._last_dual_trace)

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
                }
            )
        self._last_dual_trace = (
            row_traces[0] if len(row_traces) == 1 else {"environment_rows": row_traces}
        )
        return results
