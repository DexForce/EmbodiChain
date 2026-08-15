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

"""EEF-aware lazy provider for antipodal grasp candidates."""

from __future__ import annotations

from typing import Any, Callable

import torch

from .antipodal_generator import GraspGenerator
from .profiles import AntipodalGraspPolicy, ParallelJawEefProfile

__all__ = ["GraspCandidateProvider"]


class GraspCandidateProvider:
    """Combine object geometry, EEF geometry, and sampling policy lazily."""

    def __init__(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        eef_profile: ParallelJawEefProfile,
        sampling_policy: AntipodalGraspPolicy,
        force_reannotate: bool = False,
    ) -> None:
        self.mesh_vertices = mesh_vertices
        self.mesh_triangles = mesh_triangles
        self.eef_profile = eef_profile
        self.sampling_policy = sampling_policy
        self.force_reannotate = bool(force_reannotate)
        self._generator: GraspGenerator | None = None

    @property
    def generator(self) -> GraspGenerator:
        """Return the initialized generator and populate raw pairs when needed."""
        if self._generator is None:
            self._generator = GraspGenerator(
                vertices=self.mesh_vertices,
                triangles=self.mesh_triangles,
                cfg=self.sampling_policy.generator_config(self.eef_profile),
                gripper_collision_cfg=self.eef_profile.collision_config(
                    max_decomposition_hulls=(
                        self.sampling_policy.max_decomposition_hulls
                    )
                ),
            )
            if self.force_reannotate or self._generator._hit_point_pairs is None:
                self._generator.annotate()
        return self._generator

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return the latest filtering trace without forcing initialization."""
        if self._generator is None:
            return {}
        return self._generator.last_filter_diagnostics

    def get_valid_grasp_poses(
        self,
        *,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        object_part: str = "center",
        pose_cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        approach_attempt_id: int = 0,
    ) -> tuple[bool, torch.Tensor, torch.Tensor | float, torch.Tensor]:
        """Return single-arm candidates from the shared generator."""
        return self.generator.get_valid_grasp_poses(
            object_pose=object_pose,
            approach_direction=approach_direction,
            object_part=object_part,
            pose_cost_fn=pose_cost_fn,
            approach_attempt_id=approach_attempt_id,
        )

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        middle_empty_ratio: float,
        approach_attempt_id: int = 0,
    ) -> dict[str, Any] | None:
        """Return dual-arm candidates from the shared generator."""
        return self.generator.get_dual_arm_valid_grasp_poses(
            object_pose=object_pose,
            approach_direction=approach_direction,
            left_to_right_arm_direction=left_to_right_arm_direction,
            middle_empty_ratio=middle_empty_ratio,
            approach_attempt_id=approach_attempt_id,
        )
