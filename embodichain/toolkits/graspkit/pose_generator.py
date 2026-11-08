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

"""Backend-neutral contracts for standalone grasp-pose generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

import torch

__all__ = [
    "GraspPoseGenerator",
    "ParallelJawGraspPoseGenerator",
    "ParallelJawGripperModelCfg",
]


from .gripper_models import ParallelJawGripperModelCfg


class GraspPoseGenerator(ABC):
    """Standalone service that generates grasp poses from target geometry.

    The contract has no dependency on Gym, simulation, atomic actions, or
    Task Program. Application code may call it directly or install the same
    service instance alongside a motion generator in a higher-level runtime.
    """

    @abstractmethod
    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return candidates, optionally restricted to one projected axis end."""

    @abstractmethod
    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return success, best pose, and opening width for every object pose."""


class ParallelJawGraspPoseGenerator(GraspPoseGenerator, ABC):
    """Base service shared by two-finger parallel-jaw grippers."""

    def __init__(self, gripper_model: ParallelJawGripperModelCfg) -> None:
        if not isinstance(gripper_model, ParallelJawGripperModelCfg):
            raise TypeError(
                "gripper_model must be a ParallelJawGripperModelCfg instance."
            )
        self._gripper_model = deepcopy(gripper_model)

    @property
    def gripper_model(self) -> ParallelJawGripperModelCfg:
        """Return an owned snapshot of the physical gripper model."""
        return deepcopy(self._gripper_model)

    def get_grasp_candidates(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return aligned poses, opening widths and costs for each object row.

        Each tuple contains tensors of shape ``(K, 4, 4)``, ``(K,)`` and
        ``(K,)``. Infinite costs mark ineligible candidates. Services without
        this optional metadata capability fail explicitly rather than inventing
        opening widths for their existing pose-only candidates.
        """
        raise NotImplementedError(
            "This grasp service does not expose opening metadata."
        )

    @abstractmethod
    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
    ) -> list[dict[str, dict[str, object]] | None]:
        """Return coordinated candidate sets for two parallel-jaw grippers."""
