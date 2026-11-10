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

import torch

__all__ = [
    "GraspPoseGenerator",
    "ParallelJawGraspPoseGenerator",
]


class GraspPoseGenerator(ABC):
    """Standalone service that generates grasp poses from target geometry.

    The contract has no dependency on Gym, simulation, atomic skills, or
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

    def __init__(
        self,
        *,
        min_opening_width: float,
        max_opening_width: float,
        model_id: str = "parallel_jaw",
    ) -> None:
        """Own validated scalar constraints independently of any hardware Cfg.

        Args:
            min_opening_width: Minimum usable full contact distance in metres.
            max_opening_width: Maximum usable full contact distance in metres.
            model_id: Opaque caller identity used for provenance and compatibility.
        """
        import math

        for name, value in (
            ("min_opening_width", min_opening_width),
            ("max_opening_width", max_opening_width),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if min_opening_width >= max_opening_width:
            raise ValueError("min_opening_width must be less than max_opening_width.")
        if type(model_id) is not str or not model_id or model_id != model_id.strip():
            raise ValueError("model_id must be an explicit nonempty caller identity.")
        self._model_id = model_id
        self._opening_limits = (float(min_opening_width), float(max_opening_width))

    @property
    def model_id(self) -> str:
        """Return the opaque identity without performing model lookup."""
        return self._model_id

    @property
    def min_opening_width(self) -> float:
        """Return the minimum full contact distance in metres."""
        return self._opening_limits[0]

    @property
    def max_opening_width(self) -> float:
        """Return the maximum full contact distance in metres."""
        return self._opening_limits[1]

    @property
    def gripper_data(self) -> dict[str, object]:
        """Return an owned scalar input snapshot for composed generator views."""
        return {
            "model_id": self.model_id,
            "min_opening_width": self.min_opening_width,
            "max_opening_width": self.max_opening_width,
        }

    @property
    def paired_query_key(self) -> object:
        """Identify one service that owns both role geometries and TCP mappings.

        Distinct custom instances are incompatible unless they explicitly expose
        an equal immutable key or share the same paired provider. Model identity
        and opening limits alone cannot authorize one query for both hands.
        """
        return ("instance", id(self))

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
