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
import math
from types import MappingProxyType

import torch

from embodichain.utils import configclass

__all__ = [
    "GraspPoseGenerator",
    "ParallelJawGraspPoseGenerator",
    "ParallelJawGripperModelCfg",
    "get_parallel_jaw_gripper_model",
]


def _positive_finite(value: float, *, field_name: str) -> float:
    """Return one validated positive finite value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


@configclass
class ParallelJawGripperModelCfg:
    """Physical geometry shared by parallel-jaw grasp generators.

    ``model_id`` names one concrete end-effector model or calibration. Product
    names belong in that value (for example ``"dh_pgi_140_80"``), not in the
    generator class hierarchy.
    """

    model_id: str = "parallel_jaw"
    """Stable identifier for the concrete gripper geometry."""

    min_opening_width: float = 0.001
    """Minimum usable distance between the two fingers in metres."""

    max_opening_width: float = 0.1
    """Maximum usable distance between the two fingers in metres."""

    finger_length: float = 0.08
    """Finger length along the grasp-frame approach axis in metres."""

    finger_width: float = 0.03
    """Finger extent perpendicular to its opening and approach axes."""

    finger_thickness: float = 0.01
    """Finger extent along the opening axis in metres."""

    palm_depth: float = 0.08
    """Palm/root extent along the grasp-frame approach axis in metres."""

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or not self.model_id
            or self.model_id != self.model_id.strip()
        ):
            raise ValueError(
                "ParallelJawGripperModelCfg.model_id must be a non-empty "
                "string without outer whitespace."
            )
        self.min_opening_width = _positive_finite(
            self.min_opening_width,
            field_name="min_opening_width",
        )
        self.max_opening_width = _positive_finite(
            self.max_opening_width,
            field_name="max_opening_width",
        )
        if self.min_opening_width >= self.max_opening_width:
            raise ValueError("min_opening_width must be less than max_opening_width.")
        for field_name in (
            "finger_length",
            "finger_width",
            "finger_thickness",
            "palm_depth",
        ):
            setattr(
                self,
                field_name,
                _positive_finite(getattr(self, field_name), field_name=field_name),
            )


_PARALLEL_JAW_GRIPPER_MODELS = MappingProxyType(
    {
        "dh_pgi_140_80": MappingProxyType(
            {
                "model_id": "dh_pgi_140_80",
                "min_opening_width": 0.005,
                "max_opening_width": 0.1,
                "finger_length": 0.12,
                "finger_width": 0.04,
                "finger_thickness": 0.01,
                "palm_depth": 0.096,
            }
        ),
    }
)


def get_parallel_jaw_gripper_model(model_id: str) -> ParallelJawGripperModelCfg:
    """Return a fresh configuration for a named parallel-jaw gripper model.

    The built-in catalog contains grasp-planning geometry rather than URDF or
    downloadable asset metadata. Callers with an unregistered calibration can
    construct :class:`ParallelJawGripperModelCfg` directly.

    Args:
        model_id: Stable identifier of a built-in gripper geometry.

    Returns:
        An independently owned gripper-model configuration.

    Raises:
        ValueError: If ``model_id`` is malformed or is not built in.
    """
    if type(model_id) is not str or not model_id or model_id != model_id.strip():
        raise ValueError(
            "model_id must be a non-empty string without outer whitespace."
        )
    try:
        values = _PARALLEL_JAW_GRIPPER_MODELS[model_id]
    except KeyError as exc:
        available = sorted(_PARALLEL_JAW_GRIPPER_MODELS)
        raise ValueError(
            f"unknown parallel-jaw gripper model {model_id!r}; "
            f"available models are {available}."
        ) from exc
    return ParallelJawGripperModelCfg(**dict(values))


class GraspPoseGenerator(ABC):
    """Standalone service that generates grasp poses from target geometry.

    The contract has no dependency on Gym, simulation, atomic actions, or
    Expert Program. Application code may call it directly or install the same
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
