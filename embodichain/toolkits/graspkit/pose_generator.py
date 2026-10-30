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
    """Named antipodal contact-width limits, independent of collision geometry.

    Opening width is the full distance between opposing object contacts. Actual
    gripper surfaces, opening commands and contact/TCP transforms are mandatory
    inputs to the concrete generator; scalar box dimensions are unsupported.
    """

    model_id: str = "parallel_jaw"
    """Identity of the physical hand whose contact-width limits are supplied."""

    min_opening_width: float = 0.001
    """Minimum usable full distance between antipodal contacts in metres."""

    max_opening_width: float = 0.1
    """Maximum usable full distance between antipodal contacts in metres."""

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or not self.model_id
            or self.model_id != self.model_id.strip()
        ):
            raise ValueError(
                "model_id must be a non-empty string without outer whitespace."
            )
        self.min_opening_width = _positive_finite(
            self.min_opening_width, field_name="min_opening_width"
        )
        self.max_opening_width = _positive_finite(
            self.max_opening_width, field_name="max_opening_width"
        )
        if self.min_opening_width >= self.max_opening_width:
            raise ValueError("min_opening_width must be less than max_opening_width.")


_PARALLEL_JAW_GRIPPER_MODELS = MappingProxyType(
    {
        # Pika's native X approaches and Y opens. See README.md for the
        # source meshes, aperture measurement and adapter/TCP convention.
        "pika_gripper": MappingProxyType(
            {
                "model_id": "pika_gripper",
                "min_opening_width": 0.001,
                "max_opening_width": 0.094,
            }
        ),
        "dh_pgi_140_80": MappingProxyType(
            {
                "model_id": "dh_pgi_140_80",
                "min_opening_width": 0.005,
                "max_opening_width": 0.1,
            }
        ),
        # Geometry provenance and mesh-axis mapping: graspkit/README.md.
        # The 1 mm minimum is a planning cutoff, not a hardware closing limit.
        "franka_panda_hand": MappingProxyType(
            {
                "model_id": "franka_panda_hand",
                "min_opening_width": 0.001,
                "max_opening_width": 0.08,
            }
        ),
        "cobotmagic_v100_gripper": MappingProxyType(
            {
                "model_id": "cobotmagic_v100_gripper",
                "min_opening_width": 0.001,
                "max_opening_width": 0.1,
            }
        ),
        "tianji_marvin_gripper": MappingProxyType(
            {
                "model_id": "tianji_marvin_gripper",
                "min_opening_width": 0.001,
                "max_opening_width": 0.095,
            }
        ),
    }
)


def get_parallel_jaw_gripper_model(model_id: str) -> ParallelJawGripperModelCfg:
    """Return a fresh configuration for a named parallel-jaw gripper model.

    This catalog provides antipodal contact-width limits. Actual collision
    surfaces and TCP mappings are supplied separately through explicit geometry;
    custom hands require both their own width limits and geometry.

    Built-in IDs are ``dh_pgi_140_80``, ``franka_panda_hand`` (the hand in the
    packaged Panda asset), ``cobotmagic_v100_gripper`` (the V100 fingers), and
    ``tianji_marvin_gripper`` (both hands in the asset with end effectors),
    and ``pika_gripper`` (the standalone Pika asset).
    The V100 preset is distinct from CobotMagic's V70 hardware variant. Source
    archive checksums are recorded in GraspKit's ``README.md``. Algorithm and
    target collision policy remain generator settings.

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
