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

"""GenSim end-region selection from collision-qualified cached grasp support."""

from __future__ import annotations

from typing import Any, Literal

import torch

from embodichain.toolkits.graspkit import GraspPoseGenerator
from embodichain.utils.math import get_pc_center_box

__all__: list[str] = []


def select_supported_end(
    *,
    centers: torch.Tensor,
    pair_ids: torch.Tensor,
    pair_centers: torch.Tensor,
    axis: torch.Tensor,
    positive: bool,
    center_mode: Literal["bounds", "centroid"],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select the requested end before ranking, counting each contact pair once."""
    unique_centers = pair_centers[torch.unique(pair_ids, sorted=True)]
    projection = unique_centers @ axis
    if not torch.isfinite(projection).all() or len(unique_centers) < 2:
        return None
    low, high = projection.min(), projection.max()
    span = high - low
    if not bool(span > 0):
        return None
    fraction = 0.5 if center_mode == "bounds" else (0.65 if positive else 0.35)
    threshold = low + fraction * span
    projected = centers @ axis
    selected = projected > threshold if positive else projected < threshold
    selected_unique = projection > threshold if positive else projection < threshold
    if not selected.any():
        return None
    center = (
        get_pc_center_box(unique_centers) + (0.25 if positive else -0.25) * span * axis
        if center_mode == "bounds"
        else unique_centers[selected_unique].mean(0)
    )
    return selected, center


class _EndRegionGenerator(GraspPoseGenerator):
    """Bind a trusted selector for one request without mutating the cached service."""

    def __init__(self, delegate: GraspPoseGenerator) -> None:
        self._delegate = delegate

    def get_valid_grasp_poses(
        self, **kwargs: Any
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._delegate.get_valid_grasp_poses(
            **kwargs, axis_region_selector=select_supported_end
        )

    def get_best_grasp_poses(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._delegate.get_best_grasp_poses(**kwargs)


def for_axis_region(generator: GraspPoseGenerator, policy: str) -> GraspPoseGenerator:
    """Keep explicit mesh restrictions unchanged; opt only derived ends into recovery."""
    if policy not in ("mesh_extent", "supported_end"):
        raise ValueError("Unknown axis_region_policy.")
    return generator if policy == "mesh_extent" else _EndRegionGenerator(generator)
