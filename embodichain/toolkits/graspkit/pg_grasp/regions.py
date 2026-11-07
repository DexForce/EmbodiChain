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

"""End-region selection from collision-qualified cached grasp support."""

from __future__ import annotations

from typing import Any, Literal

import torch

from embodichain.toolkits.graspkit import GraspPoseGenerator
from embodichain.utils.math import get_pc_center_box

__all__ = ["select_supported_end", "for_axis_region"]


def select_supported_end(
    *,
    centers: torch.Tensor,
    pair_ids: torch.Tensor,
    pair_centers: torch.Tensor,
    axis: torch.Tensor,
    positive: bool,
    center_mode: Literal["bounds", "centroid"],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Select an end from collision-qualified contacts before candidate ranking.

    Args:
        centers: Candidate centers with shape ``(N, 3)``.
        pair_ids: Original contact-pair indices for the candidates.
        pair_centers: Original contact-pair centers with shape ``(P, 3)``.
        axis: The sampler's unit longitudinal axis.
        positive: Whether to select the positive end of the axis.
        center_mode: Existing bounds or centroid region policy.

    Returns:
        Candidate mask and region center, or ``None`` for unavailable support.
        Repeated orientation proposals count as one original contact pair.
    """
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
    """Select a per-request end policy without mutating a cached generator.

    Args:
        generator: Grasp service accepting the optional region-selector protocol.
        policy: ``mesh_extent`` retains the original generator; ``supported_end``
            enables selection from collision-qualified fallback contacts.

    Returns:
        The original service or a request-local selector adapter.

    Raises:
        ValueError: If the policy is not supported.
    """
    if policy not in ("mesh_extent", "supported_end"):
        raise ValueError("Unknown axis_region_policy.")
    return generator if policy == "mesh_extent" else _EndRegionGenerator(generator)
