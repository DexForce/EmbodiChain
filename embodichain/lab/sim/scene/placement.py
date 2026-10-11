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

"""Pure placement proposals for explicit scene support geometry."""

from __future__ import annotations

import numpy as np

from embodichain.compute.geometry.aabb import AABBLayoutConfig, AABBLayoutOptimizer
from embodichain.compute.geometry.placement import (
    GroupSupportClamp,
    GroupSupportClampConfig,
    group_height_offset,
)
from .geometry import SupportRegion

__all__ = ["clamp_group", "separate_aabbs", "align_group_height"]


def clamp_group(
    region: SupportRegion,
    aabbs_by_uid: dict[str, np.ndarray],
    *,
    config: GroupSupportClampConfig | None = None,
) -> np.ndarray | None:
    """Propose one planar group translation retaining all relative positions.

    Args:
        region: Supporting entity and allowed arena-local XY region.
        aabbs_by_uid: Four XY corners per entity in arena coordinates.
        config: Boundary clearance and raster-search resolution.

    Returns:
        Shared XY offset, or ``None`` if the raster search found no placement.
        Failure does not establish mathematical infeasibility.
    """
    return GroupSupportClamp(config=config).find_translation(
        region.support_polygon, aabbs_by_uid
    )


def separate_aabbs(
    region: SupportRegion,
    aabbs_by_uid: dict[str, np.ndarray],
    *,
    config: AABBLayoutConfig | None = None,
) -> dict[str, np.ndarray]:
    """Propose independent XY translations resolving projected AABB overlaps.

    Args:
        region: Supporting entity and allowed arena-local XY region.
        aabbs_by_uid: Four XY corners per entity in arena coordinates.
        config: Boundary margin, pair clearance and iteration budget.

    Returns:
        Translation per logical entity UID; no simulation is modified.

    Raises:
        ValueError: Invalid geometry or no solution found within the budget.

    .. attention::
        This checks planar AABBs, not exact mesh collisions or physical stability.
    """
    return AABBLayoutOptimizer(config=config).solve(
        region.support_polygon, aabbs_by_uid
    )


def align_group_height(
    region: SupportRegion,
    bounds_by_uid: dict[str, np.ndarray],
    *,
    clearance_m: float = 0.02,
) -> float:
    """Propose one Z translation aligning the group's lowest point to support.

    Args:
        region: Horizontal support descriptor in arena coordinates.
        bounds_by_uid: Per-entity minima/maxima with shape ``(2, 3)``.
        clearance_m: Non-negative gap above support, in metres.

    Returns:
        Shared vertical translation. Positions and orientations are unchanged.
    """
    if not bounds_by_uid or any(
        not isinstance(uid, str) or not uid for uid in bounds_by_uid
    ):
        raise ValueError("At least one non-empty entity UID and bounds are required.")
    return group_height_offset(
        region.top_z, np.asarray(list(bounds_by_uid.values())), clearance_m=clearance_m
    )
