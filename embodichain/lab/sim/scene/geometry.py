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

"""Explicit Z-up support geometry for scene placement services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from embodichain.lab.sim.scene._support_cpu import (
    SupportSurfaceConfig,
    SupportSurfaceDetector,
)

__all__ = ["SupportSurfaceConfig", "SupportRegion", "detect_support_region"]


@dataclass(frozen=True)
class SupportRegion:
    """A horizontal support identified by an existing scene entity UID.

    Geometry is measured in metres in arena-local Z-up coordinates. Polygon
    holes and disconnected regions are supported by placement; mesh detection
    selects one largest contour, retaining openings when ``preserve_holes`` is
    enabled in :class:`SupportSurfaceConfig`.
    This descriptor neither owns nor creates simulation resources.
    """

    support_uid: str
    top_z: float
    support_polygon: Polygon | MultiPolygon
    optimization_rectangle: Polygon | None = None
    frame: Literal["arena"] = "arena"

    def __post_init__(self) -> None:
        if not isinstance(self.support_uid, str) or not self.support_uid:
            raise ValueError("support_uid must be a non-empty entity UID.")
        if self.frame != "arena":
            raise ValueError("Support regions require the arena-local Z-up frame.")
        if not np.isfinite(self.top_z):
            raise ValueError("Support height must be finite.")
        polygon = self.support_polygon
        if (
            not isinstance(polygon, (Polygon, MultiPolygon))
            or polygon.is_empty
            or not polygon.is_valid
            or polygon.area <= 0
            or not np.isfinite(polygon.bounds).all()
        ):
            raise ValueError(
                "Support must be a finite, valid, non-empty polygonal region."
            )
        rectangle = self.optimization_rectangle
        if rectangle is not None and (
            not isinstance(rectangle, Polygon)
            or rectangle.is_empty
            or not rectangle.is_valid
            or not polygon.covers(rectangle)
        ):
            raise ValueError("Optimization rectangle must be contained in support.")


def detect_support_region(
    support_uid: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    config: SupportSurfaceConfig | None = None,
) -> SupportRegion:
    """Detect one main horizontal support from explicit arena-local geometry.

    Args:
        support_uid: UID of the existing supporting entity.
        vertices: Already transformed Z-up vertices in metres, shape ``(N, 3)``.
        faces: Triangle indices, shape ``(M, 3)``.
        config: Geometric support selection thresholds.

    Returns:
        Support descriptor in the same frame, without applying asset scale again.
    """
    detected = SupportSurfaceDetector(vertices, faces, config=config).detect()
    return SupportRegion(
        support_uid=support_uid,
        top_z=detected.top_z,
        support_polygon=detected.support_polygon,
        optimization_rectangle=detected.optimization_rectangle,
    )
