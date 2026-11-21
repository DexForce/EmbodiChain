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

"""Explicit opposing-side grasp regions for an upright rectangular proxy."""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.utils import configclass

__all__ = ["BoxSideGraspRegionCfg", "project_box_side_grasp"]


@configclass
class BoxSideGraspRegionCfg:
    """Declare opposed short-side faces in the object's +Z-up frame.

    Object +X is the longitudinal/front axis; jaws oppose the +Y/-Y faces.
    The canonical grasp opens along +X and approaches along +Z. This region
    supplies geometry targets; opening, IK, calibration and collision checks
    remain independent and mandatory in a deployment.
    """

    longitudinal_margin_m: float = 0.01
    grasp_height_m: float = 0.0
    height_margin_m: float = 0.002

    def __post_init__(self) -> None:
        for name in ("longitudinal_margin_m", "grasp_height_m", "height_margin_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("box grasp region parameters must be finite")
        if self.longitudinal_margin_m < 0 or self.height_margin_m < 0:
            raise ValueError("box grasp margins must be nonnegative")


def project_box_side_grasp(
    object_to_source_grasp: np.ndarray,
    object_size_m: np.ndarray,
    cfg: BoxSideGraspRegionCfg,
) -> np.ndarray:
    """Find the closest canonical grasp in the declared opposed-face region.

    Longitudinal position is projected onto the inset interval, the short-axis
    centre is fixed between the two faces, and height is explicitly declared.
    Of the two 180-degree jaw symmetries, choose the nearest source orientation.
    No clean reference path, simulator rollout or grasp oracle is read.

    Args:
        object_to_source_grasp: Observed/derived canonical grasp hint, (4,4).
        object_size_m: Verified physical box dimensions (long, short, height).
        cfg: Explicit region and vertical clearance policy.

    Returns:
        Owned object-relative canonical grasp transform.

    Raises:
        ValueError: A transform, box or declared region is invalid/empty.
    """
    if not isinstance(cfg, BoxSideGraspRegionCfg):
        raise TypeError("box grasp projection requires a typed region")
    cfg = cfg.copy()
    cfg.__post_init__()
    pose, size = np.asarray(object_to_source_grasp, dtype=np.float64), np.asarray(
        object_size_m, dtype=np.float64
    )
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)
        or np.linalg.det(pose[:3, :3]) < 0.999999
    ):
        raise ValueError("source grasp must be an SE(3) transform")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("physical box dimensions must be positive")
    extent = size[0] / 2 - cfg.longitudinal_margin_m
    if extent < 0 or abs(cfg.grasp_height_m) + cfg.height_margin_m > size[2] / 2:
        raise ValueError("declared box grasp region is empty or outside the side faces")
    candidates = np.array(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        ]
    )
    angles = Rotation.from_matrix(candidates @ pose[:3, :3].T).magnitude()
    result = np.eye(4)
    result[:3, :3] = candidates[int(np.argmin(angles))]
    result[:3, 3] = [
        float(np.clip(pose[0, 3], -extent, extent)),
        0.0,
        cfg.grasp_height_m,
    ]
    return result
