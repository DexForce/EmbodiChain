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

"""Uniform grasp-fit arithmetic independent of assets, robots and task semantics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["GraspFitScale", "compute_grasp_fit_scale"]


@dataclass(frozen=True, slots=True)
class GraspFitScale:
    """Uniform scale result retaining the original support bottom and fit margins."""

    factor: float
    body_scale: tuple[float, float, float]
    origin_z: float
    extents: tuple[float, float, float]
    support_bottom_z: float
    fit_span: float
    pad_margin: float
    usable_span: float


def compute_grasp_fit_scale(
    vertices: np.ndarray,
    *,
    body_scale: tuple[float, float, float],
    rotation_xyz_degrees: tuple[float, float, float],
    origin_z: float,
    opening: float,
    contact_clearance: float,
    fit_mode: Literal["minimum_axis", "single_arm_transverse"],
) -> GraspFitScale:
    """Fit a source mesh below a measured jaw gap while preserving support height.

    Args:
        vertices: Unscaled object-local Z-up source vertices, shape (V, 3).
        body_scale: Source positive XYZ scale.
        rotation_xyz_degrees: Source intrinsic-XYZ Euler angles, in degrees.
        origin_z: Source world origin height.
        opening: Explicit calibrated jaw opening in metres.
        contact_clearance: Explicit required separation per jaw.
        fit_mode: Minimum axis for a paired grasp or maximum transverse axis for
            a single arm. The source algorithm does not search a new orientation.

    Returns:
        Original v3 bounded uniform scale with the 2 mm pad-margin floor,
        1 mm fit margin and 0.25 minimum scale. This is a source proposal, not a
        reachability, native contact or physical retention certificate.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    old_scale = np.asarray(body_scale, dtype=np.float64)
    angles = np.asarray(rotation_xyz_degrees, dtype=np.float64)
    rotation = Rotation.from_euler("XYZ", angles, degrees=True)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not vertices.size
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Grasp-fit target has invalid vertices.")
    if (
        old_scale.shape != (3,)
        or not np.isfinite(old_scale).all()
        or (old_scale <= 0).any()
    ):
        raise ValueError("Grasp-fit target has invalid body_scale.")
    if (
        angles.shape != (3,)
        or not np.isfinite(angles).all()
        or not math.isfinite(origin_z)
    ):
        raise ValueError("Grasp-fit source pose is unavailable.")
    if (
        isinstance(contact_clearance, bool)
        or not isinstance(contact_clearance, (int, float))
        or not math.isfinite(contact_clearance)
        or contact_clearance < 0
    ):
        raise ValueError("Grasp-fit contact clearance must be finite and non-negative.")
    if fit_mode not in {"minimum_axis", "single_arm_transverse"}:
        raise ValueError("Unknown grasp-fit span mode.")
    pad_margin = max(0.002, float(contact_clearance))
    if not math.isfinite(opening) or opening <= 2 * pad_margin + 0.001:
        raise ValueError("Grasp-fit requires a finite, positive calibrated opening.")
    scaled = vertices * old_scale
    extents = np.ptp(scaled, axis=0)
    span = (
        float(np.delete(extents, int(np.argmax(extents))).max())
        if fit_mode == "single_arm_transverse"
        else float(extents.min())
    )
    if not math.isfinite(span) or span <= 0:
        raise ValueError("Grasp-fit target has no measurable span.")
    usable = opening - 2 * pad_margin
    factor = min(1.0, (usable - 0.001) / span)
    if factor < 0.25:
        raise ValueError(
            f"Grasp-fit target requires scale {factor:.6f} below minimum 0.25."
        )
    old_bottom = float(rotation.apply(scaled)[:, 2].min() + origin_z)
    new_bottom_offset = float(rotation.apply(scaled * factor)[:, 2].min())
    return GraspFitScale(
        factor,
        tuple((old_scale * factor).tolist()),
        origin_z if factor == 1.0 else old_bottom - new_bottom_offset,
        tuple(extents.tolist()),
        old_bottom,
        span,
        pad_margin,
        usable,
    )
