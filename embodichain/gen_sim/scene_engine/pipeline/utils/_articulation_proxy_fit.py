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

"""Fit a native mechanism to a settled Scene Engine proxy footprint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

__all__ = ["ArticulationProxyFit", "fit_articulation_proxy"]


@dataclass(frozen=True)
class ArticulationProxyFit:
    """Z-up runtime placement and standard-frame footprint comparison."""

    position_z_up: NDArray[np.float64]
    rotation_z_up: NDArray[np.float64]
    standard_scale: NDArray[np.float64]
    native_scale_transform: NDArray[np.float64]
    proxy_footprint_size: NDArray[np.float64]
    fitted_footprint_size: NDArray[np.float64]
    axis_coverage: NDArray[np.float64]


def fit_articulation_proxy(
    *,
    proxy_standard_vertices: ArrayLike,
    native_footprint_vertices: ArrayLike,
    native_collision_vertices: ArrayLike,
    native_to_standard: ArrayLike,
    proxy_yaw_degrees: float,
    proxy_center_xy: ArrayLike,
    support_z: float,
    clearance: float = 0.001,
) -> ArticulationProxyFit:
    """Place the mechanism level with the support while preserving the proxy yaw.

    Args:
        proxy_standard_vertices: Scaled proxy points in the leveled semantic
            frame, without its final world yaw or translation.
        native_footprint_vertices: Native root-relative points used to measure
            the footprint in the initial joint state.
        native_collision_vertices: Native root-relative collision points in the
            same initial joint state, used to determine the contact height.
        native_to_standard: Proper 3×3 rotation mapping the native root frame
            to the leveled semantic frame.
        proxy_yaw_degrees: Final proxy heading about positive world Z in degrees.
        proxy_center_xy: Center of the settled proxy's world XY bounding box.
        support_z: Height of the world support plane in metres.
        clearance: Distance from the bottom of the fitted collider to support.

    Returns:
        The native root's world pose and local scale transform.

    Raises:
        ValueError: If geometry, placement, or rotation is invalid.
    """
    proxy = _points(proxy_standard_vertices, "Proxy footprint")
    native = _points(native_footprint_vertices, "Native footprint")
    collider = _points(native_collision_vertices, "Native collision")
    standard = np.asarray(native_to_standard, dtype=float)
    if (
        standard.shape != (3, 3)
        or not np.isfinite(standard).all()
        or not np.allclose(standard.T @ standard, np.eye(3), rtol=0.0, atol=1e-6)
        or np.linalg.det(standard) <= 0
    ):
        raise ValueError("Native-to-standard rotation must be a proper 3×3 rotation.")
    target = np.asarray(proxy_center_xy, dtype=float)
    if (
        target.shape != (2,)
        or not np.isfinite(target).all()
        or not np.isfinite([proxy_yaw_degrees, support_z, clearance]).all()
        or clearance < 0.0
    ):
        raise ValueError(
            "Proxy placement requires finite XY, yaw, support height, and clearance. "
            "Clearance must be nonnegative."
        )
    proxy_size = np.ptp(proxy[:, :2], axis=0)
    native_size = np.ptp(native @ standard[:2].T, axis=0)
    if (
        not np.isfinite([proxy_size, native_size]).all()
        or (proxy_size <= 0.0).any()
        or (native_size <= 0.0).any()
    ):
        raise ValueError("Both footprints must have finite nonzero width and depth.")
    xy_scale = proxy_size / native_size
    # The footprint does not constrain height. Use the smaller ratio vertically.
    standard_scale = np.array([*xy_scale, xy_scale.min()])
    if not np.isfinite(standard_scale).all() or (standard_scale <= 0).any():
        raise ValueError("Footprint fitting produced invalid scale factors.")
    native_scale = standard.T @ np.diag(standard_scale) @ standard
    fitted_size = native_size * xy_scale

    world_rotation = (
        Rotation.from_euler("z", proxy_yaw_degrees, degrees=True).as_matrix() @ standard
    )
    scaled_to_world = world_rotation @ native_scale
    footprint_xy = native @ scaled_to_world[:2].T
    center = (footprint_xy.min(axis=0) + footprint_xy.max(axis=0)) / 2
    collision_bottom = float((collider @ scaled_to_world[2]).min())
    position = np.array([*(target - center), support_z + clearance - collision_bottom])
    if not np.isfinite(position).all():
        raise ValueError("Footprint fitting produced a nonfinite root position.")
    return ArticulationProxyFit(
        position_z_up=position,
        rotation_z_up=world_rotation,
        standard_scale=standard_scale,
        native_scale_transform=native_scale,
        proxy_footprint_size=proxy_size,
        fitted_footprint_size=fitted_size,
        axis_coverage=fitted_size / proxy_size,
    )


def _points(vertices: ArrayLike, label: str) -> NDArray[np.float64]:
    points = np.asarray(vertices, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{label} requires finite N×3 vertices.")
    return points
