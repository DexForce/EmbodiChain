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

"""Numeric declaration targets retaining original source frame arithmetic."""

from __future__ import annotations
from collections.abc import Sequence
import math

__all__ = [
    "upright_source_quaternion",
    "upright_origin_height",
    "bounded_carry_lift",
    "constrained_pick_approach",
]


def constrained_pick_approach(world_axis: Sequence[float]) -> list[float]:
    """Retain the original downward-minus-axis constrained grasp direction.

    Args:
        world_axis: Three finite components of the explicitly selected source axis.

    Returns:
        Normalized approach, or downward for the original 1e-6 degenerate norm.
        NumPy float64 arithmetic retains the original declaration values.

    Raises:
        ValueError: The declared axis is not a finite three-vector.
    """
    import numpy as np

    axis = np.asarray(world_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("Constrained Pick requires a finite three-value world axis.")
    downward = np.array([0.0, 0.0, -1.0])
    approach = downward - axis
    length = np.linalg.norm(approach)
    return (downward if length <= 1e-6 else approach / length).tolist()


def upright_source_quaternion(
    initial_euler_degrees: Sequence[float],
    local_axis: Sequence[float],
    *,
    world_yaw: float,
) -> list[float]:
    """Align a positive authored axis to +Z with the original float32 arithmetic.

    Args:
        initial_euler_degrees: Original intrinsic-XYZ source angles, in degrees.
        local_axis: Explicit unit source axis in the imported object's basis.
        world_yaw: World vertical rotation applied after positive-axis alignment.

    Returns:
        Original XYZW quaternion values as Python floats. Antiparallel sources
        retain the deterministic least-aligned-axis tie break. This computes a
        declared pose; it performs no planning or physical qualification.
    """
    import torch

    from embodichain.utils.math import (
        axis_angle_to_rotation_matrix,
        matrix_from_quat,
        quat_from_euler_xyz,
        quat_from_matrix,
    )

    init_rot = initial_euler_degrees
    if len(init_rot) != 3:
        raise ValueError("Source alignment requires three Euler angles.")
    angles = torch.deg2rad(torch.tensor(init_rot, dtype=torch.float32))
    quaternion = quat_from_euler_xyz(angles[0], angles[1], angles[2])
    rotation = matrix_from_quat(quaternion)
    axis = torch.tensor(local_axis, dtype=torch.float32)
    source_axis = torch.nn.functional.normalize(rotation @ axis, dim=0)
    target_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    cross = torch.linalg.cross(source_axis, target_axis)
    dot = torch.clamp(torch.dot(source_axis, target_axis), -1.0, 1.0)
    cross_norm = torch.linalg.vector_norm(cross)
    if float(cross_norm) <= 1.0e-6:
        if float(dot) >= 0.0:
            delta = torch.eye(3, dtype=torch.float32)
        else:
            basis = torch.eye(3, dtype=torch.float32)[
                torch.argmin(torch.abs(source_axis))
            ]
            perpendicular = torch.nn.functional.normalize(
                torch.linalg.cross(source_axis, basis), dim=0
            )
            delta = axis_angle_to_rotation_matrix(perpendicular * math.pi)
    else:
        rotation_axis = cross / cross_norm
        angle = torch.atan2(cross_norm, dot)
        delta = axis_angle_to_rotation_matrix(rotation_axis * angle)
    target_rotation = delta @ rotation
    if world_yaw:
        target_rotation = (
            axis_angle_to_rotation_matrix(
                torch.tensor([0.0, 0.0, world_yaw], dtype=torch.float32)
            )
            @ target_rotation
        )
    target_quaternion = quat_from_matrix(target_rotation)
    return [float(value) for value in target_quaternion]


def upright_origin_height(
    local_minimum: float,
    support_height: float,
    release_clearance: float,
    *,
    staging_radius: float | None = None,
    staging_margin: float = 0.0,
    staging_clearance: float = 0.0,
) -> float:
    """Keep original mesh-origin release height and optional rotation reserve.

    Args:
        local_minimum: Raw authored minimum on the selected source axis.
        support_height: Explicit world support top.
        release_clearance: Explicit release offset above support.
        staging_radius: Optional complete scaled-vertex rotation radius.
        staging_margin: Additional clearance around that rotation radius.
        staging_clearance: Fixed extra staging height when no radius is selected.

    Returns:
        Declared object-origin height; no contact/stability assurance is implied.
    """
    height = support_height + release_clearance - local_minimum
    if staging_radius is not None:
        height = max(height, support_height + staging_radius + staging_margin)
    else:
        height += staging_clearance
    return height


def bounded_carry_lift(
    bottom: float, top: float, *, minimum: float, maximum: float, reserve: float
) -> float:
    """Preserve the original vertical-span carry lift within explicit bounds.

    Args:
        bottom: Complete source geometry minimum vertical offset.
        top: Complete source geometry maximum vertical offset.
        minimum: Explicit minimum lift.
        maximum: Explicit maximum lift.
        reserve: Explicit extra clearance above half the complete span.

    Returns:
        Numeric declaration lift, without motion or collision qualification.
    """
    return max(minimum, min(maximum, 0.5 * (top - bottom) + reserve))
