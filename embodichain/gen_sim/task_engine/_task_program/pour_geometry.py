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

"""Pure source-only gravity-top proposals, never poured or physical evidence."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["staging_pose", "pour_frames"]

_UP = np.array([0.0, 0.0, 1.0])
_BEARING_TIE_M = 1e-5
_AXIS_EPS = 1e-9


def _pose(value: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("Pour geometry requires a finite real SE3 pose.")
    pose = np.array(value, dtype=np.float64, copy=True)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-5, rtol=0)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=2e-5, rtol=0)
    ):
        raise ValueError("Pour geometry requires a finite proper SE3 pose.")
    return pose


def _vertices(value: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("Pour geometry requires finite real native vertices.")
    vertices = np.array(value, dtype=np.float64, copy=True)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Pour geometry requires nonempty finite native vertices.")
    return vertices


def _gravity(value: np.ndarray) -> None:
    gravity = np.asarray(value)
    if (
        gravity.shape != (3,)
        or np.iscomplexobj(gravity)
        or not np.isfinite(gravity).all()
        or gravity[0] != 0
        or gravity[1] != 0
        or gravity[2] >= 0
    ):
        raise ValueError(
            "Pour geometry requires finite nonzero negative-Z World gravity."
        )


def _top(vertices: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = rotation.T @ _UP
    up /= np.linalg.norm(up)
    center = (vertices.min(0) + vertices.max(0)) / 2
    pivot = center + up * ((vertices @ up).max() - center @ up)
    return up, pivot


def staging_pose(
    source_vertices: np.ndarray,
    receiver_vertices: np.ndarray,
    observed_source_pose: np.ndarray,
    observed_receiver_pose: np.ndarray,
    *,
    clearance_m: float,
    gravity_world: np.ndarray,
) -> np.ndarray:
    """Preserve observed entry R in a conservative source-only staging proposal.

    Args:
        source_vertices: Scaled native source vertices in its runtime-local basis.
        receiver_vertices: Scaled native receiver vertices in its runtime-local basis.
        observed_source_pose: Current unbatched source arena SE3, not initial-R.
        observed_receiver_pose: Current unbatched receiver arena SE3.
        clearance_m: Explicit nonnegative source-envelope clearance in meters.
        gravity_world: Actual World gravity; this MVP requires finite negative-Z.

    Returns:
        A fresh float64 4x4 pose retaining every observed rotation value. The
        source sweep sphere certifies neither the hand, a receiver opening,
        a collision-free path nor native bearing. No rotation is reprojected.
    """
    _gravity(gravity_world)
    source, receiver = _pose(observed_source_pose), _pose(observed_receiver_pose)
    if (
        isinstance(clearance_m, (bool, np.bool_))
        or not isinstance(clearance_m, Real)
        or not math.isfinite(clearance_m)
        or clearance_m < 0
    ):
        raise ValueError("clearance_m must be finite and nonnegative.")
    vertices, receiving = _vertices(source_vertices), _vertices(receiver_vertices)
    _, pivot = _top(vertices, source[:3, :3])
    # Bound native-float32 matrix stretch without normalizing the observed R.
    stretch = np.sqrt(np.abs(source[:3, :3].T @ source[:3, :3]).sum(axis=1).max())
    radius = stretch * np.linalg.norm(vertices - pivot, axis=-1).max()
    center = (
        receiver[:3, :3] @ ((receiving.min(0) + receiving.max(0)) / 2) + receiver[:3, 3]
    )
    world = receiving @ receiver[:3, :3].T + receiver[:3, 3]
    anchor = center.copy()
    anchor[2] = world[:, 2].max() + radius + clearance_m
    result = source.copy()
    result[:3, 3] = anchor - source[:3, :3] @ pivot
    return result


def pour_frames(
    source_vertices: np.ndarray,
    receiver_vertices: np.ndarray,
    observed_staging_pose: np.ndarray,
    observed_receiver_pose: np.ndarray,
    jaw_line_object: np.ndarray,
    declared_angle_rad: float,
    return_pose: np.ndarray,
    *,
    sample_count: int,
    gravity_world: np.ndarray,
) -> dict[str, Any]:
    """Build one fixed-pivot tilt/restore arc using an ordered object-local jaw.

    Args:
        source_vertices: Scaled native source vertices in runtime-local meters.
        receiver_vertices: Scaled native receiver vertices in runtime-local meters.
        observed_staging_pose: Current unbatched source arena pose at Pour entry.
        observed_receiver_pose: Current unbatched receiver arena pose.
        jaw_line_object: Ordered pad-line vector in the source object's frame.
        declared_angle_rad: Nonzero tilt magnitude; no explicit sign is owned.
        return_pose: Explicit measured-entry return SE3; never inferred here.
        sample_count: Number of samples including entry, peak and restored end.
        gravity_world: Actual World gravity; this MVP requires finite negative-Z.

    Returns:
        ``object_arc`` (N,4,4), ``object_targets`` (5,4,4), ordered ``target_roles``
        and axis/pivot/bearing diagnostics. Targets are staging, peak, restore,
        return transport and return Place. ``physical_evidence`` is always False.
        The caller owns candidate G/TCP composition and semantic/physical gates.
    """
    _gravity(gravity_world)
    stage, receiver, returned = (
        _pose(observed_staging_pose),
        _pose(observed_receiver_pose),
        _pose(return_pose),
    )
    vertices, receiving = _vertices(source_vertices), _vertices(receiver_vertices)
    if np.iscomplexobj(jaw_line_object):
        raise ValueError("A finite real object-local jaw line is required.")
    jaw = np.array(jaw_line_object, dtype=np.float64, copy=True)
    norm = np.linalg.norm(jaw)
    if (
        jaw.shape != (3,)
        or not np.isfinite(jaw).all()
        or not math.isfinite(norm)
        or norm <= _AXIS_EPS
    ):
        raise ValueError("A finite nonzero object-local ordered jaw line is required.")
    if (
        isinstance(declared_angle_rad, (bool, np.bool_))
        or not isinstance(declared_angle_rad, Real)
        or not math.isfinite(declared_angle_rad)
        or declared_angle_rad == 0
    ):
        raise ValueError("A finite nonzero declared angle is required.")
    if type(sample_count) is not int or sample_count < 3:
        raise ValueError("sample_count must be an integer >=3.")
    up, pivot = _top(vertices, stage[:3, :3])
    jaw /= norm
    axis = jaw - (jaw @ up) * up
    magnitude = np.linalg.norm(axis)
    if magnitude <= _AXIS_EPS:
        raise ValueError(
            "Jaw axis is parallel to gravity; projected tilt axis is degenerate."
        )
    axis /= magnitude
    anchor = stage[:3, :3] @ pivot + stage[:3, 3]
    center = (
        receiver[:3, :3] @ ((receiving.min(0) + receiving.max(0)) / 2) + receiver[:3, 3]
    )
    visible = stage[:3, :3] @ np.cross(axis, up)
    visible /= np.linalg.norm(visible)
    bearing = float((center - anchor) @ visible)
    tie = abs(bearing) <= _BEARING_TIE_M
    # Direction-free E3 owns an ordered jaw, not a receiver-facing spout.
    # Millimetric tracking error must not reverse the qualified tilt arc.
    theta, peak = abs(float(declared_angle_rad)), sample_count // 2
    angles = np.concatenate(
        (
            np.linspace(0, theta, peak + 1),
            np.linspace(theta, 0, sample_count - peak)[1:],
        )
    )
    rotations = stage[:3, :3] @ Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
    arc = np.tile(np.eye(4), (sample_count, 1, 1))
    arc[:, :3, :3], arc[:, :3, 3] = rotations, anchor - rotations @ pivot
    arc[0], arc[-1] = stage, stage
    return {
        "object_arc": arc,
        "object_targets": np.stack((stage, arc[peak], stage, returned, returned)),
        "target_roles": (
            "staging",
            "peak",
            "restore",
            "return_transport",
            "return_place",
        ),
        "pivot_object": pivot,
        "pivot_world": anchor,
        "up_object": up,
        "tilt_axis_object": axis,
        "effective_angle_rad": theta,
        "peak_index": peak,
        "receiver_bearing_m": bearing,
        "receiver_bearing_tie": tie,
        "tilt_direction_policy": "ordered_jaw_axis_plus_angle",
        "physical_evidence": False,
    }
