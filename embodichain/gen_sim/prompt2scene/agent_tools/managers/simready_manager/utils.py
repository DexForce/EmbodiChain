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

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "_as_transform",
    "_axis_angle_rotation",
    "_axis_conversion_transform",
    "_center_aabb_bottom_xy_at_origin",
    "_center_aabb_bottom_xy_at_origin_transform",
    "_normalize",
    "_orthogonal_axis",
    "_place_above_plane_transform",
    "_request_axis",
    "_rotation_between_vectors",
    "_scale_transform",
    "_translation_transform",
]


def _request_axis(value: list[float] | None, default: tuple[float, float, float]) -> list[float]:
    if value is not None:
        return list(value)
    return list(default)


def _center_aabb_bottom_xy_at_origin(mesh: Any) -> Any:
    bounds = mesh.bounds
    bottom_center_x = (float(bounds[0][0]) + float(bounds[1][0])) * 0.5
    bottom_center_y = (float(bounds[0][1]) + float(bounds[1][1])) * 0.5
    centered = mesh.copy()
    centered.apply_translation([-bottom_center_x, -bottom_center_y, 0.0])
    return centered


def _axis_conversion_transform(source_axis: list[float], target_axis: list[float]) -> np.ndarray:
    source = _normalize(np.asarray(source_axis, dtype=np.float64))
    target = _normalize(np.asarray(target_axis, dtype=np.float64))
    return _rotation_between_vectors(source, target)


def _place_above_plane_transform(mesh: Any, clearance: float) -> np.ndarray:
    min_z = float(mesh.bounds[0][2])
    return _translation_transform(np.array([0.0, 0.0, clearance - min_z]))


def _center_aabb_bottom_xy_at_origin_transform(mesh: Any) -> np.ndarray:
    bounds = mesh.bounds
    bottom_center_x = (float(bounds[0][0]) + float(bounds[1][0])) * 0.5
    bottom_center_y = (float(bounds[0][1]) + float(bounds[1][1])) * 0.5
    return _translation_transform(np.array([-bottom_center_x, -bottom_center_y, 0.0]))


def _translation_transform(translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return transform


def _scale_transform(scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = scale
    transform[1, 1] = scale
    transform[2, 2] = scale
    return transform


def _as_transform(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.shape == (4, 4):
        return value.astype(np.float64)
    raise TypeError(f"Cannot convert {type(value)} to 4x4 transform.")


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    cos_angle = np.dot(source, target)
    if cos_angle > 1.0 - 1e-10:
        return np.eye(4, dtype=np.float64)
    if cos_angle < -1.0 + 1e-10:
        axis = _orthogonal_axis(source)
        return _axis_angle_rotation(axis, np.pi)
    axis = np.cross(source, target)
    sin_angle = np.linalg.norm(axis)
    axis = axis / sin_angle
    angle = np.arctan2(sin_angle, cos_angle)
    return _axis_angle_rotation(axis, angle)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _orthogonal_axis(vector: np.ndarray) -> np.ndarray:
    x, y, z = _normalize(vector)
    if abs(x) < 0.9:
        return np.array([1.0, 0.0, -x / (z + 1e-10)], dtype=np.float64)
    return np.array([-y / (x + 1e-10), 1.0, 0.0], dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalise zero-length vector.")
    return vector / norm
