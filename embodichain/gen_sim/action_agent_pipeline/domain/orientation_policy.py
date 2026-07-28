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

"""Canonical upright and lay-flat orientation policy for pipeline objects.

The policy is deliberately backend-independent. Generation supplies plain
mesh bounds and Runtime transfers only bounds plus one representative rotation
from its tensor device. Both stages therefore make the same semantic and
geometric decision without copying a complete runtime mesh to the CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import re

from embodichain.gen_sim.action_agent_pipeline.domain.object_semantics import (
    BOTTLE_LIKE_KEYWORDS,
    SHORT_BOTTLE_LIKE_KEYWORDS,
)

__all__ = [
    "principal_local_axis_order",
    "resolve_target_rotation",
    "rotated_local_z_min",
]

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
LocalBounds = tuple[Vector3, Vector3]

_NORMALIZED_LOCAL_Z_KEYWORDS = ("bottle", "can")
_BOTTLE_LONG_TO_MID_RATIO = 1.6
_BOTTLE_MID_TO_MIN_RATIO = 1.35
_EPSILON = 1e-6
_WORLD_X: Vector3 = (1.0, 0.0, 0.0)
_WORLD_Z: Vector3 = (0.0, 0.0, 1.0)
_LOCAL_AXES: tuple[Vector3, Vector3, Vector3] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def principal_local_axis_order(
    local_bounds: tuple[Sequence[float], Sequence[float]],
) -> tuple[int, int, int]:
    """Return local axis indices ordered from longest to shortest extent.

    Equal extents are ordered by the original local axis index. This explicit
    tie-break avoids backend-specific sorting behavior for symmetric meshes.

    Args:
        local_bounds: Local-space minimum and maximum xyz coordinates.

    Returns:
        The three local axis indices in descending extent order.

    Raises:
        ValueError: If the bounds are invalid or contain non-finite values.
    """
    mins, maxs = _validated_local_bounds(local_bounds)
    extents = tuple(maxs[index] - mins[index] for index in range(3))
    return tuple(sorted(range(3), key=lambda index: (-extents[index], index)))


def resolve_target_rotation(
    *,
    orientation_goal: str,
    local_bounds: tuple[Sequence[float], Sequence[float]],
    current_rotation: Sequence[Sequence[float]],
    object_label: str,
) -> Matrix3:
    """Resolve one canonical upright or lay-flat local-to-world rotation.

    The object label is the Runtime UID in Generation and
    ``ObjectSemantics.label`` in Runtime. No generation-only description or
    mesh filename participates in policy selection.

    Args:
        orientation_goal: Either ``"upright"`` or ``"lay_flat"``.
        local_bounds: Local-space minimum and maximum xyz coordinates.
        current_rotation: Current local-to-world 3x3 rotation.
        object_label: Canonical runtime object label.

    Returns:
        A deterministic local-to-world 3x3 rotation.

    Raises:
        ValueError: If the goal, bounds, or rotation is invalid.
    """
    bounds = _validated_local_bounds(local_bounds)
    rotation = _validated_matrix3(current_rotation, "current_rotation")
    axis_order = principal_local_axis_order(bounds)
    long_axis = _LOCAL_AXES[axis_order[0]]
    short_axis = _LOCAL_AXES[axis_order[2]]

    if orientation_goal == "lay_flat":
        # The shortest dimension is the only canonical vertical thickness.
        return _rotation_from_axis_targets(
            local_primary=long_axis,
            world_primary=_WORLD_X,
            local_secondary=short_axis,
            world_secondary=_WORLD_Z,
        )
    if orientation_goal != "upright":
        raise ValueError(
            "Canonical orientation policy supports only 'upright' and 'lay_flat'; "
            f"got {orientation_goal!r}."
        )

    if _is_normalized_local_z_label(object_label):
        return _preview_preserving_upright_rotation(
            primary_axis=_LOCAL_AXES[2],
            secondary_axes=(_LOCAL_AXES[0], _LOCAL_AXES[1]),
            current_rotation=rotation,
        )
    if _is_bottle_like(object_label, bounds):
        return _preview_preserving_upright_rotation(
            primary_axis=long_axis,
            secondary_axes=tuple(_LOCAL_AXES[index] for index in axis_order[1:]),
            current_rotation=rotation,
        )
    return _rotation_from_axis_targets(
        local_primary=long_axis,
        world_primary=_WORLD_Z,
        local_secondary=short_axis,
        world_secondary=_WORLD_X,
    )


def rotated_local_z_min(
    vertices: Sequence[Sequence[float]],
    rotation: Sequence[Sequence[float]],
) -> float:
    """Return the exact minimum Z after rotating local mesh vertices.

    Applying the rotation to real vertices preserves bottom-origin and
    off-center mesh frames; a half-extent approximation cannot do so.

    Args:
        vertices: Non-empty local-space mesh vertices.
        rotation: Local-to-world 3x3 rotation.

    Returns:
        Minimum rotated Z coordinate.

    Raises:
        ValueError: If vertices or rotation are invalid.
    """
    matrix = _validated_matrix3(rotation, "rotation")
    clean_vertices = [
        _validated_vector3(vertex, f"vertices[{index}]")
        for index, vertex in enumerate(vertices)
    ]
    if not clean_vertices:
        raise ValueError("vertices must contain at least one xyz coordinate.")
    return min(_matrix_vector_mul(matrix, vertex)[2] for vertex in clean_vertices)


def _is_normalized_local_z_label(object_label: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(object_label).lower()))
    return any(keyword in tokens for keyword in _NORMALIZED_LOCAL_Z_KEYWORDS)


def _is_bottle_like(object_label: str, local_bounds: LocalBounds) -> bool:
    if _has_bottle_like_keyword(str(object_label).lower()):
        return True
    mins, maxs = local_bounds
    extents = sorted(maxs[index] - mins[index] for index in range(3))
    min_extent = max(extents[0], _EPSILON)
    mid_extent = max(extents[1], _EPSILON)
    return (
        extents[2] / mid_extent >= _BOTTLE_LONG_TO_MID_RATIO
        and mid_extent / min_extent <= _BOTTLE_MID_TO_MIN_RATIO
    )


def _has_bottle_like_keyword(text: str) -> bool:
    tokens = (
        text.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    ).split()
    return any(
        keyword in tokens if keyword in SHORT_BOTTLE_LIKE_KEYWORDS else keyword in text
        for keyword in BOTTLE_LIKE_KEYWORDS
    )


def _preview_preserving_upright_rotation(
    *,
    primary_axis: Vector3,
    secondary_axes: tuple[Vector3, ...],
    current_rotation: Matrix3,
) -> Matrix3:
    candidates: list[tuple[float, Matrix3]] = []
    for secondary_axis in (
        *secondary_axes,
        *(_scale_vector(axis, -1.0) for axis in secondary_axes),
    ):
        preview_secondary = _matrix_vector_mul(current_rotation, secondary_axis)
        world_secondary = (preview_secondary[0], preview_secondary[1], 0.0)
        if _vector_norm(world_secondary) < _EPSILON:
            continue
        rotation = _rotation_from_axis_targets(
            local_primary=primary_axis,
            world_primary=_WORLD_Z,
            local_secondary=secondary_axis,
            world_secondary=world_secondary,
        )
        candidates.append(
            (_rotation_distance_score(rotation, current_rotation), rotation)
        )
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return _rotation_from_axis_targets(
        local_primary=primary_axis,
        world_primary=_WORLD_Z,
        local_secondary=secondary_axes[-1],
        world_secondary=_WORLD_X,
    )


def _rotation_from_axis_targets(
    *,
    local_primary: Sequence[float],
    world_primary: Sequence[float],
    local_secondary: Sequence[float],
    world_secondary: Sequence[float],
) -> Matrix3:
    local_primary = _normalize_vector(local_primary)
    world_primary = _normalize_vector(world_primary)
    local_secondary = _orthogonalized_axis(local_secondary, local_primary)
    world_secondary = _orthogonalized_axis(world_secondary, world_primary)
    local_basis = _columns_to_matrix(
        (
            local_primary,
            local_secondary,
            _normalize_vector(_cross(local_primary, local_secondary)),
        )
    )
    world_basis = _columns_to_matrix(
        (
            world_primary,
            world_secondary,
            _normalize_vector(_cross(world_primary, world_secondary)),
        )
    )
    return _matrix_multiply(world_basis, _matrix_transpose(local_basis))


def _orthogonalized_axis(
    axis: Sequence[float],
    reference: Sequence[float],
) -> Vector3:
    clean_axis = _validated_vector3(axis, "axis")
    clean_reference = _validated_vector3(reference, "reference")
    dot = _dot(clean_axis, clean_reference)
    projected = tuple(
        clean_axis[index] - dot * clean_reference[index] for index in range(3)
    )
    if _vector_norm(projected) < _EPSILON:
        fallback = _WORLD_X
        if abs(_dot(fallback, clean_reference)) > 0.9:
            fallback = (0.0, 1.0, 0.0)
        fallback_dot = _dot(fallback, clean_reference)
        projected = tuple(
            fallback[index] - fallback_dot * clean_reference[index]
            for index in range(3)
        )
    return _normalize_vector(projected)


def _rotation_distance_score(
    rotation: Matrix3,
    reference_rotation: Matrix3,
) -> float:
    delta = _matrix_multiply(rotation, _matrix_transpose(reference_rotation))
    return -sum(delta[index][index] for index in range(3))


def _validated_local_bounds(
    local_bounds: tuple[Sequence[float], Sequence[float]],
) -> LocalBounds:
    if not isinstance(local_bounds, Sequence) or len(local_bounds) != 2:
        raise ValueError("local_bounds must contain minimum and maximum xyz values.")
    mins = _validated_vector3(local_bounds[0], "local_bounds minimum")
    maxs = _validated_vector3(local_bounds[1], "local_bounds maximum")
    if any(maxs[index] < mins[index] for index in range(3)):
        raise ValueError(
            "local_bounds maximum values must not be below minimum values."
        )
    return mins, maxs


def _validated_matrix3(
    matrix: Sequence[Sequence[float]],
    name: str,
) -> Matrix3:
    if not isinstance(matrix, Sequence) or len(matrix) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix.")
    return tuple(
        _validated_vector3(row, f"{name}[{index}]") for index, row in enumerate(matrix)
    )


def _validated_vector3(vector: Sequence[float], name: str) -> Vector3:
    if not isinstance(vector, Sequence) or len(vector) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _columns_to_matrix(columns: Sequence[Sequence[float]]) -> Matrix3:
    return tuple(
        tuple(float(columns[column][row]) for column in range(3)) for row in range(3)
    )


def _matrix_multiply(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> Matrix3:
    return tuple(
        tuple(
            sum(
                float(left[row][index]) * float(right[index][column])
                for index in range(3)
            )
            for column in range(3)
        )
        for row in range(3)
    )


def _matrix_transpose(matrix: Sequence[Sequence[float]]) -> Matrix3:
    return tuple(
        tuple(float(matrix[column][row]) for column in range(3)) for row in range(3)
    )


def _matrix_vector_mul(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> Vector3:
    return tuple(
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )


def _normalize_vector(vector: Sequence[float]) -> Vector3:
    clean_vector = _validated_vector3(vector, "vector")
    norm = _vector_norm(clean_vector)
    if norm < _EPSILON:
        raise ValueError("Cannot normalize a near-zero vector.")
    return tuple(value / norm for value in clean_vector)


def _scale_vector(vector: Sequence[float], scale: float) -> Vector3:
    clean_vector = _validated_vector3(vector, "vector")
    return tuple(value * float(scale) for value in clean_vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))
