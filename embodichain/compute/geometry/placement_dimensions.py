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

"""Numeric source-envelope dimensions for declaration-time placement."""

from __future__ import annotations
import numpy as np

__all__ = [
    "dominant_local_axis",
    "vertical_placement_bounds",
    "horizontal_placement_extent",
    "horizontal_placement_distance",
    "horizontal_slender_axis",
]


def dominant_local_axis(vertices: np.ndarray) -> list[float]:
    """Return the unique major mesh axis in the object-local basis.

    Args:
        vertices: Scaled local vertices with shape ``(V, 3)``.

    Returns:
        Unit basis vector for the unique long axis. Near-ties retain the
        original 1.25 ratio and 1e-6 extent rejection thresholds.
    """
    extents = np.ptp(vertices, axis=0)
    order = np.argsort(extents)
    major = int(order[-1])
    if extents[major] <= max(float(extents[order[-2]]) * 1.25, 1.0e-6):
        raise ValueError(
            "Source geometry has no unique dominant "
            "local axis for semantic upright alignment."
        )
    axis = [0.0, 0.0, 0.0]
    axis[major] = 1.0
    return axis


def vertical_placement_bounds(
    vertices: np.ndarray,
    rotation: np.ndarray,
    *,
    axis_aligned: bool,
) -> tuple[float, float]:
    """Return bottom/top offsets for the intended object orientation.

    Args:
        vertices: Scaled local vertices with shape ``(V, 3)``.
        rotation: Local-to-world 3x3 rotation for the ordinary orientation.
        axis_aligned: Use the qualified long local axis as vertical.

    Returns:
        Minimum and maximum vertical coordinates relative to the object origin.
    """
    if axis_aligned:
        axis = np.asarray(dominant_local_axis(vertices), dtype=np.float64)
        heights = vertices @ axis
    else:
        heights = vertices @ rotation[2]
    return float(heights.min()), float(heights.max())


def horizontal_placement_extent(
    vertices: np.ndarray,
    rotation: np.ndarray,
    *,
    world_axis: int,
    axis_aligned: bool,
) -> float:
    """Return a conservative horizontal half extent in the intended pose.

    Args:
        vertices: Scaled local vertices with shape ``(V, 3)``.
        rotation: Local-to-world 3x3 rotation for ordinary orientation.
        world_axis: Explicit numeric horizontal axis index.
        axis_aligned: Use half the largest transverse extent of the long axis.

    Returns:
        Half-span under the selected source orientation convention.
    """
    if axis_aligned:
        dominant_axis = int(np.argmax(np.ptp(vertices, axis=0)))
        horizontal_extents = np.delete(np.ptp(vertices, axis=0), dominant_axis)
        return 0.5 * float(horizontal_extents.max())
    world_vertices = vertices @ rotation.T
    return 0.5 * float(np.ptp(world_vertices[:, world_axis]))


def horizontal_placement_distance(
    object_vertices: np.ndarray,
    object_rotation: np.ndarray,
    reference_vertices: np.ndarray,
    reference_rotation: np.ndarray,
    *,
    reference_is_articulation: bool,
    relation_clearance: float,
    axis_aligned_clearance: float,
    world_axis: int,
    object_axis_aligned: bool,
    reference_axis_aligned: bool,
    minimum: float,
    reference_direction: int = 1,
) -> float:
    """Return source-envelope center separation along an explicit numeric axis.

    Args:
        object_vertices: Scaled local payload vertices.
        object_rotation: Payload local-to-world rotation.
        reference_vertices: Scaled local reference vertices.
        reference_rotation: Reference local-to-world rotation.
        reference_is_articulation: Measure the signed edge from its root origin.
        relation_clearance: Ordinary envelope margin in metres.
        axis_aligned_clearance: Envelope margin for an aligned payload.
        world_axis: Explicit horizontal axis index.
        object_axis_aligned: Use the payload's transverse half-span.
        reference_axis_aligned: Use the rigid reference's transverse half-span.
        minimum: Minimum requested separation in metres.
        reference_direction: Signed side of the reference envelope.

    Returns:
        Maximum of requested minimum and source-envelope separation. No
        collision or native bearing qualification is implied.
    """
    if reference_is_articulation:
        # An articulation root need not be at its geometry center. Measure the
        # outward edge in its actual root frame, not half the aggregate span.
        points = reference_vertices @ reference_rotation.T
        edge = float((points[:, world_axis] * reference_direction).max())
        return max(
            minimum,
            edge
            + horizontal_placement_extent(
                object_vertices,
                object_rotation,
                world_axis=world_axis,
                axis_aligned=object_axis_aligned,
            )
            + (axis_aligned_clearance if object_axis_aligned else relation_clearance),
        )
    return max(
        minimum,
        horizontal_placement_extent(
            object_vertices,
            object_rotation,
            world_axis=world_axis,
            axis_aligned=object_axis_aligned,
        )
        + horizontal_placement_extent(
            reference_vertices,
            reference_rotation,
            world_axis=world_axis,
            axis_aligned=reference_axis_aligned,
        )
        + (axis_aligned_clearance if object_axis_aligned else relation_clearance),
    )


def horizontal_slender_axis(vertices: np.ndarray, rotation: np.ndarray) -> bool:
    """Apply the source horizontal three-to-one full-mesh extent selector.

    Args:
        vertices: Scaled object-local mesh vertices, shape (V, 3).
        rotation: Authored local-to-world object rotation, shape (3, 3).

    Returns:
        Whether the longest full-mesh extent is at least three times the middle
        positive extent and its world vertical axis component is below one half.
        This selects a proposal policy only; it is not grasp qualification.
    """
    spans = np.ptp(vertices, axis=0)
    ordered = np.sort(spans)
    world_axis = rotation[:, int(np.argmax(spans))]
    return bool(
        ordered[1] > 0 and ordered[2] >= 3 * ordered[1] and abs(world_axis[2]) < 0.5
    )
