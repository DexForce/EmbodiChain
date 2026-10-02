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

"""Small Blender building blocks exposed to Codex-authored build() functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy
from mathutils import Vector

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["box", "cylinder", "rod", "torus", "union", "difference"]


def _apply(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def box(
    size: Sequence[float], center: Sequence[float], name: str = "box"
) -> bpy.types.Object:
    """Create a box with applied dimensions in meters.

    Args:
        size: Full XYZ dimensions.
        center: Center in local asset coordinates.
        name: Blender object name.

    Returns:
        A closed Blender mesh object.
    """
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    return _apply(obj)


def cylinder(
    radius: float, depth: float, center: Sequence[float], vertices: int = 64
) -> bpy.types.Object:
    """Create a closed Z-axis cylinder in meters.

    Args:
        radius: Cross-section radius.
        depth: Total height.
        center: Center coordinates.
        vertices: Radial segment count.

    Returns:
        A closed cylinder mesh.
    """
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        end_fill_type="NGON",
        location=center,
    )
    return _apply(bpy.context.object)


def rod(
    start: Sequence[float], end: Sequence[float], radius: float, vertices: int = 48
) -> bpy.types.Object:
    """Create a cylinder between two endpoints.

    Args:
        start: First end center.
        end: Second end center.
        radius: Cylinder radius.
        vertices: Radial segments.

    Returns:
        The oriented rod mesh.
    """
    start, end = Vector(start), Vector(end)
    direction = end - start
    obj = cylinder(radius, direction.length, (start + end) / 2, vertices)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    bpy.context.view_layer.update()
    return obj


def torus(
    major_radius: float,
    minor_radius: float,
    center: Sequence[float],
    rotation: Sequence[float] = (0, 0, 0),
) -> bpy.types.Object:
    """Create a handle-like torus whose unrotated axis is Z.

    Args:
        major_radius: Centerline radius.
        minor_radius: Tube radius.
        center: Object center.
        rotation: XYZ Euler angles in radians.

    Returns:
        A closed torus mesh.
    """
    bpy.ops.mesh.primitive_torus_add(
        major_segments=64,
        minor_segments=16,
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=center,
        rotation=rotation,
    )
    return _apply(bpy.context.object)


def _boolean(
    left: bpy.types.Object, right: bpy.types.Object, operation: str
) -> bpy.types.Object:
    _apply(left)
    modifier = left.modifiers.new(name=operation, type="BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = right
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(right, do_unlink=True)
    return left


def union(objects: Sequence[bpy.types.Object]) -> bpy.types.Object:
    """Boolean-union solids, consuming all objects except the first.

    Args:
        objects: Solids; overlapping parts should intersect by at least 1 mm.

    Returns:
        One mesh with internal overlapping surfaces removed.
    """
    result, *others = objects
    for other in others:
        _boolean(result, other, "UNION")
    return result


def difference(solid: bpy.types.Object, cutter: bpy.types.Object) -> bpy.types.Object:
    """Subtract a closed cutter and remove the cutter object.

    Args:
        solid: Object to modify, such as a mug exterior.
        cutter: Volume to remove; extend beyond a face to make an opening.

    Returns:
        The modified solid.
    """
    return _boolean(solid, cutter, "DIFFERENCE")
