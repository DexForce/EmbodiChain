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

"""Pure CPU XY projection of complete triangle meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["triangle_mesh_xy_projection"]


def triangle_mesh_xy_projection(
    vertices: np.ndarray, faces: np.ndarray
) -> BaseGeometry:
    """Union all positive-area XY triangle projections in source face order.

    Args:
        vertices: Validated finite mesh vertices, shape ``(V, 3)``. Values are
            converted to float64 just as the source Trimesh mesh constructor did.
        faces: Authored triangle indices, shape ``(F, 3)``. Degenerate XY
            triangles contribute no area; empty faces produce an empty geometry.

    Returns:
        Owned Shapely geometry preserving concavities, holes and disconnected
        components. This is a complete mesh projection, not a convex hull or
        floor-region selector. Input arrays are never modified.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    points = np.asarray(vertices, dtype=np.float64)
    indices = np.asarray(faces, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or indices.ndim != 2
        or indices.shape[1] != 3
    ):
        raise ValueError(
            "Triangle projection requires (V, 3) vertices and (F, 3) faces."
        )
    triangles = (Polygon(triangle[:, :2]) for triangle in points[indices])
    return unary_union([polygon for polygon in triangles if polygon.area > 0])
