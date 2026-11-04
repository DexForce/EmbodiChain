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
"""Shared mesh topology and flat normals for portable visual formats."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import AssemblyGeometry

__all__: list[str] = []


def _mesh_surface_arrays(
    geometry: AssemblyGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the source triangle surface without a native SDK dependency."""
    return _checked_surface_arrays(
        np.asarray(geometry.vertices, dtype=np.float64).reshape(-1, 3),
        np.asarray(geometry.triangles, dtype=np.int64).reshape(-1, 3),
        None,
        name=geometry.name,
    )


def _checked_surface_arrays(
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate surfaces and preserve authored normals or compute flat normals."""
    if (
        not len(points)
        or not len(faces)
        or not np.isfinite(points).all()
        or faces.min() < 0
        or faces.max() >= len(points)
    ):
        raise ValueError(f"Invalid assembly visual surface for {name!r}.")
    if normals is None or not np.asarray(normals).size:
        # Each triangle corner references its face normal, rather than an
        # average across the sharp edges of the source convex piece.
        triangles = points[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normal_indices = np.repeat(np.arange(len(faces))[:, None], 3, axis=1)
    else:
        # Native primitive meshes already split cap and side vertices/normals.
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError(f"Invalid primitive visual normals for {name!r}.")
        normal_indices = faces.copy()
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]
    # A zero-area triangle has no defined surface direction; keep its source
    # topology without exporting an invalid zero normal.
    normals[~valid] = [0.0, 0.0, 1.0]
    return points, faces, normals, normal_indices
