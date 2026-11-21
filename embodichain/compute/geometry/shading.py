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

"""Source-compatible geometric corner-normal consistency without mesh libraries."""

from __future__ import annotations
import numpy as np

__all__ = ["has_inconsistent_shading_normals"]


def has_inconsistent_shading_normals(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Return whether a geometric corner normal opposes its incident triangle.

    Args:
        vertices: Source positions with shape ``(V, 3)``.
        faces: Triangle vertex indices with shape ``(F, 3)``.

    Returns:
        Original v3 decision using float64, equally weighted unit-face normal
        accumulation and a strict negative-dot test. Empty topology returns
        false; nonfinite derived normals return true. This is a shading check,
        not a manifold, collision or physical validity certificate.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if faces.size == 0:
        return False
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_lengths = np.linalg.norm(crosses, axis=1)
    face_normals = np.zeros_like(crosses)
    np.divide(
        crosses,
        face_lengths[:, None],
        out=face_normals,
        where=face_lengths[:, None] > 0.0,
    )
    vertex_normals = np.zeros((len(vertices), 3), dtype=np.float64)
    np.add.at(
        vertex_normals,
        faces.reshape(-1),
        np.repeat(face_normals, 3, axis=0),
    )
    vertex_lengths = np.linalg.norm(vertex_normals, axis=1)
    np.divide(
        vertex_normals,
        vertex_lengths[:, None],
        out=vertex_normals,
        where=vertex_lengths[:, None] > 0.0,
    )
    corner_normals = vertex_normals[faces]
    if not np.isfinite(face_normals).all() or not np.isfinite(corner_normals).all():
        return True
    alignments = np.einsum("fci,fi->fc", corner_normals, face_normals)
    return bool(np.any(alignments < 0.0))
