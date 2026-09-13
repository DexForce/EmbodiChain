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
"""Backend-specific articulation link geometry accessors."""

from __future__ import annotations

import numpy as np

__all__ = ["get_link_vert_face"]


def get_link_vert_face(
    entity: object,
    link_name: str,
    *,
    is_newton: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Read complete link mesh geometry through the active backend API."""
    if not is_newton:
        return entity.get_link_vert_face(link_name)

    render_body = entity.get_render_body(link_name)
    if render_body is None:
        return _empty_geometry()

    mesh_count = int(render_body.get_mesh_count())
    if mesh_count == 0:
        return _empty_geometry()

    vertex_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_id in range(mesh_count):
        vertices = np.asarray(render_body.get_vertices(mesh_id), dtype=np.float32)
        faces = np.asarray(render_body.get_triangles(mesh_id), dtype=np.int32)
        vertex_parts.append(vertices)
        face_parts.append(faces + vertex_offset)
        vertex_offset += len(vertices)
    return np.concatenate(vertex_parts), np.concatenate(face_parts)


def _empty_geometry() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.int32),
    )
