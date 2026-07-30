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

import numpy as np
from dexsim.models import MeshObject

__all__ = ["get_combined_triangles", "get_combined_vertices"]


def get_combined_vertices(entity: MeshObject) -> np.ndarray:
    """Concatenate all render-mesh vertices for one mesh object.

    Args:
        entity: Mesh object whose render meshes are combined.

    Returns:
        Vertices concatenated in render-mesh order.
    """
    render_body = entity.get_render_body()
    mesh_count = render_body.get_mesh_count()
    if mesh_count <= 1:
        return entity.get_vertices()
    return np.concatenate(
        [render_body.get_vertices(mesh_id) for mesh_id in range(mesh_count)],
        axis=0,
    )


def get_combined_triangles(entity: MeshObject) -> np.ndarray:
    """Concatenate all render-mesh faces with matching vertex offsets.

    Args:
        entity: Mesh object whose render meshes are combined.

    Returns:
        Triangle indices referencing :func:`get_combined_vertices` output.
    """
    render_body = entity.get_render_body()
    mesh_count = render_body.get_mesh_count()
    if mesh_count <= 1:
        return entity.get_triangles()

    triangles: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_id in range(mesh_count):
        triangles.append(
            render_body.get_triangles(mesh_id).astype(np.int64) + vertex_offset
        )
        vertex_offset += render_body.get_vertices(mesh_id).shape[0]
    return np.concatenate(triangles, axis=0)
