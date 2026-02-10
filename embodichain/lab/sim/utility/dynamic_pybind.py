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

import dexsim
import numpy as np

from dexsim.engine import RenderBody


def set_projective_uv(self: RenderBody, proj_direct: np.ndarray | None = None) -> None:
    """Set projective uv mapping to render body.

    Args:
        proj_direct: UV project direction. Default to be None, using svd.
    """
    import open3d as o3d
    from dexsim.kit.meshproc import get_mesh_auto_uv

    n_mesh = self.get_mesh_count()
    if n_mesh <= 0:
        return
    n_vert_list = []
    verts = np.empty((0, 3), dtype=np.float32)
    faces = np.empty((0, 3), dtype=np.int32)
    # gather all vertices
    for i in range(n_mesh):
        mesh_verts = self.get_vertices(mesh_id=i)
        n_vert_list.append(mesh_verts.shape[0])
        verts = np.vstack((verts, mesh_verts))

        mesh_faces = self.get_triangles(mesh_id=i)
        faces = np.vstack((faces, mesh_faces))
    if (verts.shape[0] == 0) or (faces.shape[0] == 0):
        return
    # project uv for all vertices
    mesh_o3dt = o3d.t.geometry.TriangleMesh()
    mesh_o3dt.vertex.positions = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
    mesh_o3dt.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    is_success, vert_uvs = get_mesh_auto_uv(mesh_o3dt, proj_direct)

    # set uv mapping for each mesh
    start_idx = 0
    for i in range(n_mesh):
        mesh_vert_uvs = vert_uvs[start_idx : start_idx + n_vert_list[i], :]
        self.set_uv_mapping(uvs=mesh_vert_uvs, mesh_id=i)
        start_idx += n_vert_list[i]


def init_dynamic_pybind() -> None:
    """Initialize dynamic pybind interface."""

    RenderBody.set_projective_uv = set_projective_uv
