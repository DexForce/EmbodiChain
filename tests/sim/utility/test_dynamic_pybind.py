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
import pytest

pytest.importorskip("dexsim")
pytest.importorskip("open3d")

from dexsim.kit import meshproc

from embodichain.lab.sim.utility.dynamic_pybind import set_projective_uv


class _FakeRenderBody:
    def __init__(self) -> None:
        self._vertices = [
            np.zeros((3, 3), dtype=np.float32),
            np.ones((3, 3), dtype=np.float32),
        ]
        self._triangles = [
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 1, 2]], dtype=np.int32),
        ]
        self.uv_mappings: dict[int, np.ndarray] = {}

    def get_mesh_count(self) -> int:
        return len(self._vertices)

    def get_vertices(self, mesh_id: int) -> np.ndarray:
        return self._vertices[mesh_id]

    def get_triangles(self, mesh_id: int) -> np.ndarray:
        return self._triangles[mesh_id]

    def set_uv_mapping(self, uvs: np.ndarray, mesh_id: int) -> None:
        self.uv_mappings[mesh_id] = uvs


def test_projective_uv_offsets_faces_for_multiple_meshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render_body = _FakeRenderBody()
    project_direction = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    generated_uv = np.arange(12, dtype=np.float32).reshape(6, 2)

    def fake_get_mesh_auto_uv(mesh, received_direction):
        np.testing.assert_array_equal(mesh.triangle.indices.numpy(), expected_faces)
        np.testing.assert_array_equal(received_direction, project_direction)
        return True, generated_uv

    monkeypatch.setattr(meshproc, "get_mesh_auto_uv", fake_get_mesh_auto_uv)

    set_projective_uv(render_body, project_direction)

    np.testing.assert_array_equal(render_body.uv_mappings[0], generated_uv[:3])
    np.testing.assert_array_equal(render_body.uv_mappings[1], generated_uv[3:])
