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
"""Source geometry preserves authored topology and owns its numeric inputs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from embodichain.lab.sim.scene.source_geometry import LinkMesh


def test_input_mesh_preserves_order_and_detaches_mutable_buffers():
    vertices = np.array(
        [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[2, 0, 1]], dtype=np.int32)
    mesh = LinkMesh("body/collider", "body", vertices, faces, "convexHull")
    copied = deepcopy(mesh)
    vertices[:] = 99
    faces[:] = 0
    for owned in (mesh, copied):
        np.testing.assert_array_equal(
            owned.vertices, [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        )
        np.testing.assert_array_equal(owned.faces, [[2, 0, 1]])
        assert owned.vertices.dtype == np.float32
        assert owned.faces.dtype == np.int32
        assert not owned.vertices.flags.writeable
        assert not owned.faces.flags.writeable
    assert not np.shares_memory(mesh.vertices, copied.vertices)
    cloud = LinkMesh(
        "cloud", "body", mesh.vertices, np.empty((0, 3), dtype=np.int64), "none"
    )
    assert cloud.faces.shape == (0, 3)


@pytest.mark.parametrize(
    "vertices,faces",
    [
        ([[float("nan"), 0, 0]], [[0, 0, 0]]),
        ([[0.0, 0.0, 0.0]], [[0, 1, 0]]),
        ([[0.0, 0.0, 0.0]], [[-1, 0, 0]]),
        ([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]),
    ],
)
def test_source_mesh_rejects_invalid_geometry(vertices, faces):
    with pytest.raises(ValueError):
        LinkMesh("collider", "link", np.asarray(vertices), np.asarray(faces), "none")
