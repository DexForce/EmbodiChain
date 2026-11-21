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
from shapely.geometry import Point

from embodichain.compute.geometry.projection import triangle_mesh_xy_projection


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_complete_projection_retains_concavity_and_ignores_vertical_triangles(dtype):
    # A concave L covers three unit squares. A convex hull would incorrectly
    # include the upper-right gap used by the physical receiver alignment test.
    storage = np.array(
        [
            [0, 0, 0, 99],
            [2, 0, 0, 99],
            [2, 1, 0, 99],
            [1, 1, 0, 99],
            [1, 2, 0, 99],
            [0, 2, 0, 99],
            [0, 0, 1, 99],
        ],
        dtype=dtype,
    )
    vertices = storage[:, :3]
    faces = np.array(
        [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5], [0, 6, 5], [0, 1, 3]],
        dtype=np.int32,
    )
    before = vertices.copy()
    projected = triangle_mesh_xy_projection(vertices, faces)
    assert projected.area == 3.0
    assert projected.covers(Point(1, 1))
    assert not projected.covers(Point(1.5, 1.5))
    np.testing.assert_array_equal(vertices, before)
    vertices[:] = 100.0
    assert projected.area == 3.0
    assert projected.covers(Point(0.5, 0.5))
    assert triangle_mesh_xy_projection(
        np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    ).is_empty
