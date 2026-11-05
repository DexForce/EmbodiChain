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

"""Table landing preserves full support, holes and obstacle clearance."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from embodichain.compute.geometry.landing import select_table_landing


def _select(obstacles=(), *, position=(0.0, 0.0, 0.16), axis=None):
    table = trimesh.creation.box(extents=[1, 1, 0.1])
    vertices = trimesh.creation.box(extents=[0.1, 0.1, 0.2]).vertices.copy()
    before = vertices.copy()
    result = select_table_landing(
        table,
        vertices,
        position,
        obstacles,
        table_top_z=0.05,
        object_bottom=-0.1,
        object_top=0.1,
        axis_aligned_axis=axis,
        placement_clearance=0.01,
        relation_clearance=0.02,
    )
    np.testing.assert_array_equal(vertices, before)
    return result


def test_clear_table_preserves_original_origin_and_ignores_vertical_nonoverlap():
    assert _select() == (0.0, 0.0, 0.16)
    assert _select((np.array([[-0.5, -0.5, 1.0], [0.5, 0.5, 2.0]]),)) == (
        0.0,
        0.0,
        0.16,
    )


def test_occupied_origin_moves_only_as_needed_and_keeps_height():
    value = _select((np.array([[-0.12, -0.12, 0.05], [0.12, 0.12, 0.4]]),))
    assert value[2] == 0.16
    assert max(abs(value[0]), abs(value[1])) == pytest.approx(0.190001)
    assert min(abs(value[0]), abs(value[1])) == pytest.approx(0.0)


def test_missing_full_support_and_exhausted_space_are_not_replaced_by_origin_checks():
    with pytest.raises(ValueError, match="full measured support"):
        _select(position=(0.49, 0.0, 0.16))
    with pytest.raises(ValueError, match="remaining legal"):
        _select((np.array([[-1, -1, 0.05], [1, 1, 0.4]]),))


def test_table_triangle_hole_is_not_filled_by_a_convex_hull():
    values = [-0.5, -0.15, 0.15, 0.5]
    vertices = np.asarray([(x, y, 0.05) for y in values for x in values])
    faces = []
    for row in range(3):
        for column in range(3):
            if (row, column) == (1, 1):
                continue
            first = row * 4 + column
            faces.extend([(first, first + 1, first + 5), (first, first + 5, first + 4)])
    table = trimesh.Trimesh(vertices, faces, process=False)
    payload = trimesh.creation.box(extents=[0.1, 0.1, 0.2]).vertices
    with pytest.raises(ValueError, match="full measured support"):
        select_table_landing(
            table,
            payload,
            (0, 0, 0.16),
            (),
            table_top_z=0.05,
            object_bottom=-0.1,
            object_top=0.1,
            axis_aligned_axis=None,
            placement_clearance=0.01,
            relation_clearance=0.02,
        )
