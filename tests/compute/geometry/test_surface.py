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

"""Surface partitions must respect topology and source vertex indices."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.compute.geometry.surface import (
    partition_surface,
    face_scores_to_vertices,
)


def test_nearby_disconnected_sheets_never_share_patches():
    # Identical squares separated by only 1 micron must remain independent.
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    vertices = np.vstack([square, square + [0, 0, 1e-6]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    labels = partition_surface(vertices, faces, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    with pytest.raises(ValueError, match="components"):
        partition_surface(vertices, faces, 1)


def test_exact_seam_duplicates_connect_without_reindexing():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    original = vertices.copy()
    assert partition_surface(vertices, faces, 1).tolist() == [0, 0]
    np.testing.assert_array_equal(vertices, original)


def test_scores_are_area_weighted_and_preserve_unreferenced_vertices():
    # Incident triangle areas are 1/2 and 1, giving shared vertices a score of 2/3.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-2, 0, 0], [9, 9, 9]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    scores, valid = face_scores_to_vertices(vertices, faces, np.array([0, 1]))
    np.testing.assert_allclose(scores, [2 / 3, 0, 2 / 3, 1, 0])
    assert valid.tolist() == [True, True, True, True, False]


@pytest.mark.parametrize("values", [[np.nan, 0], [-0.1, 0], [1.1, 0], [0]])
def test_invalid_scores_are_rejected(values):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(ValueError):
        face_scores_to_vertices(vertices, faces, np.array(values))


def test_invalid_geometry_is_rejected():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="degenerate"):
        partition_surface(vertices, np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="indexing"):
        partition_surface(vertices, np.array([[0, 1, 3]]))


def test_partition_is_deterministic_and_scale_invariant():
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=1)
    labels = partition_surface(mesh.vertices, mesh.faces, 12)
    moved = partition_surface(mesh.vertices * 10 + [2, 4, 6], mesh.faces, 12)
    # Every patch exists, even when the surface is closed.
    assert set(labels) == set(range(12))
    assert np.array_equal(labels, partition_surface(mesh.vertices, mesh.faces, 12))
    # Symmetric meshes may break exact FPS ties differently after floating-point transforms.
    assert len(set(moved)) == 12
