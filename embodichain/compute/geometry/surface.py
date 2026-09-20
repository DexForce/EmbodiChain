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

"""Topology-aware surface partitioning and face-to-vertex score transfer."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["partition_surface", "face_scores_to_vertices"]


def _geometry(
    vertices: NDArray, faces: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must have nonempty shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("faces must be nonempty integer triangles indexing vertices")
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(cross, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("mesh contains degenerate triangles")
    return (
        vertices,
        faces,
        triangles.mean(axis=1),
        cross / lengths[:, None],
        lengths / 2,
    )


def partition_surface(
    vertices: NDArray, faces: NDArray, patch_count: int = 64
) -> NDArray[np.int32]:
    """Partition triangles by geodesic distance with a bend penalty.

    Exact duplicate positions are welded only for adjacency, preserving input
    indexing. Disconnected components never share a patch. Distances follow
    mesh edges, so nearby opposite sides of a thin wall remain distinct.

    Args:
        vertices: Finite source coordinates, shape ``(N, 3)``.
        faces: Nondegenerate triangle indices, shape ``(F, 3)``.
        patch_count: Maximum number of surface patches.

    Returns:
        Contiguous zero-based patch IDs, one per input triangle.

    Raises:
        ValueError: Geometry is invalid or has more components than patches.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra

    vertices, faces, centers, normals, areas = _geometry(vertices, faces)
    if type(patch_count) is not int or patch_count < 1:
        raise ValueError("patch_count must be a positive integer")
    count = min(patch_count, len(faces))
    _, welded = np.unique(vertices, axis=0, return_inverse=True)
    triangles = welded[faces]
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, groups, multiplicity = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(groups, kind="stable")
    offsets = np.cumsum(multiplicity) - multiplicity
    pairs = []
    # Non-manifold edges are barriers; never connect unrelated overlapping sheets.
    for edge in np.flatnonzero(multiplicity == 2):
        pair = order[offsets[edge] : offsets[edge] + 2] // 3
        if pair[0] != pair[1]:
            pairs.append(pair)
    adjacency = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    left, right = adjacency.T
    distance = np.linalg.norm(centers[left] - centers[right], axis=1)
    bend = 1 + 4 * (1 - np.clip((normals[left] * normals[right]).sum(axis=1), -1, 1))
    weights = np.maximum(
        distance * bend,
        np.finfo(float).eps * np.ptp(vertices[np.unique(faces)], axis=0).max(),
    )
    graph = coo_matrix(
        (np.tile(weights, 2), (np.r_[left, right], np.r_[right, left])),
        shape=(len(faces), len(faces)),
    ).tocsr()
    components, _ = connected_components(graph)
    if components > count:
        raise ValueError(
            f"Mesh has {components} surface components but only {count} patches; "
            "increase patch_count or repair the mesh."
        )
    nearest = np.full(len(faces), np.inf)
    labels = np.full(len(faces), -1, dtype=np.int32)
    seed = int(areas.argmax())
    for patch in range(count):
        distances = dijkstra(graph, directed=False, indices=seed)
        improved = distances < nearest
        labels[improved] = patch
        nearest[improved] = distances[improved]
        seed = int(nearest.argmax())
    return labels


def face_scores_to_vertices(
    vertices: NDArray, faces: NDArray, face_scores: NDArray
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Average incident triangle scores using triangle area as the weight.

    Args:
        vertices: Source coordinates, shape ``(N, 3)``.
        faces: Triangle indices, shape ``(F, 3)``.
        face_scores: Finite values in ``[0, 1]``, shape ``(F,)``.

    Returns:
        Scores in source vertex order and a mask identifying referenced vertices.
        Unreferenced vertices have score zero and a false mask entry.
    """
    vertices, faces, _, _, areas = _geometry(vertices, faces)
    scores = np.asarray(face_scores, dtype=float)
    if scores.shape != (len(faces),) or not np.isfinite(scores).all():
        raise ValueError("face_scores must be a finite array with one value per face")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("face_scores must lie in [0, 1]")
    weights = np.bincount(
        faces.ravel(), weights=np.repeat(areas, 3), minlength=len(vertices)
    )
    total = np.bincount(
        faces.ravel(), weights=np.repeat(areas * scores, 3), minlength=len(vertices)
    )
    valid = weights > 0
    result = np.divide(total, weights, out=np.zeros_like(total), where=valid)
    return result.astype(np.float32), valid
