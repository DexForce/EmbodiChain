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
"""Official NPZ initial-state slicing and authored surface topology."""

from __future__ import annotations
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np

__all__ = ["load_sample"]


def cloth_faces(points, indices, rows):
    """Recover the display triangles from the sample's four-neighbor grid."""
    adjacency = [
        set(map(int, indices[rows[i] : rows[i + 1]])) - {i} for i in range(len(points))
    ]
    if max(map(len, adjacency), default=0) > 16:
        raise ValueError("Cloth sample is not a triangle/four-neighbor surface graph")
    faces = set()
    for a, neighbors in enumerate(adjacency):
        for b in neighbors:
            if b > a:
                faces.update((a, b, c) for c in neighbors & adjacency[b] if c > b)
    if not faces:
        quads = set()
        for a, neighbors in enumerate(adjacency):
            for b, d in combinations(sorted(neighbors), 2):
                for c in (adjacency[b] & adjacency[d]) - {a}:
                    key = tuple(sorted((a, b, c, d)))
                    if key not in quads:
                        quads.add(key)
                        faces.update(((a, b, c), (a, c, d)))
    result = np.asarray(sorted(faces), dtype=np.int32).reshape(-1, 3)
    if not len(result):
        raise ValueError("No cloth display faces recovered from the sample topology")
    normals = np.cross(
        points[result[:, 1]] - points[result[:, 0]],
        points[result[:, 2]] - points[result[:, 0]],
    )
    # Omit coincident-point display faces; preserve the original neural CSR.
    valid = np.linalg.norm(normals, axis=1) > 1e-12
    result, normals = result[valid], normals[valid]
    flip = normals[:, 1] < 0
    result[flip] = result[flip][:, [0, 2, 1]]
    return result


def load_sample(
    path: str | Path,
    profile: dict[str, Any],
    *,
    frame_start: int = 0,
    frame_end: int | None = None,
) -> tuple:
    """Separate initial dynamics inputs from optional reference-only metrics."""
    with np.load(path, allow_pickle=False) as data:
        # Match official inference: task stride, config slice, then initial state.
        positions = data["position"][:: profile["stride"]][frame_start:frame_end]
        velocities = data["velocity"][:: profile["stride"]][frame_start:frame_end]
        if frame_start < 0 or not len(positions):
            raise ValueError("Initial frame selection must be nonnegative and nonempty")
        q = np.asarray(positions[0], np.float32)
        qd = np.asarray(velocities[0], np.float32)
        raw = data["feats"].copy()
        box = np.asarray(data["box"], np.float32)
        normals = np.asarray(data["box_feats"], np.float32)
        reference = positions.copy() if profile["reference"] else None
        faces = None
        if profile["checkpoint"] == "cloth_on_ball":
            faces = cloth_faces(
                q, data["neighbors_index"], data["neighbors_row_splits"]
            )
    if q.shape != qd.shape or q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("Initial position/velocity must have matching shapes (N,3)")
    return q, qd, raw, box, normals, faces, reference
