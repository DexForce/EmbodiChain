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

"""Mesh IO without OBJ texture/normal seam vertex duplication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if path.suffix.lower() == ".obj":
        vertices, faces = [], []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError(f"Invalid OBJ vertex at line {number}")
                vertex = [float(value) for value in parts[1:4]]
                if len(parts) == 5:  # Optional homogeneous coordinate.
                    w = float(parts[4])
                    if not np.isfinite(w) or w == 0:
                        raise ValueError(f"Invalid OBJ vertex weight at line {number}")
                    vertex = [value / w for value in vertex]
                vertices.append(vertex)
            elif parts[0] == "f":
                if len(parts) != 4:
                    raise ValueError(
                        f"OBJ line {number} is not a triangle; triangulate the mesh "
                        "before scoring (without merging or reordering vertices)."
                    )
                indices = [int(part.split("/")[0]) for part in parts[1:]]
                if 0 in indices or any(index < -len(vertices) for index in indices):
                    raise ValueError(f"Invalid OBJ vertex index at line {number}")
                faces.append(
                    [
                        index - 1 if index > 0 else len(vertices) + index
                        for index in indices
                    ]
                )
        return (
            np.asarray(vertices, dtype=float),
            np.asarray(faces, dtype=np.int64),
            "obj_v_line_order",
        )
    mesh = trimesh.load(path, process=False, maintain_order=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            "Expected one mesh; export multi-object scenes as a triangle OBJ or PLY"
        )
    return (
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        "unprocessed_trimesh_vertex_order",
    )


def patch_table(
    vertices: np.ndarray, faces: np.ndarray, patch_ids: np.ndarray
) -> list[dict]:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area = np.linalg.norm(cross, axis=1) / 2
    result = []
    for patch in range(int(patch_ids.max()) + 1):
        selected = patch_ids == patch
        points = triangles[selected].reshape(-1, 3)
        normal = cross[selected].sum(axis=0)
        normal /= max(np.linalg.norm(normal), 1e-12)
        result.append(
            {
                "patch_id": patch,
                "face_count": int(selected.sum()),
                "center": np.average(
                    triangles[selected].mean(axis=1), axis=0, weights=area[selected]
                )
                .round(5)
                .tolist(),
                "bounds": [
                    points.min(axis=0).round(5).tolist(),
                    points.max(axis=0).round(5).tolist(),
                ],
                "mean_normal": normal.round(5).tolist(),
                "area": float(area[selected].sum()),
            }
        )
    return result


def score_colors(scores: np.ndarray) -> np.ndarray:
    # Blue -> cyan -> yellow -> red, with a fixed [0, 1] scale.
    anchors = np.array([[35, 65, 160], [30, 195, 215], [245, 220, 65], [215, 45, 35]])
    positions = np.linspace(0, 1, len(anchors))
    return np.column_stack(
        [np.interp(scores, positions, anchors[:, channel]) for channel in range(3)]
    ).astype(np.uint8)
