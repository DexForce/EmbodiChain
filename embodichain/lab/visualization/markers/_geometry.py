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

from .cfg import MarkerPrototypeCfg


def _revolve(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(24) * (2 * np.pi / 24)
    radius, height = profile.T
    vertices = np.stack(
        (
            radius[:, None] * np.cos(angles),
            radius[:, None] * np.sin(angles),
            np.broadcast_to(height[:, None], (len(height), 24)),
        ),
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    for row in range(len(profile) - 1):
        for col in range(24):
            a, b = row * 24 + col, row * 24 + (col + 1) % 24
            faces.extend(((a, b, a + 24), (b, b + 24, a + 24)))
    faces = np.asarray(faces, dtype=np.uint32)
    triangles = vertices[faces]
    valid = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        > 1e-10
    )
    return vertices.astype(np.float32), faces[valid]


def geometry_parts(
    cfg: MarkerPrototypeCfg,
) -> tuple[tuple[np.ndarray, np.ndarray, tuple[float, float, float] | None], ...]:
    """Generate local triangles once per prototype; RGB override denotes frame axes."""
    shape = cfg.shape
    if shape == "mesh":
        return ((cfg.vertices, cfg.faces, None),)
    if shape == "box":
        vertices = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float32,
            )
            * 0.5
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.uint32,
        )
    else:
        if shape in ("sphere", "capsule"):
            theta = np.linspace(-np.pi / 2, np.pi / 2, 17)
            profile = np.stack((0.5 * np.cos(theta), 0.5 * np.sin(theta)), axis=1)
            if shape == "capsule":
                # Duplicate the equator to join the two hemispheres with a cylinder.
                profile = np.concatenate(
                    (profile[:9] - [0, 0.5], profile[8:] + [0, 0.5])
                )
        elif shape == "cylinder":
            profile = np.array([[0, -0.5], [0.5, -0.5], [0.5, 0.5], [0, 0.5]])
        elif shape == "cone":
            profile = np.array([[0, -0.5], [0.5, -0.5], [0, 0.5]])
        else:  # arrow or frame
            profile = np.array(
                [[0, 0], [0.025, 0], [0.025, 0.75], [0.075, 0.75], [0, 1]]
            )
        vertices, faces = _revolve(profile)
        if shape in ("arrow", "frame"):
            vertices = vertices[:, [2, 0, 1]]
    if shape == "frame":
        return tuple(
            (vertices[:, order], faces, color)
            for order, color in (
                ([0, 1, 2], (1.0, 0.0, 0.0)),
                ([2, 0, 1], (0.0, 1.0, 0.0)),
                ([1, 2, 0], (0.0, 0.0, 1.0)),
            )
        )
    return ((vertices, faces, None),)
