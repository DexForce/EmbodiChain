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
"""Circular-prism proof preserves its closed ring and extra-collider boundary."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from embodichain.compute.geometry.cylinder import circular_prism_radius


def cylinder_inputs(count=32):
    angles = np.arange(count) * (2 * np.pi / count)
    ring = np.column_stack((0.03 * np.cos(angles), 0.03 * np.sin(angles)))
    vertices = np.concatenate(
        [np.column_stack((ring, np.full(count, z))) for z in (-0.01, 0.01)]
    )
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    # ConvexHull equations give outward normals, independent of simplex winding.
    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    flip = np.sum(normals * hull.equations[:, :3], axis=1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return vertices, faces


def test_closed_uniform_rings_preserve_radius_under_frame_changes():
    vertices, faces = cylinder_inputs()
    for rotation in (
        np.eye(3),
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
    ):
        origin = np.array([0.2, -0.1, 0.4])
        transformed = vertices @ rotation.T + origin
        axis = rotation[:, 2]
        radius = circular_prism_radius(
            transformed, faces, axis, origin, extra_colliders=(origin[None, :],)
        )
        assert radius == pytest.approx(0.03, abs=1e-15)


@pytest.mark.parametrize(
    "condition", ["open_surface", "short_ring", "extra_reaches_shell", "cap_center"]
)
def test_unqualified_shell_is_rejected(condition):
    vertices, faces = cylinder_inputs(8 if condition == "short_ring" else 32)
    extra = ()
    if condition == "open_surface":
        faces = faces[1:]
    elif condition == "extra_reaches_shell":
        extra = (np.array([[0.03, 0.0, 0.0]]),)
    elif condition == "cap_center":
        # The preserved proof requires every vertex on the two rings, including
        # otherwise valid capped mesh formats that add a center vertex.
        vertices = np.concatenate((vertices, [[0.0, 0.0, 0.01]]))
    with pytest.raises(ValueError):
        circular_prism_radius(
            vertices,
            faces,
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            extra_colliders=extra,
        )
