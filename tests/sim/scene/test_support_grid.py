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
from shapely.geometry import box, mapping, shape
import torch
import trimesh

from embodichain.lab.sim.scene._support_grid import _prepare_support
from embodichain.compute.geometry._hierarchical import _coverage


def _surfaces(rectangles):
    vertices, faces = [], []
    for x0, y0, x1, y1, z in rectangles:
        start = len(vertices)
        vertices.extend([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
        faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
    return np.array(vertices, dtype=float), np.array(faces)


def test_holes_disconnected_regions_and_full_cell_certificate():
    # Ring with a central hole, plus a separate island.
    vertices, faces = _surfaces(
        [
            [-0.2, -0.2, 0.2, -0.05, 0],
            [-0.2, 0.05, 0.2, 0.2, 0],
            [-0.2, -0.05, -0.05, 0.05, 0],
            [0.05, -0.05, 0.2, 0.05, 0],
            [0.25, -0.1, 0.35, 0.1, 0],
        ]
    )
    child = np.array([[-0.15, -0.15, 0.001], [-0.1, -0.1, 0.02]])
    support = _prepare_support(vertices, faces, [child], resolution=0.01, margin=0.002)
    polygon = shape(support["polygon"])
    assert len(polygon.geoms) == 2
    integral = torch.tensor(support["unsupported_integral"])
    origin = torch.tensor(support["grid_origin"], dtype=torch.float64)
    bounds = torch.tensor(
        [
            [[-0.15, -0.15], [-0.1, -0.1]],
            [[-0.15, -0.15], [0.15, 0.15]],
            [[0.27, -0.05], [0.32, 0.05]],
            [[0.19, -0.01], [0.26, 0.01]],
        ],
        dtype=torch.float64,
    )
    assert _coverage(bounds, integral, origin, 0.01).tolist() == [
        True,
        False,
        True,
        False,
    ]
    for bound, valid in zip(bounds.numpy(), [True, False, True, False]):
        assert (
            polygon.buffer(-0.002, join_style=2).covers(box(*bound[0], *bound[1]))
            == valid
        )


def test_child_contact_selects_floor_instead_of_tray_rim():
    v, f = _surfaces(
        [
            [-0.2, -0.2, 0.2, 0.2, 0],
            [-0.22, -0.22, 0.22, -0.2, 0.03],
            [-0.22, 0.2, 0.22, 0.22, 0.03],
        ]
    )
    child = np.array([[-0.03, -0.03, 0.001], [0.03, 0.03, 0.06]])
    support = _prepare_support(v, f, [child], resolution=0.01, margin=0.002)
    assert abs(np.array(support["parent_from_support"])[2, 3]) < 1e-12


def test_ambiguous_layers_and_unsupported_override_rejected():
    v, f = _surfaces([[-0.2, -0.2, 0.2, 0.2, 0], [-0.2, -0.2, 0.2, 0.2, 0.01]])
    child = np.array([[-0.03, -0.03, 0.005], [0.03, 0.03, 0.06]])
    with pytest.raises(ValueError, match="Ambiguous"):
        _prepare_support(v, f, [child], resolution=0.01, margin=0.002)
    override = {
        "parent_from_support": np.eye(4).tolist(),
        "polygon": mapping(box(-0.1, -0.1, 0.1, 0.1)),
    }
    support = _prepare_support(
        v, f, [child], resolution=0.01, margin=0.002, override=override
    )
    assert shape(support["polygon"]).area == pytest.approx(0.04)
    override["polygon"] = mapping(box(-0.4, -0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="evidence"):
        _prepare_support(
            v, f, [child], resolution=0.01, margin=0.002, override=override
        )


def test_boundary_touch_requires_whole_intersecting_cells():
    v, f = _surfaces([[0, 0, 0.1, 0.1, 0]])
    support = _prepare_support(
        v,
        f,
        [np.array([[0.03, 0.03, 0.001], [0.04, 0.04, 0.02]])],
        resolution=0.01,
        margin=0,
    )
    bounds = torch.tensor(
        [
            [[0, 0], [0.1, 0.1]],
            [[0.001, 0.001], [0.099, 0.099]],
            [[-0.000001, 0], [0.05, 0.05]],
        ],
        dtype=torch.float64,
    )
    assert _coverage(
        bounds,
        torch.tensor(support["unsupported_integral"]),
        torch.tensor(support["grid_origin"]),
        0.01,
    ).tolist() == [True, True, False]


def test_tilted_parent_local_plane_uses_explicit_support_frame():
    from scipy.spatial.transform import Rotation

    vertices, faces = _surfaces([[-0.2, -0.2, 0.2, 0.2, 0]])
    rotation = Rotation.from_euler("xyz", [8, 10, 15], degrees=True).as_matrix()
    translation = np.array([0.02, -0.03, 0.04])
    vertices = vertices @ rotation.T + translation
    child = (
        np.array([[-0.03, -0.03, 0.001], [0.03, 0.03, 0.06]]) @ rotation.T + translation
    )
    support = _prepare_support(vertices, faces, [child], resolution=0.005, margin=0.002)
    frame = np.array(support["parent_from_support"])
    np.testing.assert_allclose(frame[:3, 2], rotation[:, 2], atol=1e-12)
    assert abs(np.dot(frame[:3, 3] - translation, rotation[:, 2])) < 1e-12


def test_shallow_raised_patch_has_a_conservative_height_ceiling():
    vertices, faces = _surfaces(
        [[-0.3, -0.2, 0.3, 0.2, 0.02], [-0.18, -0.07, -0.02, 0.07, 0.0218]]
    )
    child = np.array([[-0.12, -0.02, 0.023], [-0.06, 0.02, 0.08]])
    support = _prepare_support(vertices, faces, [child], resolution=0.002, margin=0.002)
    frame = np.asarray(support["parent_from_support"])
    ceiling = support["height_ceiling_m"]
    # The nominal fitted plane is below the raised patch. Its cached upper
    # deviation must prevent the default 1 mm gap from being consumed by it.
    assert frame[2, 3] + 0.001 < 0.0218
    assert frame[2, 3] + ceiling == pytest.approx(0.0218)
    assert frame[2, 3] + ceiling + 0.001 == pytest.approx(0.0228)
