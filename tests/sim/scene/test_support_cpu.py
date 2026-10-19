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
import trimesh

from embodichain.lab.sim.scene._support_cpu import SupportSurfaceDetector


def test_detector_keeps_input_geometry_unchanged() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 0.2))
    mesh.apply_translation([0.5, -0.2, 0.7])
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    support = SupportSurfaceDetector(vertices, faces).detect()
    assert support.top_z == pytest.approx(0.8)
    assert support.support_polygon.area == pytest.approx(2.0)
    np.testing.assert_array_equal(vertices, mesh.vertices)
    np.testing.assert_array_equal(faces, mesh.faces)


@pytest.mark.parametrize(
    "faces",
    [np.array([[0, 1, 3]]), np.array([[0, 1, -1]]), np.array([[0.0, 1.0, 2.0]])],
)
def test_invalid_triangle_indices_rejected(faces: np.ndarray) -> None:
    with pytest.raises(ValueError, match="faces"):
        SupportSurfaceDetector(np.eye(3), faces)


def test_degenerate_horizontal_geometry_rejected() -> None:
    with pytest.raises(ValueError):
        SupportSurfaceDetector(np.zeros((3, 3)), np.array([[0, 1, 2]])).detect()


def test_mutated_detection_threshold_rejected_before_mesh_search() -> None:
    from embodichain.lab.sim.scene._support_cpu import SupportSurfaceConfig

    cfg = SupportSurfaceConfig()
    mesh = trimesh.creation.box()
    detector = SupportSurfaceDetector(mesh.vertices, mesh.faces, config=cfg)
    cfg.normal_z_min = float("nan")
    with pytest.raises(ValueError, match="finite"):
        detector.detect()
