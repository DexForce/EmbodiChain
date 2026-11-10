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
from shapely.geometry import box
import trimesh

from embodichain.lab.sim.scene.geometry import SupportRegion, detect_support_region


def test_support_detection_uses_uid_and_already_scaled_arena_geometry() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.1))
    mesh.apply_scale([2.0, 3.0, 0.5])
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.deg2rad(31), [0, 0, 1])
    )
    mesh.apply_translation([0.4, -0.3, 0.8])
    region = detect_support_region("shelf", mesh.vertices, mesh.faces)
    assert region.support_uid == "shelf" and region.frame == "arena"
    assert region.top_z == pytest.approx(0.825)
    assert region.support_polygon.area == pytest.approx(3.0)


def test_support_does_not_allow_unlabelled_or_world_geometry() -> None:
    with pytest.raises(ValueError, match="UID"):
        SupportRegion("", 1.0, box(0, 0, 1, 1))
    with pytest.raises(ValueError, match="arena-local"):
        SupportRegion("shelf", 1.0, box(0, 0, 1, 1), frame="world")


def test_support_rectangle_must_be_inside_region() -> None:
    with pytest.raises(ValueError, match="contained"):
        SupportRegion("shelf", 1.0, box(0, 0, 1, 1), box(0, 0, 2, 2))
