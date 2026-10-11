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
from shapely.geometry import Polygon, box

from embodichain.lab.sim.scene.geometry import SupportRegion
from embodichain.lab.sim.scene.placement import (
    align_group_height,
    clamp_group,
    separate_aabbs,
)


def test_scene_group_proposals_share_region_without_mutation() -> None:
    region = SupportRegion("tray", 0.5, box(0, 0, 2, 2))
    footprint = np.array([[-0.1, 0.1], [0.1, 0.1], [0.1, 0.3], [-0.1, 0.3]])
    before = footprint.copy()
    delta = clamp_group(region, {"part": footprint})
    assert delta is not None
    assert region.support_polygon.covers(Polygon(footprint + delta))
    height = align_group_height(
        region, {"part": np.array([[0, 0, 0.1], [0.2, 0.2, 0.3]])}, clearance_m=0.0
    )
    assert height == pytest.approx(0.4)
    np.testing.assert_array_equal(footprint, before)


def test_scene_separation_retains_uid_mapping_and_support() -> None:
    region = SupportRegion("workbench", 0.8, box(0, 0, 2, 2))
    footprint = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]])
    offsets = separate_aabbs(region, {"left": footprint, "right": footprint + 0.1})
    first = Polygon(footprint + offsets["left"])
    second = Polygon(footprint + 0.1 + offsets["right"])
    assert not first.intersects(second)
    assert region.support_polygon.covers(first) and region.support_polygon.covers(
        second
    )
