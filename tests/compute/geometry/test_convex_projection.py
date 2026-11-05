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

"""Conservative convex projections, including interior and degenerate cases."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.compute.geometry import convex_projection_distance_to_origin


@pytest.mark.parametrize(
    "points,distance",
    [
        ([[3.0, 4.0]], 5.0),
        ([[2.0, -1.0], [2.0, 1.0]], 2.0),
        ([[2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [2.0, 1.0]], 2.0),
        ([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], 0.0),
        ([[2.0, -1.0], [4.0, -1.0], [4.0, 1.0], [2.0, 1.0]], 2.0),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 0.0),
    ],
)
def test_distance_bounds_the_complete_projection(points, distance):
    value = np.asarray(points, dtype=np.float64)
    before = value.copy()
    result = convex_projection_distance_to_origin(value)
    assert max(0.0, distance - 1e-10) <= result <= distance
    assert np.array_equal(value, before)
    assert convex_projection_distance_to_origin(value[::-1]) == result


def test_near_collinear_residual_is_subtracted_conservatively():
    points = np.array([[2.0, -1.0], [2.0 - 1e-13, 0.0], [2.0, 1.0]])
    result = convex_projection_distance_to_origin(points)
    assert 2.0 - 1e-10 <= result <= 2.0 - 1e-13


@pytest.mark.parametrize(
    "points", [[], [[1.0, 2.0, 3.0]], [[np.nan, 0.0]], [[1j, 0.0]]]
)
def test_unsupported_projections_fail_without_clearance(points):
    with pytest.raises(ValueError):
        convex_projection_distance_to_origin(points)
