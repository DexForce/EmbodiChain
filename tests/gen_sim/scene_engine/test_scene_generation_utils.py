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

from embodichain.gen_sim.scene_engine.pipeline.utils import scene_generation_utils


def _aabb_corners(
    minimum: tuple[float, float], maximum: tuple[float, float]
) -> np.ndarray:
    return np.asarray(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ],
        dtype=float,
    )


def test_layout_transform_round_trip_preserves_pose_and_scale() -> None:
    layout = {
        "id": "cup",
        "rot": [20.0, -35.0, 40.0],
        "pos": [1.0, 2.0, 3.0],
        "scale": [1.0, 2.0, 3.0],
    }

    recovered = scene_generation_utils.transform_matrix_to_layout_object(
        "cup",
        scene_generation_utils.layout_object_to_transform_matrix(layout),
    )

    np.testing.assert_allclose(recovered["pos"], layout["pos"], atol=1e-8)
    np.testing.assert_allclose(recovered["scale"], layout["scale"], atol=1e-8)
    np.testing.assert_allclose(
        scene_generation_utils.layout_object_to_transform_matrix(recovered),
        scene_generation_utils.layout_object_to_transform_matrix(layout),
        atol=1e-8,
    )


def test_aabb_optimizer_resolves_overlap_inside_boundary() -> None:
    corners_by_id = {
        "first": _aabb_corners((-0.75, -0.5), (0.25, 0.5)),
        "second": _aabb_corners((-0.25, -0.5), (0.75, 0.5)),
    }

    offsets = scene_generation_utils._optimize_assets_2d_aabbs_in_rectangle(
        rectangle_min=np.asarray([-1.0, -1.0]),
        rectangle_max=np.asarray([1.0, 1.0]),
        aabb_corners_by_id=corners_by_id,
        boundary_margin=0.0,
        aabb_clearance=0.0,
    )
    first_min, first_max = scene_generation_utils._aabb_2d_bounds_from_corners(
        corners_by_id["first"] + offsets["first"],
        name="first",
        require_nonzero_extent=True,
    )
    second_min, second_max = scene_generation_utils._aabb_2d_bounds_from_corners(
        corners_by_id["second"] + offsets["second"],
        name="second",
        require_nonzero_extent=True,
    )

    assert first_min[0] >= -1.0
    assert first_max[0] <= 1.0
    assert second_min[0] >= -1.0
    assert second_max[0] <= 1.0
    assert first_max[0] <= second_min[0] or second_max[0] <= first_min[0]


def test_aabb_optimizer_rejects_asset_larger_than_boundary() -> None:
    with pytest.raises(ValueError, match="larger than the table"):
        scene_generation_utils._optimize_assets_2d_aabbs_in_rectangle(
            rectangle_min=np.asarray([-1.0, -1.0]),
            rectangle_max=np.asarray([1.0, 1.0]),
            aabb_corners_by_id={
                "oversized": _aabb_corners((-2.0, -0.5), (2.0, 0.5)),
            },
            boundary_margin=0.0,
            aabb_clearance=0.0,
        )
