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
from shapely.geometry import Point, Polygon
import trimesh

from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_layout_optimizer import (
    AssetsSupportLayoutOptimizer,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.assets_group_support_clamp import (
    AssetsGroupSupportClamp,
    AssetsGroupSupportClampConfig,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface import (
    TableSupportSurfaceDetector,
)


def _aabb(minimum_x: float, minimum_y: float, maximum_x: float, maximum_y: float) -> np.ndarray:
    return np.array(
        [
            [minimum_x, minimum_y],
            [maximum_x, minimum_y],
            [maximum_x, maximum_y],
            [minimum_x, maximum_y],
        ],
        dtype=float,
    )


def _layout(object_id: str, x: float, y: float) -> dict[str, object]:
    return {"id": object_id, "pos": [x, 0.0, -y]}


def _top_mesh(vertices_xy: list[tuple[float, float]], faces: list[list[int]], z: float) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array([[x, y, z] for x, y in vertices_xy], dtype=float),
        faces=np.array(faces, dtype=int),
        process=False,
    )


def test_support_detector_preserves_an_l_shaped_support_contour() -> None:
    mesh = _top_mesh(
        [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)],
        [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]],
        z=1.0,
    )

    region = TableSupportSurfaceDetector(table_world_mesh=mesh).detect()

    assert region.top_z == pytest.approx(1.0)
    assert region.support_polygon.area == pytest.approx(3.0)
    assert region.support_polygon.covers(Point(0.5, 1.5))
    assert not region.support_polygon.covers(Point(1.5, 1.5))


def test_support_detector_prefers_main_tabletop_over_small_higher_piece() -> None:
    main = _top_mesh(
        [(0, 0), (2, 0), (2, 2), (0, 2)], [[0, 1, 2], [0, 2, 3]], z=1.0
    )
    decoration = _top_mesh(
        [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
        [[0, 1, 2], [0, 2, 3]],
        z=1.2,
    )
    mesh = trimesh.util.concatenate([main, decoration])

    region = TableSupportSurfaceDetector(table_world_mesh=mesh).detect()

    assert region.top_z == pytest.approx(1.0)
    assert region.support_polygon.area == pytest.approx(4.0)


def test_group_clamp_preserves_relative_layout_while_moving_inside_support() -> None:
    support = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    aabbs = {
        "first": _aabb(-0.5, 1.0, 0.5, 2.0),
        "second": _aabb(1.0, 1.0, 2.0, 2.0),
    }
    layouts = [_layout("first", 0.0, 1.5), _layout("second", 1.5, 1.5)]

    refined = AssetsGroupSupportClamp(
        support_region=support,
        assets_aabb_2d_z_up_world_corners_by_id=aabbs,
        assets_layout=layouts,
        config=AssetsGroupSupportClampConfig(grid_resolution_m=0.05),
    ).clamp()

    first, second = refined
    assert first["pos"][0] > layouts[0]["pos"][0]  # type: ignore[index]
    assert second["pos"][0] - first["pos"][0] == pytest.approx(1.5)  # type: ignore[index]
    assert second["pos"][2] - first["pos"][2] == pytest.approx(0.0)  # type: ignore[index]


def test_group_clamp_returns_unchanged_layout_when_already_contained() -> None:
    layout = _layout("cup", 1.5, 1.5)
    refined = AssetsGroupSupportClamp(
        support_region=Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
        assets_aabb_2d_z_up_world_corners_by_id={"cup": _aabb(1.0, 1.0, 2.0, 2.0)},
        assets_layout=[layout],
    ).clamp()

    assert refined == [layout]


def test_group_clamp_reports_infeasible_oversized_group() -> None:
    clamp = AssetsGroupSupportClamp(
        support_region=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        assets_aabb_2d_z_up_world_corners_by_id={"large": _aabb(0, 0, 3, 3)},
        assets_layout=[_layout("large", 1.5, 1.5)],
        config=AssetsGroupSupportClampConfig(grid_resolution_m=0.1),
    )

    with pytest.raises(ValueError, match="cannot be placed"):
        clamp.clamp()


def test_layout_optimizer_resolves_a_simple_pair_overlap() -> None:
    optimizer = AssetsSupportLayoutOptimizer(
        support_region=Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        assets_aabb_2d_z_up_world_corners_by_id={
            "first": _aabb(1.0, 1.0, 2.0, 2.0),
            "second": _aabb(1.5, 1.0, 2.5, 2.0),
        },
        assets_layout=[_layout("first", 1.5, 1.5), _layout("second", 2.0, 1.5)],
    )

    refined = optimizer.optimize()

    refined_offsets = np.array(
        [
            [refined[index]["pos"][0] - optimizer.assets_layout[index]["pos"][0],  # type: ignore[index]
             optimizer.assets_layout[index]["pos"][2] - refined[index]["pos"][2]]  # type: ignore[index]
            for index in range(2)
        ]
    )
    base_aabbs = np.stack(
        [
            optimizer.assets_aabb_2d_z_up_world_corners_by_id["first"],
            optimizer.assets_aabb_2d_z_up_world_corners_by_id["second"],
        ]
    )
    assert not optimizer._overlaps(base_aabbs, refined_offsets)


def test_layout_optimizer_rejects_unresolvable_overlap() -> None:
    optimizer = AssetsSupportLayoutOptimizer(
        support_region=Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        assets_aabb_2d_z_up_world_corners_by_id={
            "first": _aabb(0, 0, 2, 2),
            "second": _aabb(0, 0, 2, 2),
        },
        assets_layout=[_layout("first", 1.0, 1.0), _layout("second", 1.0, 1.0)],
    )

    with pytest.raises(ValueError, match="cannot be resolved"):
        optimizer.optimize()
