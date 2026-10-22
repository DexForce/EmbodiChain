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

from pathlib import Path

import numpy as np
import trimesh

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
    SceneGraphRelation,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer import (
    ParentSurfaceLayoutOptimizer,
    ParentSurfaceLayoutProblem,
)
import embodichain.gen_sim.scene_engine.pipeline.utils.parent_surface_layout_optimizer as parent_surface_layout_optimizer_module
import embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_constructor as scene_layout_constructor_module
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_constructor import (
    SceneLayoutConstructor,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_surface_layout_optimizer import (
    TableSurfaceLayoutOptimizer,
    TableSurfaceLayoutProblem,
    _table_region_bounds,
)

_TABLE_BOUNDS = [
    [-2.0, -2.0],
    [2.0, -2.0],
    [2.0, 2.0],
    [-2.0, 2.0],
]
_OVERLAPPING_CENTER_XY = [0.0, 0.0]
_ASSET_SIDE_LENGTH_M = 0.2
_RELATION_CLEARANCE_M = 0.03
_COLLISION_MARGIN_M = 0.02
_BOARD_XY_SIZE_M = 0.6
_CAN_XY_SIZE_M = 0.1
_PENCIL_XY_SIZE_M = [0.04, 0.2]
_SETTLED_DYNAMIC_POS_Y_UP = [0.25, 0.5, -0.4]
_SETTLED_DYNAMIC_ROT_Y_UP = [0.0, 0.0, 0.0]


def _asset(
    *,
    object_id: str,
    glb_path: Path,
    center_xy: list[float] | None = None,
    pos: list[float] | None = None,
) -> SceneObject:
    return SceneObject(
        id=object_id,
        kind="asset",
        category=object_id,
        name=object_id,
        description=object_id,
        simready_glb_path=str(glb_path),
        rot=[0.0, 0.0, 0.0],
        pos=pos or [0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
        center_xy=center_xy,
    )


def test_table_regions_put_front_at_larger_y() -> None:
    table_bounds = np.asarray([[0.0, 0.0], [3.0, 3.0]])

    assert np.allclose(
        _table_region_bounds(
            table_bounds=table_bounds,
            table_region="back_center",
        ),
        [[1.0, 0.0], [2.0, 1.0]],
    )
    assert np.allclose(
        _table_region_bounds(
            table_bounds=table_bounds,
            table_region="front_center",
        ),
        [[1.0, 2.0], [2.0, 3.0]],
    )


def test_layout_constructor_places_new_child_on_parent_top(
    tmp_path: Path,
    monkeypatch,
) -> None:
    book_glb = tmp_path / "book.glb"
    cup_glb = tmp_path / "cup.glb"
    # SimReady GLBs are y-up, so the book's short vertical axis is y.
    trimesh.creation.box(extents=[1.0, 0.2, 1.0]).export(book_glb)
    trimesh.creation.box(extents=[0.2, 0.2, 0.2]).export(cup_glb)

    table = SceneObject(
        id="table",
        kind="table",
        category="table",
        name="table",
        description="table",
        support_surface_z=0.0,
        support_optimization_rect_xy=[
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
        ],
    )
    book = _asset(
        object_id="book_001",
        glb_path=book_glb,
        center_xy=[0.0, 0.0],
        # This y-up position maps to a z-up center at z=0.52 m.
        pos=[0.0, 0.52, 0.0],
    )
    cup = _asset(object_id="cup_001", glb_path=cup_glb)
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="book_001",
                parent_id="table",
                parent_relation="on",
            ),
            SceneGraphNode(
                object_id="cup_001",
                parent_id="book_001",
                parent_relation="on",
            ),
        ]
    )
    settler_inputs: dict[str, object] = {}

    class _FakeGravitySettler:
        def __init__(self, **kwargs: object) -> None:
            settler_inputs.update(kwargs)

        def settle(self) -> dict[str, dict[str, list[float]]]:
            return {
                "cup_001": {
                    "pos": [0.0, 0.74, 0.0],
                    "rot": [0.0, 0.0, 0.0],
                }
            }

    monkeypatch.setattr(
        parent_surface_layout_optimizer_module,
        "GravitySettler",
        _FakeGravitySettler,
    )

    post_edit_scene = SceneLayoutConstructor(
        formal_scene=Scene(objects=[table, book]),
        goal_scene_graph=graph,
        layout_variable_ids={"cup_001"},
        generated_scene_objects=[cup],
        output_root=tmp_path,
    ).construct()

    placed_cup = next(
        asset for asset in post_edit_scene.assets if asset.id == "cup_001"
    )
    assert placed_cup.center_xy == [0.0, 0.0]
    # Book top is z=0.62 m; cup half-height plus support clearance is 0.12 m.
    assert np.allclose(placed_cup.pos, [0.0, 0.74, 0.0])
    assert settler_inputs["dynamic_asset_ids"] == {"cup_001"}
    assert settler_inputs["static_asset_ids"] == {"book_001"}


def test_layout_constructor_settles_dynamic_table_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    asset_glb = tmp_path / "asset.glb"
    trimesh.creation.box(extents=[_ASSET_SIDE_LENGTH_M] * 3).export(asset_glb)
    table = SceneObject(
        id="table",
        kind="table",
        category="table",
        name="table",
        description="table",
        rot=[0.0, 0.0, 0.0],
        pos=[0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
        support_surface_z=0.0,
        support_optimization_rect_xy=_TABLE_BOUNDS,
    )
    fixed_asset = _asset(
        object_id="fixed_001",
        glb_path=asset_glb,
        center_xy=[-0.5, 0.0],
    )
    dynamic_asset = _asset(object_id="dynamic_001", glb_path=asset_glb)
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="fixed_001", parent_id="table", parent_relation="on"
            ),
            SceneGraphNode(
                object_id="dynamic_001", parent_id="table", parent_relation="on"
            ),
        ]
    )
    settler_inputs: dict[str, object] = {}

    class _FakeGravitySettler:
        def __init__(self, **kwargs: object) -> None:
            settler_inputs.update(kwargs)

        def settle(self) -> dict[str, dict[str, list[float]]]:
            return {
                "dynamic_001": {
                    "pos": _SETTLED_DYNAMIC_POS_Y_UP,
                    "rot": _SETTLED_DYNAMIC_ROT_Y_UP,
                }
            }

    monkeypatch.setattr(
        scene_layout_constructor_module,
        "GravitySettler",
        _FakeGravitySettler,
    )

    post_edit_scene = SceneLayoutConstructor(
        formal_scene=Scene(objects=[table, fixed_asset]),
        goal_scene_graph=graph,
        layout_variable_ids={"dynamic_001"},
        generated_scene_objects=[dynamic_asset],
        output_root=tmp_path,
    ).construct()

    settled_dynamic_asset = next(
        asset for asset in post_edit_scene.assets if asset.id == "dynamic_001"
    )
    assert settler_inputs["dynamic_asset_ids"] == {"dynamic_001"}
    assert settler_inputs["static_asset_ids"] == {"fixed_001"}
    assert settled_dynamic_asset.pos == _SETTLED_DYNAMIC_POS_Y_UP
    assert settled_dynamic_asset.rot == _SETTLED_DYNAMIC_ROT_Y_UP
    # y-up [x, y, z] maps to z-up tabletop XY [x, -z].
    assert settled_dynamic_asset.center_xy == [0.25, 0.4]


def test_table_optimizer_ignores_fixed_sibling_overlap(tmp_path: Path) -> None:
    asset_glb = tmp_path / "asset.glb"
    trimesh.creation.box(extents=[_ASSET_SIDE_LENGTH_M] * 3).export(asset_glb)
    first_id, second_id = "first_001", "second_001"
    optimizer = TableSurfaceLayoutOptimizer()

    solved_xy_by_id = optimizer.optimize(
        TableSurfaceLayoutProblem(
            assets_by_id={
                first_id: _asset(
                    object_id=first_id,
                    glb_path=asset_glb,
                    center_xy=_OVERLAPPING_CENTER_XY,
                ),
                second_id: _asset(
                    object_id=second_id,
                    glb_path=asset_glb,
                    center_xy=_OVERLAPPING_CENTER_XY,
                ),
            },
            root_ids=[first_id, second_id],
            root_seed_xy_by_id={
                first_id: _OVERLAPPING_CENTER_XY,
                second_id: _OVERLAPPING_CENTER_XY,
            },
            imported_root_ids={first_id, second_id},
            fixed_root_xy_by_id={
                first_id: _OVERLAPPING_CENTER_XY,
                second_id: _OVERLAPPING_CENTER_XY,
            },
            root_table_regions_by_id={first_id: None, second_id: None},
            table_optimization_rect_xy=_TABLE_BOUNDS,
            root_relations=[],
        )
    )

    assert solved_xy_by_id == {
        first_id: _OVERLAPPING_CENTER_XY,
        second_id: _OVERLAPPING_CENTER_XY,
    }


def test_table_optimizer_keeps_fixed_sibling_partly_outside_table(
    tmp_path: Path,
) -> None:
    asset_glb = tmp_path / "asset.glb"
    trimesh.creation.box(extents=[_ASSET_SIDE_LENGTH_M] * 3).export(asset_glb)
    fixed_id, variable_id = "fixed_001", "variable_001"
    fixed_xy = [1.95, 0.0]  # Its 0.2 m AABB extends beyond the x=2 m table edge.

    solved_xy_by_id = TableSurfaceLayoutOptimizer().optimize(
        TableSurfaceLayoutProblem(
            assets_by_id={
                fixed_id: _asset(
                    object_id=fixed_id,
                    glb_path=asset_glb,
                    center_xy=fixed_xy,
                ),
                variable_id: _asset(object_id=variable_id, glb_path=asset_glb),
            },
            root_ids=[fixed_id, variable_id],
            root_seed_xy_by_id={fixed_id: fixed_xy, variable_id: [0.0, 0.0]},
            imported_root_ids={fixed_id},
            fixed_root_xy_by_id={fixed_id: fixed_xy, variable_id: None},
            root_table_regions_by_id={fixed_id: None, variable_id: None},
            table_optimization_rect_xy=_TABLE_BOUNDS,
            root_relations=[],
        )
    )

    assert solved_xy_by_id[fixed_id] == fixed_xy
    assert np.all(np.abs(solved_xy_by_id[variable_id]) <= 1.9 + 1e-6)


def test_table_optimizer_separates_variable_sibling_from_fixed_sibling(
    tmp_path: Path,
) -> None:
    asset_glb = tmp_path / "asset.glb"
    trimesh.creation.box(extents=[_ASSET_SIDE_LENGTH_M] * 3).export(asset_glb)
    fixed_id, variable_id = "fixed_001", "variable_001"
    optimizer = TableSurfaceLayoutOptimizer()

    solved_xy_by_id = optimizer.optimize(
        TableSurfaceLayoutProblem(
            assets_by_id={
                fixed_id: _asset(
                    object_id=fixed_id,
                    glb_path=asset_glb,
                    center_xy=_OVERLAPPING_CENTER_XY,
                ),
                variable_id: _asset(
                    object_id=variable_id,
                    glb_path=asset_glb,
                ),
            },
            root_ids=[fixed_id, variable_id],
            root_seed_xy_by_id={
                fixed_id: _OVERLAPPING_CENTER_XY,
                variable_id: _OVERLAPPING_CENTER_XY,
            },
            imported_root_ids={fixed_id},
            fixed_root_xy_by_id={fixed_id: _OVERLAPPING_CENTER_XY, variable_id: None},
            root_table_regions_by_id={fixed_id: None, variable_id: None},
            table_optimization_rect_xy=_TABLE_BOUNDS,
            root_relations=[],
        )
    )

    assert solved_xy_by_id[fixed_id] == _OVERLAPPING_CENTER_XY
    assert np.max(np.abs(solved_xy_by_id[variable_id])) >= _ASSET_SIDE_LENGTH_M - 1e-6


def test_table_optimizer_can_reverse_a_collision_order(tmp_path: Path) -> None:
    board_glb = tmp_path / "board.glb"
    can_glb = tmp_path / "can.glb"
    pencil_glb = tmp_path / "pencil.glb"
    # SimReady GLBs are y-up, so z-up XY uses the source XZ extents.
    trimesh.creation.box(
        extents=[_BOARD_XY_SIZE_M, _ASSET_SIDE_LENGTH_M, _BOARD_XY_SIZE_M]
    ).export(board_glb)
    trimesh.creation.box(
        extents=[_CAN_XY_SIZE_M, _ASSET_SIDE_LENGTH_M, _CAN_XY_SIZE_M]
    ).export(can_glb)
    trimesh.creation.box(
        extents=[
            _PENCIL_XY_SIZE_M[0],
            _ASSET_SIDE_LENGTH_M,
            _PENCIL_XY_SIZE_M[1],
        ]
    ).export(pencil_glb)
    board_id, pencil_id, can_id = "board_001", "pencil_001", "can_001"
    board_xy, pencil_xy, can_xy = [0.0, 0.0], [-0.2, 0.0], [-0.4, 0.0]

    solved_xy_by_id = TableSurfaceLayoutOptimizer().optimize(
        TableSurfaceLayoutProblem(
            assets_by_id={
                board_id: _asset(
                    object_id=board_id,
                    glb_path=board_glb,
                    center_xy=board_xy,
                ),
                pencil_id: _asset(object_id=pencil_id, glb_path=pencil_glb),
                can_id: _asset(
                    object_id=can_id,
                    glb_path=can_glb,
                    center_xy=can_xy,
                ),
            },
            root_ids=[board_id, pencil_id, can_id],
            root_seed_xy_by_id={
                board_id: board_xy,
                pencil_id: pencil_xy,
                can_id: can_xy,
            },
            imported_root_ids={board_id, can_id},
            fixed_root_xy_by_id={
                board_id: board_xy,
                pencil_id: None,
                can_id: can_xy,
            },
            root_table_regions_by_id={
                board_id: None,
                pencil_id: None,
                can_id: None,
            },
            table_optimization_rect_xy=_TABLE_BOUNDS,
            root_relations=[],
        )
    )

    expected_pencil_x_upper_bound = (
        can_xy[0]
        - _CAN_XY_SIZE_M / 2.0
        - _PENCIL_XY_SIZE_M[0] / 2.0
        - _COLLISION_MARGIN_M
    )
    assert solved_xy_by_id[pencil_id][0] <= expected_pencil_x_upper_bound + 1e-6


def test_parent_optimizer_applies_sibling_planar_relation(tmp_path: Path) -> None:
    asset_glb = tmp_path / "asset.glb"
    trimesh.creation.box(extents=[_ASSET_SIDE_LENGTH_M] * 3).export(asset_glb)
    left_id, right_id = "left_001", "right_001"

    solved_xy_by_id = ParentSurfaceLayoutOptimizer().optimize(
        ParentSurfaceLayoutProblem(
            assets_by_id={
                left_id: _asset(object_id=left_id, glb_path=asset_glb),
                right_id: _asset(object_id=right_id, glb_path=asset_glb),
            },
            child_ids=[left_id, right_id],
            child_seed_xy_by_id={
                left_id: _OVERLAPPING_CENTER_XY,
                right_id: _OVERLAPPING_CENTER_XY,
            },
            imported_child_ids=set(),
            fixed_child_xy_by_id={left_id: None, right_id: None},
            parent_aabb_xy=_TABLE_BOUNDS,
            parent_top_z=0.0,
            child_relations=[
                SceneGraphRelation(
                    source_id=left_id,
                    relation="left_of",
                    target_id=right_id,
                )
            ],
        )
    )

    assert (
        solved_xy_by_id[right_id][0] - solved_xy_by_id[left_id][0]
        >= _ASSET_SIDE_LENGTH_M + _RELATION_CLEARANCE_M - 1e-6
    )
