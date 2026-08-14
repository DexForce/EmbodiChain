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
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_constructor import (
    SceneLayoutConstructor,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_layout_optimizer import (
    _table_region_bounds,
)


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
    # book top is z=0.62 m; cup half-height is 0.1 m and clearance is 0.02 m.
    assert np.allclose(placed_cup.pos, [0.0, 0.74, 0.0])
