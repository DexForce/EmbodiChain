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

import json

import pytest

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_edit_plan import (
    SceneEditOperation,
    SceneEditPlan,
)
from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
)
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_understanding import (
    _parse_scene_edit_operations,
)


def _scene_and_graph() -> tuple[Scene, SceneGraph]:
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="wooden table",
                description="A wooden table.",
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="blue book",
                description="A blue book.",
            ),
            SceneObject(
                id="orange_001",
                kind="asset",
                category="orange",
                name="orange",
                description="An orange.",
            ),
        ]
    )
    scene_graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="book_001",
                parent_id="table",
                parent_relation="on",
            ),
            SceneGraphNode(
                object_id="orange_001",
                parent_id="book_001",
                parent_relation="on",
            ),
        ]
    )
    return scene, scene_graph


def test_scene_edit_plan_accepts_add_without_a_position() -> None:
    scene, scene_graph = _scene_and_graph()

    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(
                op="add",
                object_id="cup_001",
                category="cup",
                name="green cup",
                description="A small green ceramic cup.",
            )
        ],
    )

    assert len(plan.operations) == 1
    assert plan.to_dict()["operations"] == [
        {
            "op": "add",
            "object_id": "cup_001",
            "target_id": None,
            "relation": None,
            "category": "cup",
            "name": "green cup",
            "description": "A small green ceramic cup.",
        }
    ]


def test_scene_edit_plan_accepts_multiple_new_objects_with_the_same_category() -> None:
    scene, scene_graph = _scene_and_graph()

    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(
                op="add",
                object_id="orange_002",
                category="orange",
                name="small orange",
                description="A small round orange with a textured peel.",
            ),
            SceneEditOperation(
                op="add",
                object_id="orange_003",
                category="orange",
                name="large orange",
                description="A large round orange with a textured peel.",
            ),
        ],
    )

    assert [operation.category for operation in plan.operations] == ["orange", "orange"]


def test_scene_edit_parser_assigns_ids_to_same_category_adds_in_order() -> None:
    scene, _ = _scene_and_graph()
    draft = {
        "operations": [
            {
                "op": "add",
                "object_id": None,
                "target_id": None,
                "relation": None,
                "category": "orange",
                "name": "small_orange",
                "description": "A small round orange with a textured peel.",
            },
            {
                "op": "add",
                "object_id": None,
                "target_id": None,
                "relation": None,
                "category": "orange",
                "name": "small_orange",
                "description": "A small round orange with a textured peel.",
            },
        ]
    }

    operations = _parse_scene_edit_operations(
        json.loads(json.dumps(draft)), scene=scene
    )

    assert [operation.object_id for operation in operations] == [
        "orange_002",
        "orange_003",
    ]


def test_scene_edit_plan_rejects_targets_outside_the_input_scene() -> None:
    scene, scene_graph = _scene_and_graph()

    with pytest.raises(ValueError, match="existing scene objects"):
        SceneEditPlan(
            scene=scene,
            scene_graph=scene_graph,
            operations=[
                SceneEditOperation(
                    op="add",
                    object_id="spoon_001",
                    target_id="new_orange_001",
                    relation="left_of",
                    category="spoon",
                    name="metal spoon",
                    description="A metal spoon.",
                )
            ],
        )


def test_scene_edit_plan_requires_deleting_all_children_of_a_deleted_parent() -> None:
    scene, scene_graph = _scene_and_graph()

    with pytest.raises(ValueError, match="all of its children"):
        SceneEditPlan(
            scene=scene,
            scene_graph=scene_graph,
            operations=[SceneEditOperation(op="delete", object_id="book_001")],
        )
