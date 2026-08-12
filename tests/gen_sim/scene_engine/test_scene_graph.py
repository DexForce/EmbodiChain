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

import pytest

from embodichain.gen_sim.scene_engine.core.scene_graph import (
    SceneGraph,
    SceneGraphNode,
    SceneGraphRelation,
)


def test_scene_graph_accepts_layered_on_relations() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
                table_region="center",
            ),
            SceneGraphNode(
                object_id="cup",
                parent_id="table",
                parent_relation="on",
                table_region="right_center",
            ),
            SceneGraphNode(
                object_id="spoon",
                parent_id="plate",
                parent_relation="on",
            ),
        ],
        relations=[
            SceneGraphRelation(
                source_id="plate",
                relation="left_of",
                target_id="cup",
            ),
        ],
    )

    graph.validate()
    assert graph.layer_by_id()["spoon"] == 2


def test_scene_graph_rejects_planar_relations_without_common_parent() -> None:
    with pytest.raises(ValueError, match="share one parent"):
        SceneGraph(
            nodes=[
                SceneGraphNode(object_id="table", parent_id=None),
                SceneGraphNode(
                    object_id="plate",
                    parent_id="table",
                    parent_relation="on",
                ),
                SceneGraphNode(
                    object_id="spoon",
                    parent_id="plate",
                    parent_relation="on",
                ),
            ],
            relations=[
                SceneGraphRelation(
                    source_id="plate",
                    relation="right_of",
                    target_id="spoon",
                ),
            ],
        )


def test_scene_graph_rejects_conflicting_planar_relations() -> None:
    with pytest.raises(ValueError, match="Conflicting planar relations"):
        SceneGraph(
            nodes=[
                SceneGraphNode(object_id="table", parent_id=None),
                SceneGraphNode(
                    object_id="plate",
                    parent_id="table",
                    parent_relation="on",
                ),
                SceneGraphNode(
                    object_id="cup",
                    parent_id="table",
                    parent_relation="on",
                ),
            ],
            relations=[
                SceneGraphRelation(
                    source_id="plate",
                    relation="left_of",
                    target_id="cup",
                ),
                SceneGraphRelation(
                    source_id="cup",
                    relation="left_of",
                    target_id="plate",
                ),
            ],
        )


def test_scene_graph_requires_explicit_parent_relation() -> None:
    with pytest.raises(ValueError, match="parent relation"):
        SceneGraph(
            nodes=[
                SceneGraphNode(object_id="table", parent_id=None),
                SceneGraphNode(
                    object_id="plate",
                    parent_id="table",
                ),
            ],
        )


def test_scene_graph_can_skip_validation_during_refresh() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="plate",
                parent_id="table",
            ),
        ],
        validate_on_refresh=False,
    )

    with pytest.raises(ValueError, match="parent relation"):
        graph.validate()


def test_scene_graph_rejects_unsupported_parent_relation() -> None:
    with pytest.raises(ValueError, match="must be on their parent"):
        SceneGraphNode(
            object_id="orange",
            parent_id="box",
            parent_relation="inside",
        )


def test_scene_graph_derives_layers_from_parent_links() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
            ),
            SceneGraphNode(
                object_id="spoon",
                parent_id="plate",
                parent_relation="on",
            ),
        ],
    )

    assert graph.layer_by_id() == {
        "table": 0,
        "plate": 1,
        "spoon": 2,
    }


def test_scene_graph_layer_by_id_requires_table_root() -> None:
    graph = SceneGraph(nodes=[], validate_on_refresh=False)

    with pytest.raises(ValueError, match="table node"):
        graph.layer_by_id()


def test_scene_graph_rejects_table_region_for_non_table_parent() -> None:
    with pytest.raises(ValueError, match="only valid for objects on the table"):
        SceneGraph(
            nodes=[
                SceneGraphNode(object_id="table", parent_id=None),
                SceneGraphNode(
                    object_id="plate",
                    parent_id="table",
                    parent_relation="on",
                ),
                SceneGraphNode(
                    object_id="spoon",
                    parent_id="plate",
                    parent_relation="on",
                    table_region="center",
                ),
            ],
        )


def test_scene_graph_derives_support_and_inverse_planar_constraints() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
            ),
            SceneGraphNode(
                object_id="cup",
                parent_id="table",
                parent_relation="on",
            ),
        ],
        relations=[
            SceneGraphRelation(
                source_id="plate",
                relation="left_of",
                target_id="cup",
            ),
            SceneGraphRelation(
                source_id="plate",
                relation="left_of",
                target_id="cup",
            ),
        ],
    )

    constraints = graph.derive_constraints()

    assert constraints == [
        {"source_id": "plate", "relation": "on", "target_id": "table"},
        {"source_id": "cup", "relation": "on", "target_id": "table"},
        {"source_id": "plate", "relation": "left_of", "target_id": "cup"},
        {"source_id": "cup", "relation": "right_of", "target_id": "plate"},
    ]


def test_scene_graph_materializes_inverse_planar_relations() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
            ),
            SceneGraphNode(
                object_id="cup",
                parent_id="table",
                parent_relation="on",
            ),
        ],
        relations=[
            SceneGraphRelation(
                source_id="plate",
                relation="left_of",
                target_id="cup",
            ),
        ],
    )

    assert [relation.to_dict() for relation in graph.relations] == [
        {"source_id": "plate", "relation": "left_of", "target_id": "cup"},
        {"source_id": "cup", "relation": "right_of", "target_id": "plate"},
    ]


def test_scene_graph_to_dict_serializes_graph_state() -> None:
    graph = SceneGraph(
        nodes=[
            SceneGraphNode(object_id="table", parent_id=None),
            SceneGraphNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
                table_region="center",
            ),
        ],
    )

    graph_dict = graph.to_dict()

    assert graph_dict == {
        "nodes": [
            {
                "object_id": "table",
                "parent_id": None,
                "parent_relation": None,
                "table_region": None,
            },
            {
                "object_id": "plate",
                "parent_id": "table",
                "parent_relation": "on",
                "table_region": "center",
            },
        ],
        "relations": [],
    }
