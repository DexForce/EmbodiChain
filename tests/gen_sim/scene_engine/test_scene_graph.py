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
    GeneratedSceneGraph,
    GeneratedSceneNode,
    GeneratedSceneRelation,
)


@pytest.mark.parametrize("object_id", ["", " cup", "cup "])
def test_scene_graph_rejects_unstable_object_ids(object_id: str) -> None:
    with pytest.raises(ValueError, match="trimmed string"):
        GeneratedSceneNode(object_id=object_id, parent_id="table")


def test_scene_graph_accepts_layered_on_relations() -> None:
    graph = GeneratedSceneGraph(
        nodes=[
            GeneratedSceneNode(object_id="table", parent_id=None),
            GeneratedSceneNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
                table_region="center",
                orientation_state="standing",
            ),
            GeneratedSceneNode(
                object_id="cup",
                parent_id="table",
                parent_relation="on",
                table_region="right_center",
            ),
            GeneratedSceneNode(
                object_id="spoon",
                parent_id="plate",
                parent_relation="on",
            ),
        ],
        relations=[
            GeneratedSceneRelation(
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
        GeneratedSceneGraph(
            nodes=[
                GeneratedSceneNode(object_id="table", parent_id=None),
                GeneratedSceneNode(
                    object_id="plate",
                    parent_id="table",
                    parent_relation="on",
                ),
                GeneratedSceneNode(
                    object_id="spoon",
                    parent_id="plate",
                    parent_relation="on",
                ),
            ],
            relations=[
                GeneratedSceneRelation(
                    source_id="plate",
                    relation="right_of",
                    target_id="spoon",
                ),
            ],
        )


def test_scene_graph_rejects_conflicting_planar_relations() -> None:
    with pytest.raises(ValueError, match="Conflicting planar relations"):
        GeneratedSceneGraph(
            nodes=[
                GeneratedSceneNode(object_id="table", parent_id=None),
                GeneratedSceneNode(
                    object_id="plate",
                    parent_id="table",
                    parent_relation="on",
                ),
                GeneratedSceneNode(
                    object_id="cup",
                    parent_id="table",
                    parent_relation="on",
                ),
            ],
            relations=[
                GeneratedSceneRelation(
                    source_id="plate",
                    relation="left_of",
                    target_id="cup",
                ),
                GeneratedSceneRelation(
                    source_id="cup",
                    relation="left_of",
                    target_id="plate",
                ),
            ],
        )


def test_scene_graph_requires_explicit_parent_relation() -> None:
    with pytest.raises(ValueError, match="parent relation"):
        GeneratedSceneGraph(
            nodes=[
                GeneratedSceneNode(object_id="table", parent_id=None),
                GeneratedSceneNode(
                    object_id="plate",
                    parent_id="table",
                ),
            ],
        )


def test_scene_graph_can_skip_validation_during_refresh() -> None:
    graph = GeneratedSceneGraph(
        nodes=[
            GeneratedSceneNode(object_id="table", parent_id=None),
            GeneratedSceneNode(
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
        GeneratedSceneNode(
            object_id="orange",
            parent_id="box",
            parent_relation="inside",
        )


def test_scene_graph_derives_layers_from_parent_links() -> None:
    graph = GeneratedSceneGraph(
        nodes=[
            GeneratedSceneNode(object_id="table", parent_id=None),
            GeneratedSceneNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
            ),
            GeneratedSceneNode(
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
    graph = GeneratedSceneGraph(nodes=[], validate_on_refresh=False)

    with pytest.raises(ValueError, match="table node"):
        graph.layer_by_id()


def test_scene_graph_rejects_table_region_for_non_table_parent() -> None:
    with pytest.raises(ValueError, match="only valid for objects on the table"):
        GeneratedSceneGraph(
            nodes=[
                GeneratedSceneNode(object_id="table", parent_id=None),
                GeneratedSceneNode(
                    object_id="plate",
                    parent_id="table",
                    parent_relation="on",
                ),
                GeneratedSceneNode(
                    object_id="spoon",
                    parent_id="plate",
                    parent_relation="on",
                    table_region="center",
                ),
            ],
        )


def test_scene_graph_derives_support_and_inverse_planar_constraints() -> None:
    graph = GeneratedSceneGraph(
        nodes=[
            GeneratedSceneNode(object_id="table", parent_id=None),
            GeneratedSceneNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
            ),
            GeneratedSceneNode(
                object_id="cup",
                parent_id="table",
                parent_relation="on",
            ),
        ],
        relations=[
            GeneratedSceneRelation(
                source_id="plate",
                relation="left_of",
                target_id="cup",
            ),
            GeneratedSceneRelation(
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
    graph = GeneratedSceneGraph(
        nodes=[
            GeneratedSceneNode(object_id="table", parent_id=None),
            GeneratedSceneNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
            ),
            GeneratedSceneNode(
                object_id="cup",
                parent_id="table",
                parent_relation="on",
            ),
        ],
        relations=[
            GeneratedSceneRelation(
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
    graph = GeneratedSceneGraph(
        nodes=[
            GeneratedSceneNode(object_id="table", parent_id=None),
            GeneratedSceneNode(
                object_id="plate",
                parent_id="table",
                parent_relation="on",
                table_region="center",
                orientation_state="standing",
            ),
        ],
    )

    graph_dict = graph.to_dict()

    assert graph_dict == {
        "schema_version": "generated_scene_graph/v1",
        "artifact_kind": "scene_authoring",
        "nodes": [
            {
                "object_id": "table",
                "parent_id": None,
                "parent_relation": None,
                "table_region": None,
                "orientation_state": None,
            },
            {
                "object_id": "plate",
                "parent_id": "table",
                "parent_relation": "on",
                "table_region": "center",
                "orientation_state": "standing",
            },
        ],
        "relations": [],
    }
