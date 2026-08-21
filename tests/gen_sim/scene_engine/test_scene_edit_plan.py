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
from pathlib import Path

import pytest
from PIL import Image
import trimesh

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
    _apply_scene_edit_plan_to_scene_graph,
    _build_updated_scene_graph,
    _parse_scene_edit_operations,
)
from embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation import (
    prepare_scene_edit_assets,
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
            "table_region": None,
            "category": "cup",
            "name": "green cup",
            "description": "A small green ceramic cup.",
            "orientation_state": None,
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
                "table_region": None,
                "category": "orange",
                "name": "small_orange",
                "description": "A small round orange with a textured peel.",
                "orientation_state": None,
            },
            {
                "op": "add",
                "object_id": None,
                "target_id": None,
                "relation": None,
                "table_region": None,
                "category": "orange",
                "name": "small_orange",
                "description": "A small round orange with a textured peel.",
                "orientation_state": None,
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


def test_scene_edit_plan_requires_a_position_for_move_operations() -> None:
    scene, scene_graph = _scene_and_graph()

    with pytest.raises(ValueError, match="must specify target_id and relation"):
        SceneEditPlan(
            scene=scene,
            scene_graph=scene_graph,
            operations=[SceneEditOperation(op="move", object_id="book_001")],
        )


def test_scene_edit_asset_preparation_skips_plans_without_adds(
    tmp_path: Path,
) -> None:
    scene, scene_graph = _scene_and_graph()
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(
                op="move",
                object_id="book_001",
                target_id="table",
                relation="on",
            )
        ],
    )
    previous_asset_output = tmp_path / "scene_editing" / "asset_preparation"
    previous_asset_output.mkdir(parents=True)
    (previous_asset_output / "previous.txt").write_text("keep", encoding="utf-8")

    prepared_assets = prepare_scene_edit_assets(
        scene_edit_plan=plan,
        output_root=tmp_path,
        image_generation_client=object(),  # type: ignore[arg-type]
        geometry_generation_client=object(),  # type: ignore[arg-type]
        image_segmentation_client=object(),  # type: ignore[arg-type]
    )

    assert prepared_assets == []
    assert (previous_asset_output / "previous.txt").is_file()


def test_scene_edit_asset_preparation_generates_one_image_per_add(
    tmp_path: Path,
) -> None:
    class ImageGenerationClient:
        def __init__(self) -> None:
            self.requests: list[tuple[str, Path]] = []

        def generate_image_by_prompt(self, *, prompt: str, output_path: Path) -> Path:
            self.requests.append((prompt, output_path))
            Image.new("RGB", (6, 6), "white").save(output_path)
            return output_path

    class ImageSegmentationClient:
        def __init__(self) -> None:
            self.requests: list[tuple[Path, str]] = []

        def segment_single_object(
            self,
            *,
            image_path: Path,
            prompt: str,
        ) -> list[dict[str, object]]:
            self.requests.append((image_path, prompt))
            return [
                {
                    "size": [6, 6],
                    "counts": [7, 4, 2, 4, 2, 4, 2, 4, 7],
                    "starts_with": 1,
                }
            ]

    class GeometryGenerationClient:
        def __init__(self) -> None:
            self.requests: list[tuple[Path, list[tuple[str, Path]], Path]] = []

        def generate_objects(
            self,
            *,
            image_path: Path,
            object_masks: list[tuple[str, Path]],
            output_root: Path,
        ) -> tuple[dict[str, object], list[dict[str, object]]]:
            self.requests.append((image_path, object_masks, output_root))
            output_root.mkdir(parents=True, exist_ok=True)
            for object_id, _ in object_masks:
                trimesh.creation.box().export(output_root / f"{object_id}.glb")
            return {}, [{"scale": [1.25, 1.5, 1.75]}]

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
    image_generation_client = ImageGenerationClient()
    image_segmentation_client = ImageSegmentationClient()
    geometry_generation_client = GeometryGenerationClient()

    prepared_assets = prepare_scene_edit_assets(
        scene_edit_plan=plan,
        output_root=tmp_path,
        image_generation_client=image_generation_client,  # type: ignore[arg-type]
        geometry_generation_client=geometry_generation_client,  # type: ignore[arg-type]
        image_segmentation_client=image_segmentation_client,  # type: ignore[arg-type]
    )

    expected_image_path = (
        tmp_path
        / "scene_editing"
        / "asset_preparation"
        / "generated_images"
        / "cup_001.png"
    )
    assert [asset.id for asset in prepared_assets] == ["cup_001"]
    assert prepared_assets[0].simready_glb_path is not None
    assert prepared_assets[0].rot == [0.0, 0.0, 0.0]
    assert prepared_assets[0].pos == [0.0, 0.0, 0.0]
    assert prepared_assets[0].scale == [1.0, 1.0, 1.0]
    assert image_generation_client.requests == [
        ("A small green ceramic cup.", expected_image_path)
    ]
    assert expected_image_path.is_file()
    assert image_segmentation_client.requests == [
        (expected_image_path, "A small green ceramic cup.")
    ]
    generated_mask_path = (
        tmp_path
        / "scene_editing"
        / "asset_preparation"
        / "generated_masks"
        / "cup_001_mask.png"
    )
    assert generated_mask_path.is_file()
    with Image.open(generated_mask_path) as mask:
        assert mask.getpixel((3, 3)) == 255
        assert mask.getpixel((0, 0)) == 0
    generated_glb_path = (
        tmp_path
        / "scene_editing"
        / "asset_preparation"
        / "coarse_geometry"
        / "cup_001.glb"
    )
    assert geometry_generation_client.requests == [
        (
            expected_image_path,
            [("cup_001", generated_mask_path)],
            generated_glb_path.parent,
        )
    ]
    assert generated_glb_path.read_bytes().startswith(b"glTF")


def test_scene_edit_graph_builder_copies_the_pre_edit_graph() -> None:
    scene, scene_graph = _scene_and_graph()
    plan = SceneEditPlan(scene=scene, scene_graph=scene_graph)

    updated_scene_graph = _build_updated_scene_graph(
        scene_graph=scene_graph,
        scene_edit_plan=plan,
    )

    assert updated_scene_graph is not scene_graph
    assert updated_scene_graph.nodes is not scene_graph.nodes
    assert updated_scene_graph.relations is not scene_graph.relations
    assert updated_scene_graph.to_dict() == scene_graph.to_dict()


def test_scene_edit_graph_builder_removes_deleted_nodes() -> None:
    scene, scene_graph = _scene_and_graph()
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(op="delete", object_id="orange_001"),
            SceneEditOperation(op="delete", object_id="book_001"),
        ],
    )

    updated_scene_graph = _build_updated_scene_graph(
        scene_graph=scene_graph,
        scene_edit_plan=plan,
    )

    assert set(updated_scene_graph.node_by_id()) == {"table"}
    assert set(scene_graph.node_by_id()) == {"table", "book_001", "orange_001"}


def test_scene_edit_graph_builder_adds_unpositioned_objects_on_the_table() -> None:
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

    updated_scene_graph = _build_updated_scene_graph(
        scene_graph=scene_graph,
        scene_edit_plan=plan,
    )

    added_node = updated_scene_graph.node_by_id()["cup_001"]
    assert added_node.parent_id == "table"
    assert added_node.parent_relation == "on"


def test_scene_edit_graph_builder_updates_move_on_parent() -> None:
    scene, scene_graph = _scene_and_graph()
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(
                op="move",
                object_id="orange_001",
                target_id="table",
                relation="on",
            )
        ],
    )

    updated_scene_graph = _build_updated_scene_graph(
        scene_graph=scene_graph,
        scene_edit_plan=plan,
    )

    assert updated_scene_graph.node_by_id()["orange_001"].parent_id == "table"


def test_scene_edit_graph_builder_adds_planar_relation_with_target_parent() -> None:
    scene, scene_graph = _scene_and_graph()
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(
                op="add",
                object_id="cup_001",
                target_id="book_001",
                relation="right_of",
                category="cup",
                name="green cup",
                description="A small green ceramic cup.",
            )
        ],
    )

    updated_scene_graph = _build_updated_scene_graph(
        scene_graph=scene_graph,
        scene_edit_plan=plan,
    )

    assert updated_scene_graph.node_by_id()["cup_001"].parent_id == "table"
    assert any(
        relation.source_id == "cup_001"
        and relation.relation == "right_of"
        and relation.target_id == "book_001"
        for relation in updated_scene_graph.relations
    )


def test_scene_edit_plan_application_adds_new_nodes_before_relationship_updates() -> (
    None
):
    scene, scene_graph = _scene_and_graph()
    plan = SceneEditPlan(
        scene=scene,
        scene_graph=scene_graph,
        operations=[
            SceneEditOperation(
                op="add",
                object_id="cup_001",
                target_id="book_001",
                relation="right_of",
                category="cup",
                name="green cup",
                description="A small green ceramic cup.",
            )
        ],
    )

    _apply_scene_edit_plan_to_scene_graph(
        scene_graph=scene_graph,
        scene_edit_plan=plan,
    )

    assert scene_graph.node_by_id()["cup_001"].parent_id == "table"
