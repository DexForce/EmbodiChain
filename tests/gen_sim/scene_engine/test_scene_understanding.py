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

from PIL import Image, ImageDraw
import pytest

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
from embodichain.gen_sim.scene_engine.task_spec_input import TaskEntity
from embodichain.gen_sim.scene_engine.pipeline.generation import scene_understanding
from embodichain.gen_sim.scene_engine.pipeline.utils import image_segmentation_utils
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    render_asset_mask_id_overlay,
    save_visible_rgba_crop,
)


def _response(*, asset_name: str = "cup") -> str:
    return json.dumps(
        {
            "table": {
                "category": "dining_table",
                "name": "wooden table",
                "description": "A rectangular wooden table.",
            },
            "assets": [
                {
                    "category": "cup",
                    "name": asset_name,
                    "description": "A small ceramic cup.",
                }
            ],
        }
    )


def test_image_object_analysis_parses_code_fence_and_assigns_stable_ids() -> None:
    scene = scene_understanding._parse_image_object_analysis_response(
        f"```json\n{_response()}\n```"
    )

    assert scene.table is not None
    assert scene.table.id == "table"
    assert [asset.id for asset in scene.assets] == ["cup_001"]
    assert scene.assets[0].is_articulated is False


@pytest.mark.parametrize(
    ("category", "required_alias"),
    [("bell", "hotel bell"), ("water_dispenser", "dispenser")],
)
def test_segmentation_aliases_cover_observed_service_vocabulary(
    category: str, required_alias: str
) -> None:
    assert required_alias in scene_understanding._SEGMENTATION_PROMPT_ALIASES[category]


@pytest.mark.parametrize("category", ["bell", "drawer", "box"])
def test_image_analysis_defaults_to_rigid_without_task_spec(
    tmp_path: Path, category: str
) -> None:
    payload = json.loads(_response())
    payload["assets"][0].update(
        category=category,
        name="gray object with a button",
        description="A gray object with a visible push button.",
    )

    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps(payload)

    image = tmp_path / "scene.png"
    Image.new("RGB", (8, 8)).save(image)
    scene = Scene()
    scene_understanding._analyze_image_objects(
        scene=scene,
        image_path=image,
        vlm_client=VLM(),  # type: ignore[arg-type]
    )
    assert scene.assets[0].is_articulated is False


def test_image_object_analysis_accepts_name_with_spatial_words() -> None:
    scene = scene_understanding._parse_image_object_analysis_response(
        _response(asset_name="left cup")
    )

    assert scene.assets[0].name == "left cup"


def test_image_object_analysis_accepts_description_with_structural_words() -> None:
    response = json.loads(_response())
    response["assets"][0]["description"] = "A small ceramic cup with a lid on top."

    scene = scene_understanding._parse_image_object_analysis_response(
        json.dumps(response)
    )

    assert scene.assets[0].description == "A small ceramic cup with a lid on top."


def test_image_object_analysis_retries_then_updates_scene(tmp_path: Path) -> None:
    class VLM:
        def __init__(self) -> None:
            self.responses = ["not-json", _response()]

        def complete(self, **_: object) -> str:
            return self.responses.pop(0)

    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    scene = Scene()

    scene_understanding._analyze_image_objects(
        scene=scene,
        image_path=image_path,
        vlm_client=VLM(),  # type: ignore[arg-type]
        json_max_attempts=2,
    )

    assert scene.table is not None
    assert [asset.id for asset in scene.assets] == ["cup_001"]


@pytest.mark.parametrize("with_task_spec", [False, True])
def test_understanding_keeps_visual_metadata_and_matches_ids_before_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_task_spec: bool
) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (16, 16)).save(image)
    payload = json.loads(_response(asset_name="blue cup"))
    payload["assets"].append(
        {"category": "spoon", "name": "silver spoon", "description": "A metal spoon."}
    )
    task_entities = (
        (
            TaskEntity("evidence_001", "blue cup", True),
            TaskEntity("evidence_002", "missing block", False),
        )
        if with_task_spec
        else ()
    )
    expected_cup_id = "evidence_001" if with_task_spec else "cup_001"
    matching_calls = []

    class VLM:
        def complete(self, *, system_prompt: str, user_prompt: str, **_: object) -> str:
            if system_prompt == scene_understanding._SYSTEM_PROMPT:
                return json.dumps(payload)
            if system_prompt == scene_understanding._TASK_ENTITY_MATCH_SYSTEM_PROMPT:
                matching_calls.append(json.loads(user_prompt))
                return json.dumps({"evidence_001": "cup_001", "evidence_002": None})
            return json.dumps(
                {
                    "nodes": [
                        {
                            "object_id": asset["id"],
                            "parent_id": "table",
                            "parent_relation": "on",
                            "pose_description": None,
                        }
                        for asset in json.loads(user_prompt)["assets"]
                    ]
                }
            )

    def segment_scene(*, scene: Scene, **_: object) -> Path:
        assert [obj.id for obj in scene.objects] == [
            "table",
            expected_cup_id,
            "spoon_001",
        ]
        assert scene.assets[0].is_articulated is with_task_spec
        assert scene.assets[0].name == "blue cup"
        assert scene.assets[0].description == "A small ceramic cup."
        assert scene.assets[1].is_articulated is False
        return image

    monkeypatch.setattr(scene_understanding, "_segment_scene", segment_scene)
    scene, graph = scene_understanding.understand_scene(
        scene=Scene(),
        image_path=image,
        output_root=tmp_path / "output",
        vlm_client=VLM(),
        image_segmentation_client=object(),
        task_entities=task_entities,
    )
    assert set(graph.node_by_id()) == {obj.id for obj in scene.objects}
    report = tmp_path / "output" / "scene_understanding" / "task_entity_matches.json"
    if with_task_spec:
        assert json.loads(report.read_text())["entity_to_scene"] == {
            "evidence_001": "cup_001",
            "evidence_002": None,
        }
        assert matching_calls[0]["task_entities"] == [
            {"id": entity.id, "description": entity.description}
            for entity in task_entities
        ]
    else:
        assert matching_calls == []
        assert not report.exists()


@pytest.mark.parametrize(
    "matches,entities,error",
    [
        ({"e1": "missing"}, (TaskEntity("e1", "cup", False),), "Unknown scene asset"),
        (
            {"e1": "cup_001", "e2": "cup_001"},
            (TaskEntity("e1", "cup", False), TaskEntity("e2", "cup", False)),
            "same scene asset",
        ),
        ({}, (TaskEntity("e1", "cup", False),), "every task entity"),
        (
            {"cup_002": "cup_001"},
            (TaskEntity("cup_002", "cup", False),),
            "duplicate scene object",
        ),
    ],
)
def test_task_entity_matching_rejects_invalid_assignments_without_mutation(
    matches: dict[str, str | None], entities: tuple[TaskEntity, ...], error: str
) -> None:
    scene = scene_understanding._parse_image_object_analysis_response(_response())
    scene.objects.append(
        SceneObject(
            id="cup_002",
            kind="asset",
            category="cup",
            name="red cup",
            description="A red cup.",
        )
    )
    original = scene.to_dict()
    with pytest.raises(ValueError, match=error):
        scene_understanding._parse_task_entity_matches(
            json.dumps(matches), scene, entities
        )
    assert scene.to_dict() == original


def test_task_entity_matching_retries_and_allows_atomic_id_swaps(
    tmp_path: Path,
) -> None:
    scene = scene_understanding._parse_image_object_analysis_response(_response())
    scene.objects.append(
        SceneObject(
            id="cup_002",
            kind="asset",
            category="cup",
            name="red cup",
            description="A red cup.",
        )
    )
    entities = (
        TaskEntity("cup_002", "blue cup", False),
        TaskEntity("cup_001", "red cup", True),
    )

    class VLM:
        responses = ["invalid JSON", '{"cup_002":"cup_001","cup_001":"cup_002"}']

        def complete(self, **_: object) -> str:
            return self.responses.pop(0)

    matches = scene_understanding._match_task_entities(
        scene=scene,
        task_entities=entities,
        image_path=tmp_path / "image.png",
        vlm_client=VLM(),
    )
    assert matches == {"cup_002": "cup_001", "cup_001": "cup_002"}
    assert [asset.id for asset in scene.assets] == ["cup_001", "cup_002"]


class _PromptedSegmentation:
    """Return one rectangular mask only for the prompts it recognizes."""

    def __init__(self, image_size: tuple[int, int], known: set[str]) -> None:
        mask = Image.new("L", image_size, 0)
        ImageDraw.Draw(mask).rectangle((8, 8, 24, 20), fill=255)
        self.mask = image_segmentation_utils._encode_binary_mask_rle(mask)
        self.known = known
        self.prompts: list[str] = []

    def segment_single_object(self, *, image_path: Path, prompt: str) -> list:
        self.prompts.append(prompt)
        return [self.mask] if prompt in self.known else []


def test_opaque_task_id_uses_visual_prompts_for_segmentation(tmp_path: Path) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image)
    cup = SceneObject(
        id="evidence_002",
        kind="asset",
        category="cup",
        name="blue cup",
        description="A blue ceramic cup.",
    )
    segmentation = _PromptedSegmentation((48, 32), {"cup"})

    class VLM:
        def complete(self, **_: object) -> str:
            return '{"assignments":[{"asset_id":"evidence_002","mask_index":1}]}'

    scene_understanding._segment_assets(
        image,
        tmp_path,
        tmp_path,
        Scene(objects=[cup]),
        vlm_client=VLM(),
        image_segmentation_client=segmentation,
    )
    assert segmentation.prompts == ["cup", "blue cup"]
    assert cup.mask_path == str(tmp_path / "evidence_002_mask.png")
    assert Image.open(cup.mask_path).getbbox() == (8, 8, 25, 21)


@pytest.mark.parametrize("table_state", ["normal", "degraded"])
def test_table_visibility_states_accept_the_surface_prompt(
    tmp_path: Path, table_state: str
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="white table",
                description="A white tabletop filling the frame.",
            )
        ]
    )
    segmentation = _PromptedSegmentation((48, 32), {"surface"})

    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps({"table_state": table_state, "selected_mask_index": 1})

    scene_understanding._segment_table(
        image_path,
        image_path,
        Image.new("L", (48, 32), 0),
        tmp_path,
        tmp_path,
        scene,
        vlm_client=VLM(),  # type: ignore[arg-type]
        image_segmentation_client=segmentation,  # type: ignore[arg-type]
    )

    assert segmentation.prompts[-1] == "surface"
    assert scene.table is not None and scene.table.mask_path is not None
    assert Image.open(scene.table.mask_path).getbbox() == (8, 8, 25, 21)
    assert json.loads((tmp_path / "table_validation_surface.json").read_text()) == {
        "table_state": table_state,
        "selected_mask_index": 1,
    }


def test_visible_table_without_a_matching_mask_tries_the_next_prompt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    scene = scene_understanding._parse_image_object_analysis_response(_response())
    segmentation = _PromptedSegmentation((48, 32), {"wooden table", "surface"})
    responses = iter(
        [
            {"table_state": "degraded", "selected_mask_index": None},
            {"table_state": "degraded", "selected_mask_index": 1},
        ]
    )

    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps(next(responses))

    scene_understanding._segment_table(
        image_path,
        image_path,
        Image.new("L", (48, 32), 0),
        tmp_path,
        tmp_path,
        scene,
        vlm_client=VLM(),  # type: ignore[arg-type]
        image_segmentation_client=segmentation,  # type: ignore[arg-type]
    )

    assert segmentation.prompts[-1] == "surface"
    assert scene.table is not None and scene.table.mask_path is not None
    assert json.loads((tmp_path / "table_validation_name.json").read_text()) == {
        "table_state": "degraded",
        "selected_mask_index": None,
    }


def test_absent_table_state_has_no_selected_mask() -> None:
    validation = scene_understanding._parse_table_validation_response(
        '{"table_state":null,"selected_mask_index":null}', []
    )
    assert validation.table_state is None
    assert validation.selected_mask_index is None


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"table_state": None, "selected_mask_index": 1}, "cannot select"),
        ({"table_state": "unknown", "selected_mask_index": None}, "table_state"),
        ({"selected_mask_index": 1}, "exactly table_state"),
        ({"table_state": "degraded", "selected_mask_index": True}, "integer"),
    ],
)
def test_table_validation_rejects_inconsistent_or_invalid_decisions(
    payload: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        scene_understanding._parse_table_validation_response(json.dumps(payload), [])


def test_unsegmented_asset_retries_with_vlm_appearance_prompts(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    hammer = SceneObject(
        id="tool_001",
        kind="asset",
        category="tool",
        name="hammer",
        description="A hammer seen from above.",
    )
    scene = Scene(objects=[hammer])
    segmentation = _PromptedSegmentation((48, 32), {"tool with black handle"})

    class VLM:
        def complete(self, *, system_prompt: str, **_: object) -> str:
            if "found no mask" in system_prompt:
                return json.dumps({"prompts": ["tool", "tool with black handle"]})
            return json.dumps(
                {"assignments": [{"asset_id": "tool_001", "mask_index": 1}]}
            )

    scene_understanding._segment_assets(
        image_path,
        tmp_path,
        tmp_path,
        scene,
        vlm_client=VLM(),  # type: ignore[arg-type]
        image_segmentation_client=segmentation,  # type: ignore[arg-type]
    )

    assert segmentation.prompts == ["tool", "hammer", "tool with black handle"]
    assert hammer.mask_path is not None


def test_asset_mask_id_overlay_excludes_the_table_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    table_mask_path = tmp_path / "table_mask.png"
    asset_mask_path = tmp_path / "bottle_mask.png"
    output_path = tmp_path / "asset_masks_with_ids.png"
    image_size = (512, 512)
    Image.new("RGB", image_size, "black").save(image_path)

    table_mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(table_mask).rectangle((10, 10, 100, 100), fill=255)
    table_mask.save(table_mask_path)
    asset_mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(asset_mask).rectangle((380, 180, 450, 360), fill=255)
    asset_mask.save(asset_mask_path)

    rendered_path = render_asset_mask_id_overlay(
        image_path=image_path,
        asset_masks=[("bottle_001", asset_mask_path)],
        output_path=output_path,
    )

    with Image.open(rendered_path) as overlay:
        assert overlay.getpixel((10, 10)) == (0, 0, 0)
        assert overlay.getpixel((377, 180)) != (0, 0, 0)


def test_visible_rgba_crop_preserves_object_pixels_on_a_fixed_canvas(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    mask_path = tmp_path / "cup_mask.png"
    output_path = tmp_path / "object_images" / "cup_001_rgba.png"
    image_size = (100, 60)
    Image.new("RGB", image_size, "black").save(image_path)
    with Image.open(image_path) as image:
        image = image.copy()
    ImageDraw.Draw(image).rectangle((30, 20, 70, 40), fill="red")
    image.save(image_path)

    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).rectangle((30, 20, 70, 40), fill=255)
    mask.save(mask_path)

    rendered_path = save_visible_rgba_crop(
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path,
    )

    with Image.open(rendered_path) as rgba:
        assert rgba.mode == "RGBA"
        assert rgba.size == (512, 512)
        assert rgba.getpixel((256, 256))[:3] == (255, 0, 0)
        assert rgba.getpixel((0, 0))[3] == 0


def test_visible_rgba_observations_leave_unsegmented_objects_empty(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    mask_path = tmp_path / "cup_mask.png"
    image_size = (40, 40)
    Image.new("RGB", image_size, "red").save(image_path)
    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).rectangle((10, 10, 30, 30), fill=255)
    mask.save(mask_path)
    scene = Scene(
        objects=[
            SceneObject(
                id="cup_001",
                kind="asset",
                category="cup",
                name="red cup",
                description="A red cup.",
                mask_path=str(mask_path),
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="blue book",
                description="A blue book.",
            ),
        ]
    )

    scene_understanding._save_visible_rgba_observations(
        image_path=image_path,
        output_root=tmp_path / "object_images",
        scene=scene,
    )

    assert scene.assets[0].visible_rgba_path is not None
    assert Path(scene.assets[0].visible_rgba_path).is_file()
    assert scene.assets[1].visible_rgba_path is None


def test_asset_mask_id_label_font_fits_the_mask_bbox() -> None:
    image_size = (512, 512)
    mask_bbox = (380, 180, 450, 360)
    label = "bottle_001"
    font = image_segmentation_utils._load_asset_id_label_font(
        image_size=image_size,
        mask_bbox=mask_bbox,
        label=label,
    )
    label_bounds = image_segmentation_utils._number_label_bounds(
        draw=ImageDraw.Draw(Image.new("RGBA", image_size)),
        label=label,
        center=(0.0, 0.0),
        font=font,
        minimum_padding=2,
    )

    assert label_bounds[2] - label_bounds[0] <= round(
        (mask_bbox[2] - mask_bbox[0]) * 0.9
    )


def test_initial_scene_graph_places_an_asset_on_the_table(tmp_path: Path) -> None:
    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps(
                {
                    "nodes": [
                        {
                            "object_id": "cup_001",
                            "parent_id": "table",
                            "parent_relation": "on",
                            "pose_description": None,
                        },
                    ]
                }
            )

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
                id="cup_001",
                kind="asset",
                category="cup",
                name="blue cup",
                description="A blue cup.",
            ),
        ],
    )

    overlay_path = tmp_path / "asset_masks_with_ids.png"
    overlay_path.write_bytes(b"png")
    scene_graph = scene_understanding._initialize_scene_graph_from_segmented_scene(
        scene,
        asset_mask_id_overlay_path=overlay_path,
        vlm_client=VLM(),  # type: ignore[arg-type]
    )

    assert scene_graph.to_dict() == {
        "nodes": [
            {
                "object_id": "table",
                "parent_id": None,
                "parent_relation": None,
                "table_region": None,
                "pose_description": None,
            },
            {
                "object_id": "cup_001",
                "parent_id": "table",
                "parent_relation": "on",
                "table_region": None,
                "pose_description": None,
            },
        ],
        "relations": [],
    }


def test_scene_graph_initialization_uses_image_support_and_orientation(
    tmp_path: Path,
) -> None:
    class VLM:
        def __init__(self) -> None:
            self.user_prompt: str | None = None

        def complete(self, **_: object) -> str:
            self.user_prompt = _["user_prompt"]  # type: ignore[assignment,index]
            return json.dumps(
                {
                    "nodes": [
                        {
                            "object_id": "bottle_001",
                            "parent_id": "table",
                            "parent_relation": "on",
                            "pose_description": "Stand upright on its base.",
                        },
                        {
                            "object_id": "book_001",
                            "parent_id": "table",
                            "parent_relation": "on",
                            "pose_description": "Lie flat on the support surface.",
                        },
                    ]
                }
            )

    overlay_path = tmp_path / "asset_masks_with_ids.png"
    overlay_path.write_bytes(b"png")
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
                id="bottle_001",
                kind="asset",
                category="bottle",
                name="blue bottle",
                description="A blue bottle.",
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="red book",
                description="A red book.",
            ),
        ]
    )

    vlm = VLM()
    scene_graph = scene_understanding._initialize_scene_graph_from_segmented_scene(
        scene,
        asset_mask_id_overlay_path=overlay_path,
        vlm_client=vlm,  # type: ignore[arg-type]
    )

    assert json.loads(vlm.user_prompt or "{}") == {
        "assets": [
            {
                "id": "bottle_001",
                "category": "bottle",
                "name": "blue bottle",
                "description": "A blue bottle.",
            },
            {
                "id": "book_001",
                "category": "book",
                "name": "red book",
                "description": "A red book.",
            },
        ],
    }
    assert (
        scene_graph.node_by_id()["bottle_001"].pose_description
        == "Stand upright on its base."
    )
    assert (
        scene_graph.node_by_id()["book_001"].pose_description
        == "Lie flat on the support surface."
    )


def test_scene_graph_initialization_retries_a_response_containing_table(
    tmp_path: Path,
) -> None:
    class VLM:
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "nodes": [
                            {
                                "object_id": "table",
                                "parent_id": "table",
                                "parent_relation": "on",
                                "pose_description": None,
                            },
                            {
                                "object_id": "bottle_001",
                                "parent_id": "table",
                                "parent_relation": "on",
                                "pose_description": "Stand upright on its base.",
                            },
                        ]
                    }
                ),
                json.dumps(
                    {
                        "nodes": [
                            {
                                "object_id": "bottle_001",
                                "parent_id": "table",
                                "parent_relation": "on",
                                "pose_description": "Stand upright on its base.",
                            },
                        ]
                    }
                ),
            ]

        def complete(self, **_: object) -> str:
            return self.responses.pop(0)

    overlay_path = tmp_path / "asset_masks_with_ids.png"
    overlay_path.write_bytes(b"png")
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
                id="bottle_001",
                kind="asset",
                category="bottle",
                name="blue bottle",
                description="A blue bottle.",
            ),
        ]
    )

    scene_graph = scene_understanding._initialize_scene_graph_from_segmented_scene(
        scene,
        asset_mask_id_overlay_path=overlay_path,
        vlm_client=VLM(),  # type: ignore[arg-type]
        json_max_attempts=2,
    )

    assert (
        scene_graph.node_by_id()["bottle_001"].pose_description
        == "Stand upright on its base."
    )


def test_scene_graph_initialization_retries_a_response_with_a_parent_cycle(
    tmp_path: Path,
) -> None:
    class VLM:
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "nodes": [
                            {
                                "object_id": "book_001",
                                "parent_id": "pen_001",
                                "parent_relation": "on",
                                "pose_description": None,
                            },
                            {
                                "object_id": "pen_001",
                                "parent_id": "book_001",
                                "parent_relation": "on",
                                "pose_description": "Lie flat on the support surface.",
                            },
                        ]
                    }
                ),
                json.dumps(
                    {
                        "nodes": [
                            {
                                "object_id": "book_001",
                                "parent_id": "table",
                                "parent_relation": "on",
                                "pose_description": None,
                            },
                            {
                                "object_id": "pen_001",
                                "parent_id": "book_001",
                                "parent_relation": "on",
                                "pose_description": "Lie flat on the support surface.",
                            },
                        ]
                    }
                ),
            ]

        def complete(self, **_: object) -> str:
            return self.responses.pop(0)

    overlay_path = tmp_path / "asset_masks_with_ids.png"
    overlay_path.write_bytes(b"png")
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
                name="red book",
                description="A red book.",
            ),
            SceneObject(
                id="pen_001",
                kind="asset",
                category="pen",
                name="black pen",
                description="A black pen.",
            ),
        ]
    )

    scene_graph = scene_understanding._initialize_scene_graph_from_segmented_scene(
        scene,
        asset_mask_id_overlay_path=overlay_path,
        vlm_client=VLM(),  # type: ignore[arg-type]
        json_max_attempts=2,
    )

    assert scene_graph.node_by_id()["pen_001"].parent_id == "book_001"


def test_scene_graph_initialization_requires_asset_mask_id_overlay(
    tmp_path: Path,
) -> None:
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="wooden table",
                description="A wooden table.",
            )
        ]
    )

    with pytest.raises(FileNotFoundError, match="Image input not found"):
        scene_understanding._initialize_scene_graph_from_segmented_scene(
            scene,
            asset_mask_id_overlay_path=tmp_path / "missing.png",
            vlm_client=object(),  # type: ignore[arg-type]
        )


def test_scene_graph_initialization_info_lists_asset_metadata() -> None:
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
                id="bottle_001",
                kind="asset",
                category="bottle",
                name="blue bottle",
                description="A blue bottle.",
                center_xy=[0.2, -0.1],
            ),
            SceneObject(
                id="book_001",
                kind="asset",
                category="book",
                name="red book",
                description="A red book.",
                center_xy=[-0.1, 0.2],
            ),
        ]
    )

    simplified_scene_info = (
        scene_understanding._simplify_scene_info_for_graph_initialization(scene=scene)
    )

    assert simplified_scene_info == {
        "assets": [
            {
                "id": "bottle_001",
                "category": "bottle",
                "name": "blue bottle",
                "description": "A blue bottle.",
            },
            {
                "id": "book_001",
                "category": "book",
                "name": "red book",
                "description": "A red book.",
            },
        ],
    }


def test_initial_scene_graph_places_a_lying_asset_on_an_asset(tmp_path: Path) -> None:
    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps(
                {
                    "nodes": [
                        {
                            "object_id": "book_001",
                            "parent_id": "table",
                            "parent_relation": "on",
                            "pose_description": None,
                        },
                        {
                            "object_id": "pen_001",
                            "parent_id": "book_001",
                            "parent_relation": "on",
                            "pose_description": "Lie flat on the support surface.",
                        },
                    ]
                }
            )

    overlay_path = tmp_path / "asset_masks_with_ids.png"
    overlay_path.write_bytes(b"png")
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
                name="red book",
                description="A red book.",
            ),
            SceneObject(
                id="pen_001",
                kind="asset",
                category="pen",
                name="black pen",
                description="A black pen.",
            ),
        ]
    )

    scene_graph = scene_understanding._initialize_scene_graph_from_segmented_scene(
        scene,
        asset_mask_id_overlay_path=overlay_path,
        vlm_client=VLM(),  # type: ignore[arg-type]
    )

    pen = scene_graph.node_by_id()["pen_001"]
    assert pen.parent_id == "book_001"
    assert pen.parent_relation == "on"
    assert pen.pose_description == "Lie flat on the support surface."
