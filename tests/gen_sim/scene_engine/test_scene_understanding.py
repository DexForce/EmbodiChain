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
from embodichain.gen_sim.scene_engine.pipeline.generation import scene_understanding
from embodichain.gen_sim.scene_engine.pipeline.utils import image_segmentation_utils
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    render_asset_mask_id_overlay,
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


def test_initial_scene_graph_places_every_asset_on_table(tmp_path: Path) -> None:
    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps(
                {
                    "orientation_states": [
                        {"object_id": "cup_001", "orientation_state": None},
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
                "orientation_state": None,
            },
            {
                "object_id": "cup_001",
                "parent_id": "table",
                "parent_relation": "on",
                "table_region": None,
                "orientation_state": None,
            },
        ],
        "relations": [],
    }


def test_scene_graph_initialization_uses_image_orientation_states(
    tmp_path: Path,
) -> None:
    class VLM:
        def __init__(self) -> None:
            self.user_prompt: str | None = None

        def complete(self, **_: object) -> str:
            self.user_prompt = _["user_prompt"]  # type: ignore[assignment,index]
            return json.dumps(
                {
                    "orientation_states": [
                        {
                            "object_id": "bottle_001",
                            "orientation_state": "standing",
                        },
                        {"object_id": "book_001", "orientation_state": "lying"},
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
        "asset_ids": ["bottle_001", "book_001"],
    }
    assert scene_graph.node_by_id()["bottle_001"].orientation_state == "standing"
    assert scene_graph.node_by_id()["book_001"].orientation_state == "lying"


def test_scene_graph_initialization_retries_a_response_containing_table(
    tmp_path: Path,
) -> None:
    class VLM:
        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "orientation_states": [
                            {"object_id": "table", "orientation_state": None},
                            {
                                "object_id": "bottle_001",
                                "orientation_state": "standing",
                            },
                        ]
                    }
                ),
                json.dumps(
                    {
                        "orientation_states": [
                            {
                                "object_id": "bottle_001",
                                "orientation_state": "standing",
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

    assert scene_graph.node_by_id()["bottle_001"].orientation_state == "standing"


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


def test_scene_graph_initialization_info_lists_asset_ids() -> None:
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
        "asset_ids": ["bottle_001", "book_001"],
    }
