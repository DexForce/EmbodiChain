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
from embodichain.gen_sim.scene_engine.task_spec_input import TaskEntity, TaskSpecInput
from embodichain.gen_sim.scene_engine.pipeline.generation import scene_understanding
from embodichain.gen_sim.scene_engine.pipeline.utils import image_segmentation_utils
from embodichain.gen_sim.scene_engine.pipeline.utils.image_segmentation_utils import (
    render_asset_mask_id_overlay,
    save_visible_rgba_crop,
)


def _response(*, asset_name: str = "cup", table_state: str = "normal") -> str:
    return json.dumps(
        {
            "table_state": table_state,
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


@pytest.mark.parametrize("table_state", ["normal", "degraded"])
def test_image_object_analysis_parses_code_fence_and_assigns_stable_ids(
    table_state: str,
) -> None:
    scene = scene_understanding._parse_image_object_analysis_response(
        f"```json\n{_response(table_state=table_state)}\n```"
    )

    assert scene.table is not None
    assert scene.table.id == "table"
    assert scene.table.table_state == table_state
    assert [asset.id for asset in scene.assets] == ["cup_001"]
    assert scene.assets[0].is_articulated is False


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
        analysis_output_path=tmp_path / "analysis.json",
    )

    assert json.loads((tmp_path / "analysis.json").read_text()) == json.loads(
        _response()
    )
    assert scene.table is not None
    assert [asset.id for asset in scene.assets] == ["cup_001"]


@pytest.mark.parametrize(
    ("table_state", "with_table"),
    [("unknown", True), ("normal", False), (None, True)],
)
def test_image_object_analysis_rejects_invalid_table_classification(
    table_state: str | None, with_table: bool
) -> None:
    payload = json.loads(_response())
    payload["table_state"] = table_state
    if not with_table:
        payload["table"] = None
    with pytest.raises(ValueError, match="table"):
        scene_understanding._parse_image_object_analysis_response(json.dumps(payload))


def test_image_object_analysis_requires_table_classification() -> None:
    payload = json.loads(_response())
    del payload["table_state"]
    with pytest.raises(ValueError, match="exactly"):
        scene_understanding._parse_image_object_analysis_response(json.dumps(payload))


def test_absent_support_surface_stops_before_segmentation_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    payload = json.loads(_response())
    payload.update(table_state=None, table=None)

    class VLM:
        calls = 0

        def complete(self, **_: object) -> str:
            self.calls += 1
            return json.dumps(payload)

    def unexpected_segmentation(**_: object) -> Path:
        pytest.fail("A scene without a visible support surface must not be segmented.")

    monkeypatch.setattr(scene_understanding, "_segment_scene", unexpected_segmentation)
    vlm = VLM()
    with pytest.raises(
        ValueError, match="No visible table or main scene support plane"
    ):
        scene_understanding.understand_scene(
            scene=Scene(),
            image_path=image_path,
            output_root=tmp_path / "output",
            vlm_client=vlm,
            image_segmentation_client=object(),
        )
    assert vlm.calls == 1


@pytest.mark.parametrize("task_mode", ["none", "empty", "entities"])
def test_understanding_integrates_task_ids_before_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_mode: str
) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (160, 96), "white").save(image)
    payload = json.loads(_response(asset_name="blue cup"))
    payload["assets"].append(dict(payload["assets"][0]))
    task_input = (
        TaskSpecInput(
            instruction="Pick up the blue cup on the left.",
            entities=(
                (
                    TaskEntity(
                        "evidence_001",
                        "blue cup",
                        True,
                        ("the cup on the left",),
                        ("cup_handle",),
                    ),
                    TaskEntity("evidence_002", "missing block", False),
                )
                if task_mode == "entities"
                else ()
            ),
        )
        if task_mode != "none"
        else None
    )
    task_aware = task_mode == "entities"
    expected_ids = ["evidence_001", "cup_001"] if task_aware else ["cup_001", "cup_002"]
    expected_hints = (
        {
            "evidence_001": "The blue cup on the left.",
            "cup_001": "The blue cup on the right.",
        }
        if task_aware
        else {}
    )
    analysis_calls: list[dict[str, object]] = []
    overlay_ids: list[list[str]] = []
    if task_aware:
        payload["assets"][0].update(
            task_entity_id="cup_handle", instance_hint=expected_hints["evidence_001"]
        )
        payload["assets"][1].update(
            task_entity_id=None, instance_hint=expected_hints["cup_001"]
        )

    def render_overlay(
        *,
        image_path: str | Path,
        asset_masks: list[tuple[str, str | Path]],
        output_path: str | Path,
    ) -> Path:
        overlay_ids.append([asset_id for asset_id, _ in asset_masks])
        return render_asset_mask_id_overlay(
            image_path=image_path, asset_masks=asset_masks, output_path=output_path
        )

    class VLM:
        def complete(
            self, *, system_prompt: str, user_prompt: str, image_path: Path
        ) -> str:
            if system_prompt.startswith(scene_understanding._SYSTEM_PROMPT):
                # Identity is resolved with the raw image, before any mask exists.
                assert Path(image_path) == image.resolve()
                assert overlay_ids == []
                if task_aware:
                    analysis_calls.append(json.loads(user_prompt))
                else:
                    assert system_prompt == scene_understanding._SYSTEM_PROMPT
                    assert user_prompt == scene_understanding._USER_PROMPT
                return json.dumps(payload)
            assert overlay_ids == [expected_ids]
            assets = json.loads(user_prompt)["assets"]
            assert [asset["id"] for asset in assets] == expected_ids
            return json.dumps(
                {
                    "nodes": [
                        {
                            "object_id": asset["id"],
                            "parent_id": "table",
                            "parent_relation": "on",
                            "pose_description": None,
                        }
                        for asset in assets
                    ]
                }
            )

    def segment_scene(
        *,
        scene: Scene,
        stage_output_root: Path,
        instance_hints: dict[str, str],
        **_: object,
    ) -> Path:
        assert [obj.id for obj in scene.objects] == ["table", *expected_ids]
        assert instance_hints == expected_hints
        assert [asset.is_articulated for asset in scene.assets] == [task_aware, False]
        assert all(asset.name == "blue cup" for asset in scene.assets)
        assert all(
            asset.description == "A small ceramic cup." for asset in scene.assets
        )
        masks = stage_output_root / "masks"
        masks.mkdir()
        for obj, bbox in zip(
            scene.objects, [(0, 0, 159, 95), (20, 25, 60, 70), (100, 25, 140, 70)]
        ):
            mask = Image.new("L", (160, 96), 0)
            ImageDraw.Draw(mask).rectangle(bbox, fill=255)
            obj.mask_path = str(masks / f"{obj.id}_mask.png")
            mask.save(obj.mask_path)
        scene_understanding._save_visible_rgba_observations(
            image_path=image,
            output_root=stage_output_root / "object_images",
            scene=scene,
        )
        return render_overlay(
            image_path=image,
            asset_masks=[(obj.id, obj.mask_path) for obj in scene.assets],
            output_path=masks / "asset_masks_with_ids.png",
        )

    monkeypatch.setattr(scene_understanding, "_segment_scene", segment_scene)
    scene, graph = scene_understanding.understand_scene(
        scene=Scene(),
        image_path=image,
        output_root=tmp_path / "output",
        vlm_client=VLM(),
        image_segmentation_client=object(),
        task_input=task_input,
    )
    assert [asset.id for asset in scene.assets] == expected_ids
    assert set(graph.node_by_id()) == {obj.id for obj in scene.objects}
    for asset in scene.assets:
        assert Path(asset.mask_path).name == f"{asset.id}_mask.png"
        assert Path(asset.visible_rgba_path).name == f"{asset.id}_rgba.png"
        assert Path(asset.mask_path).is_file()
        assert Path(asset.visible_rgba_path).is_file()
    report = tmp_path / "output" / "scene_understanding" / "task_entity_matches.json"
    if task_aware:
        assert json.loads(report.read_text())["entity_to_scene"] == {
            "evidence_001": "evidence_001",
            "cup_handle": "evidence_001",
            "evidence_002": None,
        }
        masked_report = report.with_name("task_entity_mask_matches.json")
        assert json.loads(masked_report.read_text())["entity_to_scene"] == {
            "evidence_001": "evidence_001",
            "cup_handle": "evidence_001",
            "evidence_002": None,
        }
        assert task_input is not None
        assert analysis_calls == [
            {
                "task_instruction": task_input.instruction,
                "task_entities": [
                    {
                        "id": "evidence_001",
                        "description": "blue cup",
                        "task_references": ["the cup on the left"],
                        "part_aliases": ["cup_handle"],
                    },
                    {
                        "id": "evidence_002",
                        "description": "missing block",
                        "task_references": [],
                    },
                ],
            }
        ]
    else:
        assert analysis_calls == []
        assert not report.exists()


@pytest.mark.parametrize(
    "invalid_case",
    ["unknown_id", "duplicate_id", "missing_id", "missing_hint", "empty_hint"],
)
def test_task_aware_analysis_rejects_invalid_identity_or_instance_hint(
    invalid_case: str,
) -> None:
    payload = json.loads(_response())
    payload["assets"][0].update(task_entity_id="e1", instance_hint="The left cup.")
    if invalid_case == "unknown_id":
        payload["assets"][0]["task_entity_id"] = "unknown"
    elif invalid_case == "duplicate_id":
        payload["assets"].append(dict(payload["assets"][0]))
    elif invalid_case == "missing_id":
        del payload["assets"][0]["task_entity_id"]
    elif invalid_case == "missing_hint":
        del payload["assets"][0]["instance_hint"]
    else:
        payload["assets"][0]["instance_hint"] = " "
    with pytest.raises(ValueError):
        scene_understanding._parse_image_object_analysis_response(
            json.dumps(payload),
            task_input=TaskSpecInput(
                "Pick up the left cup.", (TaskEntity("e1", "cup", True),)
            ),
        )


def test_unmatched_assets_reserve_all_task_entity_ids_when_allocating_ids() -> None:
    payload = json.loads(_response())
    payload["assets"][0].update(task_entity_id="cup_003", instance_hint="The left cup.")
    for hint in ("The middle cup.", "The right cup."):
        payload["assets"].append(
            {
                **payload["assets"][0],
                "task_entity_id": None,
                "instance_hint": hint,
            }
        )
    instance_hints: dict[str, str] = {}
    scene = scene_understanding._parse_image_object_analysis_response(
        json.dumps(payload),
        task_input=TaskSpecInput(
            "Pick up the left cup.",
            (
                TaskEntity("cup_001", "missing cup", False),
                TaskEntity("cup_003", "left cup", True),
            ),
        ),
        instance_hints=instance_hints,
    )
    assert [asset.id for asset in scene.assets] == ["cup_003", "cup_002", "cup_004"]
    assert [asset.is_articulated for asset in scene.assets] == [True, False, False]
    assert instance_hints == {
        "cup_003": "The left cup.",
        "cup_002": "The middle cup.",
        "cup_004": "The right cup.",
    }


def test_owner_and_part_alias_cannot_bind_two_assets() -> None:
    payload = json.loads(_response(asset_name="cabinet"))
    payload["assets"][0].update(task_entity_id="cabinet", instance_hint="The cabinet.")
    payload["assets"].append(
        {
            **payload["assets"][0],
            "task_entity_id": "drawer",
            "instance_hint": "Its front drawer.",
        }
    )
    task_input = TaskSpecInput(
        "Open the drawer.",
        (TaskEntity("cabinet", "Brown cabinet", True, aliases=("drawer",)),),
    )

    with pytest.raises(ValueError, match="cannot bind more than one scene asset"):
        scene_understanding._parse_image_object_analysis_response(
            json.dumps(payload), task_input=task_input
        )


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


def test_same_appearance_assets_keep_instance_hints_through_mask_assignment(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image)
    assets = [
        SceneObject(
            id=asset_id,
            kind="asset",
            category="cup",
            name="blue cup",
            description="A blue ceramic cup.",
        )
        for asset_id in ("evidence_001", "cup_001")
    ]
    mask_rles = []
    for bbox in ((28, 8, 43, 20), (5, 8, 19, 20)):
        mask = Image.new("L", (48, 32), 0)
        ImageDraw.Draw(mask).rectangle(bbox, fill=255)
        mask_rles.append(image_segmentation_utils._encode_binary_mask_rle(mask))
    hints = {
        "evidence_001": "The blue cup on the left.",
        "cup_001": "The blue cup on the right.",
    }

    class Segmentation:
        def segment_single_object(self, **_: object) -> list:
            return mask_rles

    class VLM:
        def complete(self, *, user_prompt: str, image_path: Path, **_: object) -> str:
            assert image_path.is_file()
            # The assignment input includes independent candidate cards, rather
            # than only the original-sized overlapping outlines.
            with Image.open(image_path) as selection_sheet:
                assert selection_sheet.height > 32
            assert "evidence_001" in user_prompt and "cup_001" in user_prompt
            for hint in hints.values():
                assert hint in user_prompt
            return json.dumps(
                {
                    "decisions": [
                        {
                            "asset_id": "evidence_001",
                            "status": "accept",
                            "mask_index": 2,
                            "reason": "The left blue cup.",
                        },
                        {
                            "asset_id": "cup_001",
                            "status": "accept",
                            "mask_index": 1,
                            "reason": "The right blue cup.",
                        },
                    ]
                }
            )

    scene_understanding._segment_assets(
        image,
        tmp_path,
        tmp_path,
        Scene(objects=assets),
        instance_hints=hints,
        vlm_client=VLM(),
        image_segmentation_client=Segmentation(),
    )
    assert Image.open(assets[0].mask_path).getbbox() == (5, 8, 20, 21)
    assert Image.open(assets[1].mask_path).getbbox() == (28, 8, 44, 21)
    assert all(asset.name == "blue cup" for asset in assets)
    assert all(asset.description == "A blue ceramic cup." for asset in assets)


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
            return _mask_decisions(("evidence_002", "accept", 1))

    scene_understanding._segment_assets(
        image,
        tmp_path,
        tmp_path,
        Scene(objects=[cup]),
        vlm_client=VLM(),
        image_segmentation_client=segmentation,
    )
    assert segmentation.prompts == ["blue cup", "cup"]
    assert cup.mask_path == str(tmp_path / "evidence_002_mask.png")
    assert Image.open(cup.mask_path).getbbox() == (8, 8, 25, 21)


@pytest.mark.parametrize("table_state", ["normal", "degraded"])
def test_table_visibility_states_route_critics_and_accept_the_surface_prompt(
    tmp_path: Path, table_state: str
) -> None:
    image_path = tmp_path / "scene.png"
    original = Image.new("RGB", (48, 32), "white")
    ImageDraw.Draw(original).rectangle((8, 8, 24, 20), fill=(220, 30, 40))
    original.save(image_path)
    validation_image_path = tmp_path / "assets_hidden.png"
    hidden_assets = original.copy()
    ImageDraw.Draw(hidden_assets).rectangle((8, 8, 24, 20), fill=(128, 128, 128))
    hidden_assets.save(validation_image_path)
    scene = scene_understanding._parse_image_object_analysis_response(
        _response(table_state=table_state)
    )
    segmentation = _PromptedSegmentation((48, 32), {"surface"})
    calls: list[str] = []

    class VLM:
        def complete(self, *, system_prompt: str, image_path: Path, **_: object) -> str:
            calls.append(system_prompt)
            # Cards must retain source RGB even when the overview has assets
            # hidden. Using the gray validation image for cards loses evidence.
            with Image.open(image_path) as critic_input:
                assert critic_input.width > 48
                colors = critic_input.convert("RGB").getcolors(
                    critic_input.width * critic_input.height
                )
                assert colors is not None
                assert (220, 30, 40) in {color for _, color in colors}
            return json.dumps(
                {
                    "selected_mask_index": 1,
                    "reason": "The mask covers the main support.",
                }
            )

    scene_understanding._segment_table(
        image_path,
        validation_image_path,
        Image.new("L", (48, 32), 0),
        tmp_path,
        tmp_path,
        scene,
        vlm_client=VLM(),  # type: ignore[arg-type]
        image_segmentation_client=segmentation,  # type: ignore[arg-type]
    )

    expected_critic = (
        scene_understanding._NORMAL_TABLE_VALIDATION_SYSTEM_PROMPT
        if table_state == "normal"
        else scene_understanding._DEGRADED_TABLE_VALIDATION_SYSTEM_PROMPT
    )
    assert calls == [expected_critic]
    assert segmentation.prompts[-1] == "surface"
    assert scene.table is not None and scene.table.mask_path is not None
    assert scene.table.table_state == table_state
    assert Image.open(scene.table.mask_path).getbbox() == (8, 8, 25, 21)
    assert json.loads((tmp_path / "table_validation_surface.json").read_text()) == {
        "table_state": table_state,
        "selected_mask_index": 1,
        "reason": "The mask covers the main support.",
    }


def test_rejected_table_mask_keeps_classification_and_tries_the_next_prompt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    scene = scene_understanding._parse_image_object_analysis_response(
        _response(table_state="degraded")
    )
    segmentation = _PromptedSegmentation((48, 32), {"wooden table", "surface"})
    responses = iter(
        [
            {
                "selected_mask_index": None,
                "reason": "This mask only covers a small lid.",
            },
            {"selected_mask_index": 1, "reason": "This mask covers the main tabletop."},
        ]
    )

    class VLM:
        def complete(self, *, system_prompt: str, **_: object) -> str:
            assert (
                system_prompt
                == scene_understanding._DEGRADED_TABLE_VALIDATION_SYSTEM_PROMPT
            )
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
    assert scene.table.table_state == "degraded"
    assert json.loads((tmp_path / "table_validation_name.json").read_text()) == {
        "table_state": "degraded",
        "selected_mask_index": None,
        "reason": "This mask only covers a small lid.",
    }


def test_rejected_table_masks_report_segmentation_failure_not_absent_support(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    scene = scene_understanding._parse_image_object_analysis_response(
        _response(table_state="degraded")
    )
    segmentation = _PromptedSegmentation((48, 32), {"wooden table", "surface"})

    class VLM:
        def complete(self, **_: object) -> str:
            return json.dumps(
                {
                    "selected_mask_index": None,
                    "reason": "Only a small lid was segmented.",
                }
            )

    with pytest.raises(ValueError, match="Support surface exists but no"):
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
    assert scene.table is not None
    assert scene.table.table_state == "degraded"
    assert scene.table.mask_path is None


def test_table_segmentation_requires_classification_from_image_analysis(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image_path)
    # Legacy scene objects may lack classification; fresh image segmentation
    # must obtain it from image analysis instead of guessing a critic branch.
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="wooden table",
                description="A rectangular wooden table.",
            )
        ]
    )
    segmentation = _PromptedSegmentation((48, 32), {"surface"})
    with pytest.raises(ValueError, match="table_state"):
        scene_understanding._segment_table(
            image_path,
            image_path,
            Image.new("L", (48, 32), 0),
            tmp_path,
            tmp_path,
            scene,
            vlm_client=object(),  # type: ignore[arg-type]
            image_segmentation_client=segmentation,  # type: ignore[arg-type]
        )
    assert segmentation.prompts == []


def test_table_critic_rejects_a_mask_without_reclassifying_the_scene() -> None:
    validation = scene_understanding._parse_table_validation_response(
        '{"selected_mask_index":null,"reason":"Candidates cover small objects only."}',
        [],
    )
    assert validation.selected_mask_index is None
    assert validation.reason == "Candidates cover small objects only."


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {"table_state": None, "selected_mask_index": None, "reason": "No table."},
            "exactly",
        ),
        ({"selected_mask_index": None}, "exactly"),
        ({"selected_mask_index": True, "reason": "A table."}, "integer"),
        ({"selected_mask_index": None, "reason": "  "}, "reason"),
        ({"selected_mask_index": 1, "reason": "A table."}, "nonexistent"),
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
        def complete(self, *, system_prompt: str, user_prompt: str, **_: object) -> str:
            if (
                system_prompt
                == scene_understanding._VISUAL_SEGMENTATION_PROMPT_SYSTEM_PROMPT
            ):
                return json.dumps({"prompts": ["tool", "tool with black handle"]})
            if not json.loads(user_prompt)["candidate_indices"]:
                return _mask_decisions(("tool_001", "retry", None))
            return _mask_decisions(("tool_001", "accept", 1))

    scene_understanding._segment_assets(
        image_path,
        tmp_path,
        tmp_path,
        scene,
        vlm_client=VLM(),  # type: ignore[arg-type]
        image_segmentation_client=segmentation,  # type: ignore[arg-type]
    )

    assert segmentation.prompts == ["hammer", "tool", "tool with black handle"]
    assert hammer.mask_path is not None


def test_asset_mask_id_overlay_includes_table_and_assets(tmp_path: Path) -> None:
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
        asset_masks=[("table", table_mask_path), ("bottle_001", asset_mask_path)],
        output_path=output_path,
    )

    with Image.open(rendered_path) as overlay:
        assert overlay.getpixel((8, 40)) != (0, 0, 0)
        # Width adapts to preview resolution; the known boundary stays visible.
        assert overlay.crop((376, 178, 381, 183)).getbbox() is not None


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


def _mask_decisions(*items: tuple[str, str, int | None]) -> str:
    return json.dumps(
        {
            "decisions": [
                {
                    "asset_id": asset_id,
                    "status": status,
                    "mask_index": index,
                    "reason": (
                        "The visible target needs a better mask."
                        if status == "retry"
                        else (
                            "The target is severely occluded inside the container."
                            if status == "skip"
                            else "The mask covers the intended visible object."
                        )
                    ),
                }
                for asset_id, status, index in items
            ]
        }
    )


def _rect_rle(size: tuple[int, int], box: tuple[int, int, int, int]) -> dict:
    mask = Image.new("L", size)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return image_segmentation_utils._encode_binary_mask_rle(mask)


def test_count_sufficient_wrong_candidate_is_repaired_and_accepted_pixels_stay_locked(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scene.png"
    size = (80, 40)
    Image.new("RGB", size, "white").save(image)
    assets = [
        SceneObject(
            id=key,
            kind="asset",
            category="cup",
            name="blue cup",
            description="Blue cup.",
        )
        for key in ("left", "right")
    ]
    left = _rect_rle(size, (5, 10, 18, 29))
    wrong = _rect_rle(size, (30, 10, 40, 29))
    # A repair result overlaps the approved left cup and would enlarge it if
    # accepted masks were re-unioned. It must never replace the saved pixels.
    enlarged_left = _rect_rle(size, (4, 10, 19, 29))
    right = _rect_rle(size, (55, 10, 69, 29))
    queries = []
    critic_calls = 0

    class Segmentation:
        def segment_single_object(self, *, prompt: str, **_: object) -> list:
            queries.append(prompt)
            return [left, wrong] if prompt == "blue cup" else [enlarged_left, right]

    class VLM:
        def complete(self, *, system_prompt: str, user_prompt: str, **_: object) -> str:
            nonlocal critic_calls
            payload = json.loads(user_prompt)
            if (
                system_prompt
                == scene_understanding._VISUAL_SEGMENTATION_PROMPT_SYSTEM_PROMPT
            ):
                assert [asset["id"] for asset in payload["unresolved_assets"]] == [
                    "right"
                ]
                assert payload["unresolved_assets"][0]["critic_feedback"]
                # The approved left mask no longer counts toward the repair pool.
                assert payload["remaining_candidate_count"] == 1
                return '{"prompts":["blue cup on the right"]}'
            critic_calls += 1
            if critic_calls == 1:
                assert len(payload["candidate_indices"]) == 2
                return _mask_decisions(("left", "accept", 1), ("right", "retry", None))
            assert payload["locked_asset_ids"] == ["left"]
            assert [asset["id"] for asset in payload["unresolved_assets"]] == ["right"]
            # Wrong middle candidate remains 1; the approved left cup is gone.
            return _mask_decisions(("right", "accept", 2))

    scene_understanding._segment_assets(
        image,
        tmp_path,
        tmp_path,
        Scene(objects=assets),
        vlm_client=VLM(),
        image_segmentation_client=Segmentation(),
    )
    assert queries == ["blue cup", "blue cup on the right"]
    with Image.open(assets[0].mask_path) as actual:
        assert (
            actual.tobytes() == image_segmentation_utils.decode_rle_mask(left).tobytes()
        )
    with Image.open(assets[1].mask_path) as actual:
        assert (
            actual.tobytes()
            == image_segmentation_utils.decode_rle_mask(right).tobytes()
        )


@pytest.mark.parametrize("skip", [False, True])
def test_no_candidates_needs_visibility_decision_and_has_bounded_repairs(
    tmp_path: Path,
    skip: bool,
) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image)
    asset = SceneObject(
        id="hidden",
        kind="asset",
        category="ball",
        name="red ball",
        description="Red ball.",
    )
    scene = Scene(objects=[asset])
    queries = []
    critics = []

    class Segmentation:
        def segment_single_object(self, *, prompt: str, **_: object) -> list:
            queries.append(prompt)
            return []

    class VLM:
        def complete(self, *, system_prompt: str, user_prompt: str, **_: object) -> str:
            if (
                system_prompt
                == scene_understanding._VISUAL_SEGMENTATION_PROMPT_SYSTEM_PROMPT
            ):
                assert json.loads(user_prompt)["remaining_candidate_count"] == 0
                return '{"prompts":["small red sphere"]}'
            assert json.loads(user_prompt)["candidate_indices"] == []
            critics.append(1)
            return _mask_decisions(("hidden", "skip" if skip else "retry", None))

    def segment() -> None:
        scene_understanding._segment_assets(
            image,
            tmp_path,
            tmp_path,
            scene,
            vlm_client=VLM(),
            image_segmentation_client=Segmentation(),
            max_repair_rounds=1,
        )

    if skip:
        segment()
        assert scene.assets == []
        assert len(critics) == 1
    else:
        with pytest.raises(
            ValueError, match="Unresolved visible assets after 1 repair"
        ):
            segment()
        assert scene.assets == [asset]
        assert len(critics) == 2
    audit = json.loads((tmp_path / "asset_segmentation.json").read_text())
    assert ("hidden" in audit["skipped"]) is skip
    assert queries == (
        ["red ball", "ball"] if skip else ["red ball", "ball", "small red sphere"]
    )


@pytest.mark.parametrize(
    "error", ["missing", "duplicate", "reuse", "boolean", "retry_index", "status"]
)
def test_mask_decisions_reject_incomplete_or_conflicting_assignments(
    error: str,
) -> None:
    assets = [
        SceneObject(
            id=key, kind="asset", category="cup", name="cup", description="Cup."
        )
        for key in ("a", "b")
    ]
    candidates = image_segmentation_utils.build_mask_candidates(
        [_rect_rle((40, 30), box) for box in ((2, 5, 10, 20), (20, 5, 28, 20))]
    )
    payload = json.loads(_mask_decisions(("a", "accept", 1), ("b", "accept", 2)))
    values = payload["decisions"]
    if error == "missing":
        values.pop()
    elif error == "duplicate":
        values[1]["asset_id"] = "a"
    elif error == "reuse":
        values[1]["mask_index"] = 1
    elif error == "boolean":
        values[0]["mask_index"] = True
    elif error == "retry_index":
        values[0]["status"] = "retry"
    else:
        values[0]["status"] = "unknown"
    with pytest.raises(ValueError):
        scene_understanding._parse_asset_assignment_response(
            json.dumps(payload), assets, candidates
        )


def test_occluded_task_entity_keeps_analysis_binding_but_not_final_mask_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (48, 32), "white").save(image)
    payload = json.loads(_response())
    payload["assets"][0].update(
        task_entity_id="e1", instance_hint="Inside the opaque container."
    )
    task_input = TaskSpecInput(
        instruction="Pick up the cup.", entities=(TaskEntity("e1", "cup", False),)
    )

    class VLM:
        def complete(self, *, system_prompt: str, **_: object) -> str:
            if system_prompt.startswith(scene_understanding._SYSTEM_PROMPT):
                return json.dumps(payload)
            if system_prompt == scene_understanding._ASSET_ASSIGNMENT_SYSTEM_PROMPT:
                return _mask_decisions(("e1", "skip", None))
            return '{"nodes":[]}'

    class Segmentation:
        def segment_single_object(self, **_: object) -> list:
            return []

    def segment_table(*, scene: Scene, masks_output_root: Path, **_: object) -> None:
        scene.table.mask_path = str(masks_output_root / "table_mask.png")
        Image.new("L", (48, 32), 255).save(scene.table.mask_path)

    monkeypatch.setattr(scene_understanding, "_segment_table", segment_table)
    scene, graph = scene_understanding.understand_scene(
        Scene(),
        image,
        tmp_path / "out",
        vlm_client=VLM(),
        image_segmentation_client=Segmentation(),
        task_input=task_input,
    )
    stage = tmp_path / "out" / "scene_understanding"
    assert scene.assets == []
    assert set(graph.node_by_id()) == {"table"}
    assert json.loads((stage / "task_entity_matches.json").read_text())[
        "entity_to_scene"
    ] == {"e1": "e1"}
    assert json.loads((stage / "task_entity_mask_matches.json").read_text())[
        "entity_to_scene"
    ] == {"e1": None}
    assert (
        "e1"
        in json.loads((stage / "debug/asset_segmentation.json").read_text())["skipped"]
    )
