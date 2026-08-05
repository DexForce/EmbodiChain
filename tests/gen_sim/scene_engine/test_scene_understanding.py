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

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.pipeline import scene_understanding


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


def test_image_object_analysis_rejects_location_words_in_object_names() -> None:
    with pytest.raises(ValueError, match="must not contain location"):
        scene_understanding._parse_image_object_analysis_response(
            _response(asset_name="left cup")
        )


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
