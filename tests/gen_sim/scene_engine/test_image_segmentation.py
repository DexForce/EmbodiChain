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
from typing import Any

import pytest

from embodichain.gen_sim.scene_engine.clients.image_segmentation import (
    ImageSegmentationClient,
    _extract_rle_masks,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _Session:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.prompt: str | None = None

    def post(self, _url: str, *, data: dict[str, str], **_: object) -> _Response:
        self.prompt = data["prompt"]
        return _Response(self._payload)

    def close(self) -> None:
        return None


def test_extract_rle_masks_accepts_instances_response() -> None:
    mask = {"counts": [1, 2], "size": [2, 2]}

    masks = _extract_rle_masks({"result": {"instances": [{"mask_rle": mask}]}})

    assert masks == [mask]


def test_segment_single_object_strips_prompt(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    mask = {"counts": [4], "size": [2, 2]}
    session = _Session({"ok": True, "result": {"masks": [mask]}})
    client = ImageSegmentationClient(
        base_url="http://segmentation.test",
        timeout_s=1,
        max_attempts=1,
        health_path="/health",
        segment_single_object_path="/segment",
        session=session,
    )

    masks = client.segment_single_object(image_path=image_path, prompt="  table  ")

    assert session.prompt == "table"
    assert masks == [mask]


def test_segment_single_object_rejects_empty_prompt(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    client = ImageSegmentationClient(
        base_url="http://segmentation.test",
        timeout_s=1,
        max_attempts=1,
        health_path="/health",
        segment_single_object_path="/segment",
        session=_Session({"ok": True, "result": {"masks": []}}),
    )

    with pytest.raises(ValueError, match="prompt"):
        client.segment_single_object(image_path=image_path, prompt="   ")
