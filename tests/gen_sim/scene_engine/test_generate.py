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

import pytest

from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.pipeline import generate


class _Client:
    def __init__(self) -> None:
        self.closed = False

    def check_health(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_segmentation_client_closes_when_segmentation_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmentation_client = _Client()

    class FakeVLM:
        @classmethod
        def from_config(cls, _config_path: object) -> object:
            return object()

    class FakeSegmentationClient:
        @classmethod
        def from_config(cls, _config_path: object) -> _Client:
            return segmentation_client

    monkeypatch.setattr(generate, "OpenAICompatibleVLM", FakeVLM)
    monkeypatch.setattr(generate, "ImageSegmentationClient", FakeSegmentationClient)
    monkeypatch.setattr(generate, "understand_scene", lambda **_: Scene())

    def fail_segment_scene(**_: object) -> Scene:
        raise RuntimeError("segmentation failed")

    monkeypatch.setattr(generate, "segment_scene", fail_segment_scene)

    with pytest.raises(RuntimeError, match="segmentation failed"):
        generate.generate_scene_from_image(tmp_path / "image.png", tmp_path / "output")

    assert segmentation_client.closed is True


def test_geometry_client_closes_when_refinement_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmentation_client = _Client()
    geometry_client = _Client()

    class FakeVLM:
        @classmethod
        def from_config(cls, _config_path: object) -> object:
            return object()

    class FakeSegmentationClient:
        @classmethod
        def from_config(cls, _config_path: object) -> _Client:
            return segmentation_client

    class FakeGeometryClient:
        @classmethod
        def from_config(cls, _config_path: object) -> _Client:
            return geometry_client

    monkeypatch.setattr(generate, "OpenAICompatibleVLM", FakeVLM)
    monkeypatch.setattr(generate, "ImageSegmentationClient", FakeSegmentationClient)
    monkeypatch.setattr(generate, "GeometryGenerationClient", FakeGeometryClient)
    monkeypatch.setattr(generate, "understand_scene", lambda **_: Scene())
    monkeypatch.setattr(generate, "segment_scene", lambda **kwargs: kwargs["scene"])

    def fail_generate_scene_and_refine(**_: object) -> Scene:
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(
        generate, "generate_scene_and_refine", fail_generate_scene_and_refine
    )

    with pytest.raises(RuntimeError, match="refinement failed"):
        generate.generate_scene_from_image(tmp_path / "image.png", tmp_path / "output")

    assert segmentation_client.closed is True
    assert geometry_client.closed is True
