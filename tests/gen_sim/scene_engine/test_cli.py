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

from embodichain.gen_sim.scene_engine.cli import preview, start


def test_cli_scene_engine_creates_output_and_forwards_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png")
    config_path = tmp_path / "scene_engine_config.json"
    config_path.write_text("{}", encoding="utf-8")
    output_root = tmp_path / "generated"
    received: dict[str, object] = {}

    def fake_generate_scene_from_image(**kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(
        start, "generate_scene_from_image", fake_generate_scene_from_image
    )

    start.cli_scene_engine(
        image=image_path,
        output_root=output_root,
        config_path=config_path,
    )

    assert output_root.is_dir()
    assert received["image_path"] == image_path.resolve()
    assert received["output_root"] == output_root.resolve()
    assert received["llm_config_path"] == config_path
    assert received["image_segmentation_config_path"] == config_path
    assert received["geometry_generation_config_path"] == config_path


def test_cli_scene_engine_rejects_non_image_input(tmp_path: Path) -> None:
    text_path = tmp_path / "scene.txt"
    text_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="extensions"):
        start.cli_scene_engine(text_path, tmp_path / "output")


def test_main_forwards_parsed_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_cli_scene_engine(
        image: str | Path,
        output_root: str | Path,
        *,
        config_path: str | Path | None,
    ) -> None:
        received["image"] = image
        received["output_root"] = output_root
        received["config_path"] = config_path

    monkeypatch.setattr(start, "cli_scene_engine", fake_cli_scene_engine)

    start.main(
        [
            "--image",
            "input.png",
            "--output_root",
            "output",
            "--config",
            "services.json",
        ]
    )

    assert received == {
        "image": "input.png",
        "output_root": "output",
        "config_path": Path("services.json"),
    }


def test_preview_main_forwards_output_root_and_viser_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def fake_preview_scene_export(**kwargs: object) -> None:
        received.update(kwargs)

    monkeypatch.setattr(preview, "preview_scene_export", fake_preview_scene_export)

    preview.main(
        [
            "--output_root",
            "output",
            "--viser",
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            "9000",
        ]
    )

    visualization = received["visualization"]
    assert received["output_root"] == Path("output")
    assert received["device"] == "cpu"
    assert received["headless"] is False
    assert visualization.backend == "viser"
    assert visualization.viser_server.host == "0.0.0.0"
    assert visualization.viser_server.port == 9000
